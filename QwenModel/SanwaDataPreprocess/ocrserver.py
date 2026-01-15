import sys
import time
import json
import csv
import shutil
import cv2
import ollama
import os
import numpy as np
import concurrent.futures
import threading
from datetime import datetime, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= SERVER CONFIGURATION =================

# 🔴 CHANGE THIS to your actual folder path on the Ubuntu Server
# Example: "/home/ubuntu/sanwa_data" or "/mnt/data/sanwa"
SERVER_ROOT = Path("/home/ubuntu/sanwa_project") 

SOURCE_DIR = SERVER_ROOT / "input_images"
OUTPUT_DIR = SERVER_ROOT / "output_results"
DEBUG_DIR  = SERVER_ROOT / "debug_crops"

OLLAMA_MODEL = "qwen2.5vl:3b"
ROI_JSON = Path("roi.json")

ROI_PAD = 2         
UPSCALE = 2.0       
DARKNESS_THRESHOLD = 15  

# 🚀 GPU ACCELERATION SETTING
# Since you have V100s, we can process multiple crops simultaneously.
# Try 4 or 8. If you get out-of-memory errors, lower to 2.
MAX_WORKERS = 4 

CSV_GROUPS = {
    "CslotCam4result.csv":           list(range(1, 12)),
    "cam 6 snap1 Latchresult.csv":   list(range(12, 20)),
    "cam 6 snap2 nozzleresult.csv":  list(range(20, 31)),
    "terminal result.csv":           list(range(31, 51))
}
# =================================================

# Lock to prevent multiple threads writing to CSV/Console at the exact same time
print_lock = threading.Lock()

class GPUHandler(FileSystemEventHandler):
    def __init__(self, rois):
        self.rois = rois

    def on_created(self, event):
        if not event.is_directory: self.process_new_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory: self.process_new_file(Path(event.dest_path))

    def process_new_file(self, file_path: Path):
        if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}: return
        if file_path.name.startswith("."): return

        print(f"\n⚡ Processing New File: {file_path.name}")
        
        try:
            relative_path = file_path.relative_to(SOURCE_DIR)
        except ValueError:
            relative_path = Path(file_path.name)

        relative_parent = relative_path.parent
        
        # Create folder for this image's results
        image_folder_name = file_path.stem
        target_image_folder = DEBUG_DIR / relative_parent / image_folder_name
        target_image_folder.mkdir(parents=True, exist_ok=True)

        self.run_parallel_pipeline(file_path, target_image_folder, relative_parent)

    def parse_filename_time(self, filename):
        try:
            name_only = filename.rsplit('.', 1)[0]
            # Adjust format if your Linux filenames are different
            dt = datetime.strptime(name_only, "%Y-%m-%d %H.%M.%S")
            return dt.isoformat() + "Z"
        except: return filename

    def parse_machine_time(self, text_str):
        if not text_str or len(text_str) < 5 or "NA" in text_str: return ""
        try:
            clean = text_str.replace("\n", " ").replace("|", "/").strip()
            dt_local = datetime.strptime(clean, "%b/%d/%y %H:%M:%S")
            dt_utc = dt_local - timedelta(hours=8)
            return dt_utc.isoformat() + "Z"
        except: return text_str 

    def is_image_too_dark(self, img):
        if img is None or img.size == 0: return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < DARKNESS_THRESHOLD

    def process_single_roi(self, args):
        """
        Worker function to be run in parallel.
        args: (roi_data, img, save_dir)
        """
        name, x, y, w, h, img, save_dir = args
        H, W = img.shape[:2]

        # 1. Bounds Check
        if x >= W or y >= H: return name, "NA"

        # 2. Crop
        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0: return name, "NA"

        # 3. Upscale
        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        # 4. Save Crop
        crop_filename = save_dir / f"ROI_{name}.jpg"
        cv2.imwrite(str(crop_filename), crop)

        # 5. Check Darkness
        if self.is_image_too_dark(crop):
            with print_lock:
                print("D", end="", flush=True)
            return name, "NA"

        # 6. Ask AI (Ollama)
        text_val = self.ask_ollama_single(crop_filename)
        
        # Save TXT
        try:
            with open(save_dir / f"ROI_{name}.txt", "w", encoding="utf-8") as f:
                f.write(text_val)
        except: pass

        with print_lock:
            # Print real value for Timestamps, dot for others
            if name in ["51", "52"]:
                print(f" [ID {name}: {text_val}] ", end="", flush=True)
            else:
                print(".", end="", flush=True)

        return name, text_val

    def run_parallel_pipeline(self, img_path, save_dir, relative_parent):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Cannot read image: {img_path}")
            return

        # --- Debug Map ---
        try:
            vis_img = img.copy()
            for name, x, y, w, h in self.rois:
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis_img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(save_dir / "_DEBUG_MAP.jpg"), vis_img)
        except: pass

        print(f"  --> Processing {len(self.rois)} ROIs with {MAX_WORKERS} GPU threads...")
        start_t = time.time()
        
        collected_results = {}
        
        # Prepare arguments for parallel execution
        tasks = []
        for name, x, y, w, h in self.rois:
            tasks.append((name, x, y, w, h, img, save_dir))

        # --- PARALLEL EXECUTION ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map returns results in the order they were started
            results = executor.map(self.process_single_roi, tasks)
            
            for name, text_val in results:
                collected_results[name] = text_val

        duration = time.time() - start_t
        print(f"\n  --> Finished in {duration:.1f}s")

        # Save JSON
        try:
            with open(save_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=2)
        except: pass

        # Metadata
        filename_utc = self.parse_filename_time(img_path.name)
        raw_machine_time = collected_results.get("51", "")
        if not raw_machine_time or raw_machine_time == "NA": 
            raw_machine_time = collected_results.get("52", "")
        
        calc_machine_utc = self.parse_machine_time(raw_machine_time)

        # Write CSV
        for csv_name, id_range in CSV_GROUPS.items():
            self.append_to_summary_csv(
                csv_name, 
                id_range, 
                collected_results, 
                img_path.name, 
                filename_utc, 
                raw_machine_time, 
                calc_machine_utc,
                relative_parent
            )

    def ask_ollama_single(self, image_path):
        # On server, this hits localhost:11434, which manages the GPU
        prompt = "Read the text in this image. Return ONLY the value. No extra words."
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}],
                options={'num_predict': 20} 
            )
            raw = response['message']['content'].strip()
            clean = raw.replace("`", "").replace('"', '').replace("'", "")
            if not clean: return "NA"
            return clean
        except Exception as e:
            # Don't print full error stack to avoid console spam in parallel mode
            return "NA"

    def append_to_summary_csv(self, csv_name, id_list, results_dict, filename, file_utc, raw_mach, calc_mach, relative_parent):
        target_folder = OUTPUT_DIR / relative_parent / "CSV_Results"
        target_folder.mkdir(parents=True, exist_ok=True)
        csv_path = target_folder / csv_name
        
        header = ["Filename", "File_UTC", "Machine_Text", "Machine_UTC"]
        target_ids = []
        for i in id_list:
            sid = str(i)
            target_ids.append(sid)
            header.append(f"ROI_{sid}")
        
        for extra in ["51", "52"]:
            if extra not in target_ids:
                target_ids.append(extra)
                header.append(f"ROI_{extra}")

        row = [filename, file_utc, raw_mach, calc_mach]
        for tid in target_ids:
            val = results_dict.get(tid, "NA").replace("\n", " ").replace(",", ".")
            row.append(val)

        # Lock file writing to be safe
        with print_lock:
            try:
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists: writer.writerow(header)
                    writer.writerow(row)
            except Exception as e:
                print(f"  ❌ CSV Write Error: {e}")

# =================================================

def load_rois(roi_path: Path):
    if not roi_path.exists(): return []
    try:
        with open(roi_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rois = []
        data_list = data if isinstance(data, list) else [data]
        for idx, item in enumerate(data_list):
            name = item.get("name", str(idx))
            rois.append((str(name), int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])))
        return rois
    except: return []

def main():
    # Basic Server Checks
    if not SERVER_ROOT.exists():
        print(f"❌ Error: SERVER_ROOT directory does not exist: {SERVER_ROOT}")
        print("   Please create it or update the path in the script.")
        return
        
    if not SOURCE_DIR.exists():
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    
    rois = load_rois(ROI_JSON)
    if not rois:
        print("❌ roi.json missing (Put it in the same folder as this script)")
        return

    print("========================================")
    print(f"🚀 GPU Server Monitor Started")
    print(f"   Model: {OLLAMA_MODEL}")
    print(f"   Workers: {MAX_WORKERS} (Parallel Processing)")
    print(f"📂 Watch Folder: {SOURCE_DIR}")
    print("========================================")

    handler = GPUHandler(rois)

    # 1. SCAN EXISTING FILES
    print("Scanning directory tree...")
    all_files = list(SOURCE_DIR.rglob("*"))
    
    image_files = [
        f for f in all_files 
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} 
        and not f.name.startswith(".")
    ]
    image_files.sort()
    
    total = len(image_files)
    print(f"Found {total} images. Starting batch...")

    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{total}]", end=" ")
        try:
            handler.process_new_file(img_path)
        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            return

    print("\n✅ Batch done. Monitoring for NEW files...")
    
    observer = Observer()
    observer.schedule(handler, str(SOURCE_DIR), recursive=True)
    observer.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()