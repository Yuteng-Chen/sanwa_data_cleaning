import sys
import time
import json
import csv
import shutil
import cv2
import ollama
import os
import re  # <--- NEW: Added regex for cleaning
import numpy as np
import concurrent.futures
import threading
from datetime import datetime, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= SERVER CONFIGURATION =================

SERVER_ROOT = Path("/home/wanfangyuan/Desktop/share01/Wan_Fangyuan/Sanwa/Sanwadata27Dec/")
SOURCE_DIR = SERVER_ROOT / "12_16_2025/2025-12-16"
OUTPUT_DIR = SERVER_ROOT / "sanwa_ocr_output/2025-12-16_shortmould/output_results"
DEBUG_DIR  = SERVER_ROOT / "sanwa_ocr_debug/2025-12-16_shortmould/debug_crops"

# 🔴 OLLAMA CONFIGURATION
OLLAMA_MODEL = "qwen2.5vl:3b"

ROI_JSON = Path("roi_shortmould.json")

ROI_PAD = 2         
UPSCALE = 2.0       
DARKNESS_THRESHOLD = 15  

# 🚀 GPU ACCELERATION SETTING
MAX_WORKERS = 4 

# 📊 CSV GROUP DEFINITIONS
CSV_GROUPS = {
    "Cam1_NozzleDefect.csv":        list(range(0, 8)),
    "Cam2Snap1_GoldenCheck.csv":    list(range(8, 21)) + [56, 57],
    "Cam2Snap2_TerminalGap.csv":    list(range(58, 112)), 
    "Cam2Snap3_ShortMould.csv":     list(range(114, 117)),
    "Cam3_TerminalStatus.csv":      list(range(21, 56))
}

# IDs used for Machine Timestamp
TIMESTAMP_IDS = ["117", "118"]

# =================================================

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
        
        # Create folder for this image's debug crops
        image_folder_name = file_path.stem
        target_image_folder = DEBUG_DIR / relative_parent / image_folder_name
        target_image_folder.mkdir(parents=True, exist_ok=True)

        self.run_parallel_pipeline(file_path, target_image_folder, relative_parent)

    def parse_filename_time(self, filename):
        """Extract timestamp from filename (e.g. 2025-12-16 14.30.05.jpg)"""
        try:
            name_only = filename.rsplit('.', 1)[0]
            dt = datetime.strptime(name_only, "%Y-%m-%d %H.%M.%S")
            return dt.isoformat()
        except: return "NA"

    def clean_machine_time(self, t1, t2):
        """Combine ID 117 and 118 and clean up"""
        raw = f"{t1} {t2}".strip()
        # Remove any lingering tags or mess
        clean = re.sub(r'<[^>]+>', '', raw)
        clean = clean.replace("\n", " ").replace("|", "/").replace("'", "").replace('"', "")
        return clean.strip()

    def is_image_too_dark(self, img):
        if img is None or img.size == 0: return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < DARKNESS_THRESHOLD

    def process_single_roi(self, args):
        name, x, y, w, h, img, save_dir = args
        H, W = img.shape[:2]

        if x >= W or y >= H: return name, "NA"

        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0: return name, "NA"

        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        crop_filename = save_dir / f"ROI_{name}.jpg"
        cv2.imwrite(str(crop_filename), crop)

        if self.is_image_too_dark(crop):
            with print_lock: print("D", end="", flush=True)
            return name, "NA"

        # Ask AI with new robust logic
        text_val = self.ask_ollama_single(crop_filename)
        
        try:
            with open(save_dir / f"ROI_{name}.txt", "w", encoding="utf-8") as f:
                f.write(text_val)
        except: pass

        with print_lock:
            if name in TIMESTAMP_IDS:
                print(f" [Time {name}: {text_val}] ", end="", flush=True)
            else:
                print(".", end="", flush=True)

        return name, text_val

    def run_parallel_pipeline(self, img_path, save_dir, relative_parent):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Cannot read image: {img_path}")
            return

        try:
            vis_img = img.copy()
            for name, x, y, w, h in self.rois:
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imwrite(str(save_dir / "_DEBUG_MAP.jpg"), vis_img)
        except: pass

        print(f"  --> Processing {len(self.rois)} ROIs with {MAX_WORKERS} GPU threads...")
        start_t = time.time()
        
        collected_results = {}
        
        tasks = []
        for name, x, y, w, h in self.rois:
            tasks.append((name, x, y, w, h, img, save_dir))

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(self.process_single_roi, tasks)
            for name, text_val in results:
                collected_results[name] = text_val

        duration = time.time() - start_t
        print(f"\n  --> Finished in {duration:.1f}s")

        try:
            with open(save_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=2)
        except: pass

        file_ts = self.parse_filename_time(img_path.name)
        t117 = collected_results.get("117", "")
        t118 = collected_results.get("118", "")
        machine_ts = self.clean_machine_time(t117, t118)

        for csv_name, id_range in CSV_GROUPS.items():
            self.append_to_summary_csv(
                csv_name, id_range, collected_results, img_path.name, machine_ts, file_ts, relative_parent
            )

    # ---------------------------------------------------------
    # 🌟 NEW CLEANING FUNCTION
    # ---------------------------------------------------------
    def clean_ocr_text(self, text):
        if not text: return "NA"
        
        # 1. Remove HTML tags like <img> <ref> etc.
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. Remove markdown code blocks
        text = text.replace("```", "").replace("`", "").strip()

        # 3. FIX REPEATED NUMBERS (The "6.05.051" issue)
        # This regex looks for patterns where a decimal part repeats immediately
        # E.g. .05.05 -> .05
        # This is a safe approximation for common stuttering
        text = re.sub(r'(\.\d+)\1', r'\1', text)

        # 4. Remove common hallucinated phrases
        text = text.replace("image", "").replace("crop", "").strip()

        return text

    def ask_ollama_single(self, image_path):
        # Improved Prompt
        prompt = "Read the text or number in this image. Return ONLY the value. Do not repeat text. Do not output HTML."
        
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}],
                # 🌟 CRITICAL SETTINGS FOR ACCURACY 🌟
                options={
                    'temperature': 0.0,    # Makes output deterministic (no creativity)
                    'repeat_penalty': 1.2, # Penalizes the model if it repeats "6.05.05"
                    'num_predict': 20,     # Keep it short
                    'top_k': 10            # Focus on most likely tokens
                } 
            )
            raw = response['message']['content']
            
            # Apply our new cleaning function
            clean = self.clean_ocr_text(raw)
            
            if not clean: return "NA"
            return clean
        except Exception:
            return "NA"

    def append_to_summary_csv(self, csv_name, id_list, results_dict, filename, machine_ts, file_ts, relative_parent):
        target_folder = OUTPUT_DIR / relative_parent / "CSV_Results"
        target_folder.mkdir(parents=True, exist_ok=True)
        csv_path = target_folder / csv_name
        
        header = ["Machine_Timestamp", "File_Timestamp", "Filename"]
        target_ids = []
        for i in id_list:
            sid = str(i)
            target_ids.append(sid)
            header.append(f"ROI_{sid}")
        
        row = [machine_ts, file_ts, filename]
        for tid in target_ids:
            val = results_dict.get(tid, "NA").replace("\n", " ").replace(",", ".")
            row.append(val)

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
    if not SERVER_ROOT.exists():
        print(f"❌ Error: SERVER_ROOT directory does not exist: {SERVER_ROOT}")
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
    print(f"   Workers: {MAX_WORKERS}")
    print(f"📂 Watch Folder: {SOURCE_DIR}")
    print("========================================")

    handler = GPUHandler(rois)

    print("Scanning directory tree...")
    all_files = list(SOURCE_DIR.rglob("*"))
    image_files = [f for f in all_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and not f.name.startswith(".")]
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