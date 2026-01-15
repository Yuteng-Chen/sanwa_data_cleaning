import sys
import time
import json
import csv
import shutil
import cv2
import ollama
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= CONFIGURATION =================
# Update this path to your OneDrive
ONEDRIVE_ROOT = Path("/Users/pomvrp/Library/CloudStorage/OneDrive-AgencyforScience,TechnologyandResearch")

SOURCE_DIR = ONEDRIVE_ROOT / "Sanwa data/12_16_cslot/2025-12-16"
OUTPUT_DIR = ONEDRIVE_ROOT / "sanwa_ocr_output/Sanwa data/12_16_cslot/2025-12-16"
DEBUG_DIR  = ONEDRIVE_ROOT / "sanwa_ocr_debug/Sanwa data/12_16_cslot/2025-12-16"
#/Users/pomvrp/Library/CloudStorage/OneDrive-AgencyforScience,TechnologyandResearch/Sanwa data/12_16_cslot
OLLAMA_MODEL = "qwen2.5vl:3b"
ROI_JSON = Path("roi.json")
ROI_PAD = 0         
UPSCALE = 2.0 

# --- GROUPING CONFIGURATION ---
# Define which ROIs go into which CSV file.
# Note: 51 and 52 are added automatically to all of them as metadata.
CSV_GROUPS = {
    "CslotCam4result.csv":        list(range(1, 12)),   # 1-11
    "cam 6 snap1 Latchresult.csv": list(range(12, 20)),  # 12-19
    "cam 6 snap2 nozzleresult.csv": list(range(20, 31)),  # 20-30
    "terminal result.csv":        list(range(31, 51))   # 31-50
}

# The ROIs that contain the machine timestamp (to be added to every CSV)
TIMESTAMP_ROIS = ["51", "52"] 
# =================================================

class GroupedBatchHandler(FileSystemEventHandler):
    def __init__(self, rois):
        self.rois = rois

    def on_created(self, event):
        if not event.is_directory: self.process_new_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory: self.process_new_file(Path(event.dest_path))

    def process_new_file(self, file_path: Path):
        # 1. Validation
        if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}: return
        if file_path.name.startswith("."): return

        print(f"\n👀 Detected: {file_path.name}")
        if not self.wait_for_file_ready(file_path): return

        # 2. Check if already processed (Check if it exists in ANY summary csv is hard, 
        #    so we check a marker file or just rely on the processing logic).
        #    For now, we process every event.
        
        try: relative_path = file_path.relative_to(SOURCE_DIR)
        except: return

        # Setup Debug Folder (Mirror structure)
        target_debug = DEBUG_DIR / relative_path.parent
        target_debug.mkdir(parents=True, exist_ok=True)

        print("⚡ Processing...")
        self.run_pipeline(file_path, target_debug)

    def wait_for_file_ready(self, file_path, timeout=10):
        start = time.time()
        last_size = -1
        while time.time() - start < timeout:
            try:
                curr = file_path.stat().st_size
                if curr == last_size and curr > 0: return True
                last_size = curr
                time.sleep(1.0)
            except: return False
        return False

    # --- TIMESTAMP PARSING LOGIC ---
    def parse_filename_time(self, filename):
        """
        Input: "2025-12-16 14.40.09.png"
        Output: ISO UTC string
        """
        try:
            # Extract the date part (first 19 chars usually)
            clean_name = filename.split('.')[0] + "." + filename.split('.')[1] + "." + filename.split('.')[2]
            # Expected format: YYYY-MM-DD HH.MM.SS
            dt = datetime.strptime(clean_name, "%Y-%m-%d %H.%M.%S")
            return dt.isoformat() + "Z"
        except Exception as e:
            return f"Error_Parse_File: {filename}"

    def parse_machine_time(self, text_str):
        """
        Input: "Dec/16/25 14:30:01" (UTC+8)
        Output: ISO UTC String (UTC+0)
        """
        if not text_str or len(text_str) < 5: return ""
        try:
            # Clean up OCR noise (newlines, pipes instead of slashes)
            clean = text_str.replace("\n", " ").replace("|", "/").strip()
            # Try parsing "Dec/16/25 14:30:01"
            # Format: %b (Month abbr), %d, %y (2-digit year), %H:%M:%S
            dt_local = datetime.strptime(clean, "%b/%d/%y %H:%M:%S")
            
            # Adjust UTC+8 to UTC+0 (Subtract 8 hours)
            dt_utc = dt_local - timedelta(hours=8)
            return dt_utc.isoformat() + "Z"
        except Exception:
            try:
                # Fallback: Maybe year is 4 digits?
                dt_local = datetime.strptime(clean, "%b/%d/%Y %H:%M:%S")
                dt_utc = dt_local - timedelta(hours=8)
                return dt_utc.isoformat() + "Z"
            except:
                return text_str # Return raw text if parse fails

    def create_stitched_image(self, crops):
        if not crops: return None, []
        count = len(crops)
        cols = 5 # 5 Columns
        rows = math.ceil(count / cols)

        max_h, max_w = 0, 0
        for _, c in crops:
            h, w = c.shape[:2]
            max_h, max_w = max(max_h, h), max(max_w, w)

        cell_w, cell_h = max_w + 20, max_h + 40
        canvas = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
        valid_names = []

        for idx, (name, img) in enumerate(crops):
            r, c = idx // cols, idx % cols
            x_off, y_off = c * cell_w, r * cell_h
            h, w = img.shape[:2]
            
            x_pos = x_off + (cell_w - w) // 2
            y_pos = y_off + 35
            
            canvas[y_pos:y_pos+h, x_pos:x_pos+w] = img
            cv2.putText(canvas, f"ID:{name}", (x_off+5, y_off+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(canvas, (x_off, y_off), (x_off+cell_w, y_off+cell_h), (200,200,200), 1)
            valid_names.append(name)

        return canvas, valid_names

    def run_pipeline(self, img_path, debug_dir):
        img = cv2.imread(str(img_path))
        if img is None: return

        H, W = img.shape[:2]
        crop_list = []

        # 1. Collect all ROIs (1-52)
        for name, x, y, w, h in self.rois:
            if x >= W or y >= H: continue
            
            # Exact Crop
            x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
            x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
            
            crop = img[y0:y1, x0:x1]
            if crop.size == 0: continue
            
            crop_large = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
            crop_list.append((name, crop_large))

        if not crop_list: return

        # 2. Stitch & OCR
        stitched_img, valid_ids = self.create_stitched_image(crop_list)
        debug_path = debug_dir / f"{img_path.stem}_BATCH.png"
        cv2.imwrite(str(debug_path), stitched_img)

        # 3. Get JSON from Ollama
        json_result = self.ask_ollama_batch(debug_path, valid_ids)

        # 4. Prepare Metadata
        filename_utc = self.parse_filename_time(img_path.name)
        
        # Try to find machine time in ROI 51 or 52
        raw_machine_time = json_result.get("51", "")
        if not raw_machine_time: raw_machine_time = json_result.get("52", "")
        
        calc_machine_utc = self.parse_machine_time(raw_machine_time)

        # 5. Distribute to CSVs
        for csv_name, id_range in CSV_GROUPS.items():
            self.append_to_summary_csv(
                csv_name, 
                id_range, 
                json_result, 
                img_path.name, 
                filename_utc, 
                raw_machine_time, 
                calc_machine_utc
            )

    def append_to_summary_csv(self, csv_name, id_list, json_data, filename, file_utc, raw_mach, calc_mach):
        """
        Appends a row to the specific category CSV.
        Format: [Filename, File_UTC, Raw_Machine_Time, Calc_Machine_UTC, Result_ID_X, Result_ID_Y...]
        """
        # The summary files live in the ROOT output dir (or subfolder if you prefer)
        csv_path = OUTPUT_DIR / csv_name
        
        # 1. Determine Header
        # Columns: Filename, File_UTC, Machine_Raw, Machine_UTC, {ROI_IDs...}
        header = ["Filename", "File_UTC", "Machine_Text", "Machine_UTC"]
        target_ids = []
        
        # Filter relevant IDs for this CSV
        for i in id_list:
            sid = str(i)
            target_ids.append(sid)
            header.append(f"ROI_{sid}")
        
        # Add 51/52 columns if not already in range (user requested them in all files)
        for extra in ["51", "52"]:
            if extra not in target_ids:
                target_ids.append(extra)
                header.append(f"ROI_{extra}")

        # 2. Prepare Row Data
        row = [filename, file_utc, raw_mach, calc_mach]
        for tid in target_ids:
            # Clean commas/newlines so CSV doesn't break
            val = json_data.get(tid, "").replace("\n", " ").replace(",", ".")
            row.append(val)

        # 3. Append to File
        file_exists = csv_path.exists()
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header) # Write header only once
                writer.writerow(row)
            print(f"  ✅ Appended to: {csv_name}")
        except Exception as e:
            print(f"  ❌ CSV Error {csv_name}: {e}")

    def ask_ollama_batch(self, image_path, expected_ids):
        ids_str = ", ".join(expected_ids)
        prompt = (
            f"Read the text for IDs: {ids_str}. "
            f"Return JSON object: {{\"ID\": \"Text\"}}. "
            f"If an ID is a timestamp like 'Dec/16/25', read it exactly. "
            f"Output ONLY JSON."
        )
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}]
            )
            raw = response['message']['content'].strip()
            clean = re.sub(r'```json\s*|\s*```', '', raw).replace("`", "")
            return json.loads(clean)
        except: return {}

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
    if not SOURCE_DIR.exists():
        print(f"❌ Source {SOURCE_DIR} missing.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    
    rois = load_rois(ROI_JSON)
    if not rois:
        print("❌ roi.json missing")
        return

    print("========================================")
    print(f"🚀 Grouped Monitor Started")
    print(f"📂 Output Dir: {OUTPUT_DIR}")
    print("   (Summary CSVs will appear here)")
    print("========================================")

    observer = Observer()
    handler = GroupedBatchHandler(rois)
    observer.schedule(handler, str(SOURCE_DIR), recursive=True)
    observer.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()