import sys
import time
import json
import csv
import shutil
import cv2
import ollama
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= CONFIGURATION =================

ONEDRIVE_ROOT = Path("/Users/pomvrp/Library/CloudStorage/OneDrive-AgencyforScience,TechnologyandResearch")
SOURCE_DIR = ONEDRIVE_ROOT / "Sanwa data/12_16_cslot/2025-12-16"
OUTPUT_DIR = ONEDRIVE_ROOT / "sanwa_ocr_output/Sanwa data/12_16_cslot/2025-12-16"
DEBUG_DIR  = ONEDRIVE_ROOT / "sanwa_ocr_debug/Sanwa data/12_16_cslot/2025-12-16"

OLLAMA_MODEL = "qwen2.5vl:3b"
ROI_JSON = Path("roi.json")

ROI_PAD = 2         
UPSCALE = 2.0       

# [新功能] 亮度阈值
# 像素平均亮度低于此值（0-255）会被判定为"太黑/无效"，直接填 NA
# 如果你的错位图是全黑的，设为 10 或 20 比较安全。如果是深灰色，可以设高一点。
DARKNESS_THRESHOLD = 15  

CSV_GROUPS = {
    "CslotCam4result.csv":           list(range(1, 12)),
    "cam 6 snap1 Latchresult.csv":   list(range(12, 20)),
    "cam 6 snap2 nozzleresult.csv":  list(range(20, 31)),
    "terminal result.csv":           list(range(31, 51))
}
# =================================================

class SerialHandler(FileSystemEventHandler):
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
        
        # 创建图片专属文件夹
        image_folder_name = file_path.stem
        target_image_folder = DEBUG_DIR / relative_parent / image_folder_name
        target_image_folder.mkdir(parents=True, exist_ok=True)

        self.run_serial_pipeline(file_path, target_image_folder, relative_parent)

    def parse_filename_time(self, filename):
        try:
            name_only = filename.rsplit('.', 1)[0]
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
        """检查图片是否太黑/无效"""
        if img is None or img.size == 0: return True
        # 计算灰度图的平均亮度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < DARKNESS_THRESHOLD

    def run_serial_pipeline(self, img_path, save_dir, relative_parent):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Cannot read image: {img_path}")
            return

        H, W = img.shape[:2]
        collected_results = {}
        
        # 1. 保存 Debug Map (方便你肉眼检查偏移)
        try:
            vis_img = img.copy()
            for name, x, y, w, h in self.rois:
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis_img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(save_dir / "_DEBUG_MAP.jpg"), vis_img)
        except: pass

        print(f"  --> Starting OCR (Threshold: avg brightness < {DARKNESS_THRESHOLD} => NA)")
        start_t = time.time()
        
        for i, (name, x, y, w, h) in enumerate(self.rois):
            # 越界检查
            if x >= W or y >= H: 
                collected_results[name] = "NA"
                continue
            
            # 裁剪
            x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
            x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
            crop = img[y0:y1, x0:x1]
            
            if crop.size == 0:
                collected_results[name] = "NA"
                continue

            # 放大
            if UPSCALE != 1.0:
                crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

            # 保存裁剪小图（即使是黑的也存，方便你确认它确实是黑的）
            crop_filename = save_dir / f"ROI_{name}.jpg"
            cv2.imwrite(str(crop_filename), crop)

            # --- 核心修改：亮度检测 ---
            # 如果图片太黑，直接给 NA，不问 AI
            if self.is_image_too_dark(crop):
                text_val = "NA"
                # 稍微打印一下提示，比如 [D] 代表 Dark/Skipped
                print("D", end="", flush=True) 
            else:
                # 只有亮度足够才问 AI
                text_val = self.ask_ollama_single(crop_filename)
                print(".", end="", flush=True)

            collected_results[name] = text_val
            
            # 保存 TXT
            try:
                with open(save_dir / f"ROI_{name}.txt", "w", encoding="utf-8") as f:
                    f.write(text_val)
            except: pass

            # 遇到时间戳或每10个打印一次进度
            if name in ["51", "52"] or (i % 10 == 0 and i > 0):
                 # 这里加个换行美观一点
                print(f"\n    ID {name}: {text_val}", end=" ")

        print(f"\n  --> Finished in {time.time() - start_t:.1f}s")

        # 保存完整 JSON
        try:
            with open(save_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=2)
        except: pass

        # Metadata
        filename_utc = self.parse_filename_time(img_path.name)
        # 优先读取 51，没有再读 52，如果是 NA 也没关系，filename_utc 还在
        raw_machine_time = collected_results.get("51", "")
        if not raw_machine_time or raw_machine_time == "NA": 
            raw_machine_time = collected_results.get("52", "")
        
        calc_machine_utc = self.parse_machine_time(raw_machine_time)

        # 写入 CSV (NA 会被写入)
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
        prompt = "Read the text in this image. Return ONLY the value. No extra words."
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}],
                options={'num_predict': 20} 
            )
            raw = response['message']['content'].strip()
            clean = raw.replace("`", "").replace('"', '').replace("'", "")
            # 再次防护：如果 AI 没说话或返回空，也当做 NA
            if not clean: return "NA"
            return clean
        except Exception as e:
            print(f"    ⚠️ Ollama Error: {e}")
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
            # 填入值，如果是 None 则填 NA
            val = results_dict.get(tid, "NA").replace("\n", " ").replace(",", ".")
            row.append(val)

        file_exists = csv_path.exists()
        try:
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
    print(f"🚀 Serial Monitor (Auto-NA for Dark Images)")
    print(f"Threshold: {DARKNESS_THRESHOLD}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print("========================================")

    handler = SerialHandler(rois)

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