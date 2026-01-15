"""
OCR Server for C-Slot Camera - Stage 0
Uses config_pipeline.py for directory configuration
Simple data-type based prompts (no color detection)
"""

import sys
import time
import json
import csv
import cv2
import ollama
import os
import re
import numpy as np
import concurrent.futures
import threading
from datetime import datetime, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= 从config_pipeline导入目录配置 =================
from config_pipeline import (
    SOURCE_DIR,           # 输入图像目录
    STAGE_1_OCR,          # Stage 1 OCR输出目录
    SERVER_ROOT,          # 服务器根目录
    ROI_JSON,             # ROI配置文件
    ROI_PAD,              # ROI边距
    UPSCALE,              # 上采样倍数
    DARKNESS_THRESHOLD,   # 暗度阈值
    OLLAMA_MODEL_3B,      # 3B模型名称
    MAX_WORKERS_3B,       # 并行工作数
    CSV_GROUPS,           # CSV分组定义
    create_directories    # 创建目录函数
)

# ================= Stage 0 简单Prompts (本文件专用) =================
# Simple prompts for Stage 0 - NOT from config_pipeline.py
STAGE0_PROMPTS = {
    'STATUS': "What text do you see? Reply with exactly one word: OK or NG or NA",
    'INTEGER': "What integer number is shown? Reply with only the number.",
    'FLOAT': "What decimal number is shown? Reply with only the number.",
    'TIME': "What time is shown? Reply with only HH:MM:SS format.",
    'DATE': "What date is shown? Reply with only the date in Mon/DD/YY format (like Dec/19/25).",
}

# Stage 0 输出目录 (使用STAGE_1_OCR作为输出)
OUTPUT_DIR = STAGE_1_OCR
DEBUG_DIR = STAGE_1_OCR / "debug_crops"

# IDs used for Machine Timestamp
TIMESTAMP_IDS = ["51", "52"]

# =================================================

print_lock = threading.Lock()

class GPUHandler(FileSystemEventHandler):
    def __init__(self, rois):
        self.rois = rois
        self.processed_count = 0

    def on_created(self, event):
        if not event.is_directory: 
            self.process_new_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory: 
            self.process_new_file(Path(event.dest_path))

    def process_new_file(self, file_path: Path):
        if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}: 
            return
        if file_path.name.startswith("."): 
            return

        self.processed_count += 1
        print(f"\n⚡ [{self.processed_count}] Processing: {file_path.name}")
        
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
        """Extract timestamp from filename (e.g. 2025-12-19 14.30.05.png)"""
        try:
            name_only = filename.rsplit('.', 1)[0]
            dt = datetime.strptime(name_only, "%Y-%m-%d %H.%M.%S")
            return dt.isoformat() + "Z"
        except: 
            return filename

    def parse_machine_time(self, text_str):
        """解析机器时间戳"""
        if not text_str or len(text_str) < 5 or "NA" in text_str: 
            return ""
        try:
            clean = text_str.replace("\n", " ").replace("|", "/").strip()
            dt_local = datetime.strptime(clean, "%b/%d/%y %H:%M:%S")
            dt_utc = dt_local - timedelta(hours=8)
            return dt_utc.isoformat() + "Z"
        except: 
            return text_str

    def is_image_too_dark(self, img):
        if img is None or img.size == 0: 
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < DARKNESS_THRESHOLD:
            bright_pixels = np.sum(gray > 100)
            if bright_pixels < (img.size * 0.01):
                return True
        return False

    def process_single_roi(self, args):
        name, x, y, w, h, img, save_dir = args
        H, W = img.shape[:2]

        if x >= W or y >= H: 
            return name, "NA"

        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0: 
            return name, "NA"

        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        crop_filename = save_dir / f"ROI_{name}.jpg"
        cv2.imwrite(str(crop_filename), crop)

        if self.is_image_too_dark(crop):
            with print_lock: 
                print("D", end="", flush=True)
            return name, "NA"

        # Ask AI with simple prompt
        text_val = self.ask_ollama_single(crop_filename, name)
        
        try:
            with open(save_dir / f"ROI_{name}.txt", "w", encoding="utf-8") as f:
                f.write(text_val)
        except: 
            pass

        with print_lock:
            if name in TIMESTAMP_IDS:
                print(f" [{name}: {text_val}] ", end="", flush=True)
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
                cv2.putText(vis_img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(save_dir / "_DEBUG_MAP.jpg"), vis_img)
        except: 
            pass

        print(f"  --> Processing {len(self.rois)} ROIs with {MAX_WORKERS_3B} workers...")
        start_t = time.time()
        
        collected_results = {}
        
        tasks = [(name, x, y, w, h, img, save_dir) for name, x, y, w, h in self.rois]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_3B) as executor:
            results = executor.map(self.process_single_roi, tasks)
            for name, text_val in results:
                collected_results[name] = text_val

        duration = time.time() - start_t
        print(f"\n  --> Finished in {duration:.1f}s")

        try:
            with open(save_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=2, ensure_ascii=False)
        except: 
            pass

        # Parse timestamps
        filename_utc = self.parse_filename_time(img_path.name)
        raw_machine_time = collected_results.get("51", "")
        if not raw_machine_time or raw_machine_time == "NA":
            raw_machine_time = collected_results.get("52", "")
        calc_machine_utc = self.parse_machine_time(raw_machine_time)

        for csv_name, id_range in CSV_GROUPS.items():
            self.append_to_summary_csv(
                csv_name, id_range, collected_results, 
                img_path.name, filename_utc, raw_machine_time, 
                calc_machine_utc, relative_parent
            )

    def get_roi_type(self, roi_id):
        """获取ROI数据类型"""
        # C-Slot ROI类型定义
        status_ids = {'1', '3', '5', '7', '9', '10', '11', '12', '14', '15', '17', '19', 
                      '20', '22', '24', '26', '28', '30', '31', '33', '35', '37', '39',
                      '41', '43', '45', '47', '49'}
        integer_ids = {'2', '21', '32', '34', '36', '38', '40', '42', '44', '46', '48', '50'}
        float_ids = {'4', '6', '8', '13', '16', '18', '23', '25', '27', '29'}
        time_ids = {'52'}
        date_ids = {'51'}
        
        if roi_id in status_ids:
            return 'STATUS'
        elif roi_id in integer_ids:
            return 'INTEGER'
        elif roi_id in float_ids:
            return 'FLOAT'
        elif roi_id in time_ids:
            return 'TIME'
        elif roi_id in date_ids:
            return 'DATE'
        return 'STATUS'

    def clean_ocr_text(self, text, roi_type):
        """清理OCR输出"""
        if not text: 
            return "NA"
        
        # 1. Remove model special tokens
        special_tokens = [
            r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>',
            r'<\|pad\|>', r'<\|assistant\|>', r'<\|user\|>', r'<\|system\|>',
        ]
        for token in special_tokens:
            text = re.sub(token, '', text)
        
        # 2. Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3. Remove markdown
        text = text.replace("```", "").replace("`", "").strip()

        # 4. Fix repeated decimal patterns (e.g., .05.05 -> .05)
        text = re.sub(r'(\.\d+)\1', r'\1', text)

        # 5. Take first word only
        text = text.split('\n')[0].strip()
        text = text.split()[0] if text.split() else text

        # 6. Type-specific processing
        if roi_type == 'STATUS':
            upper = text.upper().strip()
            if upper == 'OK':
                return 'OK'
            if upper == 'NG':
                return 'NG'
            if upper == 'NA':
                return 'NA'
            # If it's a number, return it as-is (not a status field)
            if re.match(r'^-?\d+\.?\d*$', text.strip()):
                return text
            if upper.startswith('NG') or upper == 'N':
                return 'NG'
            if upper.startswith('OK') or upper == 'O':
                return 'OK'
            return text
        
        elif roi_type == 'INTEGER':
            match = re.search(r'-?\d+', text)
            return match.group(0) if match else text
        
        elif roi_type == 'FLOAT':
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                try:
                    val = float(match.group(0))
                    return f"{val:.3f}".rstrip('0').rstrip('.')
                except:
                    pass
            return text
        
        elif roi_type == 'TIME':
            match = re.search(r'\d{1,2}:\d{2}:\d{2}', text)
            return match.group(0) if match else text
        
        return text

    def ask_ollama_single(self, image_path, roi_id):
        """Simple OCR call based on ROI type - using STAGE0_PROMPTS"""
        roi_type = self.get_roi_type(roi_id)
        
        # Use Stage 0 simple prompts
        prompt = STAGE0_PROMPTS.get(roi_type, "Read the text. Return ONLY the value.")
        
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_3B,
                messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}],
                options={
                    'temperature': 0.0,
                    'repeat_penalty': 1.2,
                    'num_predict': 30,
                    'top_k': 10
                } 
            )
            raw = response['message']['content']
            clean = self.clean_ocr_text(raw, roi_type)
            
            if not clean: 
                return "NA"
            return clean
        except Exception:
            return "NA"

    def append_to_summary_csv(self, csv_name, id_list, results_dict, 
                             filename, file_utc, raw_mach, calc_mach, 
                             relative_parent):
        """追加结果到CSV"""
        target_folder = OUTPUT_DIR / relative_parent / "CSV_Results"
        target_folder.mkdir(parents=True, exist_ok=True)
        csv_path = target_folder / csv_name
        
        header = ["Filename", "File_UTC", "Machine_Text", "Machine_UTC"]
        target_ids = []
        for i in id_list:
            sid = str(i)
            target_ids.append(sid)
            header.append(f"ROI_{sid}")
        
        # Add timestamp columns
        for extra in ["51", "52"]:
            if extra not in target_ids:
                target_ids.append(extra)
                header.append(f"ROI_{extra}")
        
        row = [filename, file_utc, raw_mach, calc_mach]
        for tid in target_ids:
            val = results_dict.get(tid, "NA").replace("\n", " ").replace(",", ".")
            row.append(val)

        with print_lock:
            try:
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists: 
                        writer.writerow(header)
                    writer.writerow(row)
            except Exception as e:
                print(f"  ❌ CSV Write Error: {e}")

# =================================================

def load_rois(roi_path: Path):
    """加载ROI配置"""
    if not roi_path.exists(): 
        return []
    try:
        with open(roi_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rois = []
        data_list = data if isinstance(data, list) else [data]
        for idx, item in enumerate(data_list):
            name = item.get("name", str(idx))
            rois.append((str(name), int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])))
        return rois
    except Exception as e:
        print(f"❌ Error loading ROI: {e}")
        return []

def main():
    # Check paths
    if not SERVER_ROOT.exists():
        print(f"❌ Error: SERVER_ROOT does not exist: {SERVER_ROOT}")
        return
        
    if not SOURCE_DIR.exists():
        print(f"❌ Error: SOURCE_DIR does not exist: {SOURCE_DIR}")
        print("   Please update config_pipeline.py")
        return

    # 使用config_pipeline的create_directories创建目录
    create_directories()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    
    rois = load_rois(ROI_JSON)
    if not rois:
        print(f"❌ {ROI_JSON} missing or invalid")
        return

    print("="*60)
    print("🚀 Stage 0: OCR Server for C-Slot Started")
    print(f"   Model: {OLLAMA_MODEL_3B}")
    print(f"   Workers: {MAX_WORKERS_3B} (Parallel)")
    print(f"   ROIs: {len(rois)} configured")
    print(f"📂 Watch Folder: {SOURCE_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"📂 Debug: {DEBUG_DIR}")
    print("="*60)

    handler = GPUHandler(rois)

    # Scan existing files
    print("\n📁 Scanning directory tree...")
    all_files = list(SOURCE_DIR.rglob("*"))
    image_files = [
        f for f in all_files 
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} 
        and not f.name.startswith(".")
    ]
    image_files.sort()
    
    total = len(image_files)
    print(f"Found {total} images. Starting batch...\n")

    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{total}]", end=" ")
        try:
            handler.process_new_file(img_path)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")

    print("\n✅ Batch done. Monitoring for NEW files...")
    
    observer = Observer()
    observer.schedule(handler, str(SOURCE_DIR), recursive=True)
    observer.start()
    
    try:
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Server stopped.")
    observer.join()

if __name__ == "__main__":
    main()
