"""
增强版OCR服务器 - 带动态Prompt和实时Median计算
Enhanced OCR Server with Dynamic Prompts and Real-time Median Calculation

特性 Features:
1. 根据ROI类型动态生成Prompt
2. 实时计算并使用Median值作为上下文
3. 自适应精度控制
4. 完整的调试输出
"""

import sys
import time
import json
import csv
import cv2
# import ollama           <-- 【已删除】避免 pydantic 报错
import urllib.request     # <-- 【新增】使用原生 HTTP 请求
import base64             # <-- 【新增】用于图片编码
import os
import numpy as np
import pandas as pd
import concurrent.futures
import threading
from datetime import datetime, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import defaultdict

# 导入配置
from config_pipeline import *

# ================= 🔴 核心路径配置 (保留你的设置) =================
# 1. 强制指定输入图片的文件夹
SOURCE_DIR = Path("/scratch/prj0000000262/ocr_data/QwenModel/SanwaDataPreprocess/input_images/12-22-2025/2025-12-22")

# 2. 定义批次名称 (输出到 debug_crops/12-22-2025)
BATCH_NAME = "12-22-2025"

# 3. 确保项目根目录正确
PROJECT_ROOT = Path("/scratch/prj0000000262/ocr_data/QwenModel/SanwaDataPreprocess")
ROI_JSON = PROJECT_ROOT / "roi.json"

# 4. 确保输出根目录正确
PREPROCESS_ROOT = PROJECT_ROOT / "pipeline_output"
STAGE_1_OCR = PREPROCESS_ROOT / "stage1_ocr_results"
SERVER_ROOT = PROJECT_ROOT
# ===================================================================

# ================= Stage 0 专用简单Prompts =================
STAGE0_PROMPTS = {
    'STATUS': "What text do you see in this image? Reply with exactly one word: OK or NG or NA",
    'INTEGER': "What integer number is shown? Reply with only the number, nothing else.",
    'FLOAT': "What decimal number is shown? Reply with only the number (like 1.234), nothing else.",
    'TIME': "What time is shown? Reply with only HH:MM:SS format, nothing else.",
    'DATE': "What date/time is shown? Reply with only the text, nothing else.",
}

# ================= 预计算Median加载器 =================
class PrecomputedMedianLoader:
    def __init__(self, final_dataset_dir=None):
        self.medians = {}
        self.stats = {}
        self.lock = threading.Lock()
        self.loaded = False
        
        # 默认路径
        if final_dataset_dir is None:
            # 这里的路径如果不重要可以保留默认，或者指向你的 PREPROCESS_ROOT
            final_dataset_dir = PREPROCESS_ROOT / "stage6_final_dataset"
        self.final_dataset_dir = Path(final_dataset_dir)
        
        self.load_all_medians()
    
    def load_all_medians(self):
        if not self.final_dataset_dir.exists():
            print(f"⚠️  Final dataset directory not found: {self.final_dataset_dir}")
            print("   Will use real-time median calculation as fallback.")
            return
        
        print(f"\n📊 Loading pre-computed medians from: {self.final_dataset_dir}")
        final_files = list(self.final_dataset_dir.glob("*_Final.csv"))
        
        if not final_files:
            print("⚠️  No _Final.csv files found. Will use real-time calculation.")
            return
        
        for csv_path in final_files:
            self._load_from_csv(csv_path)
        
        self.loaded = True
        print(f"✅ Loaded medians for {len(self.medians)} ROI fields\n")
    
    def _load_from_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            csv_name = csv_path.name
            print(f"   📄 {csv_name}: {len(df)} rows")
            
            for col in df.columns:
                if not col.startswith('ROI_'): continue
                
                roi_id = col.replace('ROI_', '')
                roi_type = get_roi_type(roi_id)
                
                if roi_type not in ['INTEGER', 'FLOAT']: continue
                
                try:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna()
                    vals = vals[(vals > 0) & (vals < vals.quantile(0.99))]
                    
                    if len(vals) >= 10:
                        with self.lock:
                            self.medians[roi_id] = vals.median()
                            self.stats[roi_id] = {
                                'median': vals.median(), 'mean': vals.mean(),
                                'std': vals.std(), 'min': vals.min(),
                                'max': vals.max(), 'count': len(vals)
                            }
                        print(f"      ✓ ROI_{roi_id}: Median={vals.median():.3f}")
                except: pass
        except Exception as e:
            print(f"   ❌ Error loading {csv_path.name}: {e}")
    
    def get_median(self, roi_id):
        with self.lock: return self.medians.get(roi_id, None)
    
    def get_stats(self, roi_id):
        with self.lock: return self.stats.get(roi_id, None)
    
    def add_value(self, roi_id, value, data_type): pass 
    
    def print_all_stats(self):
        print("\n" + "="*60)
        print("📊 Pre-computed Median Statistics:")
        print("="*60)
        with self.lock:
            for roi_id in sorted(self.stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                stats = self.stats[roi_id]
                print(f"  ROI_{roi_id}: Median={stats['median']:.3f}, N={stats['count']}")
        print("="*60 + "\n")

median_tracker = PrecomputedMedianLoader()
print_lock = threading.Lock()

# ================= 增强的GPU处理器 =================
class EnhancedGPUHandler(FileSystemEventHandler):
    def __init__(self, rois):
        self.rois = rois
        self.processed_count = 0
        
    def on_created(self, event):
        if not event.is_directory: self.process_new_file(Path(event.src_path))
    
    def on_moved(self, event):
        if not event.is_directory: self.process_new_file(Path(event.dest_path))
    
    def process_new_file(self, file_path: Path):
        """智能处理函数"""
        if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}: return True
        if file_path.name.startswith("."): return True

        # 1. 计算相对路径
        try:
            relative_path = file_path.relative_to(SOURCE_DIR)
        except ValueError:
            relative_path = Path(file_path.name)
        
        relative_parent = relative_path.parent
        image_folder_name = file_path.stem 
        
        # ================= 🔴 路径对齐 =================
        # 基础路径: .../stage1_ocr_results/debug_crops/12-22-2025
        base_debug_path = STAGE_1_OCR / "debug_crops" / BATCH_NAME
        
        # 完整目标路径
        target_image_folder = base_debug_path / relative_parent / image_folder_name
        
        if self.processed_count == 0:
            print(f"\n🔍 [Path Check] Checking for existing results at:\n   {target_image_folder / 'results.json'}")
        # ============================================================

        # 2. 智能断点
        if (target_image_folder / "results.json").exists():
            return True

        self.processed_count += 1
        print(f"\n⚡ [{self.processed_count}] Processing: {file_path.name}")
        
        target_image_folder.mkdir(parents=True, exist_ok=True)
        self.run_parallel_pipeline(file_path, target_image_folder, relative_parent)
        
        return False
        
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
        if img is None or img.size == 0: return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < DARKNESS_THRESHOLD:
            if np.sum(gray > 100) < (img.size * 0.01): return True
        return False
    
    def ask_ollama_simple(self, image_path, roi_id):
        """
        ✅ 修复版：使用 urllib 替代 ollama 库
        """
        roi_type = get_roi_type(roi_id)
        prompt = STAGE0_PROMPTS.get(roi_type, "Read the text. Output only the value.")
        
        try:
            # 1. 准备请求地址
            url = "http://localhost:11434/api/chat"
            
            # 2. 读取并编码图片
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 3. 构建数据包
            payload = {
                "model": OLLAMA_MODEL_3B,
                "messages": [{
                    'role': 'user', 
                    'content': prompt, 
                    'images': [img_b64]
                }],
                "stream": False,
                "options": {
                    'temperature': 0.0,
                    'num_predict': 30
                }
            }
            
            # 4. 发送请求 (Python原生方式)
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                raw = result['message']['content'].strip()
                
            clean = self.clean_output(raw, roi_type)
            return clean
            
        except Exception as e:
            with print_lock:
                print(f"c", end="", flush=True)
                # print(f" Error: {e}") # 调试用
            return "NA"
    
    def clean_output(self, raw_text, roi_type):
        import re
        if not raw_text: return "NA"
        
        special_tokens = [r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>', r'<\|pad\|>']
        text = raw_text
        for token in special_tokens: text = re.sub(token, '', text)
        text = re.sub(r'<[^>]+>', '', text).replace('```', '').replace('`', '').strip()
        text = text.split('\n')[0].strip().split()[0] if text.split() else text
        
        if roi_type == 'STATUS':
            upper = text.upper().strip()
            if upper in ['OK', 'NG', 'NA']: return upper
            if re.match(r'^-?\d+\.?\d*$', text.strip()): return text 
            if upper.startswith('NG') or upper == 'N': return 'NG'
            if upper.startswith('OK') or upper == 'O': return 'OK'
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
                except: pass
            return text
        elif roi_type == 'TIME':
            match = re.search(r'\d{1,2}:\d{2}:\d{2}', text)
            return match.group(0) if match else text
        return text
    
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
        
        text_val = self.ask_ollama_simple(crop_filename, name)
        
        try:
            with open(save_dir / f"ROI_{name}.txt", "w", encoding="utf-8") as f: f.write(text_val)
        except: pass
        
        with print_lock:
            if name in ["51", "52"]: print(f" [{name}: {text_val}] ", end="", flush=True)
            else: print(".", end="", flush=True)
        
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
        
        print(f"  --> Processing {len(self.rois)} ROIs with {MAX_WORKERS_3B} workers...")
        start_t = time.time()
        
        collected_results = {}
        tasks = [(name, x, y, w, h, img, save_dir) for name, x, y, w, h in self.rois]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_3B) as executor:
            results = executor.map(self.process_single_roi, tasks)
            for name, text_val in results:
                collected_results[name] = text_val
        
        print(f"\n  --> Finished in {time.time() - start_t:.1f}s")
        
        try:
            with open(save_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=2, ensure_ascii=False)
        except: pass
        
        filename_utc = self.parse_filename_time(img_path.name)
        raw_machine_time = collected_results.get("51", collected_results.get("52", ""))
        calc_machine_utc = self.parse_machine_time(raw_machine_time)
        
        for csv_name, id_range in CSV_GROUPS.items():
            self.append_to_summary_csv(csv_name, id_range, collected_results,
                img_path.name, filename_utc, raw_machine_time, calc_machine_utc, relative_parent)
        
        if self.processed_count % 10 == 0: self.print_median_stats()
    
    def print_median_stats(self): median_tracker.print_all_stats()
    
    def append_to_summary_csv(self, csv_name, id_list, results_dict, 
                             filename, file_utc, raw_mach, calc_mach, relative_parent):
        target_folder = STAGE_1_OCR / relative_parent / "CSV_Results"
        target_folder.mkdir(parents=True, exist_ok=True)
        csv_path = target_folder / csv_name
        
        header = ["Filename", "File_UTC", "Machine_Text", "Machine_UTC"]
        target_ids = []
        for i in id_list: target_ids.append(str(i)); header.append(f"ROI_{i}")
        for ex in ["51", "52"]: 
            if ex not in target_ids: target_ids.append(ex); header.append(f"ROI_{ex}")
        
        row = [filename, file_utc, raw_mach, calc_mach]
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
            except Exception as e: print(f"  ❌ CSV Write Error: {e}")

# ================= 辅助函数 =================
def load_rois(roi_path: Path):
    if not roi_path.exists(): return []
    try:
        with open(roi_path, "r", encoding="utf-8") as f: data = json.load(f)
        rois = []
        data_list = data if isinstance(data, list) else [data]
        for idx, item in enumerate(data_list):
            rois.append((str(item.get("name", str(idx))), int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])))
        return rois
    except Exception as e:
        print(f"❌ Error loading ROI: {e}")
        return []

def main():
    if not SERVER_ROOT.exists():
        print(f"❌ Error: SERVER_ROOT does not exist: {SERVER_ROOT}")
        return
    
    if 'create_directories' in globals(): create_directories()
    else: STAGE_1_OCR.mkdir(parents=True, exist_ok=True)
    
    rois = load_rois(ROI_JSON)
    if not rois: print("❌ roi.json missing or invalid"); return
    
    print("="*60)
    print("🚀 Enhanced OCR Server Started (3B Model - urllib version)")
    print(f"   Model: {OLLAMA_MODEL_3B}")
    print(f"   Workers: {MAX_WORKERS_3B} (Parallel)")
    print(f"   ROIs: {len(rois)} configured")
    print(f"   Watch Folder: {SOURCE_DIR}")
    print(f"   Output: {STAGE_1_OCR}")
    print(f"   Batch: {BATCH_NAME}")
    print("="*60)
    
    handler = EnhancedGPUHandler(rois)
    
    print("\n📁 Scanning directory tree...")
    all_files = list(SOURCE_DIR.rglob("*"))
    image_files = [f for f in all_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and not f.name.startswith(".")]
    image_files.sort()
    
    total = len(image_files)
    print(f"Found {total} images. Starting batch...\n")
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{total}]", end=" ")
        try: handler.process_new_file(img_path)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            handler.print_median_stats()
            return
        except Exception as e: print(f"\n❌ Error processing {img_path.name}: {e}")
    
    print("\n✅ Batch done. Monitoring for NEW files...")
    handler.print_median_stats()
    
    observer = Observer()
    observer.schedule(handler, str(SOURCE_DIR), recursive=True)
    observer.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Server stopped.")
        handler.print_median_stats()
    observer.join()

if __name__ == "__main__":
    main()