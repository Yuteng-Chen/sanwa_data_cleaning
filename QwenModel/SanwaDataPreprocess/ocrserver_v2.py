"""
OCR Server v2 - 简化版OCR服务器
Simple OCR Server using config_pipeline.py directories

Features:
1. Uses same directories as ocrserver_enhanced.py
2. Simple data-type based prompts (no color detection)
3. Parallel processing with ThreadPoolExecutor
"""

import sys
import time
import json
import csv
import re
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

# 导入配置 - 使用与ocrserver_enhanced.py相同的目录配置
from config_pipeline import (
    SOURCE_DIR, STAGE_1_OCR, ROI_JSON, ROI_PAD, UPSCALE, 
    DARKNESS_THRESHOLD, OLLAMA_MODEL_3B, MAX_WORKERS_3B,
    CSV_GROUPS, SERVER_ROOT, create_directories, get_roi_type
)

# ================= Stage 0 简单Prompts =================
STAGE0_PROMPTS = {
    'STATUS': "What text do you see in this image? Reply with exactly one word: OK or NG or NA",
    'INTEGER': "What integer number is shown? Reply with only the number, nothing else.",
    'FLOAT': "What decimal number is shown? Reply with only the number (like 1.234), nothing else.",
    'TIME': "What time is shown? Reply with only HH:MM:SS format, nothing else.",
    'DATE': "What date/time is shown? Reply with only the text, nothing else.",
}

print_lock = threading.Lock()

# ================= OCR处理器 =================
class OCRHandler(FileSystemEventHandler):
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
        """处理新图像文件"""
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
        
        # 创建调试目录
        image_folder_name = file_path.stem
        target_image_folder = STAGE_1_OCR / "debug_crops" / relative_parent / image_folder_name
        target_image_folder.mkdir(parents=True, exist_ok=True)

        self.run_parallel_pipeline(file_path, target_image_folder, relative_parent)

    def parse_filename_time(self, filename):
        """从文件名解析时间"""
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
        """简单检查图像是否太暗"""
        if img is None or img.size == 0:
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < DARKNESS_THRESHOLD:
            bright_pixels = np.sum(gray > 100)
            if bright_pixels < (img.size * 0.01):
                return True
        return False

    def ask_ollama(self, image_path, roi_id):
        """OCR调用 - 基于ROI类型使用简单prompt"""
        roi_type = get_roi_type(roi_id)
        prompt = STAGE0_PROMPTS.get(roi_type, "Read the text. Output only the value.")
        
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_3B,
                messages=[{
                    'role': 'user', 
                    'content': prompt, 
                    'images': [str(image_path)]
                }],
                options={
                    'temperature': 0.0,
                    'num_predict': 30
                }
            )
            raw = response['message']['content'].strip()
            clean = self.clean_output(raw, roi_type)
            return clean
            
        except Exception as e:
            with print_lock:
                print(f"c", end="", flush=True)
            return "NA"

    def clean_output(self, raw_text, roi_type):
        """清理模型输出"""
        if not raw_text:
            return "NA"
        
        # 移除模型特殊tokens
        special_tokens = [
            r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>',
            r'<\|pad\|>', r'<\|assistant\|>', r'<\|user\|>', r'<\|system\|>',
        ]
        text = raw_text
        for token in special_tokens:
            text = re.sub(token, '', text)
        
        # 移除HTML标签和markdown
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('```', '').replace('`', '').strip()
        
        # 只取第一行第一个词
        text = text.split('\n')[0].strip()
        text = text.split()[0] if text.split() else text
        
        # 根据类型处理
        if roi_type == 'STATUS':
            upper = text.upper().strip()
            if upper == 'OK':
                return 'OK'
            if upper == 'NG':
                return 'NG'
            if upper == 'NA':
                return 'NA'
            # 检查是否是数字
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

    def process_single_roi(self, args):
        """并行处理单个ROI"""
        name, x, y, w, h, img, save_dir = args
        H, W = img.shape[:2]
        
        # 1. 边界检查
        if x >= W or y >= H: 
            return name, "NA"
        
        # 2. 裁剪
        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0: 
            return name, "NA"
        
        # 3. 上采样
        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, 
                            interpolation=cv2.INTER_CUBIC)
        
        # 4. 保存裁剪
        crop_filename = save_dir / f"ROI_{name}.jpg"
        cv2.imwrite(str(crop_filename), crop)
        
        # 5. 亮度检查
        if self.is_image_too_dark(crop):
            with print_lock:
                print("D", end="", flush=True)
            return name, "NA"
        
        # 6. OCR识别
        text_val = self.ask_ollama(crop_filename, name)
        
        # 7. 保存文本结果
        try:
            with open(save_dir / f"ROI_{name}.txt", "w", encoding="utf-8") as f:
                f.write(text_val)
        except: 
            pass
        
        # 8. 输出进度
        with print_lock:
            if name in ["51", "52"]:
                print(f" [{name}: {text_val}] ", end="", flush=True)
            else:
                print(".", end="", flush=True)
        
        return name, text_val

    def run_parallel_pipeline(self, img_path, save_dir, relative_parent):
        """并行处理管道"""
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Cannot read image: {img_path}")
            return

        # 绘制调试地图
        try:
            vis_img = img.copy()
            for name, x, y, w, h in self.rois:
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis_img, name, (x, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(save_dir / "_DEBUG_MAP.jpg"), vis_img)
        except: 
            pass

        print(f"  --> Processing {len(self.rois)} ROIs with {MAX_WORKERS_3B} workers...")
        start_t = time.time()
        
        collected_results = {}
        
        # 准备并行任务
        tasks = [(name, x, y, w, h, img, save_dir) for name, x, y, w, h in self.rois]
        
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_3B) as executor:
            results = executor.map(self.process_single_roi, tasks)
            for name, text_val in results:
                collected_results[name] = text_val

        duration = time.time() - start_t
        print(f"\n  --> Finished in {duration:.1f}s")

        # 保存JSON结果
        try:
            with open(save_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=2, ensure_ascii=False)
        except: 
            pass

        # 解析元数据
        filename_utc = self.parse_filename_time(img_path.name)
        raw_machine_time = collected_results.get("51", "")
        if not raw_machine_time or raw_machine_time == "NA":
            raw_machine_time = collected_results.get("52", "")
        calc_machine_utc = self.parse_machine_time(raw_machine_time)

        # 写入CSV
        for csv_name, id_range in CSV_GROUPS.items():
            self.append_to_summary_csv(
                csv_name, id_range, collected_results,
                img_path.name, filename_utc, raw_machine_time,
                calc_machine_utc, relative_parent
            )

    def append_to_summary_csv(self, csv_name, id_list, results_dict, 
                             filename, file_utc, raw_mach, calc_mach, 
                             relative_parent):
        """追加结果到CSV"""
        target_folder = STAGE_1_OCR / relative_parent / "CSV_Results"
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

# ================= 辅助函数 =================
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
            rois.append((
                str(name), 
                int(item["x"]), 
                int(item["y"]), 
                int(item["w"]), 
                int(item["h"])
            ))
        return rois
    except Exception as e:
        print(f"❌ Error loading ROI: {e}")
        return []

def main():
    """主函数"""
    # 检查服务器根目录
    if not SERVER_ROOT.exists():
        print(f"❌ Error: SERVER_ROOT does not exist: {SERVER_ROOT}")
        print("   Please update config_pipeline.py")
        return
    
    # 创建必要目录
    create_directories()
    
    # 加载ROI配置
    rois = load_rois(ROI_JSON)
    if not rois:
        print("❌ roi.json missing or invalid")
        return
    
    print("="*60)
    print("🚀 OCR Server v2 Started")
    print(f"   Model: {OLLAMA_MODEL_3B}")
    print(f"   Workers: {MAX_WORKERS_3B} (Parallel)")
    print(f"   ROIs: {len(rois)} configured")
    print(f"   Watch Folder: {SOURCE_DIR}")
    print(f"   Output: {STAGE_1_OCR}")
    print("="*60)
    
    handler = OCRHandler(rois)
    
    # 1. 扫描现有文件
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
            print(f"\n❌ Error processing {img_path.name}: {e}")
    
    print("\n✅ Batch done. Monitoring for NEW files...")
    
    # 2. 监控新文件
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
