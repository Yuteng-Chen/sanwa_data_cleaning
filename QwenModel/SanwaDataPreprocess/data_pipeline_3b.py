"""
3B/7B模型数据清理管道 - 完整自动化流程
Data Cleaning Pipeline - Full Automation
针对 2025-12-22 批次优化
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import glob
import shutil
import re
import cv2
from pathlib import Path
from datetime import datetime
import concurrent.futures
import threading
from collections import defaultdict

# 关键：导入 ollama 库
import ollama 

# 导入配置 (确保 config_pipeline.py 在同一目录下)
try:
    from config_pipeline import *
except ImportError:
    print("❌ Critical Error: 'config_pipeline.py' not found!")
    sys.exit(1)

print_lock = threading.Lock()

# ================= 阶段1: 数据验证和清理 =================
class DataValidator:
    """数据验证器 - 检测异常值"""
    
    def __init__(self, max_decimals=3, outlier_threshold=5.0, z_score_threshold=3.0):
        self.max_decimals = max_decimals
        self.outlier_threshold = outlier_threshold
        self.z_score_threshold = z_score_threshold
    
    def validate_value(self, val, data_type):
        """验证单个值"""
        val_str = str(val).strip()
        if pd.isna(val) or val_str == '' or val_str.lower() == 'nan':
            return False, val, "Empty/NaN"
        
        if data_type == 'STATUS':
            val_upper = val_str.upper()
            if val_upper.startswith('O') or val_upper == '0' or 'OK' in val_upper:
                return True, 'OK', None
            if val_upper.startswith('N'):
                return True, 'NG', None
            if val_upper == 'K':
                return True, 'OK', None
            if val_upper in ['', 'NAN', 'NA', 'NULL', 'NONE']:
                return False, val, "Empty/Invalid Status"
            return False, val, "Unknown Status"
        
        elif data_type == 'INTEGER':
            try:
                return True, int(float(val_str)), None
            except:
                pass
            clean_val = re.sub(r'[^\d-]', '', val_str)
            if re.match(r'^-?\d+$', clean_val):
                return True, int(clean_val), None
            return False, val, "Not an Integer"
        
        elif data_type == 'FLOAT':
            if re.match(r'^-?\d+(\.\d+)?$', val_str):
                if '.' in val_str and len(val_str.split('.')[1]) > self.max_decimals:
                    return False, val, f"Suspicious Pattern (>{self.max_decimals} decimals)"
                try:
                    return True, float(val_str), None
                except:
                    pass
            return False, val, "Invalid Float"
        
        elif data_type == 'TIME':
            if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str):
                return True, val_str, None
            return False, val, "Invalid Time"
        
        return False, val, "Unknown Type"
    
    def detect_outliers(self, series, data_type):
        """统计异常值检测 (Ratio + Z-Score)"""
        if data_type not in ['FLOAT', 'INTEGER']:
            return []
        
        nums = pd.to_numeric(series, errors='coerce').dropna()
        if len(nums) < 5:
            return []
        
        median = nums.median()
        mean = nums.mean()
        std = nums.std()
        
        outlier_results = []
        
        for idx, val in series.items():
            try:
                val_float = float(val)
                if val_float == 0 or pd.isna(val_float):
                    continue
                
                # Method 1: Ratio (针对漏小数点)
                if median != 0:
                    ratio = val_float / median
                    if ratio > self.outlier_threshold or ratio < (1.0 / self.outlier_threshold):
                        outlier_results.append((idx, "Statistical Outlier (Likely Missing Decimal)"))
                        continue
                
                # Method 2: Z-Score (针对偏离值)
                if std > 0:
                    z_score = abs((val_float - mean) / std)
                    if z_score > self.z_score_threshold:
                        outlier_results.append((idx, f"Z-Score Outlier (Z={z_score:.2f})"))
            except:
                pass
        return outlier_results

class Stage1_DataCleaning:
    """阶段1: 数据清理"""
    def __init__(self, input_dir, output_dir, crops_base):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.crops_base = Path(crops_base)
        self.validator = DataValidator(MAX_DECIMALS, OUTLIER_THRESHOLD, Z_SCORE_THRESHOLD)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_config_for_file(self, df):
        for config in ROI_CONFIGS:
            if config['Trigger_Col'] in df.columns:
                return config
        return None
    
    def copy_crop_for_review(self, csv_base_name, filename, roi_id, dest_folder):
        """复制异常图片，支持多种路径结构"""
        try:
            folder_name = os.path.splitext(filename)[0]
            # 搜索路径策略 (适配不同的截图目录结构)
            potential_paths = [
                self.crops_base / csv_base_name / folder_name / f"{roi_id}.jpg",
                self.crops_base / csv_base_name / folder_name / f"{roi_id}.png",
                self.crops_base / folder_name / f"{roi_id}.jpg",
                self.crops_base / folder_name / f"{roi_id}.png",
            ]
            
            src_file = None
            for p in potential_paths:
                if p.exists():
                    src_file = p
                    break
            
            if not src_file:
                return False
            
            target_folder = dest_folder / folder_name
            target_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, target_folder / src_file.name)
            return True
        except:
            return False

    def process_single_csv(self, csv_path):
        filename = csv_path.name
        base_name = csv_path.stem
        print(f"\n📄 Processing: {filename}...")
        
        # 断点续传: 检查是否已处理完成
        output_cleaned = self.output_dir / f"{base_name}_Cleaned.csv"
        if output_cleaned.exists():
            print(f"  ⏭️  Skipped (already processed: {output_cleaned.name})")
            return
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ❌ Error reading CSV: {e}")
            return
        
        config = self.get_config_for_file(df)
        if not config:
            print(f"  ⚠️  Skipped: Unknown CSV format (Column mismatch)")
            return
        
        roi_map = config['Columns']
        if 'Filename' in df.columns:
            df.sort_values(by='Filename', inplace=True)
        
        df_clean = df.copy()
        abnormal_records = []
        
        # 1. 格式验证
        for idx, row in df.iterrows():
            for roi_col, dtype in roi_map.items():
                if roi_col in df.columns:
                    val = row[roi_col]
                    is_valid, clean_val, reason = self.validator.validate_value(val, dtype)
                    if is_valid:
                        df_clean.at[idx, roi_col] = clean_val
                    else:
                        abnormal_records.append({
                            'Filename': row.get('Filename', 'Unknown'),
                            'ROI_ID': roi_col,
                            'Value': val,
                            'Reason': reason
                        })
        
        # 2. 统计检测
        for roi_col, dtype in roi_map.items():
            if roi_col in df_clean.columns:
                outlier_results = self.validator.detect_outliers(df_clean[roi_col], dtype)
                for idx, reason in outlier_results:
                    abnormal_records.append({
                        'Filename': df_clean.at[idx, 'Filename'],
                        'ROI_ID': roi_col,
                        'Value': df_clean.at[idx, roi_col],
                        'Reason': reason
                    })
        
        # 保存结果
        df_clean.to_csv(self.output_dir / f"{base_name}_Cleaned.csv", index=False)
        
        if abnormal_records:
            df_abn = pd.DataFrame(abnormal_records).drop_duplicates()
            df_abn.to_csv(self.output_dir / f"{base_name}_Abnormal_Log.csv", index=False)
            
            # 复制图片供检查
            # 注意：这里使用了 config_pipeline 中的 ABNORMAL_CROPS_BASE
            crop_dest = ABNORMAL_CROPS_BASE / base_name
            crop_dest.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for _, rec in df_abn.iterrows():
                if self.copy_crop_for_review(base_name, rec['Filename'], rec['ROI_ID'], crop_dest):
                    count += 1
            print(f"  ⚠️  Found {len(df_abn)} issues. Copied {count} images for review.")
        else:
            print(f"  ✅ No issues found.")

    def run(self):
        print("\n" + "="*60)
        print("STAGE 1: Data Validation")
        print("="*60)
        csv_files = list(self.input_dir.glob("*.csv"))
        # 过滤掉已经处理过的文件
        csv_files = [f for f in csv_files if not any(x in f.name for x in ['_Cleaned', '_Log', '_Fixed'])]
        
        if not csv_files:
            print(f"❌ No input CSV files found in {self.input_dir}")
            return
            
        print(f"Found {len(csv_files)} CSV files to process.\n")
        for f in csv_files:
            self.process_single_csv(f)

# ================= 阶段2: 模型异常修正 =================
class Stage2_Correction:
    """阶段2: 使用 Ollama 模型修正异常"""
    def __init__(self, cleaned_dir, crops_base, output_dir):
        self.cleaned_dir = Path(cleaned_dir)
        self.crops_base = Path(crops_base)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_medians(self, csv_path):
        """计算每一列的中位数作为 prompt 的参考"""
        try:
            df = pd.read_csv(csv_path)
            medians = {}
            for col in df.columns:
                if not col.startswith('ROI_'): continue
                roi_type = get_roi_type(col)
                if roi_type in ['INTEGER', 'FLOAT']:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna()
                    vals = vals[vals > 0]
                    if len(vals) >= 5:
                        medians[col] = vals.median()
                elif roi_type == 'STATUS':
                    vc = df[col].value_counts()
                    if not vc.empty:
                        medians[col] = vc.index[0]
            return medians
        except:
            return {}

    def clean_llm_output(self, text):
        """清理 LLM 返回的多余字符"""
        if not text: return "ERROR"
        # 移除模型特殊token
        text = re.sub(r'<\|.*?\|>', '', text)
        text = text.strip().split('\n')[0].strip()
        # 尝试只提取数字/状态
        match = re.search(r'([0-9\.]+|OK|NG)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return text

    def run_inference(self, image_path, roi_id, median_val, original_val):
        """调用 Ollama"""
        try:
            # 使用 config_pipeline 中的 get_prompt 函数
            prompt = get_prompt(roi_id, 'correction', original_val, median_val)
            
            # 使用 7B 模型进行更准确的修正
            response = ollama.chat(
                model=OLLAMA_MODEL_7B, 
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [str(image_path)]
                }],
                options={'temperature': 0.0}
            )
            return self.clean_llm_output(response['message']['content'])
        except Exception as e:
            print(f"  ❌ Inference Error: {e}")
            return "ERROR"

    def find_image(self, filename, roi_id):
        """查找图片"""
        folder = os.path.splitext(filename)[0]
        # 直接在 crops_base 下查找文件夹 (因为 config 中已指定到具体日期目录)
        for ext in ['jpg', 'png']:
            p = self.crops_base / folder / f"{roi_id}.{ext}"
            if p.exists(): return p
        return None

    def process_log(self, log_path):
        filename = log_path.name
        print(f"\n🔧 Correcting Abnormalities: {filename}")
        
        try:
            df_bad = pd.read_csv(log_path)
            if df_bad.empty: return

            # 加载对应的 Cleaned CSV 以计算 Context
            base_name = filename.replace("_Abnormal_Log.csv", "")
            cleaned_path = self.cleaned_dir / f"{base_name}_Cleaned.csv"
            medians = {}
            if cleaned_path.exists():
                medians = self.calculate_medians(cleaned_path)
            
            # 断点续传: 检查是否有已保存的进度
            out_name = filename.replace(".csv", "_AI_Fixed.csv")
            out_path = self.output_dir / out_name
            
            if out_path.exists():
                df_progress = pd.read_csv(out_path)
                # 检查是否所有行都已处理完成
                if 'AI_Fixed' in df_progress.columns:
                    # 统计未处理的行数 (AI_Fixed 为空或 NaN)
                    unprocessed_mask = df_progress['AI_Fixed'].isna() | (df_progress['AI_Fixed'] == "")
                    unprocessed_count = unprocessed_mask.sum()
                    
                    if unprocessed_count == 0:
                        print(f"  ⏭️  Skipped (all {len(df_progress)} items already processed)")
                        return
                    
                    print(f"  📂 Resuming from checkpoint: {len(df_progress) - unprocessed_count}/{len(df_progress)} done")
                    df_bad = df_progress
                else:
                    df_bad['AI_Fixed'] = ""
            else:
                df_bad['AI_Fixed'] = ""
            
            # 处理计数器 (用于定期保存)
            save_interval = 5  # 每处理5条保存一次
            processed_since_save = 0
            
            for idx, row in df_bad.iterrows():
                # 断点续传: 跳过已处理的行
                existing_val = row['AI_Fixed'] if 'AI_Fixed' in row.index else ""
                # 检查是否已处理: 非空、非NaN、非ERROR
                if pd.notna(existing_val) and str(existing_val).strip() not in ["", "ERROR"]:
                    continue
                
                roi_id = row['ROI_ID']
                img_path = self.find_image(row['Filename'], roi_id)
                
                if not img_path:
                    df_bad.at[idx, 'AI_Fixed'] = "Image Not Found"
                    processed_since_save += 1
                else:
                    curr_median = medians.get(roi_id, None)
                    fixed_val = self.run_inference(img_path, roi_id, curr_median, row['Value'])
                    
                    # 计算当前进度
                    done_count = (df_bad['AI_Fixed'].notna() & (df_bad['AI_Fixed'] != "")).sum() + 1
                    print(f"  [{done_count}/{len(df_bad)}] {roi_id}: {row['Value']} → {fixed_val}")
                    df_bad.at[idx, 'AI_Fixed'] = fixed_val
                    processed_since_save += 1
                
                # 定期保存 checkpoint
                if processed_since_save >= save_interval:
                    df_bad.to_csv(out_path, index=False)
                    processed_since_save = 0
            
            # 最终保存
            df_bad.to_csv(out_path, index=False)
            print(f"  ✅ Saved corrections to {out_name}")
            
        except Exception as e:
            print(f"  ❌ Error processing log: {e}")

    def run(self):
        print("\n" + "="*60)
        print(f"STAGE 2: Model Correction (Using {OLLAMA_MODEL_7B})")
        print("="*60)
        # 查找由 Stage 1 生成的 Abnormal Logs
        logs = list(self.cleaned_dir.glob("*_Abnormal_Log.csv"))
        if not logs:
            print("✅ No abnormalities to correct.")
            return
        
        print(f"Found {len(logs)} logs to process.\n")
        for log in logs:
            self.process_log(log)

# ================= 阶段3: 合并结果 =================
class Stage3_Merge:
    """阶段3: 合并"""
    def __init__(self, cleaned_dir, fixed_dir, output_dir):
        self.cleaned_dir = Path(cleaned_dir)
        self.fixed_dir = Path(fixed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print("\n" + "="*60)
        print("STAGE 3: Merging Results")
        print("="*60)
        
        fixed_logs = list(self.fixed_dir.glob("*_AI_Fixed.csv"))
        if not fixed_logs:
            print("No corrections to merge.")
            return

        for log in fixed_logs:
            base_name = log.name.replace("_Abnormal_Log_AI_Fixed.csv", "")
            cleaned_csv = self.cleaned_dir / f"{base_name}_Cleaned.csv"
            
            if not cleaned_csv.exists(): continue
            
            # 断点续传: 检查是否已合并
            out_path = self.output_dir / f"{base_name}_Final.csv"
            if out_path.exists():
                print(f"⏭️  Skipped merge: {base_name} (already exists: {out_path.name})")
                continue
            
            print(f"🔀 Merging: {base_name}")
            df_clean = pd.read_csv(cleaned_csv)
            df_fixed = pd.read_csv(log)
            
            count = 0
            for _, row in df_fixed.iterrows():
                val = row['AI_Fixed']
                if pd.isna(val) or val in ["ERROR", "Image Not Found"]: continue
                
                # 查找并更新
                mask = df_clean['Filename'] == row['Filename']
                if mask.any():
                    df_clean.loc[mask, row['ROI_ID']] = val
                    count += 1
            
            df_clean.to_csv(out_path, index=False)
            print(f"  ✅ Updated {count} values → {out_path.name}")

# ================= 主程序 =================
def main():
    # 1. 路径配置 (来自 Config，但在这里具体化)
    # BATCH_NAME 和 CROP_DIR_NAME 已经在 config_pipeline.py 中定义好了
    # 我们直接使用 config 中的 OUTPUT_BASE 下的目录
    
    # 输入CSV: 如果有 'CSV_Results' 子目录则用之，否则用 stage1 根目录
    CSV_SOURCE = STAGE_1_OCR / "CSV_Results"
    if not CSV_SOURCE.exists():
        CSV_SOURCE = STAGE_1_OCR
        
    CROPS_SOURCE = DEBUG_CROPS_BASE  # 来自 config, 对应 12-22-2025
    
    # 阶段 2 输出目录
    STAGE2_OUT = STAGE_2_CLEANED
    # 阶段 3 输出目录
    STAGE3_OUT = STAGE_3_3B_CORRECTED
    
    print("\n" + "="*80)
    print("🚀 AUTOMATED DATA PIPELINE START")
    print(f"📂 CSV Source:   {CSV_SOURCE}")
    print(f"🖼️  Crops Source: {CROPS_SOURCE}")
    print(f"🤖 Model:        {OLLAMA_MODEL_7B}")  # 使用 7B 进行修正
    print("="*80)
    
    if not CSV_SOURCE.exists():
        print(f"❌ Error: Input directory {CSV_SOURCE} does not exist!")
        return

    # --- Step 1: Validate ---
    s1 = Stage1_DataCleaning(CSV_SOURCE, STAGE2_OUT, CROPS_SOURCE)
    s1.run()
    
    # --- Step 2: Correct ---
    # 注意：Stage 2 读取 Stage 1 输出的 Log
    s2 = Stage2_Correction(STAGE2_OUT, CROPS_SOURCE, STAGE3_OUT)
    s2.run()
    
    # --- Step 3: Merge ---
    # 注意：Stage 3 读取 Stage 2 输出的 Fixed Log 和 Stage 1 的 Cleaned CSV
    s3 = Stage3_Merge(STAGE2_OUT, STAGE3_OUT, STAGE3_OUT)
    s3.run()
    
    print("\n✅ PIPELINE FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()