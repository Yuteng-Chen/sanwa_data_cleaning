"""
3B模型数据清理管道 - 完整自动化流程
3B Model Data Cleaning Pipeline - Full Automation

流程 Pipeline:
1. 数据验证和清理（基于增强逻辑）
2. 异常检测和标记
3. 使用3B模型修正异常值
4. 合并修正结果
5. 重新标记和分类

输入 Input: Stage 1 OCR结果
输出 Output: Stage 4 已标记的清理数据
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
import urllib.request     # <-- 新增
import base64
from pathlib import Path
from datetime import datetime
import concurrent.futures
import threading
from collections import defaultdict

# 导入配置
from config_pipeline import *

print_lock = threading.Lock()

# ================= 阶段1: 数据验证和清理 =================
class DataValidator:
    """数据验证器 - 检测异常值"""
    
    def __init__(self, max_decimals=3, outlier_threshold=5.0, z_score_threshold=3.0):
        self.max_decimals = max_decimals
        self.outlier_threshold = outlier_threshold  # 比率阈值 (5x median)
        self.z_score_threshold = z_score_threshold  # Z-Score 阈值 (默认 3.0)
    
    def validate_value(self, val, data_type):
        """
        验证单个值
        返回: (is_valid, clean_val, reason)
        """
        val_str = str(val).strip()
        if pd.isna(val) or val_str == '' or val_str.lower() == 'nan':
            return False, val, "Empty/NaN"
        
        if data_type == 'STATUS':
            val_upper = val_str.upper()
            # 分类规则：只有OK或NG两种输出
            # 以O开头 → OK (O, OH, OK, 0)
            # 以N开头 → NG (N, NG, NH, NO, NaN)
            
            if val_upper.startswith('O') or val_upper == '0' or 'OK' in val_upper:
                return True, 'OK', None
            if val_upper.startswith('N'):
                # N, NG, NH, NO, NaN 都分类为NG
                return True, 'NG', None
            if val_upper == 'K':
                return True, 'OK', None
            if val_upper == 'G':
                return True, 'NG', None
            # 空白或无法识别 → 标记为需要检查
            if val_upper in ['', 'NAN', 'NA', 'NULL', 'NONE']:
                return False, val, "Empty/Invalid Status - needs review"
            # 其他情况标记为需要检查
            return False, val, "Unknown Status - needs review"
        
        elif data_type == 'INTEGER':
            # 先尝试处理浮点数格式（如 '95.0' → 95，避免 '95.0' → '950' 的bug）
            try:
                return True, int(float(val_str)), None
            except:
                pass
            # 如果上面失败，再用正则清理非数字字符
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
        """
        统计异常值检测 - 双重检测机制
        
        Method 1: Ratio-based (5x median) - 检测严重的数量级错误
                  例如: 177 vs median=1.18 → ratio=150x → 检测到
        
        Method 2: Z-Score (标准差) - 检测偏离正常范围的值
                  例如: 2.03 vs mean=1.2, std=0.05 → Z=16.6 → 检测到
                  
        返回: [(index, reason), ...] 包含异常原因的元组列表
        """
        if data_type not in ['FLOAT', 'INTEGER']:
            return []
        
        nums = pd.to_numeric(series, errors='coerce').dropna()
        if len(nums) < 5:
            return []
        
        median = nums.median()
        mean = nums.mean()
        std = nums.std()
        
        outlier_results = []  # [(index, reason), ...]
        
        for idx, val in series.items():
            try:
                val_float = float(val)
                
                # Skip zero/invalid values
                if val_float == 0 or pd.isna(val_float):
                    continue
                
                # Method 1: Ratio-based detection (5x median)
                # 检测严重的数量级错误，如缺少小数点 (177 vs 1.77)
                if median != 0:
                    ratio = val_float / median
                    if ratio > self.outlier_threshold or ratio < (1.0 / self.outlier_threshold):
                        outlier_results.append((idx, "Statistical Outlier (Likely Missing Decimal)"))
                        continue  # 已检测，跳过 Z-Score
                
                # Method 2: Z-Score detection
                # 检测偏离正常范围的值，如 2.03 vs 正常范围 1.1-1.3
                if std > 0:
                    z_score = abs((val_float - mean) / std)
                    if z_score > self.z_score_threshold:
                        outlier_results.append((idx, f"Z-Score Outlier (Z={z_score:.2f}, threshold={self.z_score_threshold})"))
                
            except:
                pass
        
        return outlier_results
    
    def detect_outliers_zscore_only(self, series, data_type):
        """
        仅使用 Z-Score 检测异常值
        适用于需要更精细检测的场景
        
        Z-Score 阈值参考:
        - Z > 2.0: ~5% 异常 (95% 置信区间)
        - Z > 2.5: ~1.2% 异常 
        - Z > 3.0: ~0.3% 异常 (99.7% 置信区间) [推荐]
        - Z > 3.5: ~0.05% 异常 (更保守)
        """
        if data_type not in ['FLOAT', 'INTEGER']:
            return []
        
        nums = pd.to_numeric(series, errors='coerce').dropna()
        if len(nums) < 5:
            return []
        
        mean = nums.mean()
        std = nums.std()
        
        if std == 0:
            return []
        
        outlier_results = []
        
        for idx, val in series.items():
            try:
                val_float = float(val)
                if val_float == 0 or pd.isna(val_float):
                    continue
                    
                z_score = abs((val_float - mean) / std)
                if z_score > self.z_score_threshold:
                    outlier_results.append((idx, f"Z-Score Outlier (Z={z_score:.2f})"))
            except:
                pass
        
        return outlier_results

class Stage1_DataCleaning:
    """阶段1: 数据清理和异常检测"""
    
    def __init__(self, input_dir, output_dir, crops_base):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.crops_base = Path(crops_base)
        self.validator = DataValidator(MAX_DECIMALS, OUTLIER_THRESHOLD, Z_SCORE_THRESHOLD)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_config_for_file(self, df):
        """识别文件对应的ROI配置"""
        for config in ROI_CONFIGS:
            if config['Trigger_Col'] in df.columns:
                return config
        return None
    
    def copy_crop_for_review(self, csv_base_name, filename, roi_id, dest_folder):
        """复制异常裁剪图像供人工检查"""
        try:
            folder_name = os.path.splitext(filename)[0]
            
            # 搜索路径
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
        """处理单个CSV文件"""
        filename = csv_path.name
        base_name = csv_path.stem
        
        print(f"\n📄 Processing: {filename}...")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ❌ Error reading CSV: {e}")
            return
        
        config = self.get_config_for_file(df)
        if not config:
            print(f"  ⚠️  Skipped: Unknown format")
            return
        
        roi_map = config['Columns']
        
        # 排序
        if 'Filename' in df.columns:
            df.sort_values(by='Filename', inplace=True)
        
        df_clean = df.copy()
        abnormal_records = []
        
        # 阶段1: 逐行验证
        print(f"  🔍 Validating {len(df)} rows...")
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
                            'Timestamp': row.get('ROI_52', ''),
                            'ROI_ID': roi_col,
                            'Value': val,
                            'Reason': reason
                        })
        
        # 阶段2: 统计异常检测 (双重检测: Ratio-based + Z-Score)
        print(f"  📊 Detecting statistical outliers (Ratio + Z-Score)...")
        for roi_col, dtype in roi_map.items():
            if roi_col in df_clean.columns:
                # 返回格式: [(index, reason), ...]
                outlier_results = self.validator.detect_outliers(df_clean[roi_col], dtype)
                for idx, reason in outlier_results:
                    abnormal_records.append({
                        'Filename': df_clean.at[idx, 'Filename'],
                        'Timestamp': df_clean.at[idx, 'ROI_52'] if 'ROI_52' in df_clean.columns else '',
                        'ROI_ID': roi_col,
                        'Value': df_clean.at[idx, roi_col],
                        'Reason': reason  # 现在包含具体的检测方法和详情
                    })
        
        # 保存结果
        df_clean.to_csv(self.output_dir / f"{base_name}_Cleaned.csv", index=False)
        
        if abnormal_records:
            df_abn = pd.DataFrame(abnormal_records).drop_duplicates()
            df_abn.to_csv(self.output_dir / f"{base_name}_Abnormal_Log.csv", index=False)
            
            # 复制异常图像
            crop_dest = ABNORMAL_CROPS_BASE / base_name
            crop_dest.mkdir(parents=True, exist_ok=True)
            
            count = sum(1 for _, rec in df_abn.iterrows() 
                       if self.copy_crop_for_review(base_name, rec['Filename'], 
                                                   rec['ROI_ID'], crop_dest))
            
            print(f"  ⚠️  Found {len(df_abn)} issues. Copied {count} images.")
        else:
            print(f"  ✅ No issues found.")
        
        print(f"  💾 Saved: {base_name}_Cleaned.csv")
    
    def run(self):
        """运行清理流程"""
        print("\n" + "="*60)
        print("STAGE 1: Data Validation and Cleaning")
        print("="*60)
        
        csv_files = list(self.input_dir.glob("**/*.csv"))
        csv_files = [f for f in csv_files if not any(x in f.name for x in ['_Cleaned', '_Log', '_Abnormal'])]
        
        if not csv_files:
            print("❌ No CSV files found in input directory")
            return
        
        print(f"Found {len(csv_files)} CSV files\n")
        
        for csv_file in csv_files:
            self.process_single_csv(csv_file)
        
        print("\n✅ Stage 1 Complete")

# ================= 阶段2: 3B模型异常修正 =================
class Stage2_3BCorrection:
    """阶段2: 使用3B模型修正异常值"""
    
    def __init__(self, cleaned_dir, abnormal_logs_dir, crops_base, output_dir):
        self.cleaned_dir = Path(cleaned_dir)
        self.abnormal_logs_dir = Path(abnormal_logs_dir)
        self.crops_base = Path(crops_base)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_roi_medians(self, csv_path):
        """
        从CSV计算每个ROI的median值
        返回: {roi_id: median_value}
        """
        try:
            df = pd.read_csv(csv_path)
            roi_medians = {}
            
            print(f"  📊 Calculating medians from {len(df)} rows...")
            
            for col in df.columns:
                if not col.startswith('ROI_'):
                    continue
                
                roi_type = get_roi_type(col)
                
                # 只对数值类型计算median
                if roi_type in ['INTEGER', 'FLOAT']:
                    try:
                        # 转换为数值，忽略错误
                        vals = pd.to_numeric(df[col], errors='coerce').dropna()
                        # 过滤掉0值（可能是缺陷）
                        vals = vals[vals > 0]
                        
                        if len(vals) >= 5:  # 至少5个有效样本
                            roi_medians[col] = vals.median()
                            print(f"    ✓ {col}: Median={roi_medians[col]:.3f} (from {len(vals)} samples)")
                    except Exception as e:
                        print(f"    ⚠️  {col}: Could not calculate median - {e}")
                
                elif roi_type == 'STATUS':
                    # 对于STATUS，找出最常见的值
                    try:
                        value_counts = df[col].value_counts()
                        if not value_counts.empty:
                            roi_medians[col] = value_counts.index[0]  # 最常见的值
                            print(f"    ✓ {col}: Most common={roi_medians[col]}")
                    except Exception as e:
                        print(f"    ⚠️  {col}: Could not find mode - {e}")
            
            print(f"  📊 Calculated medians for {len(roi_medians)} ROI fields")
            return roi_medians
            
        except Exception as e:
            print(f"  ❌ Error calculating medians: {e}")
            return {}
    
    def clean_model_output(self, text, roi_type='FLOAT'):
        """
        保留模型原始输出，只移除模型内部控制tokens
        """
        if not text:
            return "ERROR"
        
        # 移除模型内部控制tokens（这些是模型格式标记，不是数据）
        import re
        special_tokens = [
            r'<\|im_start\|>',
            r'<\|im_end\|>',
            r'<\|endoftext\|>',
            r'<\|pad\|>',
            r'<\|assistant\|>',
            r'<\|user\|>',
            r'<\|system\|>',
        ]
        for token in special_tokens:
            text = re.sub(token, '', text)
        
        # 只取第一行第一个词（不修改数值内容）
        text = text.strip()
        text = text.split('\n')[0].strip()
        text = text.split()[0] if text.split() else text
        
        return text if text else "ERROR"
    
    def run_3b_inference(self, image_path, roi_id, median_val, ocr_value):
        """使用7B模型重新识别"""
        try:
            prompt = get_prompt(roi_id, 'correction', ocr_value, median_val)
            roi_type = get_roi_type(roi_id)
            
            response = ollama.chat(
                model=OLLAMA_MODEL_7B,  # Using 7B for better accuracy
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [str(image_path)]
                }],
                options={'temperature': 0.0, 'num_predict': 30}
            )
            
            text = response['message']['content']
            
            # 使用增强的清理函数
            text = self.clean_model_output(text, roi_type)
            
            # 后处理：额外检查
            text = self.post_process_number(text, roi_type, median_val)
            
            return text if text else "ERROR"
            
        except Exception as e:
            print(f"  [3B Error] {e}")
            return "ERROR"
    
    def post_process_number(self, text, roi_type, median_val):
        """
        后处理数字输出 - 只做基本清理，不自动修复重复模式
        - 清理markdown标记
        - 截断小数位数到3位
        - 检测问题并警告（不自动修复）
        """
        if not text or text in ["ERROR", "NA", "Image Not Found"]:
            return text
        
        original = text
        
        if roi_type == 'FLOAT':
            # 1. 检测多小数点 (e.g., '5.7.726') - 只警告，不自动修复
            decimal_count = text.count('.')
            if decimal_count > 1:
                print(f"    ⚠️ [Warning] Multiple decimals detected: '{text}' - keeping as-is for review")
            
            # 2. 检测可能的重复模式 - 只警告，不自动修复
            repeat_match = re.match(r'^(-?\d+\.\d{1,3})\1+', text)
            if repeat_match:
                print(f"    ⚠️ [Warning] Possible repeat pattern: '{text}' - keeping as-is for review")
            
            # 3. 只截断过长的小数位数（这是格式标准化，不是修复）
            if '.' in text:
                parts = text.split('.')
                if len(parts) == 2 and len(parts[1]) > 3:
                    text = f"{parts[0]}.{parts[1][:3]}"
                    if text != original:
                        print(f"    [Truncate] '{original}' → '{text}' (max 3 decimals)")
        
        elif roi_type == 'INTEGER':
            # 1. 检测小数点 - 只警告
            if '.' in text:
                print(f"    ⚠️ [Warning] Decimal in INTEGER: '{text}' - keeping for review")
            
            # 2. 检测可能的重复模式 - 只警告，不自动修复
            clean_text = re.sub(r'[^\d-]', '', text)
            if clean_text:
                length = len(clean_text.lstrip('-'))
                for repeat_len in range(1, length // 2 + 1):
                    base = clean_text[:repeat_len + (1 if clean_text.startswith('-') else 0)]
                    if clean_text.startswith('-'):
                        pattern = base + base[1:] * ((length // repeat_len) - 1)
                    else:
                        pattern = base * (length // repeat_len)
                    if pattern == clean_text and length >= repeat_len * 2:
                        print(f"    ⚠️ [Warning] Possible repeat pattern: '{text}' - keeping as-is for review")
                        break
        
        return text
    
    def find_crop_image(self, csv_base, filename, roi_id):
        """查找裁剪图像 - 直接从DEBUG_CROPS_BASE获取 (flattened结构)"""
        folder_name = os.path.splitext(filename)[0]
        
        # Flattened结构: debug_crops/2025-12-16 17.20.09/ROI_1.jpg
        for ext in ['jpg', 'png']:
            img_path = DEBUG_CROPS_BASE / folder_name / f"{roi_id}.{ext}"
            print(img_path)
            if img_path.exists():
                return img_path
        return None
    
    def process_abnormal_log(self, log_path, cleaned_csv_path):
        """处理异常日志"""
        filename = log_path.name
        csv_base = filename.replace("_Abnormal_Log.csv", "")
        
        print(f"\n🔧 Correcting: {filename}")
        
        try:
            df_bad = pd.read_csv(log_path)
            if df_bad.empty:
                return
            
            # 从输入CSV加载并计算Median值
            roi_medians = {}
            if cleaned_csv_path.exists():
                roi_medians = self.calculate_roi_medians(cleaned_csv_path)
            else:
                print(f"  ⚠️  Cleaned CSV not found, proceeding without median context")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return
        
        df_bad['AI_3B_Corrected'] = ""
        
        for idx, row in df_bad.iterrows():
            roi_id = row['ROI_ID']
            
            # 查找图像
            img_path = self.find_crop_image(csv_base, row['Filename'], roi_id)
            
            if not img_path:
                print(f"Line 560 - img_path: {img_path}")
                
                df_bad.at[idx, 'AI_3B_Corrected'] = "Image Not Found"
                continue
            
            # 获取median
            curr_median = roi_medians.get(roi_id, None)
            
            # 3B推理
            fixed_val = self.run_3b_inference(img_path, roi_id, curr_median, row['Value'])
            
            print(f"  [{idx+1}/{len(df_bad)}] {roi_id}: {row['Value']} → {fixed_val} (Median: {curr_median})")
            
            df_bad.at[idx, 'AI_3B_Corrected'] = fixed_val
            
            # 动态更新median（加权平均）
            try:
                val_num = float(fixed_val)
                if val_num > 0:
                    if curr_median:
                        roi_medians[roi_id] = (curr_median * 0.9) + (val_num * 0.1)
                    else:
                        roi_medians[roi_id] = val_num
            except:
                pass
        
        # 保存
        out_name = filename.replace(".csv", "_AI_3B_Fixed.csv")
        df_bad.to_csv(self.output_dir / out_name, index=False)
        print(f"  ✅ Saved: {out_name}")
    
    def run(self):
        """运行3B修正流程"""
        print("\n" + "="*60)
        print("STAGE 2: 3B Model Correction")
        print("="*60)
        
        abnormal_logs = list(self.abnormal_logs_dir.glob("*_Abnormal_Log.csv"))
        
        if not abnormal_logs:
            print("✅ No abnormal logs found - data is clean!")
            return
        
        print(f"Found {len(abnormal_logs)} abnormal logs\n")
        
        for log_path in abnormal_logs:
            base_name = log_path.name.replace("_Abnormal_Log.csv", "")
            cleaned_path = self.cleaned_dir / f"{base_name}_Cleaned.csv"
            
            if cleaned_path.exists():
                self.process_abnormal_log(log_path, cleaned_path)
            else:
                print(f"⚠️  Cleaned CSV not found for {base_name}")
        
        print("\n✅ Stage 2 Complete")

# ================= 阶段3: 合并修正结果 =================
class Stage3_MergeCorrections:
    """阶段3: 将3B修正结果合并回原数据集"""
    
    def __init__(self, cleaned_dir, fixed_logs_dir, output_dir):
        self.cleaned_dir = Path(cleaned_dir)
        self.fixed_logs_dir = Path(fixed_logs_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def merge_single_file(self, fixed_log_path, cleaned_csv_path):
        """合并单个文件的修正"""
        filename = fixed_log_path.name
        
        print(f"\n🔀 Merging: {filename}")
        
        try:
            df_fixed = pd.read_csv(fixed_log_path)
            df_original = pd.read_csv(cleaned_csv_path)
        except Exception as e:
            print(f"  ❌ Read error: {e}")
            return
        
        if 'Filename' not in df_fixed.columns or 'ROI_ID' not in df_fixed.columns:
            print(f"  ⚠️  Missing columns")
            return
        
        update_count = 0
        
        for _, row in df_fixed.iterrows():
            filename_val = row['Filename']
            roi_col = row['ROI_ID']
            new_val = row.get('AI_3B_Corrected', '')
            
            # 跳过无效值
            if pd.isna(new_val) or str(new_val).strip() in ["", "Image Not Found", "ERROR"]:
                continue
            
            new_val = str(new_val).strip().replace("'", "").replace('"', '')
            
            if roi_col not in df_original.columns:
                continue
            
            # 更新
            match_mask = df_original['Filename'] == filename_val
            if match_mask.any():
                df_original.loc[match_mask, roi_col] = new_val
                update_count += 1
        
        # 保存
        base_name = cleaned_csv_path.stem.replace("_Cleaned", "")
        save_path = self.output_dir / f"{base_name}_3B_Corrected.csv"
        df_original.to_csv(save_path, index=False)
        
        print(f"  ✅ Updated {update_count} cells → {save_path.name}")
    
    def run(self):
        """运行合并流程"""
        print("\n" + "="*60)
        print("STAGE 3: Merge 3B Corrections")
        print("="*60)
        
        fixed_logs = list(self.fixed_logs_dir.glob("*_AI_3B_Fixed.csv"))
        
        if not fixed_logs:
            print("✅ No fixed logs to merge")
            return
        
        print(f"Found {len(fixed_logs)} fixed logs\n")
        
        for log_path in fixed_logs:
            base_name = log_path.name.replace("_Abnormal_Log_AI_3B_Fixed.csv", "")
            cleaned_path = self.cleaned_dir / f"{base_name}_Cleaned.csv"
            
            if cleaned_path.exists():
                self.merge_single_file(log_path, cleaned_path)
        
        print("\n✅ Stage 3 Complete")

# ================= 主流程 =================
def main():
    """3B管道主流程 - 针对 2025-12-19cslot 批次"""
    
    # ================= 路径配置 (Path Configuration) =================
    
    # 1. 定义批次名称（这将作为【最终输出文件夹】的名字）
    BATCH_NAME = "2025-12-19cslot"
    
    # 2. 定义内部日期文件夹（对应输入路径中的日期子目录）
    INNER_DATE = "2025-12-19"
    
    # 3. CSV 输入目录
    # 基础路径: .../stage1_ocr_results
    BASE_INPUT_DIR = Path("/scratch/prj0000000262/ocr_data/QwenModel/SanwaDataPreprocess/pipeline_output/stage1_ocr_results")
    # 完整路径: .../stage1_ocr_results/2025-12-19cslot/2025-12-19/CSV_Results
    TARGET_CSV_DIR = BASE_INPUT_DIR / BATCH_NAME / "CSV_Results"
    
    # 4. Debug Crops 根目录 (固定路径)
    DEBUG_CROPS_ROOT = Path("/scratch/prj0000000262/ocr_data/QwenModel/SanwaDataPreprocess/pipeline_output/stage1_ocr_results/debug_crops")
    
    # 5. 构建实际的截图搜索路径
    # 完整路径: .../debug_crops/2025-12-19cslot/2025-12-19
    TARGET_CROPS_DIR = DEBUG_CROPS_ROOT / BATCH_NAME 
    
    # 6. 【关键】定义输出目录 - 直接使用 BATCH_NAME 作为文件夹名
    # 结果将存放在: .../stage3_corrected/2025-12-19cslot/
    DATED_STAGE_2_DIR = Path(STAGE_2_CLEANED) / BATCH_NAME
    DATED_STAGE_3_DIR = Path(STAGE_3_3B_CORRECTED) / BATCH_NAME
    
    # ==============================================================

    print("\n" + "="*80)
    print("🤖 3B MODEL DATA CLEANING PIPELINE")
    print(f"📂 Batch Name:    {BATCH_NAME}")
    print(f"📄 Target CSV:    {TARGET_CSV_DIR}")
    print(f"🖼️  Target Crops:  {TARGET_CROPS_DIR}")
    print(f"💾 Output Dir:    {DATED_STAGE_3_DIR}")
    print("="*80)
    
    # 路径检查
    if not TARGET_CSV_DIR.exists():
        print(f"❌ Error: CSV Input directory not found: {TARGET_CSV_DIR}")
        return
    if not TARGET_CROPS_DIR.exists():
        print(f"⚠️ Warning: Crops directory not found: {TARGET_CROPS_DIR}")
        print("   (Proceeding, but Stage 2 3B-Correction will skip image lookups)")

    # 阶段1: 数据验证和清理
    # 注意：output_dir 指向 DATED_STAGE_2_DIR
    stage1 = Stage1_DataCleaning(
        input_dir=TARGET_CSV_DIR,
        output_dir=DATED_STAGE_2_DIR,
        crops_base=TARGET_CROPS_DIR
    )
    stage1.run()
    
    # 阶段2: 3B模型异常修正
    # 注意：输入从 DATED_STAGE_2_DIR 读取，输出到 DATED_STAGE_3_DIR
    stage2 = Stage2_3BCorrection(
        cleaned_dir=DATED_STAGE_2_DIR,
        abnormal_logs_dir=DATED_STAGE_2_DIR,
        crops_base=TARGET_CROPS_DIR,
        output_dir=DATED_STAGE_3_DIR
    )
    stage2.run()
    
    # 阶段3: 合并修正结果
    # 注意：输入输出都指向 DATED_STAGE_3_DIR (或从 STAGE 2 读取)
    stage3 = Stage3_MergeCorrections(
        cleaned_dir=DATED_STAGE_2_DIR,
        fixed_logs_dir=DATED_STAGE_3_DIR,
        output_dir=DATED_STAGE_3_DIR
    )
    stage3.run()
    
    print("\n" + "="*80)
    print("🎉 3B PIPELINE COMPLETE")
    print(f"📂 Final Output: {DATED_STAGE_3_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()