"""
7B模型验证管道 - 高精度验证和冗余消除
7B Model Verification Pipeline - High Precision Verification and Redundancy Removal

流程 Pipeline:
1. 标记数据状态（时间冻结、数据冗余）
2. 7B模型验证冗余不匹配
3. 应用7B修正
4. 重新标记
5. 消除冗余行
6. 生成最终数据集

输入 Input: Stage 3 - 3B修正后的数据
输出 Output: Stage 6 - 最终清洁数据集
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import glob
import shutil
import re
import urllib.request     # ✅ 已保留：用于无依赖HTTP请求
import base64             # ✅ 已保留：用于图片编码
from pathlib import Path
from datetime import datetime
import threading

# 导入配置
from config_pipeline import *

print_lock = threading.Lock()

# ================= 阶段4: 数据标记 =================
class Stage4_DataLabeling:
    """阶段4: 标记数据状态（时间、冗余）"""
    
    def __init__(self, input_dir, output_dir, crops_base):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.crops_base = Path(crops_base)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_pc_filename_time(self, filename):
        """从文件名提取时间戳"""
        try:
            match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}\.\d{2}\.\d{2})', str(filename))
            if match:
                clean_str = match.group(1).replace('.', ':')
                return datetime.strptime(clean_str, '%Y-%m-%d %H:%M:%S')
        except:
            return None
        return None
    
    def get_positional_data(self, row, columns_list):
        """获取用于比较的位置数据"""
        start_idx = 4
        if 'ROI_51' in columns_list:
            end_idx = columns_list.index('ROI_51')
        elif 'ROI_52' in columns_list:
            end_idx = columns_list.index('ROI_52')
        else:
            end_idx = len(columns_list)
        
        if start_idx >= end_idx:
            return [], []
        
        cols_to_check = columns_list[start_idx:end_idx]
        values = [str(row.get(col, '')).strip() for col in cols_to_check]
        return values, cols_to_check
    
    def calculate_similarity(self, list_a, list_b):
        """计算相似度"""
        if not list_a or not list_b or len(list_a) != len(list_b):
            return 0.0
        matches = sum(1 for a, b in zip(list_a, list_b) if a == b)
        return matches / len(list_a)
    
    def get_config_for_file(self, df):
        """获取文件配置"""
        for config in ROI_CONFIGS:
            if config['Trigger_Col'] in df.columns:
                return config
        return None
    
    def copy_crop_for_review(self, csv_base, filename, roi_id, dest_folder):
        """复制裁剪图像"""
        try:
            folder_name = os.path.splitext(filename)[0]
            
            potential_paths = [
                self.crops_base / csv_base / folder_name / f"{roi_id}.jpg",
                self.crops_base / csv_base / folder_name / f"{roi_id}.png",
                self.crops_base / folder_name / f"{roi_id}.jpg",
                self.crops_base / folder_name / f"{roi_id}.png",
            ]
            
            for p in potential_paths:
                if p.exists():
                    target_folder = dest_folder / folder_name
                    target_folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy(p, target_folder / p.name)
                    return True
            return False
        except:
            return False
    
    def process_single_csv(self, csv_path):
        """处理单个CSV并标记"""
        filename = csv_path.name
        base_name = csv_path.stem.replace("_Final", "")
        
        # 获取自适应阈值
        similarity_threshold = get_similarity_threshold(filename)
        
        print(f"\n🏷️  Labeling: {filename} (Threshold: {similarity_threshold})")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return
        
        config = self.get_config_for_file(df)
        if not config:
            print(f"  ⚠️  Skipped: Unknown format")
            return
        
        if 'Filename' in df.columns:
            df.sort_values(by='Filename', inplace=True)
        
        df_clean = df.copy()
        
        # 初始化新列
        df_clean['Time_Status'] = 'Unknown'
        df_clean['Data_Redundancy'] = 'Unknown'
        df_clean['Matched_File'] = ''
        df_clean['Duration_Since_Change'] = 0.0
        
        redundancy_mismatch_records = []
        
        # 转换为记录列表
        rows_list = df_clean.to_dict('records')
        columns_list = df_clean.columns.tolist()
        all_row_data = [self.get_positional_data(row, columns_list) for row in rows_list]
        
        prev_plc_time_str = None
        state_start_pc_time = None
        
        print(f"  📊 Analyzing {len(rows_list)} rows for time/redundancy patterns...")
        
        for i in range(len(rows_list)):
            curr_row = rows_list[i]
            curr_idx = df_clean.index[i]
            curr_filename = curr_row.get('Filename', '')
            
            curr_pc_obj = self.parse_pc_filename_time(curr_filename)
            curr_plc_str = str(curr_row.get('ROI_52', '')).strip()
            
            curr_vals, curr_cols = all_row_data[i]
            prev_vals, _ = all_row_data[i-1] if i > 0 else ([], [])
            next_vals, _ = all_row_data[i+1] if i < len(rows_list)-1 else ([], [])
            
            prev_filename = df_clean.at[df_clean.index[i-1], 'Filename'] if i > 0 else ""
            next_filename = df_clean.at[df_clean.index[i+1], 'Filename'] if i < len(rows_list)-1 else ""
            
            # 时间状态逻辑
            time_status = "New Time State"
            duration = 0.0
            
            if i == 0 or curr_pc_obj is None:
                time_status = "New Time State (Start)"
                state_start_pc_time = curr_pc_obj
            else:
                if curr_plc_str == prev_plc_time_str:
                    time_status = "Time Static"
                    if state_start_pc_time and curr_pc_obj:
                        duration = (curr_pc_obj - state_start_pc_time).total_seconds()
                    if duration > FROZEN_THRESHOLD_SECONDS:
                        time_status = "Time Frozen (>10s)"
                else:
                    state_start_pc_time = curr_pc_obj
            prev_plc_time_str = curr_plc_str
            
            # 模糊冗余逻辑
            is_redundant_prev = False
            is_redundant_next = False
            
            similarity_prev = self.calculate_similarity(curr_vals, prev_vals)
            if i > 0 and similarity_prev >= similarity_threshold:
                is_redundant_prev = True
                for k in range(len(curr_vals)):
                    if curr_vals[k] != prev_vals[k]:
                        redundancy_mismatch_records.append({
                            'Filename_Current': curr_filename,
                            'Filename_Compared': prev_filename,
                            'ROI_ID': curr_cols[k],
                            'Value_Current': curr_vals[k],
                            'Value_Compared': prev_vals[k],
                            'Similarity_Score': round(similarity_prev, 2),
                            'Reason': 'Redundant Row Value Mismatch'
                        })
            
            similarity_next = self.calculate_similarity(curr_vals, next_vals)
            if i < len(rows_list)-1 and similarity_next >= similarity_threshold:
                is_redundant_next = True
            
            data_redundancy_list = []
            matched_files_list = []
            if not is_redundant_prev and not is_redundant_next:
                data_redundancy = "Unique"
            else:
                if is_redundant_prev:
                    data_redundancy_list.append(f"Redundant Prev ({int(similarity_prev*100)}%)")
                    matched_files_list.append(f"Prev: {prev_filename}")
                if is_redundant_next:
                    data_redundancy_list.append(f"Redundant Next ({int(similarity_next*100)}%)")
                    matched_files_list.append(f"Next: {next_filename}")
                data_redundancy = " & ".join(data_redundancy_list)
            
            df_clean.at[curr_idx, 'Time_Status'] = time_status
            df_clean.at[curr_idx, 'Data_Redundancy'] = data_redundancy
            df_clean.at[curr_idx, 'Matched_File'] = " | ".join(matched_files_list)
            df_clean.at[curr_idx, 'Duration_Since_Change'] = round(duration, 2)
        
        # 保存结果
        df_clean.to_csv(self.output_dir / f"{base_name}_Labeled.csv", index=False)
        
        # 保存冗余不匹配日志
        if redundancy_mismatch_records:
            df_mis = pd.DataFrame(redundancy_mismatch_records).drop_duplicates()
            df_mis.to_csv(self.output_dir / f"{base_name}_Redundancy_Mismatch_Log.csv", index=False)
            
            # 【注意】这里使用了 crops_base
            mis_dest = self.output_dir / "mismatch_crops" / base_name
            mis_dest.mkdir(parents=True, exist_ok=True)
            
            count = sum(1 for _, r in df_mis.iterrows() 
                       if self.copy_crop_for_review(base_name, r['Filename_Current'], 
                                                    r['ROI_ID'], mis_dest))
            
            print(f"  ⚠️  Redundancy Mismatches: {len(df_mis)} (Copied {count} images)")
        
        print(f"  ✅ Labeled: {base_name}_Labeled.csv")
    
    def run(self):
        """运行标记流程"""
        print("\n" + "="*60)
        print("STAGE 4: Data Labeling (Time/Redundancy Analysis)")
        print("="*60)
        
        csv_files = list(self.input_dir.glob("*_Final.csv"))
        
        if not csv_files:
            print("❌ No 3B corrected files found")
            return
        
        print(f"Found {len(csv_files)} files\n")
        
        for csv_file in csv_files:
            self.process_single_csv(csv_file)
        
        print("\n✅ Stage 4 Complete")

# ================= 阶段5: 7B验证 =================
class Stage5_7BVerification:
    """阶段5: 使用7B模型验证冗余不匹配"""
    
    def __init__(self, labeled_dir, output_dir, crops_base):
        self.labeled_dir = Path(labeled_dir)
        self.output_dir = Path(output_dir)
        self.crops_base = Path(crops_base)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_roi_medians(self, csv_path):
        """从CSV计算每个ROI的median值"""
        try:
            df = pd.read_csv(csv_path)
            roi_medians = {}
            
            print(f"  📊 Calculating medians from {len(df)} rows...")
            
            for col in df.columns:
                if not col.startswith('ROI_'):
                    continue
                
                roi_type = get_roi_type(col)
                
                if roi_type in ['INTEGER', 'FLOAT']:
                    try:
                        vals = pd.to_numeric(df[col], errors='coerce').dropna()
                        vals = vals[vals > 0]
                        if len(vals) >= 5:
                            roi_medians[col] = vals.median()
                            print(f"    ✓ {col}: Median={roi_medians[col]:.3f} (from {len(vals)} samples)")
                    except:
                        pass
                elif roi_type == 'STATUS':
                    try:
                        value_counts = df[col].value_counts()
                        if not value_counts.empty:
                            roi_medians[col] = value_counts.index[0]
                            print(f"    ✓ {col}: Most common={roi_medians[col]}")
                    except:
                        pass
            
            print(f"  📊 Calculated medians for {len(roi_medians)} ROI fields")
            return roi_medians
            
        except Exception as e:
            print(f"  ⚠️  Error calculating medians: {e}")
            return {}
    
    def get_prompt_7b_enhanced(self, roi_id, current_val, compared_val, median_val,
                              prev_filename='', curr_filename=''):
        """生成增强的7B验证prompt"""
        return get_prompt(
            roi_id=roi_id,
            prompt_type='mismatch',
            median_value=median_val,
            compared_value=compared_val,
            current_value=current_val,
            prev_filename=prev_filename,
            curr_filename=curr_filename
        )
    
    def clean_model_output(self, text, roi_type='FLOAT'):
        """清理模型输出，根据ROI类型提取正确的值"""
        if not text:
            return "ERROR"
        
        import re
        # 移除特殊tokens
        special_tokens = [
            r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>',
            r'<\|pad\|>', r'<\|assistant\|>', r'<\|user\|>', r'<\|system\|>',
        ]
        for token in special_tokens:
            text = re.sub(token, '', text)
        
        text = text.strip()
        text = text.split('\n')[0].strip()
        
        # 根据类型提取值
        if roi_type in ['FLOAT', 'INTEGER']:
            # 提取数字（包括负数和小数）
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                return match.group()
            return "ERROR"
        elif roi_type == 'STATUS':
            # 提取 OK 或 NG
            text_upper = text.upper()
            if 'OK' in text_upper:
                return 'OK'
            elif 'NG' in text_upper:
                return 'NG'
            return text.split()[0] if text.split() else "ERROR"
        elif roi_type == 'TIME':
            # 提取时间格式 HH:MM:SS
            match = re.search(r'\d{1,2}:\d{2}:\d{2}', text)
            if match:
                return match.group()
            return text.split()[0] if text.split() else "ERROR"
        else:
            # 默认取第一个单词
            return text.split()[0] if text.split() else "ERROR"
    
    def run_7b_inference_dual(self, image_path_prev, image_path_curr, prompt, roi_type='FLOAT'):
        """
        使用7B模型推理（双图像输入）- ✅ 修改为 urllib 实现
        """
        try:
            url = "http://localhost:11434/api/chat"
            
            images_b64 = []
            # 加载第一张图
            if os.path.exists(image_path_prev):
                with open(image_path_prev, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
            else:
                return "Image Not Found"
                
            # 加载第二张图
            if os.path.exists(image_path_curr):
                with open(image_path_curr, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
            else:
                return "Image Not Found"
            
            payload = {
                "model": OLLAMA_MODEL_7B,
                "messages": [{
                    'role': 'user',
                    'content': prompt,
                    'images': images_b64
                }],
                "stream": False,
                "options": {'temperature': 0.1, 'num_predict': 30}
            }
            
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result['message']['content']
                
            return self.clean_model_output(text, roi_type)
            
        except Exception as e:
            print(f"  [7B Dual Error] {e}")
            return "ERROR"
    
    def run_7b_inference(self, image_path, prompt, roi_type='FLOAT'):
        """
        使用7B模型推理（单图像）- ✅ 修改为 urllib 实现
        """
        try:
            url = "http://localhost:11434/api/chat"
            
            images_b64 = []
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
            else:
                return "Image Not Found"
            
            payload = {
                "model": OLLAMA_MODEL_7B,
                "messages": [{
                    'role': 'user',
                    'content': prompt,
                    'images': images_b64
                }],
                "stream": False,
                "options": {'temperature': 0.1, 'num_predict': 30}
            }
            
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result['message']['content']
                
            return self.clean_model_output(text, roi_type)
            
        except Exception as e:
            print(f"  [7B Error] {e}")
            return "ERROR"
    
    def find_crop_image(self, csv_base, filename, roi_id):
        """查找裁剪图像"""
        folder_name = os.path.splitext(filename)[0]
        for ext in ['jpg', 'png']:
            img_path = self.crops_base / folder_name / f"{roi_id}.{ext}"
            if img_path.exists():
                return img_path
            
            # 备用：尝试在子目录查找（如果结构不同）
            img_path_sub = self.crops_base / csv_base / folder_name / f"{roi_id}.{ext}"
            if img_path_sub.exists():
                return img_path_sub
        return None
    
    def process_mismatch_log(self, log_path):
        """处理冗余不匹配日志"""
        filename = log_path.name
        csv_base = filename.replace("_Redundancy_Mismatch_Log.csv", "")
        
        print(f"\n🔍 Verifying with 7B (Dual Image Comparison): {filename}")
        
        try:
            df = pd.read_csv(log_path)
        except:
            print(f"  ❌ Could not read CSV")
            return
        
        if df.empty:
            print(f"  ✅ Log is empty")
            return
        
        # 加载对应的labeled CSV来计算median
        labeled_csv = self.labeled_dir / f"{csv_base}_Labeled.csv"
        roi_medians = {}
        
        if labeled_csv.exists():
            print(f"  📊 Calculating median values from {labeled_csv.name}...")
            roi_medians = self.calculate_roi_medians(labeled_csv)
        else:
            print(f"  ⚠️  Labeled CSV not found, proceeding without median context")
        
        df['AI_7B_Read'] = ""
        df['Verdict'] = ""
        df['Comparison_Mode'] = ""
        
        for idx, row in df.iterrows():
            roi_id = str(row['ROI_ID'])
            roi_type = get_roi_type(roi_id)
            current_filename = str(row['Filename_Current'])
            compared_filename = str(row['Filename_Compared'])
            
            img_path_prev = self.find_crop_image(csv_base, compared_filename, roi_id)
            img_path_curr = self.find_crop_image(csv_base, current_filename, roi_id)
            
            if not img_path_prev or not img_path_curr:
                df.at[idx, 'AI_7B_Read'] = "Image Not Found"
                print(f"  [{idx+1}/{len(df)}] {roi_id}: ❌ Image(s) missing")
                continue
            
            median_val = roi_medians.get(roi_id, None)
            val_curr = row['Value_Current']
            val_prev = row['Value_Compared']
            
            # 生成 Prompt
            prompt = self.get_prompt_7b_enhanced(
                roi_id, val_curr, val_prev, median_val,
                prev_filename=compared_filename,
                curr_filename=current_filename
            )
            
            # 推理
            if roi_type in ['INTEGER', 'FLOAT']:
                ai_result = self.run_7b_inference_dual(img_path_prev, img_path_curr, prompt, roi_type)
                df.at[idx, 'Comparison_Mode'] = "Dual Image"
            else:
                ai_result = self.run_7b_inference(img_path_curr, prompt, roi_type)
                df.at[idx, 'Comparison_Mode'] = "Single Image"
            
            # 显示信息
            mode_icon = "🔬" if roi_type in ['INTEGER', 'FLOAT'] else "📷"
            print(f"  [{idx+1}/{len(df)}] {mode_icon} {roi_id}: Prev={val_prev} | Curr={val_curr} | 7B={ai_result}")
            
            df.at[idx, 'AI_7B_Read'] = ai_result
            
            # 判定逻辑
            ai_clean = str(ai_result).strip().lower()
            prev_clean = str(val_prev).strip().lower()
            curr_clean = str(val_curr).strip().lower()
            
            if ai_clean == prev_clean:
                df.at[idx, 'Verdict'] = "Confirmed Redundant (OCR Error)"
            elif ai_clean == curr_clean:
                df.at[idx, 'Verdict'] = "Genuine Change (OCR Correct)"
            else:
                df.at[idx, 'Verdict'] = "New Value (7B Disagrees)"
        
        # 保存
        out_name = filename.replace(".csv", "_AI_7B_Verified.csv")
        df.to_csv(self.output_dir / out_name, index=False)
        print(f"  ✅ Saved: {out_name}")
    
    def run(self):
        """运行7B验证流程"""
        print("\n" + "="*60)
        print("STAGE 5: 7B Model Verification")
        print("="*60)
        
        mismatch_logs = list(self.labeled_dir.glob("*_Redundancy_Mismatch_Log.csv"))
        
        if not mismatch_logs:
            print("✅ No mismatch logs found - data is consistent!")
            return
        
        print(f"Found {len(mismatch_logs)} mismatch logs\n")
        
        for log_path in mismatch_logs:
            self.process_mismatch_log(log_path)
        
        print("\n✅ Stage 5 Complete")

# ================= 阶段6: 应用7B修正并消除冗余 =================
class Stage6_FinalConsolidation:
    """阶段6: 应用7B修正并消除冗余行，生成最终数据集"""
    
    def __init__(self, labeled_dir, verified_logs_dir, output_dir, crops_base):
        self.labeled_dir = Path(labeled_dir)
        self.verified_logs_dir = Path(verified_logs_dir)
        self.output_dir = Path(output_dir)
        self.crops_base = Path(crops_base) # ✅ 传入 crops_base
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_model_output(self, text, roi_type='FLOAT'):
        """清理模型输出，根据ROI类型提取正确的值"""
        if not text: return "ERROR"
        import re
        special_tokens = [
            r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>',
            r'<\|pad\|>', r'<\|assistant\|>', r'<\|user\|>', r'<\|system\|>',
        ]
        for token in special_tokens:
            text = re.sub(token, '', text)
        text = text.strip().split('\n')[0].strip()
        
        # 根据类型提取值
        if roi_type in ['FLOAT', 'INTEGER']:
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                return match.group()
            return "ERROR"
        elif roi_type == 'STATUS':
            text_upper = text.upper()
            if 'OK' in text_upper:
                return 'OK'
            elif 'NG' in text_upper:
                return 'NG'
            return text.split()[0] if text.split() else "ERROR"
        elif roi_type == 'TIME':
            match = re.search(r'\d{1,2}:\d{2}:\d{2}', text)
            if match:
                return match.group()
            return text.split()[0] if text.split() else "ERROR"
        else:
            return text.split()[0] if text.split() else "ERROR"

    def run_7b_inference(self, image_path, prompt, roi_type='FLOAT'):
        """
        Stage 6 专用的推理函数 - ✅ 修改为 urllib 实现
        """
        try:
            url = "http://localhost:11434/api/chat"
            images_b64 = []
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
            else:
                return "Image Not Found"
            
            payload = {
                "model": OLLAMA_MODEL_7B,
                "messages": [{
                    'role': 'user', 'content': prompt, 'images': images_b64
                }],
                "stream": False,
                "options": {'temperature': 0.1, 'num_predict': 30}
            }
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                text = result['message']['content']
            return self.clean_model_output(text, roi_type)
        except Exception as e:
            return "ERROR"

    def detect_format_issues(self, value, roi_type='FLOAT', median_val=None):
        """检测值是否有格式问题"""
        if pd.isna(value): return None
        val_str = str(value).strip()
        
        if '<|' in val_str or '|>' in val_str: return "special_token"
        
        if roi_type == 'INTEGER':
            if median_val is not None and median_val > 0:
                try:
                    val = abs(float(val_str.replace('.', '')[:10]))
                    median_digits = len(str(int(abs(median_val))))
                    val_digits = len(str(int(val))) if val > 0 else 1
                    is_3x_more = val > abs(median_val) * 3
                    has_more_digits = val_digits > median_digits
                    if not (is_3x_more and has_more_digits): return None
                except: return None
            else: return None
        
        if val_str.count('.') > 1: return "multiple_decimals"
        if roi_type == 'FLOAT' and '.' in val_str:
            try:
                decimal_part = val_str.split('.')[-1]
                if len(decimal_part) > 3 and decimal_part.isdigit(): return "excess_decimals"
            except: pass
        return None
    
    def fix_format_issues_with_7b(self, df, csv_base):
        """使用7B模型修复格式问题"""
        print(f"  🔍 Checking for format issues...")
        
        roi_cols = [c for c in df.columns if c.startswith('ROI_')]
        issues_found = []
        roi_medians = {}
        
        for col in roi_cols:
            roi_type = get_roi_type(col)
            if roi_type in ['INTEGER', 'FLOAT']:
                try:
                    numeric_vals = pd.to_numeric(df[col], errors='coerce')
                    valid_vals = numeric_vals[(numeric_vals != 0) & (numeric_vals.notna())]
                    if len(valid_vals) >= 5:
                        roi_medians[col] = valid_vals.median()
                except: pass
        
        for col in roi_cols:
            roi_type = get_roi_type(col)
            median_val = roi_medians.get(col)
            for idx, value in df[col].items():
                issue = self.detect_format_issues(value, roi_type, median_val)
                if issue:
                    filename = df.at[idx, 'Filename'] if 'Filename' in df.columns else f"Row_{idx}"
                    issues_found.append({
                        'idx': idx, 'roi': col, 'value': value,
                        'issue': issue, 'filename': filename, 'roi_type': roi_type
                    })
        
        if not issues_found:
            print(f"  ✅ No format issues detected")
            return df
        
        print(f"  ⚠️  Found {len(issues_found)} format issues, re-verifying with 7B...")
        
        fixed_count = 0
        for item in issues_found:
            idx = item['idx']
            roi = item['roi']
            filename = item['filename']
            roi_type = item['roi_type']
            median_val = roi_medians.get(roi)
            
            # ✅ 使用传入的 crops_base 查找图片
            folder_name = Path(filename).stem
            image_path = None
            for ext in ['jpg', 'png']:
                test_path = self.crops_base / folder_name / f"{roi}.{ext}"
                if test_path.exists():
                    image_path = test_path
                    break
            
            if not image_path: continue
            
            prompt = (
                f"Task: Extract the {'number' if roi_type in ['FLOAT', 'INTEGER'] else 'value'} from this image.\n"
                f"⚠️ The previous OCR result '{item['value']}' has formatting errors.\n"
                f"STRICT RULES:\n"
                f"1. Output ONLY the clean value you see.\n"
                f"2. For decimals: ONLY ONE decimal point, MAXIMUM 3 digits after.\n"
                f"3. NO special tokens like <|im_start|>, NO HTML.\n"
                f"Output format: Just the number (e.g., 9.128, 1.823, 0)"
            )
            
            new_value = self.run_7b_inference(image_path, prompt, roi_type)
            if new_value and new_value not in ["ERROR", "Image Not Found"]:
                if not self.detect_format_issues(new_value, roi_type, median_val):
                    df.at[idx, roi] = new_value
                    fixed_count += 1
                    print(f"    ✓ Fixed {roi} in {filename}: '{item['value']}' → '{new_value}'")
        
        print(f"  ✅ Fixed {fixed_count}/{len(issues_found)} format issues")
        return df
    
    def apply_7b_corrections(self, labeled_csv_path, verified_log_path):
        """应用7B修正（根据 Verdict 决定更新策略）"""
        print(f"\n🔧 Applying 7B corrections: {labeled_csv_path.name}")
        try:
            df_main = pd.read_csv(labeled_csv_path)
            df_log = pd.read_csv(verified_log_path)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
        
        if df_log.empty: return df_main
        
        df_main['Filename'] = df_main['Filename'].astype(str)
        corrections = []
        for _, row in df_log.iterrows():
            ai_val = str(row.get('AI_7B_Read', '')).strip()
            verdict = str(row.get('Verdict', '')).strip()
            
            if ai_val in ["", "nan", "Image Not Found", "ERROR"]: continue
            corrections.append({
                'curr': row['Filename_Current'],
                'comp': row['Filename_Compared'],
                'roi': row['ROI_ID'],
                'val': ai_val,
                'verdict': verdict
            })
        
        patch_count = 0
        skip_count = 0
        
        for item in corrections:
            target_roi = item['roi']
            new_val = item['val']
            verdict = item['verdict']
            
            if target_roi not in df_main.columns: continue
            
            # 根据 Verdict 决定更新策略
            if "Genuine Change" in verdict:
                # curr 是正确的（真实变化），不需要更新任何行
                skip_count += 1
                continue
            elif "Confirmed Redundant" in verdict:
                # prev 正确，curr 是 OCR 错误，只更新 curr 行
                mask_curr = df_main['Filename'] == item['curr']
                if mask_curr.any():
                    df_main.loc[mask_curr, target_roi] = new_val
                    patch_count += 1
            else:
                # "New Value" 或其他情况，更新两行
                mask_curr = df_main['Filename'] == item['curr']
                if mask_curr.any():
                    df_main.loc[mask_curr, target_roi] = new_val
                    patch_count += 1
                
                mask_comp = df_main['Filename'] == item['comp']
                if mask_comp.any():
                    df_main.loc[mask_comp, target_roi] = new_val
                    patch_count += 1
        
        print(f"  ✅ Patched {patch_count} cells (Skipped {skip_count} genuine changes)")
        return df_main
    
    def parse_pc_filename_time(self, filename):
        try:
            match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}\.\d{2}\.\d{2})', str(filename))
            if match:
                clean_str = match.group(1).replace('.', ':')
                return datetime.strptime(clean_str, '%Y-%m-%d %H:%M:%S')
        except: return None
        return None
    
    def get_data_columns(self, row, all_columns):
        """获取用于比较的数据列（与 Stage 4 逻辑一致）"""
        # 与 Stage4_DataLabeling.get_positional_data 保持一致
        start_idx = 4
        if 'ROI_51' in all_columns:
            end_idx = all_columns.index('ROI_51')
        elif 'ROI_52' in all_columns:
            end_idx = all_columns.index('ROI_52')
        else:
            end_idx = len(all_columns)
        
        if start_idx >= end_idx:
            return []
        
        cols_to_check = all_columns[start_idx:end_idx]
        
        # 排除 Stage 4 添加的标记列（这些列在 Stage 4 之后才存在）
        exclude = ['Time_Status', 'Data_Redundancy', 'Matched_File', 
                   'Duration_Since_Change', 'Real_Freeze_Duration_Sec',
                   'Redundancy_Action', 'Redundancy_Group_ID', 'Redundancy_Reason']
        cols_to_check = [c for c in cols_to_check if c not in exclude]
        
        return [str(row.get(c, '')).strip() for c in cols_to_check]
    
    def values_are_same(self, vals1, vals2):
        if len(vals1) != len(vals2): return False
        return vals1 == vals2
    
    def consolidate_redundancy(self, df):
        """标记冗余行（不删除，仅添加标记列）"""
        print(f"  🏷️  Labeling redundancy (mark mode)...")
        df.sort_values(by='Filename', inplace=True)
        df.reset_index(drop=True, inplace=True)
        rows = df.to_dict('records')
        all_columns = df.columns.tolist()
        total_rows = len(rows)
        
        row_times = []
        row_data_vals = []
        for row in rows:
            row_times.append(self.parse_pc_filename_time(row.get('Filename', '')))
            row_data_vals.append(self.get_data_columns(row, all_columns))
        
        # 初始化标记列
        for row in rows:
            row['Redundancy_Action'] = 'Keep'
            row['Redundancy_Group_ID'] = 0
            row['Redundancy_Reason'] = ''
        
        group_id = 0
        redundant_count = 0
        
        i = 0
        while i < total_rows:
            curr_row = rows[i]
            curr_time = row_times[i]
            curr_vals = row_data_vals[i]
            curr_roi52 = str(curr_row.get('ROI_52', '')).strip()
            
            group_id += 1
            curr_row['Redundancy_Group_ID'] = group_id
            curr_row['Redundancy_Action'] = 'Keep'
            
            j = i + 1
            
            while j < total_rows:
                next_row = rows[j]
                next_time = row_times[j]
                next_vals = row_data_vals[j]
                next_time_status = str(next_row.get('Time_Status', ''))
                next_redundancy = str(next_row.get('Data_Redundancy', ''))
                next_roi52 = str(next_row.get('ROI_52', '')).strip()
                
                time_gap = 0
                if curr_time and next_time:
                    time_gap = (next_time - curr_time).total_seconds()
                
                is_frozen = 'Time Frozen' in next_time_status or 'Time Static' in next_time_status
                is_redundant = 'Redundant' in next_redundancy
                same_machine_time = (curr_roi52 == next_roi52) and curr_roi52 != ''
                same_values = self.values_are_same(curr_vals, next_vals)
                
                if time_gap >= 9: break
                
                should_mark_redundant = False
                reason_parts = []
                
                if is_frozen and same_machine_time and same_values:
                    should_mark_redundant = True
                    reason_parts.append("Time Frozen")
                if is_redundant and same_values and time_gap < 9:
                    should_mark_redundant = True
                    reason_parts.append("Data Redundant")
                if same_machine_time and same_values and time_gap < 9:
                    should_mark_redundant = True
                    if "Same Machine Time" not in reason_parts:
                        reason_parts.append("Same Machine Time")
                
                if should_mark_redundant:
                    next_row['Redundancy_Action'] = 'Redundant'
                    next_row['Redundancy_Group_ID'] = group_id
                    next_row['Redundancy_Reason'] = f"{' + '.join(reason_parts)} (Gap: {time_gap:.1f}s)"
                    redundant_count += 1
                    j += 1
                else:
                    break
            
            i = j
        
        # 计算 Keep 行之间的时间间隔
        keep_rows = [r for r in rows if r['Redundancy_Action'] == 'Keep']
        for k in range(len(keep_rows)):
            curr_item = keep_rows[k]
            curr_time = self.parse_pc_filename_time(curr_item['Filename'])
            step_duration = 0.0
            if k > 0:
                prev_item = keep_rows[k-1]
                prev_time = self.parse_pc_filename_time(prev_item['Filename'])
                if curr_time and prev_time:
                    step_duration = (curr_time - prev_time).total_seconds()
            curr_item['Real_Freeze_Duration_Sec'] = round(step_duration, 2)
        
        # Redundant 行的 Real_Freeze_Duration_Sec 设为 0
        for row in rows:
            if row['Redundancy_Action'] == 'Redundant':
                row['Real_Freeze_Duration_Sec'] = 0.0
        
        df_final = pd.DataFrame(rows)
        keep_count = total_rows - redundant_count
        print(f"  ✅ Labeling complete: {total_rows} rows total (Keep: {keep_count}, Redundant: {redundant_count})")
        return df_final, redundant_count
    
    def process_single_file(self, labeled_csv_path):
        """处理单个文件"""
        base_name = labeled_csv_path.stem.replace("_Labeled", "")
        print(f"\n📦 Finalizing: {labeled_csv_path.name}")
        
        verified_log = self.verified_logs_dir / f"{base_name}_Redundancy_Mismatch_Log_AI_7B_Verified.csv"
        
        df_corrected = None
        if verified_log.exists():
            df_corrected = self.apply_7b_corrections(labeled_csv_path, verified_log)
        else:
            df_corrected = pd.read_csv(labeled_csv_path)
            print(f"  ℹ️  No 7B corrections needed")
        
        if df_corrected is None: return
        
        # 格式验证和修复
        df_corrected = self.fix_format_issues_with_7b(df_corrected, base_name)
        
        # 标记冗余（不删除，仅添加标记列）
        df_final, redundant_count = self.consolidate_redundancy(df_corrected)
        
        out_name = f"{base_name}_Final.csv"
        df_final.to_csv(self.output_dir / out_name, index=False)
        print(f"  💾 Saved: {out_name} (Redundant rows marked: {redundant_count})")
    
    def run(self):
        """运行最终整合流程"""
        print("\n" + "="*60)
        print("STAGE 6: Final Consolidation (Apply 7B + Mark Redundancy)")
        print("="*60)
        
        labeled_files = list(self.labeled_dir.glob("*_Labeled.csv"))
        if not labeled_files:
            print("❌ No labeled files found")
            return
        
        print(f"Found {len(labeled_files)} labeled files\n")
        for csv_file in labeled_files:
            self.process_single_file(csv_file)
        print("\n✅ Stage 6 Complete")

# ================= 主流程 =================
def main():
    """7B管道主流程 - 使用 config_pipeline.py 中的配置"""
    
    # ================= 路径配置 (Path Configuration) =================
    # 直接使用 config_pipeline.py 中的变量，避免路径重复
    
    # 输入目录：Stage 3 (3B修正后) 的结果
    TARGET_INPUT_DIR = STAGE_3_3B_CORRECTED
    
    # 截图目录 (Crops) - 7B模型需要回头看原始截图进行验证
    TARGET_CROPS_DIR = DEBUG_CROPS_BASE
    
    # 输出目录 - 直接使用 config 中已包含 BATCH_NAME 的路径
    DATED_STAGE_4_DIR = STAGE_4_LABELED
    DATED_STAGE_5_DIR = STAGE_5_7B_VERIFIED
    DATED_STAGE_6_DIR = STAGE_6_FINAL
    
    # ==============================================================

    print("\n" + "="*80)
    print("🤖 7B MODEL VERIFICATION PIPELINE")
    print(f"📂 Batch Name:    {BATCH_NAME}")
    print(f"📄 Input (Stg3):  {TARGET_INPUT_DIR}")
    print(f"🖼️  Crops Source:  {TARGET_CROPS_DIR}")
    print(f"💾 Final Output:  {DATED_STAGE_6_DIR}")
    print("="*80)
    
    # 路径检查
    if not TARGET_INPUT_DIR.exists():
        print(f"❌ Error: Input directory not found: {TARGET_INPUT_DIR}")
        return
    if not TARGET_CROPS_DIR.exists():
        print(f"⚠️ Warning: Crops directory not found: {TARGET_CROPS_DIR}")
        print("   (Stage 5 7B-Verification requires images. If missing, it will skip validation.)")

    # 阶段4: 数据标记 (准备数据)
    stage4 = Stage4_DataLabeling(
        input_dir=TARGET_INPUT_DIR,      # 读取 Stage 3 的结果
        output_dir=DATED_STAGE_4_DIR,    # 输出到带日期的文件夹
        crops_base=TARGET_CROPS_DIR      # 传入具体的图片路径
    )
    stage4.run()
    
    # 阶段5: 7B验证 (核心推理)
    stage5 = Stage5_7BVerification(
        labeled_dir=DATED_STAGE_4_DIR,   # 读取 Stage 4 的结果
        output_dir=DATED_STAGE_5_DIR,    # 输出到 Stage 5 文件夹
        crops_base=TARGET_CROPS_DIR      # 传入具体的图片路径
    )
    stage5.run()
    
    # 阶段6: 最终整合 (合并所有更改)
    stage6 = Stage6_FinalConsolidation(
        labeled_dir=DATED_STAGE_4_DIR,   # 原始输入 (从Stg4读取基准)
        verified_logs_dir=DATED_STAGE_5_DIR, # 7B的验证日志
        output_dir=DATED_STAGE_6_DIR,    # 最终成品
        crops_base=TARGET_CROPS_DIR      # ✅ 传入 crops_base
    )
    stage6.run()
    
    print("\n" + "="*80)
    print("🎉 7B PIPELINE COMPLETE")
    print(f"📂 Final Clean Dataset: {DATED_STAGE_6_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()