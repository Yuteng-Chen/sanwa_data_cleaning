import pandas as pd
import re
import os
import glob
import shutil
import numpy as np
from datetime import datetime

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Final_Cleaned_Dataset_1606'
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640_1125/'

# 1. Destination for Abnormal Value Crops
MANUAL_CHECK_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16/'

# 2. SOURCE for Crops (The specific path you requested)
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# 3. Destination for Redundancy Mismatch Crops
REDUNDANCY_CHECK_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Redundancy_Mismatch_Recheck0640/'

# Settings
MAX_DECIMALS = 3
OUTLIER_THRESHOLD = 5.0
FROZEN_THRESHOLD_SECONDS = 10.0
SIMILARITY_THRESHOLD = 0.8  # 80% Match

# ================= ROI CONFIGURATION =================
ROI_CONFIGS = [
    {
        'Trigger_Col': 'ROI_12',
        'Columns': {
            'ROI_12': 'STATUS', 'ROI_14': 'STATUS', 'ROI_15': 'STATUS', 'ROI_17': 'STATUS', 'ROI_19': 'STATUS',
            'ROI_13': 'INTEGER', 'ROI_16': 'FLOAT', 'ROI_18': 'FLOAT', 'ROI_52': 'TIME'
        }
    },
    {
        'Trigger_Col': 'ROI_20',
        'Columns': {
            'ROI_20': 'STATUS', 'ROI_22': 'STATUS', 'ROI_24': 'STATUS', 'ROI_25': 'STATUS', 
            'ROI_26': 'STATUS', 'ROI_27': 'STATUS', 'ROI_28': 'STATUS', 'ROI_29': 'STATUS', 'ROI_30': 'STATUS',
            'ROI_21': 'INTEGER', 'ROI_23': 'FLOAT', 'ROI_52': 'TIME'
        }
    },
    {
        'Trigger_Col': 'ROI_1',
        'Columns': {
            'ROI_1': 'STATUS', 'ROI_3': 'STATUS', 'ROI_5': 'STATUS', 'ROI_7': 'STATUS', 
            'ROI_9': 'STATUS', 'ROI_10': 'STATUS', 'ROI_11': 'STATUS',
            'ROI_2': 'INTEGER', 'ROI_4': 'FLOAT', 'ROI_6': 'FLOAT', 'ROI_8': 'FLOAT', 'ROI_52': 'TIME'
        }
    },
    {
        'Trigger_Col': 'ROI_31',
        'Columns': {
            'ROI_31': 'STATUS', 'ROI_33': 'STATUS', 'ROI_34': 'STATUS', 'ROI_36': 'STATUS', 
            'ROI_38': 'STATUS', 'ROI_40': 'STATUS', 'ROI_42': 'STATUS', 'ROI_44': 'STATUS', 
            'ROI_46': 'STATUS', 'ROI_48': 'STATUS', 'ROI_50': 'STATUS',
            'ROI_32': 'INTEGER', 'ROI_35': 'INTEGER', 'ROI_37': 'INTEGER', 'ROI_39': 'INTEGER', 
            'ROI_41': 'INTEGER', 'ROI_43': 'INTEGER', 'ROI_45': 'INTEGER', 'ROI_47': 'INTEGER', 'ROI_49': 'INTEGER',
            'ROI_52': 'TIME'
        }
    }
]

# ================= HELPER FUNCTIONS =================
def copy_abnormal_crop(csv_base_name, filename_png, roi_id, specific_dest_folder):
    """
    Copies crop from DEBUG_CROPS_BASE to specific_dest_folder.
    Searches in:
      1. DEBUG_CROPS_BASE / CSV_NAME / IMAGE_NAME / ROI.jpg (Nested - Most likely)
      2. DEBUG_CROPS_BASE / IMAGE_NAME / ROI.jpg (Flat - Fallback)
    """
    try:
        folder_name_no_ext = os.path.splitext(filename_png)[0]
        
        # --- PATH SEARCH STRATEGY ---
        potential_src_files = []
        
        # 1. Nested: DEBUG_BASE / {CSV_Name} / {Image_Name} / {ROI}.jpg
        path_nested_jpg = os.path.join(DEBUG_CROPS_BASE, csv_base_name, folder_name_no_ext, f"{roi_id}.jpg")
        path_nested_png = os.path.join(DEBUG_CROPS_BASE, csv_base_name, folder_name_no_ext, f"{roi_id}.png")
        
        # 2. Flat: DEBUG_BASE / {Image_Name} / {ROI}.jpg
        path_flat_jpg = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext, f"{roi_id}.jpg")
        path_flat_png = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext, f"{roi_id}.png")
        
        # Check in order of likelihood
        potential_src_files = [path_nested_jpg, path_nested_png, path_flat_jpg, path_flat_png]
        
        src_file = None
        for p in potential_src_files:
            if os.path.exists(p):
                src_file = p
                break
        
        if not src_file:
            # Uncomment below to debug specific missing files
            # print(f"    [Warn] Crop not found for {roi_id} in {folder_name_no_ext}")
            return False
            
        # Target Path: specific_dest_folder / folder_name_no_ext / ROI.jpg
        target_folder = os.path.join(specific_dest_folder, folder_name_no_ext)
        if not os.path.exists(target_folder): os.makedirs(target_folder)
            
        shutil.copy(src_file, os.path.join(target_folder, os.path.basename(src_file)))
        return True
    except Exception as e:
        print(f"    [Error] Copy failed: {e}")
        return False

def validate_value(val, data_type):
    val_str = str(val).strip()
    if pd.isna(val) or val_str == '' or val_str.lower() == 'nan': return False, val, "Empty/NaN"

    if data_type == 'STATUS':
        val_upper = val_str.upper()
        if val_upper.startswith('N'): return True, 'NG', None
        if 'O' in val_upper or 'K' in val_upper or '0' in val_upper: return True, 'OK', None
        return False, val, "Invalid Status"
    elif data_type == 'INTEGER':
        clean_val = re.sub(r'[^\d-]', '', val_str) 
        if re.match(r'^-?\d+$', clean_val): return True, int(clean_val), None
        return False, val, "Not an Integer"
    elif data_type == 'FLOAT':
        if re.match(r'^-?\d+(\.\d+)?$', val_str):
            try: return True, float(val_str), None
            except: pass
        return False, val, "Invalid Float"
    elif data_type == 'TIME':
        if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str): return True, val_str, None
        return False, val, "Invalid Time Format"
    return False, val, "Unknown Type"

def detect_outliers_in_column(series, data_type):
    if data_type not in ['FLOAT', 'INTEGER']: return []
    nums = pd.to_numeric(series, errors='coerce').dropna()
    if len(nums) < 5 or nums.median() == 0: return []
    median = nums.median()
    outlier_indices = []
    for idx, val in series.items():
        try:
            ratio = float(val) / median
            if ratio > OUTLIER_THRESHOLD or ratio < (1.0/OUTLIER_THRESHOLD): 
                outlier_indices.append(idx)
        except: pass
    return outlier_indices

def parse_pc_filename_time(filename):
    try:
        match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}\.\d{2}\.\d{2})', filename)
        if match:
            clean_str = match.group(1).replace('.', ':')
            return datetime.strptime(clean_str, '%Y-%m-%d %H:%M:%S')
    except: return None
    return None

def get_positional_data(row, columns_list):
    start_idx = 4 
    if 'ROI_51' in columns_list: end_idx = columns_list.index('ROI_51')
    elif 'ROI_52' in columns_list: end_idx = columns_list.index('ROI_52')
    else: end_idx = len(columns_list)
        
    if start_idx >= end_idx: return [], []
        
    cols_to_check = columns_list[start_idx:end_idx]
    values = [str(row.get(col, '')).strip() for col in cols_to_check]
    return values, cols_to_check

def calculate_similarity(list_a, list_b):
    if not list_a or not list_b or len(list_a) != len(list_b): return 0.0
    matches = sum(1 for a, b in zip(list_a, list_b) if a == b)
    return matches / len(list_a)

# ================= MAIN PROCESSING LOGIC =================
def get_config_for_file(df):
    for config in ROI_CONFIGS:
        if config['Trigger_Col'] in df.columns: return config
    return None

def process_single_file(input_csv_path, output_folder):
    filename = os.path.basename(input_csv_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\nProcessing: {filename}...")
    
    try: df = pd.read_csv(input_csv_path)
    except Exception as e: print(f"  Error: {e}"); return

    config = get_config_for_file(df)
    if not config: print("  Skipped: Unknown format."); return

    roi_map = config['Columns']
    if 'Filename' in df.columns: df.sort_values(by='Filename', inplace=True)

    df_clean = df.copy()
    
    # Init new columns
    df_clean['Time_Status'] = 'Unknown'
    df_clean['Data_Redundancy'] = 'Unknown'
    df_clean['Matched_File'] = ''
    df_clean['Duration_Since_Change'] = 0.0
    
    abnormal_records = []
    redundancy_mismatch_records = []

    # --- PHASE 1: BASIC VALIDATION ---
    print("  Validating patterns...")
    for idx, row in df.iterrows():
        for roi_col, dtype in roi_map.items():
            if roi_col in df.columns:
                val = row[roi_col]
                is_valid, clean_val, reason = validate_value(val, dtype)
                if is_valid: df_clean.at[idx, roi_col] = clean_val
                else:
                    abnormal_records.append({'Filename': row.get('Filename'), 'ROI_ID': roi_col, 'Value': val, 'Reason': reason})

    # --- PHASE 2: STATISTICS ---
    print("  Checking Statistics...")
    for roi_col, dtype in roi_map.items():
        if roi_col in df_clean.columns:
            outliers = detect_outliers_in_column(df_clean[roi_col], dtype)
            for idx in outliers:
                abnormal_records.append({
                    'Filename': df_clean.at[idx, 'Filename'], 'ROI_ID': roi_col, 
                    'Value': df_clean.at[idx, roi_col], 'Reason': "Statistical Outlier"
                })

    # --- PHASE 3: TIME-STATE ANALYSIS & FUZZY REDUNDANCY ---
    print(f"  Analyzing Similarity (Threshold: {SIMILARITY_THRESHOLD*100}%)...")
    
    rows_list = df_clean.to_dict('records')
    columns_list = df_clean.columns.tolist()
    all_row_data = [get_positional_data(row, columns_list) for row in rows_list]
    
    prev_plc_time_str = None
    state_start_pc_time = None
    
    for i in range(len(rows_list)):
        curr_row = rows_list[i]
        curr_idx = df_clean.index[i]
        curr_filename = curr_row.get('Filename', '')
        
        curr_pc_obj = parse_pc_filename_time(curr_filename)
        curr_plc_str = str(curr_row.get('ROI_52', '')).strip()
        
        # Get Data Lists
        curr_vals, curr_cols = all_row_data[i]
        prev_vals, _ = all_row_data[i-1] if i > 0 else ([], [])
        next_vals, _ = all_row_data[i+1] if i < len(rows_list)-1 else ([], [])
        
        prev_filename = df_clean.at[df_clean.index[i-1], 'Filename'] if i > 0 else ""
        next_filename = df_clean.at[df_clean.index[i+1], 'Filename'] if i < len(rows_list)-1 else ""

        # --- 1. TIME STATUS LOGIC ---
        time_status = "New Time State"
        duration = 0.0
        
        if i == 0 or curr_pc_obj is None:
            time_status = "New Time State (Start)"
            state_start_pc_time = curr_pc_obj
        else:
            if curr_plc_str == prev_plc_time_str:
                time_status = "Time Static"
                if state_start_pc_time and curr_pc_obj: duration = (curr_pc_obj - state_start_pc_time).total_seconds()
                if duration > FROZEN_THRESHOLD_SECONDS: time_status = "Time Frozen (>10s)"
            else:
                state_start_pc_time = curr_pc_obj
        prev_plc_time_str = curr_plc_str

        # --- 2. FUZZY REDUNDANCY LOGIC ---
        is_redundant_prev = False
        is_redundant_next = False
        
        similarity_prev = calculate_similarity(curr_vals, prev_vals)
        if i > 0 and similarity_prev >= SIMILARITY_THRESHOLD:
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
            
        similarity_next = calculate_similarity(curr_vals, next_vals)
        if i < len(rows_list)-1 and similarity_next >= SIMILARITY_THRESHOLD:
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

    # --- SAVE OUTPUTS ---
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    df_clean.to_csv(os.path.join(output_folder, f"{base_name}_Cleaned.csv"), index=False)
    
    # 1. Hanged Log
    hanged = df_clean[(df_clean['Time_Status'].str.contains("Static")) & (df_clean['Data_Redundancy'] == 'Unique')]
    if not hanged.empty:
        hanged[['Filename', 'ROI_52', 'Time_Status', 'Data_Redundancy']].to_csv(
            os.path.join(output_folder, f"{base_name}_Hanged_Logic_Log.csv"), index=False)

    # 2. Abnormal Values Log
    if abnormal_records:
        df_abn = pd.DataFrame(abnormal_records).drop_duplicates()
        df_abn.to_csv(os.path.join(output_folder, f"{base_name}_Abnormal_Log.csv"), index=False)
        crop_dest = os.path.join(MANUAL_CHECK_BASE, base_name)
        if not os.path.exists(crop_dest): os.makedirs(crop_dest)
        
        # [FIX] Passing base_name to help find the crop folder
        cnt = sum(1 for _, r in df_abn.iterrows() if copy_abnormal_crop(base_name, r['Filename'], r['ROI_ID'], crop_dest))
        print(f"  -> Value Issues: {len(df_abn)}. Images copied: {cnt}")

    # 3. Redundancy Mismatch Log
    if redundancy_mismatch_records:
        df_mis = pd.DataFrame(redundancy_mismatch_records).drop_duplicates()
        df_mis.to_csv(os.path.join(output_folder, f"{base_name}_Redundancy_Mismatch_Log.csv"), index=False)
        
        mis_dest = os.path.join(REDUNDANCY_CHECK_BASE, base_name)
        if not os.path.exists(mis_dest): os.makedirs(mis_dest)
        
        # [FIX] Passing base_name here too
        cnt_mis = sum(1 for _, r in df_mis.iterrows() if copy_abnormal_crop(base_name, r['Filename_Current'], r['ROI_ID'], mis_dest))
        print(f"  -> Redundancy Mismatches: {len(df_mis)}. Images copied: {cnt_mis}")

    print(f"  -> Processed {len(df)} rows.")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if os.path.isdir(INPUT_FOLDER):
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
        print(f"Found {len(files)} files.")
        for f in files: process_single_file(f, OUTPUT_DIR)
    else: print(f"Invalid input folder: {INPUT_FOLDER}")

if __name__ == "__main__":
    main()