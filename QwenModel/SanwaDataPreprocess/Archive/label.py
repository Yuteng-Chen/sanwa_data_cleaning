import pandas as pd
import re
import os
import glob
import shutil
import numpy as np
from datetime import datetime

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Final_Cleaned_Dataset_1606'
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output/'

# Crop locations
MANUAL_CHECK_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16/'
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# Settings
MAX_DECIMALS = 3
OUTLIER_THRESHOLD = 5.0
FROZEN_THRESHOLD_SECONDS = 10.0

# ================= ROI CONFIGURATION (Validation Only) =================
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
def copy_abnormal_crop(filename_png, roi_id, specific_dest_folder):
    try:
        folder_name_no_ext = os.path.splitext(filename_png)[0]
        src_folder = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext)
        src_jpg = os.path.join(src_folder, f"{roi_id}.jpg")
        src_png = os.path.join(src_folder, f"{roi_id}.png")
        
        src_file = src_jpg if os.path.exists(src_jpg) else (src_png if os.path.exists(src_png) else None)
        if not src_file: return False
            
        target_folder = os.path.join(specific_dest_folder, folder_name_no_ext)
        if not os.path.exists(target_folder): os.makedirs(target_folder)
            
        shutil.copy(src_file, os.path.join(target_folder, os.path.basename(src_file)))
        return True
    except: return False

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
    except:
        return None
    return None

def parse_plc_time_safe(time_str):
    try:
        return datetime.strptime(str(time_str).strip(), '%H:%M:%S')
    except:
        return None

def get_positional_fingerprint(row, columns_list):
    """
    Creates a fingerprint from the 5th column (Index 4) up to (but not including) ROI_51.
    Strictly enforces "Value Numeric Column between 5th to the column before ROI 51".
    """
    start_idx = 4 # 5th column (Index 0,1,2,3, [4]...)
    
    # Dynamically find ROI_51 index
    if 'ROI_51' in columns_list:
        end_idx = columns_list.index('ROI_51')
    elif 'ROI_52' in columns_list:
        end_idx = columns_list.index('ROI_52')
    else:
        end_idx = len(columns_list)
        
    if start_idx >= end_idx:
        return ""
        
    fingerprint = []
    # Slice columns by index range
    cols_to_check = columns_list[start_idx:end_idx]
    
    for col in cols_to_check:
        val = str(row.get(col, ''))
        fingerprint.append(val)
        
    return "|".join(fingerprint)

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
    
    # 1. SORT BY FILENAME
    if 'Filename' in df.columns:
        df.sort_values(by='Filename', inplace=True)

    df_clean = df.copy()
    
    # Init new columns
    df_clean['Time_Status'] = 'Unknown'
    df_clean['Data_Redundancy'] = 'Unknown'
    df_clean['Matched_File'] = ''
    df_clean['Duration_Since_Change'] = 0.0
    
    abnormal_records = []

    # --- PHASE 1: BASIC VALIDATION ---
    print("  Validating patterns...")
    for idx, row in df.iterrows():
        for roi_col, dtype in roi_map.items():
            if roi_col in df.columns:
                val = row[roi_col]
                is_valid, clean_val, reason = validate_value(val, dtype)
                if is_valid: 
                    df_clean.at[idx, roi_col] = clean_val
                else:
                    abnormal_records.append({
                        'Filename': row.get('Filename', 'Unknown'),
                        'ROI_ID': roi_col,
                        'Value': val,
                        'Reason': reason
                    })

    # --- PHASE 2: STATISTICS ---
    print("  Checking Statistics...")
    for roi_col, dtype in roi_map.items():
        if roi_col in df_clean.columns:
            outliers = detect_outliers_in_column(df_clean[roi_col], dtype)
            for idx in outliers:
                abnormal_records.append({
                    'Filename': df_clean.at[idx, 'Filename'],
                    'ROI_ID': roi_col,
                    'Value': df_clean.at[idx, roi_col],
                    'Reason': "Statistical Outlier"
                })

    # --- PHASE 3: TIME-STATE ANALYSIS (Independent Checks) ---
    print("  Analyzing Time vs Content (Positional Fingerprint)...")
    
    rows_list = df_clean.to_dict('records')
    columns_list = df_clean.columns.tolist()
    
    # Pre-calculate ALL fingerprints first
    all_fingerprints = [get_positional_fingerprint(row, columns_list) for row in rows_list]
    
    prev_plc_time_str = None
    state_start_pc_time = None
    
    for i in range(len(rows_list)):
        curr_row = rows_list[i]
        curr_idx = df_clean.index[i]
        
        curr_pc_obj = parse_pc_filename_time(curr_row.get('Filename', ''))
        curr_plc_str = str(curr_row.get('ROI_52', '')).strip()
        
        # Get Fingerprints for Prev, Current, Next
        curr_content = all_fingerprints[i]
        prev_content = all_fingerprints[i-1] if i > 0 else None
        next_content = all_fingerprints[i+1] if i < len(rows_list)-1 else None
        
        # Get Filenames for matches
        prev_filename = df_clean.at[df_clean.index[i-1], 'Filename'] if i > 0 else ""
        next_filename = df_clean.at[df_clean.index[i+1], 'Filename'] if i < len(rows_list)-1 else ""

        # --- 1. TIME STATUS LOGIC (Backward only) ---
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
                time_status = "New Time State"
                state_start_pc_time = curr_pc_obj
                duration = 0.0

        # --- 2. DATA REDUNDANCY LOGIC (Independent Checks) ---
        is_redundant_prev = False
        is_redundant_next = False
        
        # Check Backward
        if i > 0 and curr_content == prev_content:
            is_redundant_prev = True
            
        # Check Forward
        if i < len(rows_list)-1 and curr_content == next_content:
            is_redundant_next = True
            
        # Construct Status String & Matched File
        data_redundancy_list = []
        matched_files_list = []
        
        if not is_redundant_prev and not is_redundant_next:
            data_redundancy = "Unique"
        else:
            if is_redundant_prev:
                data_redundancy_list.append("Redundant (Previous)")
                matched_files_list.append(f"Prev: {prev_filename}")
            
            if is_redundant_next:
                data_redundancy_list.append("Redundant (Next)")
                matched_files_list.append(f"Next: {next_filename}")
                
            data_redundancy = " & ".join(data_redundancy_list)
        
        matched_file_str = " | ".join(matched_files_list)

        # Update Pointers for next loop
        prev_plc_time_str = curr_plc_str
        
        # Write to DataFrame
        df_clean.at[curr_idx, 'Time_Status'] = time_status
        df_clean.at[curr_idx, 'Data_Redundancy'] = data_redundancy
        df_clean.at[curr_idx, 'Matched_File'] = matched_file_str
        df_clean.at[curr_idx, 'Duration_Since_Change'] = round(duration, 2)

    # --- SAVE OUTPUTS ---
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    df_clean.to_csv(os.path.join(output_folder, f"{base_name}_Cleaned.csv"), index=False)
    
    # Hanged Log (Time Static + Unique Data)
    hanged_items = df_clean[(df_clean['Time_Status'].str.contains("Static")) & (df_clean['Data_Redundancy'] == 'Unique')]
    if not hanged_items.empty:
        log_path = os.path.join(output_folder, f"{base_name}_Hanged_Logic_Log.csv")
        hanged_items[['Filename', 'ROI_52', 'Time_Status', 'Data_Redundancy']].to_csv(log_path, index=False)
        print(f"  -> Found {len(hanged_items)} non-redundant Hanged items.")

    if abnormal_records:
        df_abnormal = pd.DataFrame(abnormal_records).drop_duplicates()
        df_abnormal.to_csv(os.path.join(output_folder, f"{base_name}_Abnormal_Log.csv"), index=False)
        
        csv_specific_crop_folder = os.path.join(MANUAL_CHECK_BASE, base_name)
        if not os.path.exists(csv_specific_crop_folder): os.makedirs(csv_specific_crop_folder)
        count = sum(1 for _, rec in df_abnormal.iterrows() if copy_abnormal_crop(rec['Filename'], rec['ROI_ID'], csv_specific_crop_folder))
        print(f"  -> Value Issues: {len(df_abnormal)}. Images copied: {count}")

    print(f"  -> Processed {len(df)} rows.")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    if os.path.isdir(INPUT_FOLDER):
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
        print(f"Found {len(files)} files in {INPUT_FOLDER}")
        for f in files: process_single_file(f, OUTPUT_DIR)
    else:
        print(f"Invalid input folder: {INPUT_FOLDER}")

if __name__ == "__main__":
    main()