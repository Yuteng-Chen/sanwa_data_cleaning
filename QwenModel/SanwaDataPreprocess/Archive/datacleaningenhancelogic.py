import pandas as pd
import re
import os
import glob
import shutil
import numpy as np
from datetime import datetime

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Cut_preprocesseddata'
OUTPUT_DIR = 'Archive/Archive/Cut_preprocesseddata_Output_1458/'

# Where to save the abnormal crops
MANUAL_CHECK_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16_1458/'
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# Settings
MAX_DECIMALS = 3
OUTLIER_THRESHOLD = 5.0

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
    """
    Returns: (is_valid, clean_val, reason)
    """
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
        # Check structure
        if re.match(r'^-?\d+(\.\d+)?$', val_str):
            if '.' in val_str and len(val_str.split('.')[1]) > MAX_DECIMALS:
                return False, val, "Suspicious Pattern (>3 decimals)"
            try: return True, float(val_str), None
            except: pass
        return False, val, "Invalid Float"
    
    elif data_type == 'TIME':
        if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str): return True, val_str, None
        return False, val, "Invalid Time"
        
    return False, val, "Unknown Type"

def detect_outliers_in_column(series, data_type):
    """
    Standard statistical outlier detection (Median +/- threshold)
    """
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

# ================= PROCESSING CORE =================
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

    # Standardize Metadata
    if 'Filename' in df.columns:
        df['Extracted_Date'] = df['Filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        if not df['Extracted_Date'].isna().all():
            date_str = pd.to_datetime(df['Extracted_Date']).dt.strftime('%m/%d/%y')
            if 'ROI_51' not in df.columns: df['ROI_51'] = date_str

    roi_map = config['Columns']
    
    # Sort by Filename just for visual tidiness (does not merge)
    if 'Filename' in df.columns:
        df.sort_values(by='Filename', inplace=True)

    # We work on a copy to create the Cleaned version
    df_clean = df.copy()
    
    abnormal_records = []

    # --- PHASE 1: PATTERN VALIDATION (Row by Row) ---
    print("  Validating patterns (Row by Row)...")
    for idx, row in df.iterrows():
        for roi_col, dtype in roi_map.items():
            if roi_col in df.columns:
                val = row[roi_col]
                is_valid, clean_val, reason = validate_value(val, dtype)
                
                # If valid, update the Clean CSV with the standardized value (e.g. OK, Integer)
                if is_valid:
                    df_clean.at[idx, roi_col] = clean_val
                # If invalid, Log it, but KEEP original value in Clean CSV (or update if you prefer)
                else:
                    abnormal_records.append({
                        'Filename': row.get('Filename', 'Unknown'),
                        'Timestamp': row.get('ROI_52', ''),
                        'ROI_ID': roi_col,
                        'Value': val,
                        'Reason': reason
                    })
                    # df_clean.at[idx, roi_col] = val # Keep original if invalid

    # --- PHASE 2: STATISTICAL OUTLIERS (Column by Column) ---
    print("  Checking for Statistical Outliers...")
    for roi_col, dtype in roi_map.items():
        if roi_col in df_clean.columns:
            outliers = detect_outliers_in_column(df_clean[roi_col], dtype)
            for idx in outliers:
                abnormal_records.append({
                    'Filename': df_clean.at[idx, 'Filename'],
                    'Timestamp': df_clean.at[idx, 'ROI_52'] if 'ROI_52' in df_clean.columns else '',
                    'ROI_ID': roi_col,
                    'Value': df_clean.at[idx, roi_col],
                    'Reason': "Statistical Outlier (Likely Missing Decimal)"
                })

    # --- SAVE OUTPUTS ---
    # 1. Save Cleaned File (Contains ALL rows, no deletions)
    df_clean.to_csv(os.path.join(output_folder, f"{base_name}_Cleaned.csv"), index=False)
    
    # 2. Save Abnormal Log & Copy Crops
    if abnormal_records:
        df_abnormal = pd.DataFrame(abnormal_records).drop_duplicates()
        df_abnormal.to_csv(os.path.join(output_folder, f"{base_name}_Abnormal_Log.csv"), index=False)
        
        # Copy Images for verification
        csv_specific_crop_folder = os.path.join(MANUAL_CHECK_BASE, base_name)
        if not os.path.exists(csv_specific_crop_folder): os.makedirs(csv_specific_crop_folder)
        
        count = sum(1 for _, rec in df_abnormal.iterrows() if copy_abnormal_crop(rec['Filename'], rec['ROI_ID'], csv_specific_crop_folder))
        print(f"  -> Found {len(df_abnormal)} issues. Copied {count} images.")

    print(f"  -> Input Rows: {len(df)} | Output Rows: {len(df_clean)}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(MANUAL_CHECK_BASE): os.makedirs(MANUAL_CHECK_BASE)
    
    if os.path.isdir(INPUT_FOLDER):
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
        # Filter out existing outputs to avoid re-processing loops
        files = [f for f in files if not any(x in f for x in ['_Cleaned', '_Log'])]
        print(f"Found {len(files)} files.")
        for f in files: process_single_file(f, OUTPUT_DIR)
    else: print("Invalid input.")

if __name__ == "__main__":
    main()