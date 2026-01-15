import pandas as pd
import re
import os
import glob
import shutil
import numpy as np

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Cut_preprocesseddata'
OUTPUT_DIR = 'Archive/Archive/Cut_preprocesseddata_Output/'

# Where to save the abnormal crops (Subfolders will be created here)
MANUAL_CHECK_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16/'

# Original Debug Crops Source
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# Validation Settings
MAX_DECIMALS = 3  # If >3 decimals, flag as "Suspicious Pattern"
OUTLIER_THRESHOLD = 5.0 # If value is 5x larger than Median, flag as "Outlier"

# ================= ROI CONFIGURATION =================
ROI_CONFIGS = [
    {
        # CAM 6 SNAP 1 (Latch)
        'Trigger_Col': 'ROI_12',
        'Columns': {
            'ROI_12': 'STATUS', 'ROI_14': 'STATUS', 'ROI_15': 'STATUS', 'ROI_17': 'STATUS', 'ROI_19': 'STATUS',
            'ROI_13': 'INTEGER', 'ROI_16': 'FLOAT', 'ROI_18': 'FLOAT',
            'ROI_52': 'TIME'
        }
    },
    {
        # CAM 6 SNAP 2 (Nozzle)
        'Trigger_Col': 'ROI_20',
        'Columns': {
            'ROI_20': 'STATUS', 'ROI_22': 'STATUS', 'ROI_24': 'STATUS', 'ROI_25': 'STATUS', 
            'ROI_26': 'STATUS', 'ROI_27': 'STATUS', 'ROI_28': 'STATUS', 'ROI_29': 'STATUS', 'ROI_30': 'STATUS',
            'ROI_21': 'INTEGER', 'ROI_23': 'FLOAT',
            'ROI_52': 'TIME'
        }
    },
    {
        # CSLOT CAM 4
        'Trigger_Col': 'ROI_1',
        'Columns': {
            'ROI_1': 'STATUS', 'ROI_3': 'STATUS', 'ROI_5': 'STATUS', 'ROI_7': 'STATUS', 
            'ROI_9': 'STATUS', 'ROI_10': 'STATUS', 'ROI_11': 'STATUS',
            'ROI_2': 'INTEGER', 'ROI_4': 'FLOAT', 'ROI_6': 'FLOAT', 'ROI_8': 'FLOAT',
            'ROI_52': 'TIME'
        }
    },
    {
        # TERMINAL RESULT
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

# ================= 1. HELPER: COPY CROP =================
def copy_abnormal_crop(filename_png, roi_id, specific_dest_folder):
    """
    Copies ROI crop to .../Abnormal_Crops_Recheck/CSV_NAME/ParentImage/ROI.jpg
    """
    try:
        folder_name_no_ext = os.path.splitext(filename_png)[0]
        
        # Source Paths
        src_folder = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext)
        src_jpg = os.path.join(src_folder, f"{roi_id}.jpg")
        src_png = os.path.join(src_folder, f"{roi_id}.png")
        
        src_file = None; ext = ""
        if os.path.exists(src_jpg): src_file = src_jpg; ext = ".jpg"
        elif os.path.exists(src_png): src_file = src_png; ext = ".png"
        else: return False
            
        # Destination
        target_folder = os.path.join(specific_dest_folder, folder_name_no_ext)
        if not os.path.exists(target_folder): os.makedirs(target_folder)
            
        dst_file = os.path.join(target_folder, f"{roi_id}{ext}")
        shutil.copy(src_file, dst_file)
        return True
    except Exception: return False

# ================= 2. VALIDATION LOGIC =================
def validate_value(val, data_type):
    """
    Returns: (is_valid, clean_val, reason)
    Does NOT truncate numbers anymore. Just flags them.
    """
    val_str = str(val).strip()
    if pd.isna(val) or val_str == '' or val_str.lower() == 'nan':
        return False, val, "Empty/NaN"

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
            if '.' in val_str:
                decimals = len(val_str.split('.')[1])
                # LOGIC CHANGE: If >3 decimals, flag as suspicious pattern. Do NOT fix.
                if decimals > MAX_DECIMALS: 
                    return False, val, f"Suspicious Pattern (>3 decimals)"
            
            # Check if it converts to float safely
            try:
                num_val = float(val_str)
                return True, num_val, None
            except:
                return False, val, "Float Conversion Error"
        return False, val, "Invalid Float"

    elif data_type == 'TIME':
        if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str): return True, val_str, None
        return False, val, "Invalid Time"

    return False, val, "Unknown Type"

# ================= 3. OUTLIER DETECTION (Missing Decimal) =================
def detect_outliers_in_column(series, data_type):
    """
    Compares each value to the Median of the column.
    If Value > Median * 5 OR Value < Median / 5 -> Flag it.
    Catches 188 vs 1.88 error.
    """
    if data_type not in ['FLOAT', 'INTEGER']: return []
    
    # Get valid numbers for statistics
    nums = pd.to_numeric(series, errors='coerce').dropna()
    if len(nums) < 5: return [] # Need enough data to determine a trend
    
    median = nums.median()
    if median == 0: return [] 
    
    outlier_indices = []
    
    for idx, val in series.items():
        try:
            num = float(val)
            # Calculate ratio difference
            ratio = num / median
            
            # Check if it deviates significantly
            if ratio > OUTLIER_THRESHOLD or ratio < (1.0 / OUTLIER_THRESHOLD):
                outlier_indices.append(idx)
        except: 
            pass
            
    return outlier_indices

# ================= 4. SMART MERGE LOGIC =================
def select_best_roi_value(series, data_type):
    """
    Picks the best value from duplicates.
    Prioritizes: Clean floats (3 decimals) > Valid Ints > Suspicious Patterns > Invalid
    """
    valid_values = []
    
    for val in series:
        is_valid, clean_val, reason = validate_value(val, data_type)
        
        score = 0
        if is_valid:
            if data_type == 'STATUS' and clean_val == 'OK': score = 10
            elif data_type == 'FLOAT':
                s_val = str(clean_val)
                if '.' in s_val:
                    decimals = len(s_val.split('.')[1])
                    if decimals == 3: score = 20 # Perfect
                    elif decimals == 2: score = 10
            elif data_type == 'INTEGER': score = 5
        else:
            # If invalid, it gets score 0.
            # But we still store it in case ALL rows are bad.
            # We want to return the original Bad value so it can be flagged later.
            clean_val = val 

        valid_values.append({'val': clean_val, 'score': score})

    if not valid_values: return series.iloc[0]
    
    # Sort: Highest score wins
    valid_values.sort(key=lambda x: x['score'], reverse=True)
    return valid_values[0]['val']

# ================= 5. PROCESSING CORE =================
def get_config_for_file(df):
    for config in ROI_CONFIGS:
        if config['Trigger_Col'] in df.columns: return config
    return None

def process_single_file(input_csv_path, output_folder):
    filename = os.path.basename(input_csv_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\nProcessing: {filename}...")
    
    # Specific Crop Folder: .../Abnormal_Crops/Filename_No_Ext/
    csv_specific_crop_folder = os.path.join(MANUAL_CHECK_BASE, base_name)

    try: df = pd.read_csv(input_csv_path)
    except Exception as e: print(f"  Error: {e}"); return

    config = get_config_for_file(df)
    if not config: print("  Skipped: Unknown format."); return

    if 'Filename' in df.columns:
        df['Extracted_Date'] = df['Filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        if not df['Extracted_Date'].isna().all():
            date_str = pd.to_datetime(df['Extracted_Date']).dt.strftime('%m/%d/%y')
            if 'ROI_51' not in df.columns: df['ROI_51'] = date_str

    if 'ROI_52' not in df.columns: print("  Warning: No timestamp."); return

    roi_map = config['Columns']
    meta_cols = ['Filename', 'File_UTC', 'Machine_Text', 'Machine_UTC', 'ROI_51', 'ROI_52']
    
    # --- PHASE 1: MERGE DUPLICATES ---
    print("  Merging duplicates (Selecting best candidates)...")
    merged_rows = []
    grouped = df.groupby('ROI_52')
    
    for timestamp, group in grouped:
        best_row = {}
        last_entry = group.iloc[-1]
        for meta in meta_cols:
            if meta in last_entry: best_row[meta] = last_entry[meta]
        
        for roi_col, dtype in roi_map.items():
            if roi_col == 'ROI_52': continue 
            if roi_col in group.columns:
                best_row[roi_col] = select_best_roi_value(group[roi_col], dtype)
        merged_rows.append(best_row)
        
    df_clean = pd.DataFrame(merged_rows)
    # Ensure column order
    df_clean = df_clean[df.columns.intersection(df_clean.columns)]

    # --- PHASE 2: OUTLIER DETECTION (Missing Decimal) ---
    print("  Checking for Statistical Outliers (e.g. 188 vs 1.88)...")
    abnormal_records = []
    
    for roi_col, dtype in roi_map.items():
        if roi_col in df_clean.columns:
            outliers = detect_outliers_in_column(df_clean[roi_col], dtype)
            for idx in outliers:
                val = df_clean.at[idx, roi_col]
                abnormal_records.append({
                    'Filename': df_clean.at[idx, 'Filename'],
                    'Timestamp': df_clean.at[idx, 'ROI_52'],
                    'ROI_ID': roi_col,
                    'Value': val,
                    'Reason': f"Statistical Outlier (Likely Missing Decimal)"
                })

    # --- PHASE 3: PATTERN & FORMAT VALIDATION ---
    print("  Checking for Suspicious Patterns (e.g. 9.18181)...")
    for idx, row in df_clean.iterrows():
        for roi_col, dtype in roi_map.items():
            if roi_col in df_clean.columns:
                val = row[roi_col]
                is_valid, clean_val, reason = validate_value(val, dtype)
                
                if not is_valid:
                    abnormal_records.append({
                        'Filename': row.get('Filename', 'Unknown'),
                        'Timestamp': row.get('ROI_52', 'Unknown'),
                        'ROI_ID': roi_col,
                        'Value': val,
                        'Reason': reason # Will include "Suspicious Pattern (>3 decimals)"
                    })
                else:
                    df_clean.at[idx, roi_col] = clean_val

    # --- SAVE OUTPUTS ---
    df_clean.to_csv(os.path.join(output_folder, f"{base_name}_Cleaned.csv"), index=False)
    
    if abnormal_records:
        df_abnormal = pd.DataFrame(abnormal_records)
        df_abnormal.drop_duplicates(inplace=True)
        df_abnormal.to_csv(os.path.join(output_folder, f"{base_name}_Abnormal_Log.csv"), index=False)
        
        print(f"  -> Found {len(df_abnormal)} abnormal items. Copying crops...")
        
        if not os.path.exists(csv_specific_crop_folder):
            os.makedirs(csv_specific_crop_folder)

        copied_count = 0
        for _, rec in df_abnormal.iterrows():
            if copy_abnormal_crop(rec['Filename'], rec['ROI_ID'], csv_specific_crop_folder):
                copied_count += 1
        
        print(f"  -> Copied {copied_count} debug images to: {csv_specific_crop_folder}")

    print(f"  -> Orig: {len(df)} | Final: {len(df_clean)} | Abnormal: {len(abnormal_records)}")

# ================= 6. RUNNER =================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(MANUAL_CHECK_BASE): os.makedirs(MANUAL_CHECK_BASE)
    
    if os.path.isdir(INPUT_FOLDER):
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
        files = [f for f in files if not any(x in f for x in ['_Cleaned', '_Log'])]
        print(f"Found {len(files)} files to process.")
        for f in files: process_single_file(f, OUTPUT_DIR)
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()