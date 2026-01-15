import pandas as pd
import re
import os
import glob
import shutil
import numpy as np

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Cut_preprocesseddata'
OUTPUT_DIR = 'Archive/Archive/Cleaned_Cut_Results_Output/'
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/Cut_preprocesseddata_12_16'
MAX_DECIMALS = 3

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
def copy_abnormal_crop(filename_png, roi_id):
    try:
        folder_name_no_ext = os.path.splitext(filename_png)[0]
        src_folder = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext)
        src_jpg = os.path.join(src_folder, f"{roi_id}.jpg")
        src_png = os.path.join(src_folder, f"{roi_id}.png")
        
        src_file = None; ext = ""
        if os.path.exists(src_jpg): src_file = src_jpg; ext = ".jpg"
        elif os.path.exists(src_png): src_file = src_png; ext = ".png"
        else: return False
            
        abnormal_crops_dir = os.path.join(OUTPUT_DIR, "Abnormal_Crops_Recheck")
        target_folder = os.path.join(abnormal_crops_dir, folder_name_no_ext)
        if not os.path.exists(target_folder): os.makedirs(target_folder)
        shutil.copy(src_file, os.path.join(target_folder, f"{roi_id}{ext}"))
        return True
    except Exception: return False

# ================= 2. VALIDATION LOGIC =================
def validate_value(val, data_type):
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
        if re.match(r'^-?\d+(\.\d+)?$', val_str):
            if '.' in val_str:
                decimals = len(val_str.split('.')[1])
                if decimals > MAX_DECIMALS: return False, val, "Suspicious Pattern (>3 decimals)"
            return True, val_str, None
        return False, val, "Invalid Float"

    elif data_type == 'TIME':
        if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str): return True, val_str, None
        return False, val, "Invalid Time"

    return False, val, "Unknown Type"

# ================= 3. SMART MERGE LOGIC =================
def select_best_roi_value(series, data_type):
    """
    Given a pandas Series of values for ONE specific ROI (from duplicate rows),
    select the single best value.
    """
    valid_values = []
    
    # Filter and score values
    for val in series:
        is_valid, clean_val, reason = validate_value(val, data_type)
        if is_valid:
            score = 0
            # Preference Logic
            if data_type == 'STATUS' and clean_val == 'OK': score = 10
            elif data_type == 'FLOAT' and '.' in str(clean_val):
                decimals = len(str(clean_val).split('.')[1])
                if decimals == 3: score = 20 # Best
                elif decimals == 2: score = 10 # Okay
            elif data_type == 'INTEGER': score = 5 # Valid integer
            
            valid_values.append({'val': clean_val, 'score': score})

    if not valid_values:
        # If no valid values exist, return the first one (will be flagged later)
        return series.iloc[0]

    # Sort by score descending
    valid_values.sort(key=lambda x: x['score'], reverse=True)
    return valid_values[0]['val']


# ================= 4. PROCESSING CORE =================
def get_config_for_file(df):
    for config in ROI_CONFIGS:
        if config['Trigger_Col'] in df.columns: return config
    return None

def process_single_file(input_csv_path, output_folder):
    filename = os.path.basename(input_csv_path)
    print(f"\nProcessing: {filename}...")
    try: df = pd.read_csv(input_csv_path)
    except Exception as e: print(f"  Error: {e}"); return

    config = get_config_for_file(df)
    if not config: print("  Skipped: Unknown format."); return

    # Standardize Date
    if 'Filename' in df.columns:
        df['Extracted_Date'] = df['Filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        if not df['Extracted_Date'].isna().all():
            date_str = pd.to_datetime(df['Extracted_Date']).dt.strftime('%m/%d/%y')
            if 'ROI_51' not in df.columns: df['ROI_51'] = date_str

    if 'ROI_52' not in df.columns: print("  Warning: No timestamp to merge."); return

    # --- MERGE STEP: Group by Timestamp & Construct Best Row ---
    print("  Merging duplicates by selecting best ROI values...")
    
    # Columns we will process
    roi_map = config['Columns']
    meta_cols = ['Filename', 'File_UTC', 'Machine_Text', 'Machine_UTC', 'ROI_51', 'ROI_52']
    
    # Initialize container for merged rows
    merged_rows = []
    deleted_rows_log = [] # Just tracking how many were dropped

    grouped = df.groupby('ROI_52')
    
    for timestamp, group in grouped:
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
            continue
            
        # MULTIPLE ROWS FOUND -> COMBINE THEM
        best_row = {}
        
        # 1. Handle Metadata (Just take from the last file, usually most recent)
        last_entry = group.iloc[-1]
        for meta in meta_cols:
            if meta in last_entry: best_row[meta] = last_entry[meta]

        # 2. Handle ROIs (Pick the Winner for EACH ROI)
        for roi_col, dtype in roi_map.items():
            if roi_col == 'ROI_52': continue # Already grouped key
            if roi_col in group.columns:
                best_val = select_best_roi_value(group[roi_col], dtype)
                best_row[roi_col] = best_val

        merged_rows.append(best_row)
        
        # Log duplicates for stats
        for idx, row in group.iloc[:-1].iterrows():
            deleted_rows_log.append({'ROI_52': timestamp, 'Dropped_File': row.get('Filename', 'Unknown')})

    # Create the clean dataframe from the merged list
    df_clean = pd.DataFrame(merged_rows)
    
    # Re-order columns to match input
    df_clean = df_clean[df.columns.intersection(df_clean.columns)]

    # --- FINAL VALIDATION PASS (On the Merged Data) ---
    abnormal_records = []
    
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
                        'Reason': reason
                    })
                else:
                    # Final update (e.g. normalizing string formats)
                    df_clean.at[idx, roi_col] = clean_val

    # --- SAVE OUTPUTS ---
    base_name = os.path.splitext(filename)[0]
    
    # 1. Cleaned File
    df_clean.to_csv(os.path.join(output_folder, f"{base_name}_Cleaned.csv"), index=False)
    
    # 2. Abnormal Log & Crops
    if abnormal_records:
        pd.DataFrame(abnormal_records).to_csv(os.path.join(output_folder, f"{base_name}_Abnormal_Log.csv"), index=False)
        print(f"  -> Flagging {len(abnormal_records)} abnormal items (Copying crops...)")
        for rec in abnormal_records:
            copy_abnormal_crop(rec['Filename'], rec['ROI_ID'])

    print(f"  -> Original Rows: {len(df)} | Merged Rows: {len(df_clean)} | Duplicates Combined: {len(deleted_rows_log)}")

# ================= 5. RUNNER =================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    if os.path.isdir(INPUT_FOLDER):
        files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
        files = [f for f in files if not any(x in f for x in ['_Cleaned', '_Log'])]
        print(f"Found {len(files)} files.")
        for f in files: process_single_file(f, OUTPUT_DIR)
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()