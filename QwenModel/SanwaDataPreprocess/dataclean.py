import pandas as pd
import re
import os
import glob

# --- 1. Updated Validation Logic ---
def validate_status(val):
    """
    Applies user-defined rules for Status columns:
    1. If it starts with 'N' -> 'NG'
    2. If it contains 'O' or 'K' -> 'OK'
    3. Else -> Invalid/Keep Original
    """
    if pd.isna(val):
        return False, val
    
    val_str = str(val).strip().upper() # Normalize to uppercase
    
    # Priority 1: Check for NG (Starts with N)
    if val_str.startswith('N'):
        return True, 'NG'
    
    # Priority 2: Check for OK (Contains O or K)
    if 'O' in val_str or 'K' in val_str:
        return True, 'OK'
        
    return False, val # Original value if neither rule matches

def validate_numeric(val):
    val_str = str(val).strip()
    # Matches integers (97) or floats (1.822)
    if re.match(r'^\d+(\.\d+)?$', val_str): return True
    return False

def validate_time(val):
    val_str = str(val).strip()
    # Matches HH:MM:SS
    if re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str): return True
    return False

# --- 2. Core Processing Function (Per File) ---
def process_single_file(input_csv_path):
    print(f"Processing file: {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading {input_csv_path}: {e}")
        return

    # A. Standardize Date (ROI_51) from Filename
    if 'Filename' in df.columns:
        # Extracts YYYY-MM-DD and converts to MM/DD/YY
        df['Extracted_Date'] = df['Filename'].str.extract(r'(\d{4}-\d{2}-\d{2})')
        df['ROI_51'] = pd.to_datetime(df['Extracted_Date']).dt.strftime('%m/%d/%y')
        df['Machine_Text'] = df['ROI_51']

    # B. Define Columns to Validate
    status_cols = ['ROI_12', 'ROI_14', 'ROI_15', 'ROI_17', 'ROI_19']
    numeric_cols = ['ROI_13', 'ROI_16', 'ROI_18']
    time_cols = ['ROI_52']

    abnormal_records = []
    df['total_invalid_rois'] = 0

    # C. Validation Loop
    for idx, row in df.iterrows():
        # Validate Status
        for col in status_cols:
            if col in df.columns:
                is_valid, clean_val = validate_status(row[col])
                if not is_valid:
                    abnormal_records.append({'Filename': row['Filename'], 'ROI_52': row.get('ROI_52'), 'ROI_ID': col, 'Value': row[col], 'Reason': 'Invalid Status'})
                    df.at[idx, 'total_invalid_rois'] += 1
                else:
                    df.at[idx, col] = clean_val 

        # Validate Numbers
        for col in numeric_cols:
            if col in df.columns:
                if not validate_numeric(row[col]):
                    abnormal_records.append({'Filename': row['Filename'], 'ROI_52': row.get('ROI_52'), 'ROI_ID': col, 'Value': row[col], 'Reason': 'Invalid Number'})
                    df.at[idx, 'total_invalid_rois'] += 1
        
        # Validate Time
        for col in time_cols:
            if col in df.columns:
                if not validate_time(row[col]):
                    abnormal_records.append({'Filename': row['Filename'], 'ROI_52': row.get('ROI_52'), 'ROI_ID': col, 'Value': row[col], 'Reason': 'Invalid Time'})
                    df.at[idx, 'total_invalid_rois'] += 1

    # D. Deduplication
    # Create sorting helpers
    df['sort_roi_13'] = pd.to_numeric(df['ROI_13'], errors='coerce').fillna(-1)
    
    # Count OKs in updated status columns
    exist_status_cols = [c for c in status_cols if c in df.columns]
    df['ok_count'] = df[exist_status_cols].apply(lambda x: (x == 'OK').sum(), axis=1)

    # Sort Priority: 
    # 1. Fewest Invalid Fields (asc)
    # 2. Highest Confidence (desc)
    # 3. Most OKs (desc)
    # 4. Latest Filename (desc)
    df_sorted = df.sort_values(
        by=['ROI_52', 'total_invalid_rois', 'sort_roi_13', 'ok_count', 'Filename'],
        ascending=[True, True, False, False, False]
    )

    df_sorted['is_duplicate'] = df_sorted.duplicated(subset=['ROI_52'], keep='first')
    df_clean = df_sorted[~df_sorted['is_duplicate']].copy()
    df_deleted = df_sorted[df_sorted['is_duplicate']].copy()

    # E. Generate Deletion Reasons
    winners = df_clean.set_index('ROI_52')
    def get_reason(row):
        if row['ROI_52'] not in winners.index: return "Unknown"
        winner = winners.loc[row['ROI_52']]
        if winner['total_invalid_rois'] < row['total_invalid_rois']: return "Contains Invalid Values"
        if winner['sort_roi_13'] > row['sort_roi_13']: return "Lower Confidence"
        if winner['ok_count'] > row['ok_count']: return "Fewer OK Statuses"
        return "Duplicate Timestamp"

    df_deleted['Reason'] = df_deleted.apply(get_reason, axis=1)
    df_deleted['Kept_File'] = df_deleted['ROI_52'].map(winners['Filename'])

    # F. Save Files
    base_name = os.path.splitext(input_csv_path)[0]
    
    # 1. Clean File
    cols_to_remove = ['Extracted_Date', 'total_invalid_rois', 'sort_roi_13', 'ok_count', 'is_duplicate']
    out_cols = [c for c in df.columns if c not in cols_to_remove]
    df_clean[out_cols].to_csv(f"{base_name}_Cleaned.csv", index=False)
    
    # 2. Deleted Log
    log_cols = ['Filename', 'ROI_52', 'Kept_File', 'Reason', 'ROI_13', 'ROI_16', 'ROI_18']
    df_deleted[[c for c in log_cols if c in df_deleted.columns]].to_csv(f"{base_name}_Deleted_Log.csv", index=False)
    
    # 3. Abnormal Log
    if abnormal_records:
        pd.DataFrame(abnormal_records).to_csv(f"{base_name}_Abnormal_Log.csv", index=False)
    
    print(f"Done. Processed {os.path.basename(input_csv_path)}")
    print(f"  - Cleaned Rows: {len(df_clean)}")
    print(f"  - Deleted Rows: {len(df_deleted)}")
    print(f"  - Abnormal Records: {len(abnormal_records)}")

# --- 3. Wrapper to Handle File OR Folder ---
def process_path(input_path):
    """
    Checks if input_path is a folder or a file.
    """
    if os.path.isdir(input_path):
        print(f"Directory detected: {input_path}")
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        if not csv_files:
            print("No CSV files found in this directory.")
            return
        
        print(f"Found {len(csv_files)} files. Starting batch process...")
        for file_path in csv_files:
            # Skip generated reports to avoid infinite loops
            if any(x in file_path for x in ['_Cleaned.csv', '_Deleted_Log.csv', '_Abnormal_Log.csv']):
                continue
            process_single_file(file_path)
            
    elif os.path.isfile(input_path):
        print(f"Single file detected: {input_path}")
        process_single_file(input_path)
        
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")

# --- 4. Execution ---
# Replace with your Folder Path or File Name
input_path = 'cam 6 snap1 Latchresult cleanup.csv' 
process_path(input_path)