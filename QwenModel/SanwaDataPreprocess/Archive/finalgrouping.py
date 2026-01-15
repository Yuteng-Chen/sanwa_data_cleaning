import pandas as pd
import re
import os
import glob
import numpy as np

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_final' # Result from V16
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_Chained/'

# Settings
MAX_DECIMALS = 3

# ================= ROI CONFIGURATION (For Scoring) =================
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

def get_config_for_file(df):
    for config in ROI_CONFIGS:
        if config['Trigger_Col'] in df.columns: return config
    return None

def calculate_global_medians(df, roi_map):
    medians = {}
    for col, dtype in roi_map.items():
        if dtype in ['INTEGER', 'FLOAT'] and col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            valid_series = series[(series != 0) & (~series.isna())]
            if not valid_series.empty:
                medians[col] = valid_series.median()
            else:
                medians[col] = 0
    return medians

def calculate_outlier_score(row, medians, roi_map):
    score = 0.0
    for col, dtype in roi_map.items():
        if dtype in ['INTEGER', 'FLOAT'] and col in row and col in medians:
            try:
                val = float(row[col])
                median = medians[col]
                if val == 0: continue 
                if median == 0: continue 
                deviation = abs(val - median) / median
                score += deviation
            except: pass
    return score

def get_positional_fingerprint(row, columns_list):
    start_idx = 4 
    if 'ROI_51' in columns_list: end_idx = columns_list.index('ROI_51')
    elif 'ROI_52' in columns_list: end_idx = columns_list.index('ROI_52')
    else: end_idx = len(columns_list)
        
    if start_idx >= end_idx: return ""
    fingerprint = []
    cols_to_check = columns_list[start_idx:end_idx]
    for col in cols_to_check:
        val = str(row.get(col, ''))
        fingerprint.append(val)
    return "|".join(fingerprint)

# ================= CHAINING LOGIC =================

def process_single_file(input_csv_path, output_folder):
    filename = os.path.basename(input_csv_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\nProcessing Chains: {filename}...")
    
    try: df = pd.read_csv(input_csv_path)
    except Exception as e: print(f"  Error: {e}"); return

    config = get_config_for_file(df)
    if not config: print("  Skipped: Unknown format."); return
    roi_map = config['Columns']

    if 'Filename' in df.columns:
        df.sort_values(by='Filename', inplace=True)

    # Calculate scores to pick the best row in a chain
    medians = calculate_global_medians(df, roi_map)
    df['_Outlier_Score'] = df.apply(lambda row: calculate_outlier_score(row, medians, roi_map), axis=1)

    # --- IDENTIFY CHAINS ---
    # We assign a 'Chain_ID' to every consecutive block of identical data.
    
    chain_ids = []
    current_chain_id = 0
    prev_fingerprint = None
    columns_list = df.columns.tolist()
    
    for idx, row in df.iterrows():
        curr_fingerprint = get_positional_fingerprint(row, columns_list)
        
        if prev_fingerprint is None:
            current_chain_id = 0
        else:
            if curr_fingerprint != prev_fingerprint:
                # content changed -> New Chain
                current_chain_id += 1
            else:
                # content same -> Same Chain
                pass
        
        chain_ids.append(current_chain_id)
        prev_fingerprint = curr_fingerprint
        
    df['Chain_ID'] = chain_ids
    
    # --- CONSOLIDATE CHAINS ---
    final_rows = []
    
    for c_id, group in df.groupby('Chain_ID'):
        # 1. Start and End filenames
        start_row = group.iloc[0]
        end_row = group.iloc[-1]
        
        # 2. Best Data Row (Lowest Outlier Score)
        best_data_row = group.sort_values(by='_Outlier_Score').iloc[0].copy()
        
        # 3. Construct Final Row
        # Use Metadata from START (Earliest Time)
        # Use Data from BEST (Cleanest OCR)
        
        final_row = best_data_row
        
        # Overwrite metadata with the Start of the chain
        if 'Filename' in start_row: final_row['Filename'] = start_row['Filename']
        if 'File_UTC' in start_row: final_row['File_UTC'] = start_row['File_UTC']
        if 'ROI_52' in start_row: final_row['ROI_52'] = start_row['ROI_52']
        if 'Time_Status' in start_row: final_row['Time_Status'] = start_row['Time_Status']
        
        # Add Chain Metadata
        final_row['Chain_Start_File'] = start_row['Filename']
        final_row['Chain_End_File'] = end_row['Filename']
        final_row['Chain_Size'] = len(group)
        
        # Clean up Redundancy Status for the consolidated row
        if len(group) > 1:
            final_row['Data_Redundancy'] = f"Consolidated ({len(group)} rows)"
        else:
            final_row['Data_Redundancy'] = "Unique"
            
        final_rows.append(final_row)
        
    df_final = pd.DataFrame(final_rows)
    
    # Remove temp cols
    cols_to_drop = ['_Outlier_Score', 'Chain_ID', 'Matched_File']
    df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], inplace=True)

    # SAVE
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    df_final.to_csv(os.path.join(output_folder, f"{base_name}_Final_Chained.csv"), index=False)
    
    print(f"  -> Original Rows: {len(df)} | Final Events: {len(df_final)}")

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