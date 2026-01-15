import pandas as pd
import re
import os
import glob
import shutil
import numpy as np
from datetime import datetime

# ================= USER CONFIGURATION =================
INPUT_FOLDER = 'Archive/Archive/Final_Cleaned_Dataset_1606'
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_Cleaned/'

# Settings
MAX_DECIMALS = 3
OUTLIER_THRESHOLD = 5.0
FROZEN_THRESHOLD_SECONDS = 10.0

# ================= ROI CONFIGURATION =================
# We use this to identify which columns are Numbers vs Status
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

def get_positional_fingerprint(row, columns_list):
    """
    Creates a fingerprint from the 5th column (Index 4) up to (but not including) ROI_51.
    Strictly ignores Time and Metadata.
    """
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

def calculate_global_medians(df, roi_map):
    """
    Calculates the Median for every numeric column.
    Ignores 0 and NaN values to find the 'True Normal'.
    """
    medians = {}
    for col, dtype in roi_map.items():
        if dtype in ['INTEGER', 'FLOAT'] and col in df.columns:
            # Convert to numeric, turn errors to NaN
            series = pd.to_numeric(df[col], errors='coerce')
            # Filter out 0 and NaN
            valid_series = series[(series != 0) & (~series.isna())]
            
            if not valid_series.empty:
                medians[col] = valid_series.median()
            else:
                medians[col] = 0 # Default if empty
    return medians

def calculate_outlier_score(row, medians, roi_map):
    """
    Scoring: Lower is Better.
    0 = Perfect Match or Zero Value.
    High Score = Far from Median (Likely OCR Error).
    """
    score = 0.0
    for col, dtype in roi_map.items():
        if dtype in ['INTEGER', 'FLOAT'] and col in row and col in medians:
            try:
                val = float(row[col])
                median = medians[col]
                
                # Rule: "except 0... keep normal one"
                if val == 0:
                    continue # 0 is considered safe/neutral
                
                if median == 0:
                    continue # Cannot judge if median is 0
                
                # Calculate percent deviation
                deviation = abs(val - median) / median
                score += deviation
            except:
                pass # Non-numeric, ignore
    return score

# ================= CORE LOGIC =================

def process_single_file(input_csv_path, output_folder):
    filename = os.path.basename(input_csv_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\nProcessing: {filename}...")
    
    try: df = pd.read_csv(input_csv_path)
    except Exception as e: print(f"  Error: {e}"); return

    config = get_config_for_file(df)
    if not config: print("  Skipped: Unknown format."); return

    roi_map = config['Columns']
    
    # 1. SORT BY FILENAME (PC Sequence)
    if 'Filename' in df.columns:
        df.sort_values(by='Filename', inplace=True)
    
    # 2. CALCULATE MEDIANS (For Outlier Detection)
    global_medians = calculate_global_medians(df, roi_map)
    
    # 3. CONFLICT RESOLUTION (Group by Time -> Pick Best)
    # We will build a list of "Winners" from each timestamp group
    resolved_rows = []
    
    # Group by PLC Time (ROI_52)
    # Note: We must preserve order, so we iterate manually
    current_time_group = []
    current_plc_time = None
    
    for idx, row in df.iterrows():
        plc_time = str(row.get('ROI_52', '')).strip()
        
        if plc_time != current_plc_time:
            # PROCESS PREVIOUS GROUP
            if current_time_group:
                # Logic: Pick the best row from the group
                if len(current_time_group) == 1:
                     resolved_rows.append(current_time_group[0])
                else:
                    # Multiple rows with same time. 
                    # 1. Check if they are already identical (Redundant)
                    # 2. If different, check Outlier Score
                    
                    # Calculate scores for all in group
                    scored_rows = []
                    for g_row in current_time_group:
                        score = calculate_outlier_score(g_row, global_medians, roi_map)
                        scored_rows.append({'row': g_row, 'score': score})
                    
                    # SORT: Ascending Score (Lower is better/more normal)
                    scored_rows.sort(key=lambda x: x['score'])
                    
                    # The Winner is the one with lowest score
                    winner = scored_rows[0]['row']
                    resolved_rows.append(winner)
                    
                    # (Optional: You could log that you deleted others here)

            # RESET GROUP
            current_time_group = [row]
            current_plc_time = plc_time
        else:
            # Same time, add to group
            current_time_group.append(row)
            
    # Process final group
    if current_time_group:
        scored_rows = []
        for g_row in current_time_group:
            score = calculate_outlier_score(g_row, global_medians, roi_map)
            scored_rows.append({'row': g_row, 'score': score})
        scored_rows.sort(key=lambda x: x['score'])
        resolved_rows.append(scored_rows[0]['row'])

    # 4. DEDUPLICATION (Filter Redundant States)
    # Now that we have the "Best" row for every second, we remove sequential duplicates.
    
    final_rows = []
    columns_list = df.columns.tolist()
    
    prev_fingerprint = None
    
    for row in resolved_rows:
        curr_fingerprint = get_positional_fingerprint(row, columns_list)
        
        if curr_fingerprint == prev_fingerprint:
            # REDUNDANT: Same data as previous valid row. SKIP IT.
            continue
        else:
            # UNIQUE: Keep it.
            final_rows.append(row)
            prev_fingerprint = curr_fingerprint
            
    # 5. CREATE OUTPUT DATAFRAME
    df_final = pd.DataFrame(final_rows)
    
    # --- SAVE ---
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    df_final.to_csv(os.path.join(output_folder, f"{base_name}_Cleaned.csv"), index=False)
    
    removed_count = len(df) - len(df_final)
    print(f"  -> Original: {len(df)} | Final: {len(df_final)} | Removed: {removed_count} rows")

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