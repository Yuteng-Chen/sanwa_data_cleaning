import pandas as pd
import os
import re
import glob
from datetime import datetime

# ================= USER CONFIGURATION =================
# 1. INPUT: The "Relabeled" CSVs
INPUT_FOLDER = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/Final_7B_Relabeled_Verified/'

# 2. OUTPUT: The final condensed dataset
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/Final_Consolidated_Dataset/'

# ================= HELPER FUNCTIONS =================
def parse_pc_filename_time(filename):
    """ Extracts timestamp from filename: '2025-12-16 17.20.09.png' -> datetime obj """
    try:
        match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}\.\d{2}\.\d{2})', str(filename))
        if match:
            clean_str = match.group(1).replace('.', ':')
            return datetime.strptime(clean_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None
    return None

def check_redundancy_pair(curr_row, next_row):
    """
    Returns True if the rows form a [Redundant Next -> Redundant Prev] pair.
    """
    curr_status = str(curr_row.get('Data_Redundancy', ''))
    next_status = str(next_row.get('Data_Redundancy', ''))
    
    # Condition: Current has "Next", Next has "Prev"
    has_next = "Redundant Next" in curr_status
    has_prev = "Redundant Prev" in next_status or "Redundant Previous" in next_status
    
    return has_next and has_prev

# ================= CORE LOGIC =================
def consolidate_file(csv_path):
    filename = os.path.basename(csv_path)
    
    # SAFETY CHECK: Only process verified relabeled files
    if "_Relabeled_Verified.csv" not in filename:
        return

    print(f"\nProcessing: {filename}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [Error] {e}")
        return

    if df.empty or 'Filename' not in df.columns: return

    # Sort to ensure time order
    df.sort_values(by='Filename', inplace=True)
    rows = df.to_dict('records')
    total_rows = len(rows)
    
    kept_rows = []
    deletion_log = []
    
    i = 0
    while i < total_rows:
        curr_row = rows[i]
        
        # Look ahead if not at the last row
        if i < total_rows - 1:
            next_row = rows[i+1]
            
            # === PAIRWISE COMPRESSION LOGIC ===
            # If current is "Redundant Next" and next is "Redundant Prev" -> Collapse them
            if check_redundancy_pair(curr_row, next_row):
                
                # 1. Keep the Current Row (The first of the pair)
                # We append it now, but we will calculate time duration in Pass 2
                kept_rows.append(curr_row)
                
                # 2. Mark Next Row for Deletion
                deletion_log.append({
                    'Deleted_Filename': next_row['Filename'],
                    'Reason': "Pairwise Compression (Matches Previous)",
                    'Original_Status': next_row.get('Data_Redundancy')
                })
                
                # 3. Skip the next row (since we deleted it)
                i += 2 
                continue

        # If no pair matched, just keep the current row and move to next
        kept_rows.append(curr_row)
        i += 1

    # === PASS 2: CALCULATE TIME DURATION ===
    # Now that we have the final list of rows, we calculate the time gap
    # between each row and its predecessor. This accurately reflects "Freeze Time".
    
    for k in range(len(kept_rows)):
        curr_item = kept_rows[k]
        curr_time = parse_pc_filename_time(curr_item['Filename'])
        
        step_duration = 0.0
        
        if k > 0:
            prev_item = kept_rows[k-1]
            prev_time = parse_pc_filename_time(prev_item['Filename'])
            
            if curr_time and prev_time:
                # Calculate total seconds elapsed since the last KEPT row
                step_duration = (curr_time - prev_time).total_seconds()
        
        # Log this duration
        curr_item['Real_Freeze_Duration_Sec'] = round(step_duration, 2)
        
        # Update Status to reflect compression if needed
        if "Redundant" in str(curr_item.get('Data_Redundancy', '')):
             curr_item['Data_Redundancy'] = f"Redundant Pair (Kept 1st) | Gap: {step_duration}s"

    # --- SAVE CONSOLIDATED DATA ---
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    out_name = filename.replace("_Relabeled_Verified.csv", "_Consolidated.csv")
    if out_name == filename: out_name = filename.replace(".csv", "_Consolidated.csv")
    
    df_final = pd.DataFrame(kept_rows)
    df_final.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
    
    # --- SAVE DELETION LOG ---
    if deletion_log:
        log_name = filename.replace("_Relabeled_Verified.csv", "_Deletion_Log.csv")
        df_log = pd.DataFrame(deletion_log)
        df_log.to_csv(os.path.join(OUTPUT_DIR, log_name), index=False)
        print(f"  -> Deleted {len(df_log)} rows (Pairs compressed). Log saved.")
    
    print(f"✅ Saved: {out_name}")
    print(f"   Compression: {total_rows} -> {len(df_final)} rows.")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    if os.path.isdir(INPUT_FOLDER):
        files = glob.glob(os.path.join(INPUT_FOLDER, "*_Relabeled_Verified.csv"))
        
        if not files:
            print(f"No correct input files found in {INPUT_FOLDER}")
            return
            
        print(f"Found {len(files)} files.")
        for f in files: consolidate_file(f)
    else:
        print(f"Invalid input folder: {INPUT_FOLDER}")

if __name__ == "__main__":
    main()