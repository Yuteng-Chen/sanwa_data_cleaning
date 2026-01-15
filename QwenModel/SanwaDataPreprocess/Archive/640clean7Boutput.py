import pandas as pd
import os
import glob

# ================= USER CONFIGURATION =================
# 1. Source "Cleaned" CSVs (Output from the Fuzzy Redundancy step)
SOURCE_CLEANED_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/'

# 2. AI Verified Logs (Output from the 7B Judge step)
VERIFIED_LOGS_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/AI_Verified_Logs/'

# 3. Final Destination for Fixed Data
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/Final_7B_Corrected/'

# ================= PROCESSING LOGIC =================
def apply_redundancy_fixes(source_csv_path, verified_log_path):
    filename = os.path.basename(source_csv_path)
    print(f"\nProcessing: {filename}")
    
    try:
        # Load Main Dataset
        df_main = pd.read_csv(source_csv_path)
        # Load the 7B Judge Results
        df_log = pd.read_csv(verified_log_path)
    except Exception as e:
        print(f"  [Error] Could not read files: {e}")
        return

    if df_log.empty:
        print("  No redundancy conflicts to fix.")
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        save_path = os.path.join(OUTPUT_DIR, filename.replace("_Cleaned.csv", "_7B_Corrected.csv"))
        df_main.to_csv(save_path, index=False)
        return

    # --- BUILD CORRECTION MAP ---
    # Stores: (Filename_Current, Filename_Compared, ROI_ID) -> AI_Genius_Value
    corrections = [] 
    
    for _, row in df_log.iterrows():
        ai_val = str(row.get('AI_7B_Read', '')).strip()
        
        # 1. Filter out invalid AI readings
        if ai_val in ["", "nan", "Image Not Found", "ERROR"]:
            continue

        file_curr = row['Filename_Current']
        file_comp = row['Filename_Compared']
        target_roi = row['ROI_ID']
        
        corrections.append({
            'curr': file_curr,
            'comp': file_comp,
            'roi': target_roi,
            'val': ai_val
        })

    print(f"  Found {len(corrections)} conflicts. Applying to BOTH rows (Current & Compared)...")

    # --- APPLY PATCHES (DOUBLE UPDATE) ---
    patch_count = 0
    
    # Ensure Filename is string for accurate matching
    df_main['Filename'] = df_main['Filename'].astype(str)
    
    for item in corrections:
        target_roi = item['roi']
        new_val = item['val']
        
        # Check if ROI column exists
        if target_roi not in df_main.columns:
            print(f"    [Warn] ROI {target_roi} missing in main CSV.")
            continue
            
        # 1. Update Current Row
        mask_curr = df_main['Filename'] == item['curr']
        if mask_curr.any():
            df_main.loc[mask_curr, target_roi] = new_val
            patch_count += 1
            
        # 2. Update Compared Row (The Redundant one)
        mask_comp = df_main['Filename'] == item['comp']
        if mask_comp.any():
            df_main.loc[mask_comp, target_roi] = new_val
            patch_count += 1

    # --- SAVE RESULT ---
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    out_name = filename.replace("_Cleaned.csv", "_7B_Corrected.csv")
    if out_name == filename: out_name = filename.replace(".csv", "_7B_Corrected.csv")
    
    final_path = os.path.join(OUTPUT_DIR, out_name)
    df_main.to_csv(final_path, index=False)
    
    print(f"✅ Saved: {out_name}")
    print(f"   -> Successfully patched {patch_count} cells (Current + Compared).")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Get all verified logs
    verified_logs = glob.glob(os.path.join(VERIFIED_LOGS_DIR, "*_AI_Verified.csv"))
    
    if not verified_logs:
        print(f"No Verified Logs found in {VERIFIED_LOGS_DIR}")
        return
        
    print(f"Found {len(verified_logs)} verified logs...")
    
    for log_path in verified_logs:
        base_name = os.path.basename(log_path).replace("_Redundancy_Mismatch_Log_AI_Verified.csv", "")
        source_path = os.path.join(SOURCE_CLEANED_DIR, f"{base_name}_Cleaned.csv")
        
        if os.path.exists(source_path):
            apply_redundancy_fixes(source_path, log_path)
        else:
            print(f"⚠️ Source CSV missing for: {base_name}")

if __name__ == "__main__":
    main()