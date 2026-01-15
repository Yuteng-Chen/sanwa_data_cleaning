import pandas as pd
import os
import glob
import sys

# ================= USER CONFIGURATION =================
# 1. Where your Original Cleaned CSVs are (The files WITH the defects)
ORIGINAL_CLEANED_DIR = 'Archive/Archive/Cut_preprocesseddata_Output'

# 2. Where your AI Fixed CSVs are (The output from the previous step)
AI_FIXED_DIR = 'Archive/Archive/Final_AI_Corrected_Logs_1511'

# 3. New Folder for the Final, Perfected Data
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606'
# ======================================================

def get_original_filename(fixed_filename):
    """
    Converts '..._Abnormal_Log_AI_Fixed.csv' back to '..._Cleaned.csv'
    """
    # Remove the suffix added by the fix script
    base = fixed_filename.replace("_Abnormal_Log_AI_Fixed.csv", "")
    base = base.replace("_AI_Fixed.csv", "") # Fallback if naming changed
    
    # Add the suffix for the original cleaned file
    return f"{base}_Cleaned.csv"

def merge_corrections():
    # Create Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Created Output Folder: {OUTPUT_DIR}")

    # Find all AI Fixed Files
    fixed_files = glob.glob(os.path.join(AI_FIXED_DIR, "*_AI_Fixed.csv"))
    
    if not fixed_files:
        print(f"❌ No fixed files found in {AI_FIXED_DIR}")
        return

    print(f"🔍 Found {len(fixed_files)} fixed logs. Starting merge...\n")

    for fixed_path in fixed_files:
        fixed_filename = os.path.basename(fixed_path)
        
        # 1. Determine Original File Path
        original_filename = get_original_filename(fixed_filename)
        original_path = os.path.join(ORIGINAL_CLEANED_DIR, original_filename)
        
        if not os.path.exists(original_path):
            print(f"⚠️  Skipping: Original file not found for {fixed_filename}")
            print(f"    (Looked for: {original_filename})")
            continue

        print(f"🔄 Merging: {fixed_filename}  >>>  {original_filename}")

        # 2. Load DataFrames
        try:
            df_fixed = pd.read_csv(fixed_path)
            df_original = pd.read_csv(original_path)
        except Exception as e:
            print(f"    ❌ Read Error: {e}")
            continue

        # 3. Check for necessary columns
        if 'Filename' not in df_fixed.columns or 'ROI_ID' not in df_fixed.columns or 'AI_Corrected' not in df_fixed.columns:
            print(f"    ⚠️  Skipping: Fixed file missing columns (Filename, ROI_ID, AI_Corrected)")
            continue
        
        if 'Filename' not in df_original.columns:
            print(f"    ⚠️  Skipping: Original file missing 'Filename' column")
            continue

        # 4. Perform Updates
        update_count = 0
        
        # Iterate through every correction
        for _, row in df_fixed.iterrows():
            filename = row['Filename']
            roi_col = row['ROI_ID'] # e.g., "ROI_13"
            new_val = row['AI_Corrected']
            
            # Skip if AI didn't provide a value or said "Image Not Found"
            if pd.isna(new_val) or str(new_val).strip() in ["", "Image Not Found", "Read Error", "ERROR"]:
                continue
                
            # Clean up value (ensure string doesn't have extra quotes)
            new_val = str(new_val).strip().replace("'", "").replace('"', '')

            # Check if ROI column exists in original
            if roi_col not in df_original.columns:
                print(f"      ⚠️  Column {roi_col} not found in original CSV.")
                continue

            # UPDATE THE CELL
            # Find the row index where Filename matches
            match_mask = df_original['Filename'] == filename
            
            if match_mask.any():
                # Update the specific cell
                df_original.loc[match_mask, roi_col] = new_val
                update_count += 1

        # 5. Save the Merged File
        save_path = os.path.join(OUTPUT_DIR, original_filename)
        df_original.to_csv(save_path, index=False)
        
        print(f"    ✅ Updates Applied: {update_count}")
        print(f"    💾 Saved to: {save_path}\n")

    print("==========================================")
    print("🎉 All Merges Complete.")

if __name__ == "__main__":
    merge_corrections()