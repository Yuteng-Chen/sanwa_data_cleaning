import pandas as pd
import shutil
import os
import re

# ================= USER CONFIGURATION =================
# 1. Input: The ORIGINAL Cleaned CSV (Update filename if needed)
INPUT_CSV = 'Archive/Archive/cam 6 snap1 Latchresult cleanup.csv'

# 2. Source: Where the debug crops currently exist
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# 3. Destination: Folder for items that fail validation
DESTINATION_ROOT = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/manual_validate_original_recheck1/'

# 4. Output CSV name (For Qwen)
OUTPUT_RECHECK_CSV = 'Qwen_Recheck_Input.csv'

# 5. Output Master CSV (The Auto-Corrected version)
OUTPUT_MASTER_CSV = 'Archive/Archive/cam_6_snap1_Latchresult_AutoCorrected.csv'
# ======================================================

def smart_clean_and_truncate(val):
    """
    1. If multiple dots exist (e.g. '9.15.15'), return original -> Flag as INVALID.
    2. If single dot but >3 decimals (e.g. '0.0001'), truncate -> '0.000' -> VALID.
    3. Cleans stray non-numeric chars.
    """
    val_str = str(val).strip()
    
    # 1. Check for multiple decimal points IMMEDIATELY (Structural Error)
    if val_str.count('.') > 1:
        return val_str # Return as-is so validation fails below
    
    # 2. Clean non-numeric characters (assuming single dot or integer)
    clean_chars = re.sub(r'[^\d.]', '', val_str)
    
    # 3. Truncate to 3 decimal places if needed
    if '.' in clean_chars:
        dot_index = clean_chars.find('.')
        # Cut string at dot + 4 characters (1 for dot + 3 decimals)
        clean_chars = clean_chars[:dot_index + 4]
        
    return clean_chars

def validate_status(val):
    val_str = str(val).strip().upper()
    # Checks if it is exactly OK or NG
    return val_str in ['OK', 'NG']

def validate_numeric(val):
    val_str = str(val).strip()
    # Strict Regex: Integers OR Decimals with MAX 3 digits
    return bool(re.match(r'^\d+(\.\d{1,3})?$', val_str))

def validate_time(val):
    val_str = str(val).strip()
    # Matches HH:MM:SS
    return bool(re.match(r'^\d{1,2}:\d{2}:\d{2}$', val_str))

def get_qwen_prompt(roi_id):
    if roi_id in ['ROI_13', 'ROI_16', 'ROI_18']: 
        return "Identify the single numeric value. Limit to 3 decimal places max. Ignore artifacts. Output ONLY the number."
    elif roi_id == 'ROI_52': 
        return "Read the timestamp in HH:MM:SS format. Fix common OCR errors. Output ONLY the time string."
    elif roi_id in ['ROI_12', 'ROI_14', 'ROI_15', 'ROI_17', 'ROI_19']: 
        return "Classify status. If text contains 'O' or 'K' -> 'OK'. If text starts with 'N' -> 'NG'. Output ONLY 'OK' or 'NG'."
    return "Read the text accurately."

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Create destination directory
    if not os.path.exists(DESTINATION_ROOT):
        try:
            os.makedirs(DESTINATION_ROOT)
            print(f"Created folder: {DESTINATION_ROOT}")
        except OSError as e:
            print(f"Error creating destination root: {e}")
            return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    status_cols = ['ROI_12', 'ROI_14', 'ROI_15', 'ROI_17', 'ROI_19']
    numeric_cols = ['ROI_13', 'ROI_16', 'ROI_18']
    time_cols = ['ROI_52']

    recheck_list = []
    autocorrect_count = 0
    
    print("Validating data (Smart Cleaning + Strict Format Check)...")

    for idx, row in df.iterrows():
        filename_png = row['Filename']
        rois_to_flag = []
        
        # 1. Status Check
        for col in status_cols:
            if col in df.columns:
                if not validate_status(row[col]):
                    rois_to_flag.append((col, row[col], "Invalid Status Format"))
        
        # 2. Numeric Check (With Auto-Correction)
        for col in numeric_cols:
            if col in df.columns:
                original_val = str(row[col])
                
                # A. Apply Smart Clean & Truncate
                fixed_val = smart_clean_and_truncate(original_val)
                
                # If value changed (e.g. 0.0001 -> 0.000), update Master DF
                if fixed_val != original_val:
                    df.at[idx, col] = fixed_val
                    autocorrect_count += 1
                
                # B. Validate the result
                # If fixed_val is still bad (e.g. "9.15.15"), it fails here
                if not validate_numeric(fixed_val):
                    reason = "Multiple Dots" if fixed_val.count('.') > 1 else "Invalid Numeric Format"
                    rois_to_flag.append((col, fixed_val, reason))
                    
        # 3. Time Check
        for col in time_cols:
            if col in df.columns:
                if not validate_time(row[col]):
                    rois_to_flag.append((col, row[col], "Invalid Time Format"))

        # 4. Handle Invalid Items (Prepare Crops)
        for roi_id, val, reason in rois_to_flag:
            
            # Locate Source Image
            folder_name_no_ext = os.path.splitext(filename_png)[0]
            src_folder = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext)
            
            # Try JPG then PNG
            src_file = None
            ext = ""
            if os.path.exists(os.path.join(src_folder, f"{roi_id}.jpg")):
                src_file = os.path.join(src_folder, f"{roi_id}.jpg")
                ext = ".jpg"
            elif os.path.exists(os.path.join(src_folder, f"{roi_id}.png")):
                src_file = os.path.join(src_folder, f"{roi_id}.png")
                ext = ".png"
            else:
                print(f"Missing crop: {folder_name_no_ext}/{roi_id}")
                continue

            # Prepare Destination
            target_folder = os.path.join(DESTINATION_ROOT, folder_name_no_ext)
            if not os.path.exists(target_folder):
                try: os.makedirs(target_folder)
                except OSError: continue
                
            dst_file = os.path.join(target_folder, f"{roi_id}{ext}")

            # Copy Crop
            try:
                shutil.copy(src_file, dst_file)
            except Exception as e:
                print(f"Failed to copy {src_file}: {e}")
                continue

            # Add to Recheck List
            recheck_list.append({
                'Parent_Filename': filename_png,
                'ROI_ID': roi_id,
                'Current_Value': val,
                'Reason': reason,
                'Validation_Image_Path': dst_file,
                'Prompt': get_qwen_prompt(roi_id),
                'Validated_Value': '' 
            })

    # Output 1: Save Recheck List (For Qwen)
    if recheck_list:
        csv_path = os.path.join(DESTINATION_ROOT, OUTPUT_RECHECK_CSV)
        pd.DataFrame(recheck_list).to_csv(csv_path, index=False)
        print("------------------------------------------------")
        print(f"⚠️ Found {len(recheck_list)} items failing strict validation.")
        print(f"📂 Re-validation folders created at: {DESTINATION_ROOT}")
        print(f"📄 Qwen Input CSV saved at: {csv_path}")
    else:
        print("------------------------------------------------")
        print("✅ All items valid! No Qwen re-check needed.")

    # Output 2: Save Auto-Corrected Master File
    df.to_csv(OUTPUT_MASTER_CSV, index=False)
    print("------------------------------------------------")
    print(f"✅ Auto-Corrected {autocorrect_count} values (Long decimals truncated).")
    print(f"✅ Saved NEW MASTER dataset to: {OUTPUT_MASTER_CSV}")

if __name__ == "__main__":
    main()