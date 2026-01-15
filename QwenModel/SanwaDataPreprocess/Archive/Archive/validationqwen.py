import pandas as pd
import shutil
import os
import re

# ================= USER CONFIGURATION =================
# 1. Input: The CSV from the previous Qwen run
INPUT_CSV = 'Archive/Archive/qwen_prompt_enhance_validation1417.csv'

# 2. Source: Where the debug crops currently exist
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# 3. Destination: New folder for the re-check round
DESTINATION_ROOT = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/manual_validate_qwen_recheck1426/'

# 4. Output CSV name (saved inside the destination root)
OUTPUT_CSV_NAME = 'Qwen_Recheck_Input1417.csv'

# 5. Auto-Corrected Master File Name
AUTOCORRECTED_CSV_NAME = 'Archive/Archive/qwen_prompt_enhance_validation_AutoCorrected1426.csv'

# 🔴 CONFIGURATION FLAG
# True  = Automatically change 0.0001 -> 0.000 (Considered Fixed)
# False = Flag 0.0001 as INVALID -> Save crop & Add to manual check list
TRUNCATE_DECIMALS = False 
# ======================================================

def smart_clean_and_truncate(val, truncate=True):
    """
    1. If multiple dots exist (e.g. '9.15.15'), return original -> Flag as INVALID.
    2. Clean stray non-numeric chars.
    3. If truncate=True: Cut >3 decimals (e.g. '0.0001' -> '0.000').
    """
    val_str = str(val).strip()
    
    # 1. Check for multiple decimal points IMMEDIATELY
    if val_str.count('.') > 1:
        return val_str # Return as-is so validation fails below
    
    # 2. Clean non-numeric characters
    clean_chars = re.sub(r'[^\d.]', '', val_str)
    
    # 3. Truncate if enabled
    if truncate and '.' in clean_chars:
        dot_index = clean_chars.find('.')
        # Cut string at dot + 4 characters (1 for dot + 3 decimals)
        clean_chars = clean_chars[:dot_index + 4]
        
    return clean_chars

def validate_status(val):
    val_str = str(val).strip().upper()
    return val_str in ['OK', 'NG']

def validate_numeric(val):
    val_str = str(val).strip()
    # Strict Regex: Integers OR Decimals with MAX 3 digits
    return bool(re.match(r'^\d+(\.\d{1,3})?$', val_str))

def validate_time(val):
    val_str = str(val).strip()
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
    
    mode_msg = "Truncating Decimals (Auto-Fix)" if TRUNCATE_DECIMALS else "Flagging Long Decimals (Manual Check)"
    print(f"Validating data... Mode: {mode_msg}")

    for idx, row in df.iterrows():
        filename_png = row['Filename']
        rois_to_flag = []
        
        # 1. Status
        for col in status_cols:
            if col in df.columns:
                if not validate_status(row[col]):
                    rois_to_flag.append((col, row[col], "Invalid Status Format"))
        
        # 2. Numeric
        for col in numeric_cols:
            if col in df.columns:
                original_val = str(row[col])
                
                # A. Apply Smart Clean (Truncate dependent on flag)
                fixed_val = smart_clean_and_truncate(original_val, truncate=TRUNCATE_DECIMALS)
                
                # If changed (e.g. removed noise, or truncated if enabled)
                if fixed_val != original_val:
                    df.at[idx, col] = fixed_val
                    if TRUNCATE_DECIMALS: # Only count as "Auto-Corrected" if we actually truncated
                         autocorrect_count += 1
                
                # B. Validate
                # If TRUNCATE_DECIMALS is False, "0.0001" remains "0.0001"
                # validate_numeric checks for MAX 3 decimals.
                # So "0.0001" will return False -> Added to recheck_list
                if not validate_numeric(fixed_val):
                    reason = "Multiple Dots" if fixed_val.count('.') > 1 else "Invalid Numeric (>3 decimals or bad format)"
                    rois_to_flag.append((col, fixed_val, reason))
                    
        # 3. Time
        for col in time_cols:
            if col in df.columns:
                if not validate_time(row[col]):
                    rois_to_flag.append((col, row[col], "Invalid Time Format"))

        # Process Failures
        for roi_id, val, reason in rois_to_flag:
            folder_name_no_ext = os.path.splitext(filename_png)[0]
            src_folder = os.path.join(DEBUG_CROPS_BASE, folder_name_no_ext)
            
            # Check jpg/png
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

            target_folder = os.path.join(DESTINATION_ROOT, folder_name_no_ext)
            if not os.path.exists(target_folder):
                try: os.makedirs(target_folder)
                except OSError: continue
                
            dst_file = os.path.join(target_folder, f"{roi_id}{ext}")

            try:
                shutil.copy(src_file, dst_file)
            except Exception as e:
                print(f"Failed to copy {src_file}: {e}")
                continue

            recheck_list.append({
                'Parent_Filename': filename_png,
                'ROI_ID': roi_id,
                'Current_Value': val,
                'Reason': reason,
                'Validation_Image_Path': dst_file,
                'Prompt': get_qwen_prompt(roi_id),
                'Validated_Value': '' 
            })

    # Save Output 1: Recheck List
    if recheck_list:
        csv_path = os.path.join(DESTINATION_ROOT, OUTPUT_CSV_NAME)
        pd.DataFrame(recheck_list).to_csv(csv_path, index=False)
        print("------------------------------------------------")
        print(f"Found {len(recheck_list)} items failing validation.")
        print(f"Input CSV for Qwen saved at: {csv_path}")
    else:
        print("------------------------------------------------")
        print("All items valid! No re-check needed.")

    # Save Output 2: The Master File (with any non-truncation cleanups applied)
    df.to_csv(AUTOCORRECTED_CSV_NAME, index=False)
    if TRUNCATE_DECIMALS and autocorrect_count > 0:
        print(f"✅ Auto-Corrected {autocorrect_count} values (Long decimals truncated).")
    print(f"✅ Saved processed dataset to: {AUTOCORRECTED_CSV_NAME}")

if __name__ == "__main__":
    main()