import pandas as pd
import os
import shutil
import numpy as np
import glob

# ================= USER CONFIGURATION =================
# 1. Input: Your Final Cleaned Dataset
INPUT_DATASET_DIR = 'Archive/Archive/Final_Cleaned_Dataset'

# 2. Source of Crops: Central folder where ALL Image folders are located
#    Structure expected: EXISTING_CROPS_DIR / Image_Name / ROI_ID.jpg
EXISTING_CROPS_DIR = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# 3. Output: Where to save the Audit Log and the "Bad" images
OUTPUT_AUDIT_DIR = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/QA_Manual_Check_Folder_1411/'

# 4. Threshold: 1.0 = 100% deviation allowed
DEVIATION_THRESHOLD = 0.5 

# ================= ROI CONFIGURATION =================
ROI_CONFIGS = [
    {
        'Trigger_Col': 'ROI_12',
        'Columns': {'ROI_13': 'INTEGER', 'ROI_16': 'FLOAT', 'ROI_18': 'FLOAT'}
    },
    {
        'Trigger_Col': 'ROI_20',
        'Columns': {'ROI_21': 'INTEGER', 'ROI_23': 'FLOAT'}
    },
    {
        'Trigger_Col': 'ROI_1',
        'Columns': {'ROI_2': 'INTEGER', 'ROI_4': 'FLOAT', 'ROI_6': 'FLOAT', 'ROI_8': 'FLOAT'}
    },
    {
        'Trigger_Col': 'ROI_31',
        'Columns': {
            'ROI_32': 'INTEGER', 'ROI_35': 'INTEGER', 'ROI_37': 'INTEGER', 'ROI_39': 'INTEGER', 
            'ROI_41': 'INTEGER', 'ROI_43': 'INTEGER', 'ROI_45': 'INTEGER', 'ROI_47': 'INTEGER', 'ROI_49': 'INTEGER'
        }
    }
]

# Flatten config
ROI_TYPES = {}
for cfg in ROI_CONFIGS:
    ROI_TYPES.update(cfg['Columns'])

# ================= MAIN LOGIC =================

def find_crop_image(filename, roi_id):
    """
    Attempts to locate the crop image in the central crops folder.
    New Logic: Ignores CSV name. Looks directly for Image_Folder / ROI.jpg
    """
    parent_name = os.path.splitext(filename)[0] # Remove .jpg
    
    # Path: Source / ImageName / ROI.jpg
    path = os.path.join(EXISTING_CROPS_DIR, parent_name, f"{roi_id}.jpg")
    
    if os.path.exists(path):
        return path
    
    return None

def audit_files():
    # Create Main Output Folder
    if not os.path.exists(OUTPUT_AUDIT_DIR):
        os.makedirs(OUTPUT_AUDIT_DIR)
        
    # We will create subfolders inside this dynamically
    base_review_dir = os.path.join(OUTPUT_AUDIT_DIR, "Review_Images_Structured")
    if not os.path.exists(base_review_dir):
        os.makedirs(base_review_dir)

    csv_files = glob.glob(os.path.join(INPUT_DATASET_DIR, "*_Cleaned.csv"))
    
    if not csv_files:
        print(f"❌ No cleaned CSVs found in {INPUT_DATASET_DIR}")
        return

    print(f"🔍 Auditing {len(csv_files)} files...")
    print(f"⚙️  Settings: Threshold > {DEVIATION_THRESHOLD*100}%, Ignoring 0s.")
    
    audit_log = []

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        # CSV Name Base (e.g., "Batch1") used for output folder structure
        csv_name_base = filename.replace(".csv", "") 
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ❌ Error reading {filename}: {e}")
            continue

        print(f"  📄 Scanning: {filename}")
        
        # Calculate Medians for this file
        medians = {}
        for col in ROI_TYPES.keys():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                valid_vals = series[series > 0] 
                medians[col] = valid_vals.median() if not valid_vals.empty else 0

        # Check Rows
        for idx, row in df.iterrows():
            img_filename = row.get('Filename', f"Row_{idx}")
            parent_img_folder = os.path.splitext(img_filename)[0]
            
            for col in ROI_TYPES.keys():
                if col not in df.columns: continue
                
                try:
                    val = float(row[col])

                    # Logic: Skip 0s, Skip if Median is 0
                    if val == 0: continue
                    median = medians.get(col, 0)
                    if median == 0: continue 

                    # Logic: Check Threshold
                    diff = abs(val - median)
                    percent_diff = diff / median
                    
                    if percent_diff > DEVIATION_THRESHOLD:
                        
                        # === NEW COPY LOGIC ===
                        roi_id = col 
                        found_path = find_crop_image(img_filename, roi_id)
                        
                        saved_status = "Image_Not_Found"
                        
                        if found_path:
                            # Create Structured Path:
                            # Output / CSV_Name / Image_Name / ROI.jpg
                            dest_folder = os.path.join(base_review_dir, csv_name_base, parent_img_folder)
                            
                            if not os.path.exists(dest_folder):
                                os.makedirs(dest_folder)
                                
                            dest_path = os.path.join(dest_folder, f"{roi_id}.jpg")
                            shutil.copy(found_path, dest_path)
                            
                            saved_status = "Saved"
                        
                        # Log Issue
                        audit_log.append({
                            'CSV_File': filename,
                            'Row_Index': idx,
                            'Image_Filename': img_filename,
                            'ROI_ID': col,
                            'Value': val,
                            'Median': median,
                            'Diff_Percent': round(percent_diff * 100, 1),
                            'Image_Status': saved_status
                        })

                except (ValueError, TypeError):
                    continue 

    # Save Audit Report
    if audit_log:
        df_log = pd.DataFrame(audit_log)
        report_path = os.path.join(OUTPUT_AUDIT_DIR, "_QA_Audit_Report.csv")
        df_log.to_csv(report_path, index=False)
        
        print("\n==========================================")
        print(f"🚩 Issues Found: {len(audit_log)}")
        print(f"📝 Report Saved: {report_path}")
        print(f"📂 Images Saved in: {base_review_dir}")
        print("   (Structure: CSV_Name -> Image_Name -> ROI.jpg)")
        print("==========================================")
    else:
        print(f"\n✅ QA Passed! No values > {DEVIATION_THRESHOLD*100}% deviation found.")

if __name__ == "__main__":
    audit_files()