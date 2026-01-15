import pandas as pd
import cv2
import json
import os
import shutil
from pathlib import Path

# ================= USER CONFIGURATION =================

# 1. Source Directory: Where the ORIGINAL full images are located
#    (The script needs to read these to generate the crops)
SOURCE_IMAGES_DIR = Path("/home/wanfangyuan/Desktop/share01/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/12_16_cslot/2025-12-16")

# 2. Input Logs Directory: Where your *_AI_Fixed.csv files are located
INPUT_LOGS_DIR = Path("Archive/Archive/Final_AI_Corrected_Logs/")

# 3. Output Directory: Where to save the recovered crops
#    (This should match the 'CROPS_DIR_BASE' from your correction script)
RECOVERED_CROPS_DIR = Path("/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16/")

# 4. ROI JSON: To know the coordinates for cropping
ROI_JSON_PATH = Path("roi_cslot.json") 

# Crop Settings (Must match your original generation settings)
ROI_PAD = 2
UPSCALE = 2.0

# ======================================================

def load_rois(json_path):
    """Loads ROI coordinates into a dictionary map."""
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found.")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    roi_map = {}
    items = data if isinstance(data, list) else [data]
    
    for item in items:
        # Key: "13", Value: [x, y, w, h]
        name = str(item.get('name', ''))
        roi_map[name] = [int(item['x']), int(item['y']), int(item['w']), int(item['h'])]
        
    return roi_map

def perform_crop(img_path, roi_coords, save_path):
    """Reads source image, crops ROI, and saves it."""
    if not img_path.exists():
        print(f"  ❌ Source image missing: {img_path.name}")
        return False

    try:
        img = cv2.imread(str(img_path))
        if img is None: return False

        x, y, w, h = roi_coords
        H, W = img.shape[:2]

        # Padding
        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)

        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0: return False

        # Upscale
        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), crop)
        return True

    except Exception as e:
        print(f"  ⚠️ Crop failed: {e}")
        return False

def main():
    # 1. Setup
    if not INPUT_LOGS_DIR.exists():
        print(f"❌ Log directory not found: {INPUT_LOGS_DIR}")
        return

    roi_map = load_rois(ROI_JSON_PATH)
    if not roi_map: return

    # 2. Find all AI Fixed CSVs
    csv_files = list(INPUT_LOGS_DIR.glob("*_AI_Fixed.csv"))
    if not csv_files:
        print("No *_AI_Fixed.csv files found.")
        return

    print(f"🔍 Scanning {len(csv_files)} log files for missing images...")
    
    total_recovered = 0
    total_errors = 0

    for csv_file in csv_files:
        print(f"\n📄 Reading: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Determine the subfolder name required by the correction script
            # Logic: Remove "_Abnormal_Log_AI_Fixed.csv" to get the base name
            # Example: "cam 6 snap1..._Abnormal_Log_AI_Fixed.csv" -> "cam 6 snap1..."
            csv_folder_name = csv_file.name.replace("_Abnormal_Log_AI_Fixed.csv", "").replace("_AI_Fixed.csv", "")
            
            # Remove any trailing _Abnormal_Log if the replace above didn't catch it perfectly
            if "_Abnormal_Log" in csv_folder_name:
                csv_folder_name = csv_folder_name.replace("_Abnormal_Log", "")

            # Filter for rows where Image was not found
            # Check both possible column names depending on script version
            col_name = 'AI_Corrected' if 'AI_Corrected' in df.columns else 'AI_Corrected_Value'
            
            if col_name not in df.columns:
                print("  ⚠️ Column 'AI_Corrected' not found, skipping.")
                continue

            # Filter logic: Look for "Image Not Found" or empty values that should be there
            missing_rows = df[
                (df[col_name].astype(str).str.contains("Image Not Found", case=False, na=False)) |
                (df[col_name].astype(str).str.contains("Missing Image", case=False, na=False))
            ]

            if missing_rows.empty:
                print("  ✅ No missing images in this file.")
                continue

            print(f"  ⚠️ Found {len(missing_rows)} missing items. Regenerating...")

            for idx, row in missing_rows.iterrows():
                filename = row['Filename']
                roi_id_str = str(row['ROI_ID']) # e.g. "ROI_16" or "16"
                
                # Clean ROI ID to match JSON key (remove "ROI_")
                roi_key = roi_id_str.replace("ROI_", "")
                
                if roi_key not in roi_map:
                    print(f"    Skipping unknown ROI ID: {roi_id_str}")
                    continue

                # Paths
                src_img_path = SOURCE_IMAGES_DIR / filename
                
                # Target Structure: RECOVERED_CROPS_DIR / CSV_FOLDER / PARENT_IMAGE_STEM / ROI_ID.jpg
                parent_stem = Path(filename).stem
                target_path = RECOVERED_CROPS_DIR / csv_folder_name / parent_stem / f"{roi_id_str}.jpg"

                # Execute Crop
                if perform_crop(src_img_path, roi_map[roi_key], target_path):
                    print(f"    ✅ Restored: {target_path.name}")
                    total_recovered += 1
                else:
                    total_errors += 1

        except Exception as e:
            print(f"  ❌ Error processing CSV: {e}")

    print("\n========================================")
    print(f"🎉 Recovery Run Complete.")
    print(f"   Total Images Restored: {total_recovered}")
    print(f"   Errors (Source Missing): {total_errors}")
    print("========================================")

if __name__ == "__main__":
    main()