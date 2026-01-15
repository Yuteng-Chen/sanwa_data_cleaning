import pandas as pd
import cv2
import json
import os
import shutil
from pathlib import Path

# ================= USER CONFIGURATION =================

# 1. Input: The Audit Report
AUDIT_REPORT_CSV = Path('/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/QA_Manual_Check_Folder_1252/_QA_Audit_Report.csv')

# 2. Source Images: Original full images
SOURCE_IMAGES_DIR = Path("/home/wanfangyuan/Desktop/share01/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/12_16_cslot/2025-12-16")

# 3. Output: Where to save the debug crops
#    (Will create folder: .../debug_crops/Image_Name/ROI_XX.jpg)
RECOVERED_CROPS_DIR = Path('/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/')

# 4. ROI JSON: Your file containing the 53 ROIs
ROI_JSON_PATH = Path("roi_cslot.json") 

# Crop Settings
ROI_PAD = 2
UPSCALE = 2.0

# ======================================================

def load_rois(json_path):
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found.")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    roi_map = {}
    items = data if isinstance(data, list) else [data]
    
    # Load all items from the JSON list
    for item in items:
        name = str(item.get('name', ''))
        roi_map[name] = [int(item['x']), int(item['y']), int(item['w']), int(item['h'])]
        
    print(f"✅ Loaded {len(roi_map)} ROIs from JSON (IDs: {list(roi_map.keys())[:5]} ... {list(roi_map.keys())[-1]})")
    return roi_map

def perform_crop(img, roi_coords, save_path):
    try:
        x, y, w, h = roi_coords
        H, W = img.shape[:2]

        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)

        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0: return False

        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), crop)
        return True

    except Exception as e:
        # print(f"    ⚠️ Crop failed: {e}")
        return False

def main():
    # 1. Setup
    if not AUDIT_REPORT_CSV.exists():
        print(f"❌ Audit Report not found: {AUDIT_REPORT_CSV}")
        return

    roi_map = load_rois(ROI_JSON_PATH)
    if not roi_map: return

    # 2. Read Audit Log
    print(f"📄 Reading Audit Report: {AUDIT_REPORT_CSV.name}")
    try:
        df = pd.read_csv(AUDIT_REPORT_CSV)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 3. Find Unique Missing Images
    if 'Image_Status' not in df.columns:
        print("❌ CSV missing 'Image_Status' column.")
        return

    # Filter for rows that failed to find an image
    missing_df = df[df['Image_Status'] == 'Image_Not_Found']
    
    if missing_df.empty:
        print("✅ No 'Image_Not_Found' items to recover.")
        return

    # Get list of unique images that need regeneration
    unique_images = missing_df['Image_Filename'].unique()

    print(f"🔍 Found {len(unique_images)} unique images marked as 'Image_Not_Found'.")
    print(f"🚀 Generating FULL SET (All {len(roi_map)} ROIs) for each image...")
    
    total_crops = 0

    # 4. Loop over every missing image
    for img_filename in unique_images:
        img_filename = str(img_filename)
        img_folder_base = Path(img_filename).stem # Remove .png
        
        # Locate Source Image
        src_img_path = SOURCE_IMAGES_DIR / img_filename
        if not src_img_path.exists():
            found = list(SOURCE_IMAGES_DIR.rglob(img_filename))
            if found:
                src_img_path = found[0]
            else:
                print(f"❌ Source Missing: {img_filename}")
                continue

        # Load Source Image
        img = cv2.imread(str(src_img_path))
        if img is None:
            print(f"❌ Failed to load: {img_filename}")
            continue

        print(f"  📸 Processing: {img_filename} ... ", end="")
        
        # === LOOP THROUGH ALL JSON ROIs (0 to 52) ===
        crops_done = 0
        for roi_key, coords in roi_map.items():
            # roi_key will be "0", "1", ... "52"
            
            # Save Path: .../debug_crops/ImageName/ROI_0.jpg
            target_name = f"ROI_{roi_key}.jpg"
            target_path = RECOVERED_CROPS_DIR / img_folder_base / target_name
            
            if perform_crop(img, coords, target_path):
                crops_done += 1
        
        print(f"Generated {crops_done} ROIs.")
        total_crops += crops_done

    print("\n========================================")
    print(f"🎉 Recovery Complete.")
    print(f"   Images Restored: {len(unique_images)}")
    print(f"   Total Crops Generated: {total_crops}")
    print(f"   Location: {RECOVERED_CROPS_DIR}")
    print("========================================")

if __name__ == "__main__":
    main()