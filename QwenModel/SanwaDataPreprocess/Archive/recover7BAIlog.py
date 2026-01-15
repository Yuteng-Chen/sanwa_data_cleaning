import pandas as pd
import cv2
import json
import os
import glob
from pathlib import Path

# ================= USER CONFIGURATION =================

# 1. Input: Folder containing the AI Verified Logs
INPUT_LOGS_DIR = Path('Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/AI_Verified_Logs/')

# 2. Source Images: Where the original full screenshots are stored
#    (It will look here to generate the fresh crop)
SOURCE_IMAGES_DIR = Path("/home/wanfangyuan/Desktop/share01/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/12_16_cslot/2025-12-16")

# 3. Output: The Debug Crops Base Folder
#    (Target: .../debug_crops/Image_Name/ROI_XX.jpg)
DEBUG_CROPS_BASE = Path('/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/')

# 4. ROI JSON: Your coordinate map
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
    
    # Map 'name': [x, y, w, h]
    # IMPORTANT: Ensure your JSON keys match the ROI_ID format (e.g. "12" or "ROI_12")
    for item in items:
        name = str(item.get('name', '')) # e.g., "12"
        roi_map[name] = [int(item['x']), int(item['y']), int(item['w']), int(item['h'])]
        
    print(f"✅ Loaded {len(roi_map)} ROIs from JSON.")
    return roi_map

def get_roi_coords(roi_id_str, roi_map):
    """
    Handles formats like 'ROI_12' -> looks up '12' in JSON map.
    """
    # 1. Try exact match
    if roi_id_str in roi_map:
        return roi_map[roi_id_str]
    
    # 2. Try stripping 'ROI_' prefix (e.g. "ROI_12" -> "12")
    if roi_id_str.upper().startswith("ROI_"):
        clean_id = roi_id_str.upper().replace("ROI_", "")
        if clean_id in roi_map:
            return roi_map[clean_id]
            
    return None

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
        return False

def process_log_file(csv_path, roi_map):
    try:
        df = pd.read_csv(csv_path)
    except:
        print(f"❌ Error reading {csv_path.name}")
        return 0

    if 'AI_7B_Read' not in df.columns:
        return 0
    
    # Filter: Only rows where Image was NOT Found
    missing_df = df[df['AI_7B_Read'] == 'Image Not Found']
    
    if missing_df.empty:
        return 0

    recovered_count = 0
    print(f"\n📂 Processing Log: {csv_path.name}")
    print(f"   found {len(missing_df)} missing items.")

    for idx, row in missing_df.iterrows():
        roi_id = str(row['ROI_ID'])
        filename = str(row['Filename_Current'])
        
        # 1. Locate Source Image
        src_path = SOURCE_IMAGES_DIR / filename
        if not src_path.exists():
            # Try recursive search if not flat
            found = list(SOURCE_IMAGES_DIR.rglob(filename))
            if found:
                src_path = found[0]
            else:
                # print(f"    ❌ Source Img Missing: {filename}")
                continue

        # 2. Load Image
        img = cv2.imread(str(src_path))
        if img is None: continue

        # 3. Get Coords
        coords = get_roi_coords(roi_id, roi_map)
        if not coords:
            print(f"    ⚠️ ROI Key '{roi_id}' not in JSON.")
            continue

        # 4. Define Target Path
        # Target: DEBUG_CROPS_BASE / {Image_Name_No_Ext} / {ROI_ID}.jpg
        img_folder_name = Path(filename).stem # "2025-12-16..."
        
        # Ensure ROI ID format for filename (e.g. ensure it is "ROI_12.jpg")
        if not roi_id.upper().startswith("ROI_"):
            save_filename = f"ROI_{roi_id}.jpg"
        else:
            save_filename = f"{roi_id}.jpg"
            
        target_path = DEBUG_CROPS_BASE / img_folder_name / save_filename
        
        # 5. Crop & Save
        if perform_crop(img, coords, target_path):
            recovered_count += 1
            # print(f"    ✅ Recovered: {target_path.name}")

    print(f"   -> Recovered {recovered_count} crops.")
    return recovered_count

def main():
    if not INPUT_LOGS_DIR.exists():
        print(f"❌ Logs dir not found: {INPUT_LOGS_DIR}")
        return

    # 1. Load ROI Coordinates
    roi_map = load_rois(ROI_JSON_PATH)
    if not roi_map: return

    # 2. Find Logs
    log_files = list(INPUT_LOGS_DIR.glob("*_AI_Verified.csv"))
    if not log_files:
        print(f"No logs found in {INPUT_LOGS_DIR}")
        return

    print(f"🔍 Found {len(log_files)} logs. Scanning for 'Image Not Found'...")
    
    total_recovered = 0
    for log_file in log_files:
        total_recovered += process_log_file(log_file, roi_map)

    print("\n========================================")
    print(f"🎉 Recovery Batch Complete.")
    print(f"   Total Crops Regenerated: {total_recovered}")
    print(f"   Check Folder: {DEBUG_CROPS_BASE}")
    print("========================================")

if __name__ == "__main__":
    main()