import pandas as pd
import os
import re
import cv2
import numpy as np
import ollama
import glob
import sys
import shutil  # Added for fallback copying

# ================= USER CONFIGURATION =================
# 1. Base Folder containing your Result CSVs
INPUT_LOGS_DIR = 'Archive/Archive/Cut_preprocesseddata_Output'

# 2. Base Folder containing the Original Crop Subfolders
CROPS_DIR_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16/'

# 3. Output Folder for Final Fixed Logs
OUTPUT_DIR = 'Archive/Archive/Final_AI_Corrected_Logs/'

# 4. DEBUG FOLDER: Where Paddle-Optimized crops will be saved
DEBUG_PADDLE_OUTPUT_DIR = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Debug_Paddle_Optimized_Crops/'

# 5. Model Config
OLLAMA_MODEL = "qwen2.5vl:3b"

# ================= ROI CONFIGURATION =================
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

ROI_TYPE_MAP = {}
for cfg in ROI_CONFIGS:
    ROI_TYPE_MAP.update(cfg['Columns'])

# ================= INITIALIZATION =================
ocr_engine = None
try:
    from paddleocr import PaddleOCR
    # use_angle_cls=False is faster and better for digital screens
    ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    print("✅ PaddleOCR Initialized successfully.")
except Exception as e:
    print(f"⚠️  Warning: PaddleOCR failed to initialize: {e}. Running in 'Ollama Only' mode.")

# ================= 1. ENHANCED PROMPT GENERATOR =================
def get_prompt(roi_id, median_val=None):
    roi_id_str = str(roi_id).upper()
    roi_type = ROI_TYPE_MAP.get(roi_id_str, 'STATUS') 
    
    # Context Hint: Only suggest the median, do not force it.
    context = ""
    if median_val and isinstance(median_val, (int, float)) and median_val > 0:
        context = (
            f"CONTEXT: Similar sensors read around {median_val}. "
            "Use this ONLY to fix formatting errors (e.g. '188' -> '1.88'). "
            "If the image clearly shows '0' or is blank, IGNORE context and output '0'. "
        )

    if roi_type == 'FLOAT':
        return (
            f"Task: Extract the digital number.\n{context}\n"
            "RULES:\n"
            "1. Read the exact number in the image.\n"
            "2. FORMATTING ERROR: If you see '188' but context is '1.88', fix it to '1.88'.\n"
            "3. DEFECTS: If you see '0', '0.0', or blank, output '0'. Do NOT match the context.\n"
            "4. TRUNCATE: Max 3 decimals (e.g. '9.18181' -> '9.181').\n"
            "5. NO HTML/Markdown."
        )
    elif roi_type == 'INTEGER':
        return (
            f"Task: Extract the integer.\n{context}\n"
            "Rules: Output ONLY the integer value. If 0 or blank, output 0. NO HTML."
        )
    elif roi_type == 'STATUS':
        return "Task: Classify as 'OK' or 'NG'. (0/OK/OH -> OK, N/NG -> NG). Output ONLY one word."
    elif roi_type == 'TIME':
        return "Task: Read Timestamp (HH:MM:SS). Remove trailing text. NO HTML."
    
    return "Read the text. Output only value."

# ================= 2. PADDLE OCR REFINER (FIXED) =================
def refine_crop(image_path, debug_save_path=None):
    """
    1. Detects text area using Paddle.
    2. Crops to that area (removing black borders).
    3. Saves the crop to debug_save_path for manual verification.
    """
    # Create debug folder if needed
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)

    # 1. CHECK IF FILE EXISTS
    if not os.path.exists(image_path):
        return None

    # 2. CHECK IF PADDLE IS LOADED
    if ocr_engine is None:
        # Fallback: Just copy original to debug folder so we can see it
        if debug_save_path:
            shutil.copy(image_path, debug_save_path)
        return image_path

    try:
        img = cv2.imread(image_path)
        if img is None: 
            if debug_save_path: shutil.copy(image_path, debug_save_path)
            return image_path
        
        # Detect Text
        result = ocr_engine.ocr(img, cls=False, det=True, rec=False)
        
        # If no text found, use original image and save to debug
        if not result or result[0] is None: 
            if debug_save_path:
                cv2.imwrite(debug_save_path, img) 
            return image_path 

        # Calculate Bounding Box of all detected text
        boxes = [line[0] for line in result[0]]
        all_x = [pt[0] for box in boxes for pt in box]
        all_y = [pt[1] for box in boxes for pt in box]
        
        if not all_x or not all_y: 
            if debug_save_path: cv2.imwrite(debug_save_path, img)
            return image_path

        h, w, _ = img.shape
        # Add slight padding (2px)
        x_min = max(0, int(min(all_x)) - 2)
        y_min = max(0, int(min(all_y)) - 2)
        x_max = min(w, int(max(all_x)) + 2)
        y_max = min(h, int(max(all_y)) + 2)

        # Execute Crop
        cropped = img[y_min:y_max, x_min:x_max]
        
        # Save Debug Image (This is exactly what the LLM will see)
        if debug_save_path:
            cv2.imwrite(debug_save_path, cropped)

        # Save Temp for Qwen Input
        temp_path = image_path.replace(".jpg", "_paddle.jpg").replace(".png", "_paddle.png")
        cv2.imwrite(temp_path, cropped)
        return temp_path

    except Exception as e:
        print(f"  ⚠️ Paddle Error: {e}")
        # Fallback copy on error
        if debug_save_path and os.path.exists(image_path):
            shutil.copy(image_path, debug_save_path)
        return image_path

# ================= 3. QWEN INFERENCE & CLEANING =================
def run_qwen_inference(image_path, prompt):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
            options={'temperature': 0.0, 'num_predict': 20}
        )
        text = response['message']['content']
        
        # --- Python-Side Safety Cleaning ---
        text = re.sub(r'<[^>]+>', '', text).replace('```', '').replace('`', '').strip()
        text = re.sub(r'^(Output:|Result:)', '', text, flags=re.IGNORECASE).strip()
        
        # Hard Truncate Decimals (Double Safety)
        decimal_match = re.match(r'^(-?\d+\.\d{4,})', text)
        if decimal_match:
            parts = text.split('.')
            text = f"{parts[0]}.{parts[1][:3]}" # Force 3 decimals
            
        return text
    except Exception as e:
        print(f"  [Ollama Error] {e}")
        return "ERROR"

# ================= 4. MAIN PROCESSOR =================
def process_log_file(abnormal_csv_path, cleaned_csv_path):
    filename = os.path.basename(abnormal_csv_path)
    print(f"\nProcessing Log: {filename}")
    
    try:
        df_bad = pd.read_csv(abnormal_csv_path)
        if df_bad.empty: return
        
        # Load Medians from Cleaned file
        roi_medians = {}
        if os.path.exists(cleaned_csv_path):
            df_clean = pd.read_csv(cleaned_csv_path)
            for col in df_clean.columns:
                try:
                    vals = pd.to_numeric(df_clean[col], errors='coerce').dropna()
                    if not vals.empty: roi_medians[col] = vals.median()
                except: pass
    except Exception as e:
        print(f"  [File Error] {e}")
        return

    # Determine Folder Names
    csv_folder_name = filename.replace("_Abnormal_Log.csv", "")
    
    df_bad['AI_Corrected'] = ""
    
    for idx, row in df_bad.iterrows():
        roi_id = row['ROI_ID']
        
        # Construct Source Path
        parent_img_name = os.path.splitext(row['Filename'])[0]
        base_path = os.path.join(CROPS_DIR_BASE, csv_folder_name, parent_img_name, roi_id)
        
        img_path = None
        if os.path.exists(base_path + ".jpg"): img_path = base_path + ".jpg"
        elif os.path.exists(base_path + ".png"): img_path = base_path + ".png"
        
        # Fallback search
        if not img_path:
            alt_folder = csv_folder_name.replace("_cut", "")
            base_path_alt = os.path.join(CROPS_DIR_BASE, alt_folder, parent_img_name, roi_id)
            if os.path.exists(base_path_alt + ".jpg"): img_path = base_path_alt + ".jpg"
        
        if not img_path:
            df_bad.at[idx, 'AI_Corrected'] = "Image Not Found"
            continue

        # --- PREPARE DEBUG SAVE PATH ---
        debug_dest = os.path.join(DEBUG_PADDLE_OUTPUT_DIR, csv_folder_name, parent_img_name, f"{roi_id}.jpg")

        # --- EXECUTE PIPELINE ---
        curr_median = roi_medians.get(roi_id, None)
        
        # A. Refine (Paddle) -> Saves to Debug Folder
        refined_img = refine_crop(img_path, debug_save_path=debug_dest)
        
        if refined_img is None:
             df_bad.at[idx, 'AI_Corrected'] = "Image Read Error"
             continue

        # B. Prompt (Enhanced Logic)
        prompt = get_prompt(roi_id, curr_median)
        
        # C. Inference
        fixed_val = run_qwen_inference(refined_img, prompt)
        
        print(f"  [{idx+1}/{len(df_bad)}] {roi_id}: {row['Value']} -> {fixed_val} (Median: {curr_median})")
        
        # D. Update CSV
        df_bad.at[idx, 'AI_Corrected'] = fixed_val
        
        # E. Update Median (Weighted update)
        try:
            val_num = float(fixed_val)
            # Only update median if value > 0
            if val_num > 0:
                if curr_median: roi_medians[roi_id] = (curr_median * 0.9) + (val_num * 0.1)
                else: roi_medians[roi_id] = val_num
        except: pass

        # Cleanup Temp (Delete the _paddle.jpg used for Qwen, but keep the one in Debug Folder)
        if refined_img != img_path and "_paddle" in refined_img and os.path.exists(refined_img):
            os.remove(refined_img)

    # 4. Save
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    out_name = filename.replace(".csv", "_AI_Fixed.csv")
    df_bad.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
    print(f"✅ Saved: {out_name}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(DEBUG_PADDLE_OUTPUT_DIR): os.makedirs(DEBUG_PADDLE_OUTPUT_DIR)
    
    logs = glob.glob(os.path.join(INPUT_LOGS_DIR, "*_Abnormal_Log.csv"))
    
    if not logs:
        print(f"No logs found in {INPUT_LOGS_DIR}")
        return

    print(f"Found {len(logs)} logs.")
    for log in logs:
        cleaned = log.replace("_Abnormal_Log.csv", "_Cleaned.csv")
        process_log_file(log, cleaned)

if __name__ == "__main__":
    main()