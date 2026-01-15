import pandas as pd
import os
import re
import ollama
import glob
import sys

# ================= USER CONFIGURATION =================
# 1. Base Folder containing your Result CSVs
INPUT_LOGS_DIR = 'Archive/Archive/Cut_preprocesseddata_Output_1458'

# 2. Base Folder containing the Crop Subfolders
#    (Structure: .../CSV_NAME/PARENT_IMAGE/ROI.jpg)
CROPS_DIR_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Abnormal_Crops_Recheck12_16/'

# 3. Output Folder for Final Fixed Logs
OUTPUT_DIR = 'Archive/Archive/Final_AI_Corrected_Logs_1511/'

# 4. Model Config
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

# Flatten configs for easy lookup
ROI_TYPE_MAP = {}
for cfg in ROI_CONFIGS:
    ROI_TYPE_MAP.update(cfg['Columns'])

# ================= 1. PROMPT GENERATOR =================
def get_prompt(roi_id, median_val=None):
    roi_id_str = str(roi_id).upper()
    roi_type = ROI_TYPE_MAP.get(roi_id_str, 'STATUS') 
    
    # Context Hint logic:
    # We provide the median as a hint to fix typos, but explicitly tell LLM 
    # to trust its eyes if it sees a "0" (defect).
    context = ""
    if median_val and isinstance(median_val, (int, float)) and median_val > 0:
        context = (
            f"CONTEXT: Similar sensors usually read around {median_val}. "
            "Use this context ONLY to fix obvious formatting errors (e.g. '188' -> '1.88'). "
            "However, if the image clearly shows '0' or is blank, IGNORE context and output '0'. "
        )

    if roi_type == 'FLOAT':
        return (
            f"Task: Extract the digital number from the image.\n{context}\n"
            "STRICT RULES:\n"
            "1. Output ONLY the number.\n"
            "2. FORMATTING: If the number is large (e.g. 188) but context is small (e.g. 1.88), add the decimal.\n"
            "3. DEFECTS: If the value is '0', '0.0', or blank, output '0'.\n"
            "4. TRUNCATE: Max 3 decimal places (e.g. '9.18181' -> '9.181').\n"
            "5. NO HTML, NO MARKDOWN."
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

# ================= 2. QWEN INFERENCE =================
def run_qwen_inference(image_path, prompt):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
            options={'temperature': 0.0, 'num_predict': 20}
        )
        text = response['message']['content']
        
        # --- Python-Side Safety Cleaning ---
        # Remove Markdown artifacts
        text = re.sub(r'<[^>]+>', '', text).replace('```', '').replace('`', '').strip()
        text = re.sub(r'^(Output:|Result:)', '', text, flags=re.IGNORECASE).strip()
        
        # Hard Truncate Decimals (Double Safety)
        # Keeps only 3 decimal places
        decimal_match = re.match(r'^(-?\d+\.\d{4,})', text)
        if decimal_match:
            parts = text.split('.')
            text = f"{parts[0]}.{parts[1][:3]}" 
            
        return text
    except Exception as e:
        print(f"  [Ollama Error] {e}")
        return "ERROR"

# ================= 3. MAIN PROCESSOR =================
def process_log_file(abnormal_csv_path, cleaned_csv_path):
    filename = os.path.basename(abnormal_csv_path)
    print(f"\nProcessing Log: {filename}")
    
    # 1. Load Data
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

    # 2. Determine Crop Folder Name
    # e.g., "cam 6 snap1_Abnormal_Log.csv" -> "cam 6 snap1"
    csv_folder_name = filename.replace("_Abnormal_Log.csv", "")
    
    # 3. Iterate Abnormal Rows
    df_bad['AI_Corrected'] = ""
    
    for idx, row in df_bad.iterrows():
        roi_id = row['ROI_ID']
        
        # Construct Path: CROPS_DIR_BASE / CSV_FOLDER / PARENT_IMG / ROI.jpg
        parent_img_name = os.path.splitext(row['Filename'])[0]
        base_path = os.path.join(CROPS_DIR_BASE, csv_folder_name, parent_img_name, roi_id)
        
        img_path = None
        if os.path.exists(base_path + ".jpg"): img_path = base_path + ".jpg"
        elif os.path.exists(base_path + ".png"): img_path = base_path + ".png"
        
        # Fallback Logic: Try searching without "_cut" suffix if main folder fails
        if not img_path:
            alt_folder = csv_folder_name.replace("_cut", "")
            base_path_alt = os.path.join(CROPS_DIR_BASE, alt_folder, parent_img_name, roi_id)
            if os.path.exists(base_path_alt + ".jpg"): img_path = base_path_alt + ".jpg"
        
        if not img_path:
            df_bad.at[idx, 'AI_Corrected'] = "Image Not Found"
            continue

        # --- EXECUTE PIPELINE ---
        curr_median = roi_medians.get(roi_id, None)
        
        # A. Prompt
        prompt = get_prompt(roi_id, curr_median)
        
        # B. Inference (Directly on the crop image)
        fixed_val = run_qwen_inference(img_path, prompt)
        
        print(f"  [{idx+1}/{len(df_bad)}] {roi_id}: {row['Value']} -> {fixed_val} (Median: {curr_median})")
        
        # C. Update DataFrame
        df_bad.at[idx, 'AI_Corrected'] = fixed_val
        
        # D. Weighted Median Update
        # Only update median if the new value is > 0 (Normal). 
        # We don't want "0" (Defects) to drag down the median calculation.
        try:
            val_num = float(fixed_val)
            if val_num > 0:
                if curr_median: roi_medians[roi_id] = (curr_median * 0.9) + (val_num * 0.1)
                else: roi_medians[roi_id] = val_num
        except: pass

    # 4. Save Final CSV
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    out_name = filename.replace(".csv", "_AI_Fixed.csv")
    df_bad.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
    print(f"✅ Saved: {out_name}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Process all Abnormal Logs found in INPUT_LOGS_DIR
    logs = glob.glob(os.path.join(INPUT_LOGS_DIR, "*_Abnormal_Log.csv"))
    
    if not logs:
        print(f"No logs found in {INPUT_LOGS_DIR}")
        return

    print(f"Found {len(logs)} logs.")
    for log in logs:
        # Find matching Cleaned CSV to get Medians
        cleaned = log.replace("_Abnormal_Log.csv", "_Cleaned.csv")
        process_log_file(log, cleaned)

if __name__ == "__main__":
    main()