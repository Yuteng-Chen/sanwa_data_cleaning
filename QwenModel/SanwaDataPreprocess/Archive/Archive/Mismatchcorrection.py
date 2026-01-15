import pandas as pd
import os
import re
import ollama
import glob
import sys

# ================= USER CONFIGURATION =================
# 1. Folder containing the "_Redundancy_Mismatch_Log.csv" files
INPUT_LOGS_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/'

# 2. PRIMARY PATH: Where the previous script *should* have copied images
REDUNDANCY_IMGS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/Cleaned_Results_Output12_16/Redundancy_Mismatch_Recheck0640/'

# 3. FALLBACK PATH: The original debug crop location (Flattened or Nested)
DEBUG_CROPS_BASE = '/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops/'

# 4. Output for the AI verified logs
OUTPUT_DIR = 'Archive/Archive/Final_Cleaned_Dataset_1606_Output_0640/AI_Verified_Logs/'

# 5. Model Config
OLLAMA_MODEL = "qwen2.5vl:7b"

# ================= HELPER: PROMPT GENERATOR =================
def get_prompt(roi_id, current_val, compared_val):
    roi_id_str = str(roi_id).upper()
    
    # Simple Heuristic for Status vs Value
    is_status = "STATUS" in roi_id_str or str(current_val).upper() in ['OK', 'NG']
    
    if is_status:
        return (
            f"Task: Read the text in this image strictly.\n"
            f"Options: Usually 'OK' or 'NG'.\n"
            f"Context: The previous row was '{compared_val}', but OCR read '{current_val}'.\n"
            f"Command: Output ONLY the text visible in the image. No markdown."
        )
    else:
        return (
            f"Task: Extract the number from this image with high precision.\n"
            f"Context: There is a dispute. Previous row was '{compared_val}'. Current OCR says '{current_val}'.\n"
            f"Instructions:\n"
            f"1. Look closely for decimal points (e.g., 1.88 vs 188).\n"
            f"2. Look closely for negative signs.\n"
            f"3. Output ONLY the number found in the image.\n"
            f"4. If empty or black, output '0'."
        )

# ================= HELPER: INFERENCE =================
def run_qwen_inference(image_path, prompt):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
            options={'temperature': 0.1}
        )
        text = response['message']['content']
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('```', '').replace('`', '').strip()
        text = text.split('\n')[0] 
        return text.strip()
    except Exception as e:
        print(f"  [Ollama Error] {e}")
        return "ERROR"

# ================= MAIN PROCESSOR =================
def process_mismatch_log(csv_path):
    filename = os.path.basename(csv_path)
    # Extract base name to help finding nested folders if needed (e.g. "Cam 6")
    # File format: "{BaseName}_Redundancy_Mismatch_Log.csv"
    csv_base_name = filename.replace("_Redundancy_Mismatch_Log.csv", "")
    
    print(f"\nProcessing Mismatch Log: {filename}")
    
    try: df = pd.read_csv(csv_path)
    except: print("  Could not read CSV."); return

    if df.empty: print("  Log is empty. Skipping."); return

    df['AI_7B_Read'] = ""
    df['Verdict'] = "" 
    df['Image_Source'] = "" # To track where we found the image

    for idx, row in df.iterrows():
        roi_id = str(row['ROI_ID'])
        current_filename = str(row['Filename_Current']) # e.g., "2025-12-16 17.20.09.png"
        
        # 1. Get Folder Name from Image Filename (Strip extension)
        folder_name = os.path.splitext(current_filename)[0] # "2025-12-16 17.20.09"
        
        # --- ROBUST PATH SEARCH ---
        potential_paths = [
            # 1. Primary: Specific Redundancy Folder (Nested by base name)
            os.path.join(REDUNDANCY_IMGS_BASE, csv_base_name, folder_name, f"{roi_id}.jpg"),
            os.path.join(REDUNDANCY_IMGS_BASE, csv_base_name, folder_name, f"{roi_id}.png"),
            
            # 2. Primary: Specific Redundancy Folder (Directly inside)
            os.path.join(REDUNDANCY_IMGS_BASE, folder_name, f"{roi_id}.jpg"),
            
            # 3. Fallback: Debug Base (Flattened)
            os.path.join(DEBUG_CROPS_BASE, folder_name, f"{roi_id}.jpg"),
            os.path.join(DEBUG_CROPS_BASE, folder_name, f"{roi_id}.png"),

            # 4. Fallback: Debug Base (Nested under CSV name)
            os.path.join(DEBUG_CROPS_BASE, csv_base_name, folder_name, f"{roi_id}.jpg")
        ]
        
        final_img_path = None
        for p in potential_paths:
            if os.path.exists(p):
                final_img_path = p
                break
            
        if not final_img_path:
            # print(f"  [Missing] Could not find {roi_id} for {folder_name}")
            df.at[idx, 'AI_7B_Read'] = "Image Not Found"
            continue

        # 2. Run Inference
        val_curr = row['Value_Current']
        val_prev = row['Value_Compared']
        prompt = get_prompt(roi_id, val_curr, val_prev)
        
        ai_result = run_qwen_inference(final_img_path, prompt)
        
        print(f"  [{idx+1}/{len(df)}] {roi_id} | OCR: {val_curr} | Prev: {val_prev} | AI: {ai_result}")
        
        # 3. Determine Verdict
        df.at[idx, 'AI_7B_Read'] = ai_result
        df.at[idx, 'Image_Source'] = "Fallback" if DEBUG_CROPS_BASE in final_img_path else "Primary"
        
        ai_clean = str(ai_result).strip().lower()
        prev_clean = str(val_prev).strip().lower()
        curr_clean = str(val_curr).strip().lower()
        
        if ai_clean == prev_clean:
            df.at[idx, 'Verdict'] = "Confirmed Redundant (OCR Error)"
        elif ai_clean == curr_clean:
            df.at[idx, 'Verdict'] = "Genuine Change (OCR Correct)"
        else:
            df.at[idx, 'Verdict'] = "New Value (AI Disagrees with both)"

    # 4. Save Output
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    out_name = filename.replace(".csv", "_AI_Verified.csv")
    df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
    print(f"✅ Saved Verified Log: {out_name}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    logs = glob.glob(os.path.join(INPUT_LOGS_DIR, "*_Redundancy_Mismatch_Log.csv"))
    
    if not logs:
        print(f"No mismatch logs found in {INPUT_LOGS_DIR}")
        return
        
    print(f"Found {len(logs)} mismatch logs.")
    for log in logs:
        process_mismatch_log(log)

if __name__ == "__main__":
    main()