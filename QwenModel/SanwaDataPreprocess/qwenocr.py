import json
import time
import shutil
import cv2
import numpy as np
import sys
import ollama
from pathlib import Path

# ================= Configuration =================
INPUT_FOLDER = Path("Input_Images")
OUTPUT_FOLDER = Path("Output_Results")
PROCESSED_FOLDER = Path("Processed_History")
ERROR_FOLDER = Path("Error_Images")

ROI_JSON = Path("roi.json")
RESULT_FILENAME = "all_results.txt" 

# [FIXED] Removed the hyphen to match your installed model
OLLAMA_MODEL = "qwen2.5vl:3b"

# [CROP SETTINGS]
ROI_PAD = 0         
UPSCALE = 3.0       
# =================================================

def load_rois(roi_path: Path):
    if not roi_path.exists():
        return []
    try:
        with open(roi_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rois = []
        data_list = data if isinstance(data, list) else [data]
        for idx, item in enumerate(data_list):
            name = item.get("name", str(idx))
            rois.append((str(name), int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])))
        return rois
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return []

def read_image(path: Path):
    try:
        stream = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def prepare_image_for_llm(img, save_path):
    if img is None: return False
    try:
        img_large = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        is_success, buffer = cv2.imencode(".png", img_large)
        if is_success:
            buffer.tofile(str(save_path))
            return True
    except Exception as e:
        print(f"Failed to save image: {e}")
    return False

def run_ollama_inference(image_path, model_name):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': 'Output ONLY the text visible in this image. Do not include markdown or explanations.',
                'images': [str(image_path)]
            }]
        )
        text = response['message']['content'].strip()
        text = text.replace("**", "").replace("`", "") 
        return text
    except Exception as e:
        print(f"  ❌ Ollama Error: {e}")
        return "ERROR"

def process_single_image(img_path: Path, rois, result_file_path: Path, root_out_dir: Path):
    print(f"Processing: {img_path.name} ...")
    
    img_origin = read_image(img_path)
    if img_origin is None:
        raise ValueError(f"Could not decode image: {img_path}")

    H, W = img_origin.shape[:2]
    dir_crops = root_out_dir / "ROI_Crops" / img_path.stem
    extracted_data = []

    for name, x, y, w, h in rois:
        if x >= W or y >= H: continue
        
        x0 = max(0, x - ROI_PAD)
        y0 = max(0, y - ROI_PAD)
        x1 = min(W, x + w + ROI_PAD)
        y1 = min(H, y + h + ROI_PAD)
        
        crop = img_origin[y0:y1, x0:x1]
        if crop.size == 0: continue

        crop_filename = dir_crops / f"{name}.png"
        if not prepare_image_for_llm(crop, crop_filename): continue

        ocr_text = run_ollama_inference(crop_filename, OLLAMA_MODEL)
        print(f"    [{name}]: {ocr_text}")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        extracted_data.append(f"{timestamp}\t{img_path.name}\t{name}\t{ocr_text}\n")

    with open(result_file_path, "a", encoding="utf-8-sig") as f_out:
        for line in extracted_data:
            f_out.write(line)

def main():
    for p in [INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
        p.mkdir(parents=True, exist_ok=True)
    
    (OUTPUT_FOLDER / "ROI_Crops").mkdir(exist_ok=True)
    result_txt_path = OUTPUT_FOLDER / RESULT_FILENAME

    print("--------------------------------")
    print(f"Initializing connection to Ollama ({OLLAMA_MODEL})...")
    try:
        ollama.list()
        print("✅ Connected to Ollama.")
    except Exception as e:
        print("❌ CRITICAL: Ollama app is not running.")
        return

    rois = load_rois(ROI_JSON)
    if not rois: return

    print(f"Monitoring: {INPUT_FOLDER.absolute()}")
    print("--------------------------------")

    while True:
        try:
            files = list(INPUT_FOLDER.glob("*"))
            image_files = [f for f in files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and not f.name.startswith(".")]
            
            if not image_files:
                time.sleep(1)
                continue

            for img_f in image_files:
                time.sleep(0.5) 
                try:
                    process_single_image(img_f, rois, result_txt_path, OUTPUT_FOLDER)
                    ts_name = f"{img_f.stem}_{int(time.time())}{img_f.suffix}"
                    shutil.move(str(img_f), str(PROCESSED_FOLDER / ts_name))
                    print(f"✅ Done.\n")
                except Exception as e:
                    print(f"❌ Error processing {img_f.name}: {e}")
                    try: shutil.move(str(img_f), str(ERROR_FOLDER / img_f.name))
                    except: pass

        except KeyboardInterrupt: break
        except Exception as e: time.sleep(3)

if __name__ == "__main__":
    main()