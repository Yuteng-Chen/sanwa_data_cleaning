import json
import time
import shutil
import cv2
import numpy as np
import sys
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

# ================= Configuration =================
INPUT_FOLDER = Path("Input_Images")
OUTPUT_FOLDER = Path("Output_Results")
PROCESSED_FOLDER = Path("Processed_History")
ERROR_FOLDER = Path("Error_Images")

ROI_JSON = Path("roi.json")
RESULT_FILENAME = "all_results.txt" 

# [IMPORTANT] Increased padding to 20. 
# We capture a larger area so the text is never cut off.
# The AI detection will find the text inside this box.
ROI_PAD = 20         
UPSCALE = 3.0       # Higher upscale (3x) for blurry screenshots
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
    """ Cross-platform image reader (handles Unicode paths) """
    try:
        stream = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def save_image(img, path: Path):
    """ Cross-platform image saver """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            buffer.tofile(str(path))
    except Exception as e:
        print(f"Failed to save image {path}: {e}")

def preprocess_image_adaptive(img):
    """
    Advanced Enhancement Logic:
    1. Upscale
    2. Auto-Invert (Detects Black vs Light BG)
    3. CLAHE (Fixes 'NG' Red-on-Gray contrast)
    4. Noise Removal (Removes thin lines)
    """
    if img is None: return None
    
    # 1. Upscale
    h, w = img.shape[:2]
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Auto-Invert Check
    # Check border pixels. If dark (<127), invert to make text dark on light.
    border_mean = np.mean(np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ]))
    
    if border_mean < 127:
        gray = cv2.bitwise_not(gray) # Invert Black BG -> White
    
    # 4. CLAHE (Adaptive Contrast) - Crucial for "Red text on Gray BG"
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 5. Otsu Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 6. Noise Cleaning (Optional)
    # This helps remove thin lines or noise dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Invert so text is white (standard for morphological ops)
    binary_inv = cv2.bitwise_not(binary)
    # 'Open' removes small white noise
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    # Invert back to standard Black Text / White BG
    final = cv2.bitwise_not(cleaned)

    return final

def extract_text_safely(ocr_result):
    """
    [CRITICAL FIX] 
    Handles the nested list format from RapidOCR when use_det=True.
    Prevents the 'sequence item 0: expected str' crash.
    """
    if not ocr_result: return ""
    
    found_texts = []
    for item in ocr_result:
        # Format usually: [[[x,y..], "TEXT", Score]]
        if isinstance(item, (list, tuple)):
            if len(item) == 0: continue
            
            # If use_det=True, the text is usually at index 1
            if len(item) >= 2 and isinstance(item[1], str):
                found_texts.append(item[1])
            # If use_det=False, text might be at index 0
            elif isinstance(item[0], str):
                found_texts.append(item[0])
            # Handle nested list case
            elif isinstance(item[0], (list, tuple)) and len(item) > 1:
                 found_texts.append(str(item[1]))

    return " ".join(found_texts).strip()

def process_single_image(img_path: Path, engine, rois, result_file_path: Path, root_out_dir: Path):
    print(f"Processing: {img_path.name} ...")
    
    img_origin = read_image(img_path)
    if img_origin is None:
        raise ValueError(f"Could not decode image: {img_path}")

    H, W = img_origin.shape[:2]
    
    # Define debug folders
    dir_original = root_out_dir / "ROI_Original" / img_path.stem
    dir_processed = root_out_dir / "ROI_Processed" / img_path.stem

    extracted_data = []

    for name, x, y, w, h in rois:
        if x >= W or y >= H:
            print(f"  ⚠️ Skip [{name}]: ROI out of bounds")
            continue
        
        # --- LOOSE CROP (Add Padding) ---
        x0 = max(0, x - ROI_PAD)
        y0 = max(0, y - ROI_PAD)
        x1 = min(W, x + w + ROI_PAD)
        y1 = min(H, y + h + ROI_PAD)
        
        crop = img_origin[y0:y1, x0:x1]
        if crop.size == 0: continue

        # 1. Save Raw Original
        save_image(crop, dir_original / f"{name}.png")

        # 2. Preprocess
        processed_crop = preprocess_image_adaptive(crop)
        save_image(processed_crop, dir_processed / f"{name}.png")

        # 3. OCR with Auto-Detection
        ocr_text = ""
        try:
            # use_det=True allows AI to find the text inside the loose crop
            result, _ = engine(processed_crop, use_det=True, use_cls=False, use_rec=True)
            ocr_text = extract_text_safely(result)
        except Exception as e:
            print(f"  ❌ OCR Fail [{name}]: {e}")

        print(f"    [{name}]: {ocr_text}")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        extracted_data.append(f"{timestamp}\t{img_path.name}\t{name}\t{ocr_text}\n")

    # Write to text file
    with open(result_file_path, "a", encoding="utf-8-sig") as f_out:
        for line in extracted_data:
            f_out.write(line)

def main():
    # Initialize Directories
    for p in [INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
        p.mkdir(parents=True, exist_ok=True)
    
    # Initialize Debug Sub-directories
    (OUTPUT_FOLDER / "ROI_Original").mkdir(exist_ok=True)
    (OUTPUT_FOLDER / "ROI_Processed").mkdir(exist_ok=True)
    
    result_txt_path = OUTPUT_FOLDER / RESULT_FILENAME

    print("--------------------------------")
    print("Initializing RapidOCR...")
    engine = RapidOCR()
    
    rois = load_rois(ROI_JSON)
    if not rois:
        print("❌ Error: roi.json not found.")
        return

    print(f"Monitoring: {INPUT_FOLDER.absolute()}")
    print("--------------------------------")

    while True:
        try:
            # Filter files (ignore hidden files starting with .)
            files = list(INPUT_FOLDER.glob("*"))
            image_files = [
                f for f in files 
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} 
                and not f.name.startswith(".")
            ]
            
            if not image_files:
                time.sleep(1)
                continue

            for img_f in image_files:
                time.sleep(0.5) 
                
                try:
                    process_single_image(img_f, engine, rois, result_txt_path, OUTPUT_FOLDER)
                    
                    # Move to History
                    ts_name = f"{img_f.stem}_{int(time.time())}{img_f.suffix}"
                    shutil.move(str(img_f), str(PROCESSED_FOLDER / ts_name))
                    print(f"✅ Success.\n")

                except Exception as e:
                    print(f"❌ Error processing {img_f.name}: {e}")
                    # Move to Error Folder
                    try:
                        shutil.move(str(img_f), str(ERROR_FOLDER / img_f.name))
                        print("-> Moved to Error_Images\n")
                    except: pass

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()