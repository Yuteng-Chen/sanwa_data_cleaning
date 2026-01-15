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

# [REVERTED] Set padding to 0 or small number.
# With use_det=True below, we can handle small padding safely.
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

def save_image(img, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            buffer.tofile(str(path))
    except Exception as e:
        print(f"Failed to save image {path}: {e}")

def preprocess_image_simple(img):
    """
    [RESTORED] Simpler preprocessing logic.
    Removed the 'Noise Removal' that was eating your text.
    """
    if img is None: return None
    
    # 1. Upscale (Essential for small text)
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Auto-Invert
    # Check if background is dark. If so, invert to make it White.
    # OCR models work best with Black Text on White Background.
    border_mean = np.mean(np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ]))
    
    if border_mean < 127:
        gray = cv2.bitwise_not(gray) # Invert Black BG -> White
    
    # 4. Simple Binarization (Otsu)
    # This makes the text sharp black and background pure white.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary

def extract_text_safely(ocr_result):
    """
    Prevents the crash: 'sequence item 0: expected str instance, list found'
    """
    if not ocr_result: return ""
    
    found_texts = []
    for item in ocr_result:
        # RapidOCR format: [[coords], "Text", Confidence]
        if isinstance(item, (list, tuple)):
            if len(item) == 0: continue
            
            # Smart Check: Is the text at index 0 or 1?
            if len(item) >= 2 and isinstance(item[1], str):
                found_texts.append(item[1]) # Text is usually here when use_det=True
            elif isinstance(item[0], str):
                found_texts.append(item[0]) # Text is here when use_det=False
            elif isinstance(item[0], (list, tuple)) and len(item) > 1:
                 found_texts.append(str(item[1])) # Handle nested structure

    return " ".join(found_texts).strip()

def process_single_image(img_path: Path, engine, rois, result_file_path: Path, root_out_dir: Path):
    print(f"Processing: {img_path.name} ...")
    
    img_origin = read_image(img_path)
    if img_origin is None:
        raise ValueError(f"Could not decode image: {img_path}")

    H, W = img_origin.shape[:2]
    
    dir_debug = root_out_dir / "ROI_Processed" / img_path.stem

    extracted_data = []

    for name, x, y, w, h in rois:
        if x >= W or y >= H: continue
        
        # Exact Crop + Pad
        x0 = max(0, x - ROI_PAD)
        y0 = max(0, y - ROI_PAD)
        x1 = min(W, x + w + ROI_PAD)
        y1 = min(H, y + h + ROI_PAD)
        
        crop = img_origin[y0:y1, x0:x1]
        if crop.size == 0: continue

        # --- RESTORED PREPROCESSING ---
        processed_crop = preprocess_image_simple(crop)
        save_image(processed_crop, dir_debug / f"{name}.png")

        # --- ENABLE DETECTION (Crucial Fix) ---
        ocr_text = ""
        try:
            # [CHANGED] use_det=True
            # This is safer. It finds the text box inside your crop.
            # If you use use_det=False, your crop must be PERFECT with NO borders.
            result, _ = engine(processed_crop, use_det=True, use_cls=False, use_rec=True)
            ocr_text = extract_text_safely(result)
        except Exception as e:
            print(f"  ❌ OCR Fail [{name}]: {e}")

        print(f"    [{name}]: {ocr_text}")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        extracted_data.append(f"{timestamp}\t{img_path.name}\t{name}\t{ocr_text}\n")

    with open(result_file_path, "a", encoding="utf-8-sig") as f_out:
        for line in extracted_data:
            f_out.write(line)

def main():
    for p in [INPUT_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
        p.mkdir(parents=True, exist_ok=True)
    
    result_txt_path = OUTPUT_FOLDER / RESULT_FILENAME

    print("--------------------------------")
    print("Initializing RapidOCR...")
    engine = RapidOCR()
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
                    process_single_image(img_f, engine, rois, result_txt_path, OUTPUT_FOLDER)
                    ts_name = f"{img_f.stem}_{int(time.time())}{img_f.suffix}"
                    shutil.move(str(img_f), str(PROCESSED_FOLDER / ts_name))
                    print(f"✅ Success.\n")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    try: shutil.move(str(img_f), str(ERROR_FOLDER / img_f.name))
                    except: pass

        except KeyboardInterrupt: break
        except Exception: time.sleep(3)

if __name__ == "__main__":
    main()