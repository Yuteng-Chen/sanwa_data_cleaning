import cv2
import json
import ollama
import time
import os
from pathlib import Path

# ================= CONFIGURATION =================
# 1. PASTE THE FULL PATH TO YOUR IMAGE HERE
TARGET_IMAGE = "/Users/pomvrp/Library/CloudStorage/OneDrive-AgencyforScience,TechnologyandResearch/Sanwa data/12_16_cslot/2025-12-16/2025-12-16 14.49.06.png"

ROI_JSON = Path("roi.json")
OLLAMA_MODEL = "qwen2.5vl:3b"
ROI_PAD = 2           # Added small padding to help OCR see edges
UPSCALE = 2.0         # Slight upscale helps small numbers
# =================================================

def load_rois(roi_path):
    if not roi_path.exists():
        print(f"❌ Error: {roi_path} not found.")
        return []
    with open(roi_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rois = []
    data_list = data if isinstance(data, list) else [data]
    for idx, item in enumerate(data_list):
        name = item.get("name", str(idx))
        rois.append((str(name), int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])))
    return rois

def run_serial_test():
    print(f"--- STARTING SERIAL TEST (One-by-One) ---")
    
    # 1. SETUP
    rois = load_rois(ROI_JSON)
    if not rois: return

    img_path = Path(TARGET_IMAGE)
    if not img_path.exists():
        print(f"❌ Image path not found: {TARGET_IMAGE}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ OpenCV could not read image.")
        return

    H, W = img.shape[:2]
    temp_crop_path = "temp_current_crop.jpg"
    
    print(f"✅ Loaded Image: {W}x{H}")
    print(f"⚡ Processing {len(rois)} ROIs individually...\n")

    # 2. LOOP THROUGH EVERY ROI
    for i, (name, x, y, w, h) in enumerate(rois):
        # Boundary Checks
        if x >= W or y >= H: continue
        
        # Crop with Padding
        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)
        
        crop = img[y0:y1, x0:x1]
        if crop.size == 0: continue

        # Upscale (Critical for small text)
        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        # Save Temp Image
        cv2.imwrite(temp_crop_path, crop)

        # Send to Ollama
        # We use a very simple prompt because there is only 1 thing to read
        prompt = "Read the text in this image. Return ONLY the value. No extra words."
        
        start_t = time.time()
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt, 'images': [temp_crop_path]}],
                options={'num_predict': 20} # Limit output to 20 tokens for speed
            )
            text = response['message']['content'].strip()
            # Clean unwanted markdown
            text = text.replace("`", "").replace('"', '').replace("JSON", "")
            
            print(f"ID {name}: \t{text}")

        except Exception as e:
            print(f"ID {name}: \t❌ Error {e}")

    # Cleanup
    if os.path.exists(temp_crop_path):
        os.remove(temp_crop_path)
    print("\n✅ Done.")

if __name__ == "__main__":
    run_serial_test()