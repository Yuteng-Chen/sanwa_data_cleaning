import cv2
import json
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
IMAGE_PATH = "2025-12-16_14.40.20.png"    # Image to edit
OUTPUT_JSON = "roi.json"                  # JSON file to load/save
OVERVIEW_IMAGE = "roi_overview.png"       # Output overview

WIN_NAME = "ROI Editor (Load & Edit)"
# =============================================

drawing = False
ix, iy = -1, -1
current_rect = None
rois = []  # List of dicts: {'name': '0', 'x': 100, ...}

scale = 1.0
SCALE_STEP = 1.25
SCALE_MIN = 0.25
SCALE_MAX = 5.0

def load_existing_data():
    """Loads ROIs from JSON if file exists."""
    global rois
    json_path = Path(OUTPUT_JSON)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and single object formats
                if isinstance(data, list):
                    rois = data
                else:
                    rois = [data]
            print(f"✅ Loaded {len(rois)} existing ROIs from {OUTPUT_JSON}")
        except Exception as e:
            print(f"⚠️ Error loading JSON: {e}")
            rois = []
    else:
        print(f"ℹ️ No existing {OUTPUT_JSON} found. Starting fresh.")

def to_img_coord(x_disp, y_disp):
    global scale
    return int(x_disp / scale), int(y_disp / scale)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_rect, rois

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = to_img_coord(x, y)
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x_img, y_img = to_img_coord(x, y)
        x0, y0 = min(ix, x_img), min(iy, y_img)
        w, h = abs(x_img - ix), abs(y_img - iy)
        current_rect = (x0, y0, w, h)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_img, y_img = to_img_coord(x, y)
        x0, y0 = min(ix, x_img), min(iy, y_img)
        w, h = abs(x_img - ix), abs(y_img - iy)
        if w > 0 and h > 0:
            # Auto-assign next available number as ID
            # Check existing names to find the highest number
            max_id = -1
            for r in rois:
                try:
                    num = int(r["name"])
                    if num > max_id: max_id = num
                except: pass
            
            roi_name = str(max_id + 1)
            
            rois.append({
                "name": roi_name,
                "x": x0, "y": y0, "w": w, "h": h
            })
            print(f"➕ Added ROI {roi_name}")
            current_rect = None

def draw_rois(base_img, scale_factor=1.0):
    h, w = base_img.shape[:2]
    disp_w, disp_h = int(w * scale_factor), int(h * scale_factor)
    display = cv2.resize(base_img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    # Draw Saved ROIs
    for roi in rois:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        name = roi["name"]

        sx, sy = int(x * scale_factor), int(y * scale_factor)
        sw, sh = int(rw * scale_factor), int(rh * scale_factor)

        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv2.putText(display, name, (sx, max(0, sy - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw Current Drag
    if current_rect is not None:
        x, y, rw, rh = current_rect
        sx, sy = int(x * scale_factor), int(y * scale_factor)
        sw, sh = int(rw * scale_factor), int(rh * scale_factor)
        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 1)

    return display

def save_results(img):
    if not rois:
        print("⚠️ No ROIs to save.")
        return

    # Sort ROIs by name (numeric) just to be tidy
    try:
        rois.sort(key=lambda x: int(x["name"]))
    except: pass

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rois, f, ensure_ascii=False, indent=2)
    
    overview_img = draw_rois(img, scale_factor=1.0)
    cv2.imwrite(OVERVIEW_IMAGE, overview_img)
    print(f"💾 Saved {len(rois)} ROIs to {OUTPUT_JSON}")

def main():
    global scale, rois

    img_path = Path(IMAGE_PATH)
    if not img_path.is_file():
        print(f"❌ Image not found: {img_path}")
        return

    # Load ROI Data FIRST
    load_existing_data()

    img = cv2.imread(str(img_path))
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)

    print("=== CONTROLS ===")
    print(" [Mouse Drag] : Create ROI")
    print(" [u]          : Undo last ROI")
    print(" [c]          : Clear ALL")
    print(" [+ / -]      : Zoom In/Out")
    print(" [s]          : SAVE and Exit")
    print(" [q]          : Quit without saving")
    print("================")

    while True:
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break

        display = draw_rois(img, scale_factor=scale)
        cv2.imshow(WIN_NAME, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 255: continue

        if key == ord("u"):
            if rois:
                removed = rois.pop()
                print(f"↩️ Undo ROI {removed['name']}")
        elif key == ord("c"):
            rois.clear()
            print("🗑️ Cleared all ROIs")
        elif key in (ord("+"), ord("=")):
            scale = min(SCALE_MAX, scale * SCALE_STEP)
        elif key in (ord("-"), ord("_")):
            scale = max(SCALE_MIN, scale / SCALE_STEP)
        elif key == ord("s"):
            save_results(img)
            break
        elif key == ord("q"):
            print("👋 Exiting without save.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()