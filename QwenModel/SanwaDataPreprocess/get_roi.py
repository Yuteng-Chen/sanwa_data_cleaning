import cv2
import json
from pathlib import Path

IMAGE_PATH = "2025-12-16_14.40.20.png"    # 原始图片
OUTPUT_JSON = "roi.json"             # ROI 配置输出
OVERVIEW_IMAGE = "2025-12-16_14.40.20roi_overview.png"  # 带编号 ROI 的总览图

WIN_NAME = "Select ROI"

drawing = False          # 是否正在画框
ix, iy = -1, -1          # 鼠标按下时（原图坐标）
current_rect = None      # 当前正在画的矩形（原图坐标）: (x, y, w, h)
rois = []                # 已确认的 ROI 列表（原图坐标）

scale = 1.0              # 当前缩放比例（显示用）
SCALE_STEP = 1.25
SCALE_MIN = 0.25
SCALE_MAX = 5.0


def to_img_coord(x_disp, y_disp):
    """显示坐标 -> 原图坐标"""
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
            current_rect = (x0, y0, w, h)
            roi_name = str(len(rois))  # 编号只用数字
            rois.append(
                {
                    "name": roi_name,
                    "x": x0,
                    "y": y0,
                    "w": w,
                    "h": h,
                }
            )
            print(f"添加 ROI {roi_name}: x={x0}, y={y0}, w={w}, h={h}")
            current_rect = None


def draw_rois(base_img, scale_factor=1.0):
    """在 base_img 上画出所有 ROI 和当前矩形，并按 scale_factor 缩放用于显示。"""
    h, w = base_img.shape[:2]
    # 等比例缩放：宽高都乘同一个 scale_factor
    disp_w = int(w * scale_factor)
    disp_h = int(h * scale_factor)
    display = cv2.resize(base_img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    # 已确认的 ROI
    for roi in rois:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        name = roi["name"]

        sx = int(x * scale_factor)
        sy = int(y * scale_factor)
        sw = int(rw * scale_factor)
        sh = int(rh * scale_factor)

        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        text_pos = (sx, max(0, sy - 3))
        cv2.putText(
            display,
            name,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # 正在绘制的矩形（黄色）
    if current_rect is not None:
        x, y, rw, rh = current_rect
        sx = int(x * scale_factor)
        sy = int(y * scale_factor)
        sw = int(rw * scale_factor)
        sh = int(rh * scale_factor)
        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 1)

    return display


def save_results(img):
    """保存 roi.json 和 带编号总览图"""
    if not rois:
        print("没有任何 ROI，无法保存。")
        return

    output_path = Path(OUTPUT_JSON)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rois, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(rois)} 个 ROI 到 {output_path.resolve()}")

    overview_img = draw_rois(img, scale_factor=1.0)
    overview_path = Path(OVERVIEW_IMAGE)
    cv2.imwrite(str(overview_path), overview_img)
    print(f"已生成带编号的总览图: {overview_path.resolve()}")


def main():
    global scale, rois

    img_path = Path(IMAGE_PATH)
    if not img_path.is_file():
        raise FileNotFoundError(f"找不到图片: {img_path.resolve()}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    # 改这里：用 AUTOSIZE，禁止用户用鼠标拉伸窗口
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)

    print("操作说明：")
    print("  鼠标左键按下并拖动：画一个 ROI")
    print("  鼠标左键松开：确认该 ROI，并自动编号为 0,1,2,...")
    print("  键盘 u：撤销最近一个 ROI")
    print("  键盘 c：清空所有 ROI")
    print("  键盘 + / = / ]：放大图像（等比例）")
    print("  键盘 - / _ / [：缩小图像（等比例）")
    print("  键盘 s：保存 roi.json 和 roi_overview.png，然后退出")
    print("  键盘 q：不保存直接退出")
    print("  点击右上角关闭窗口：直接退出程序")

    while True:
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("窗口已关闭，退出程序。")
            break

        display = draw_rois(img, scale_factor=scale)
        cv2.imshow(WIN_NAME, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        if key == ord("u"):
            if rois:
                removed = rois.pop()
                print(f"撤销 ROI {removed['name']}")
            else:
                print("当前没有可撤销的 ROI。")

        elif key == ord("c"):
            if rois:
                rois.clear()
                print("已清空所有 ROI。")
            else:
                print("当前没有任何 ROI。")

        elif key in (ord("+"), ord("="), ord("]")):
            new_scale = min(SCALE_MAX, scale * SCALE_STEP)
            if new_scale != scale:
                scale = new_scale
                print(f"放大，当前缩放比例: {scale:.2f}")

        elif key in (ord("-"), ord("_"), ord("[")):
            new_scale = max(SCALE_MIN, scale / SCALE_STEP)
            if new_scale != scale:
                scale = new_scale
                print(f"缩小，当前缩放比例: {scale:.2f}")

        elif key == ord("s"):
            if not rois:
                print("还没有 ROI，先圈一个再保存。")
                continue
            save_results(img)
            break

        elif key == ord("q"):
            print("未保存，直接退出。")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
