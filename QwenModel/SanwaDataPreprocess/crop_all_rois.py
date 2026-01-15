"""
批量裁剪所有ROI / Batch Crop All ROIs
从原始截图中裁剪 ROI 1-52 的所有图像

用途：生成完整的 debug_crops 目录
"""

import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

# ================= 配置 / Configuration =================

# 原始截图目录
SOURCE_IMAGES_DIR = Path("/home/wanfangyuan/Desktop/share01/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/12_16_cslot/2025-12-16")

# 输出目录 (debug_crops)
DEBUG_CROPS_BASE = Path("/home/wanfangyuan/Desktop/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/sanwa_ocr_debug/Sanwadata/12_16_cslot/2025-12-16/debug_crops")

# ROI坐标配置文件
ROI_JSON_PATH = Path("/home/wanfangyuan/Documents/Sanwa/deploy_version/roi_cslot.json")

# 裁剪设置
ROI_PAD = 2      # 边距像素
UPSCALE = 2.0    # 放大倍数

# 要裁剪的ROI范围
ROI_START = 1
ROI_END = 52

# ================= 函数 / Functions =================

def load_rois(json_path):
    """加载ROI坐标配置"""
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found.")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    roi_map = {}
    for item in data:
        name = str(item.get('name', ''))
        roi_map[name] = {
            'x': int(item['x']),
            'y': int(item['y']),
            'w': int(item['w']),
            'h': int(item['h'])
        }
        
    print(f"✅ Loaded {len(roi_map)} ROIs from JSON.")
    return roi_map

def perform_crop(img, roi_coords, save_path):
    """执行裁剪并保存"""
    try:
        x, y, w, h = roi_coords['x'], roi_coords['y'], roi_coords['w'], roi_coords['h']
        H, W = img.shape[:2]

        x0, y0 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        x1, y1 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)

        crop = img[y0:y1, x0:x1]
        
        if crop.size == 0:
            return False

        if UPSCALE != 1.0:
            crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), crop)
        return True

    except Exception as e:
        return False

def find_all_images(source_dir):
    """查找所有图像文件"""
    images = []
    
    # 支持的扩展名
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    for ext in extensions:
        images.extend(source_dir.glob(ext))
        images.extend(source_dir.rglob(ext))
    
    # 去重
    images = list(set(images))
    
    return sorted(images)

def process_single_image(img_path, roi_map, roi_ids):
    """处理单张图像，裁剪所有指定的ROI"""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    
    # 输出文件夹名 = 图像文件名（不含扩展名）
    folder_name = img_path.stem
    output_folder = DEBUG_CROPS_BASE / folder_name
    
    cropped_count = 0
    
    for roi_id in roi_ids:
        roi_key = str(roi_id)
        
        if roi_key not in roi_map:
            continue
        
        coords = roi_map[roi_key]
        save_path = output_folder / f"ROI_{roi_id}.jpg"
        
        # 如果已存在则跳过
        if save_path.exists():
            cropped_count += 1
            continue
        
        if perform_crop(img, coords, save_path):
            cropped_count += 1
    
    return cropped_count

def main():
    print("\n" + "="*60)
    print("🔧 BATCH ROI CROPPING TOOL")
    print("="*60)
    
    print(f"\n📁 Source Images: {SOURCE_IMAGES_DIR}")
    print(f"📁 Output Crops:  {DEBUG_CROPS_BASE}")
    print(f"📁 ROI Config:    {ROI_JSON_PATH}")
    print(f"🎯 ROI Range:     {ROI_START} - {ROI_END}")
    
    # 检查源目录
    if not SOURCE_IMAGES_DIR.exists():
        print(f"\n❌ Source images directory not found!")
        return
    
    # 加载ROI坐标
    roi_map = load_rois(ROI_JSON_PATH)
    if not roi_map:
        return
    
    # 要裁剪的ROI列表
    roi_ids = list(range(ROI_START, ROI_END + 1))
    print(f"📊 Will crop {len(roi_ids)} ROIs per image")
    
    # 查找所有图像
    print("\n🔍 Scanning for images...")
    images = find_all_images(SOURCE_IMAGES_DIR)
    print(f"   Found {len(images)} images")
    
    if not images:
        print("❌ No images found!")
        return
    
    # 创建输出目录
    DEBUG_CROPS_BASE.mkdir(parents=True, exist_ok=True)
    
    # 处理所有图像
    print("\n🚀 Starting batch crop...")
    
    total_crops = 0
    processed_images = 0
    
    for img_path in tqdm(images, desc="Processing images"):
        crops = process_single_image(img_path, roi_map, roi_ids)
        if crops > 0:
            total_crops += crops
            processed_images += 1
    
    # 结果
    print("\n" + "="*60)
    print("🎉 BATCH CROP COMPLETE")
    print("="*60)
    print(f"   Images Processed: {processed_images}")
    print(f"   Total ROI Crops:  {total_crops}")
    print(f"   Expected per img: {len(roi_ids)} ROIs")
    print(f"\n📂 Output: {DEBUG_CROPS_BASE}")
    print("="*60)

if __name__ == "__main__":
    main()
