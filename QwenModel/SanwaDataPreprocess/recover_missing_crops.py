"""
从CSV文件名列表重新生成所有ROI裁剪图像
Regenerate all ROI crops (1-52) for each filename in CSV

用途：读取CSV中的Filename列，为每个图像生成 ROI_1 到 ROI_52 的裁剪
"""

import pandas as pd
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

# 导入配置
from config_pipeline import (
    DEBUG_CROPS_BASE, 
    CSV_INPUT_DIR,
    PROJECT_ROOT
)

# ================= 配置 / Configuration =================

# 原始截图目录（用于重新裁剪）
SOURCE_IMAGES_DIR = Path("/home/wanfangyuan/Desktop/share01/Wan_Fangyuan/Sanwa/Sanwa Data2/ASTAR/Sanwadata/12_16_cslot/2025-12-16")

# ROI坐标配置文件
ROI_JSON_PATH = PROJECT_ROOT / "roi_cslot.json"

# 裁剪设置
ROI_PAD = 2      # 边距像素
UPSCALE = 2.0    # 放大倍数

# ROI范围
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
        roi_map[name] = [int(item['x']), int(item['y']), int(item['w']), int(item['h'])]
        
    print(f"✅ Loaded {len(roi_map)} ROIs from JSON.")
    return roi_map

def perform_crop(img, roi_coords, save_path):
    """执行裁剪并保存"""
    try:
        x, y, w, h = roi_coords
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

def find_source_image(filename):
    """查找原始图像"""
    # 直接路径
    src_path = SOURCE_IMAGES_DIR / filename
    if src_path.exists():
        return src_path
    
    # 递归搜索
    found = list(SOURCE_IMAGES_DIR.rglob(filename))
    if found:
        return found[0]
    
    # 尝试不同扩展名
    base_name = Path(filename).stem
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG']:
        src_path = SOURCE_IMAGES_DIR / f"{base_name}.{ext}"
        if src_path.exists():
            return src_path
        found = list(SOURCE_IMAGES_DIR.rglob(f"{base_name}.{ext}"))
        if found:
            return found[0]
    
    return None

def crop_all_rois_for_image(img_path, roi_map, roi_ids):
    """为单张图像裁剪缺失的ROI（已存在的跳过）"""
    # 输出文件夹名 = 图像文件名（不含扩展名）
    folder_name = img_path.stem
    output_folder = DEBUG_CROPS_BASE / folder_name
    
    # 先检查哪些ROI缺失
    missing_rois = []
    for roi_id in roi_ids:
        save_path = output_folder / f"ROI_{roi_id}.jpg"
        if not save_path.exists():
            missing_rois.append(roi_id)
    
    # 如果没有缺失，直接返回
    if not missing_rois:
        return 0, 0, len(roi_ids)  # cropped, skipped, already_exist
    
    # 有缺失才加载图像
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, len(missing_rois), len(roi_ids) - len(missing_rois)
    
    cropped = 0
    skipped = 0
    
    for roi_id in missing_rois:
        roi_key = str(roi_id)
        
        if roi_key not in roi_map:
            skipped += 1
            continue
        
        coords = roi_map[roi_key]
        save_path = output_folder / f"ROI_{roi_id}.jpg"
        
        if perform_crop(img, coords, save_path):
            cropped += 1
        else:
            skipped += 1
    
    already_exist = len(roi_ids) - len(missing_rois)
    return cropped, skipped, already_exist

def get_filenames_from_csv(csv_path):
    """从CSV提取所有唯一的文件名"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        return []
    
    filenames = set()
    
    # 尝试多个可能的文件名列
    filename_cols = ['Filename', 'filename', 'Filename_Current', 'Image', 'image']
    
    for col in filename_cols:
        if col in df.columns:
            valid_names = df[col].dropna().astype(str).unique()
            filenames.update(valid_names)
    
    return list(filenames)

def process_csv_file(csv_path, roi_map, roi_ids):
    """处理单个CSV文件"""
    print(f"\n📂 Processing CSV: {csv_path.name}")
    
    # 获取所有文件名
    filenames = get_filenames_from_csv(csv_path)
    
    if not filenames:
        print(f"   ⚠️ No filenames found in CSV")
        return 0, 0, 0, 0
    
    print(f"   Found {len(filenames)} unique filenames")
    
    total_cropped = 0
    total_skipped = 0
    total_existed = 0
    images_with_missing = 0
    
    for filename in tqdm(filenames, desc="   Checking & Cropping"):
        # 查找源图像
        src_path = find_source_image(filename)
        if not src_path:
            total_skipped += len(roi_ids)
            continue
        
        # 裁剪缺失的ROI（已存在的跳过）
        cropped, skipped, existed = crop_all_rois_for_image(src_path, roi_map, roi_ids)
        
        if cropped > 0:
            images_with_missing += 1
        total_cropped += cropped
        total_skipped += skipped
        total_existed += existed
    
    print(f"   ✅ Generated {total_cropped} new crops | Skipped {total_existed} existing | {images_with_missing} images had missing ROIs")
    
    return images_with_missing, total_cropped, total_skipped, total_existed

def main():
    print("\n" + "="*60)
    print("🔧 ROI CROP REGENERATION TOOL")
    print("="*60)
    
    print(f"\n📁 Source Images: {SOURCE_IMAGES_DIR}")
    print(f"📁 Output Crops:  {DEBUG_CROPS_BASE}")
    print(f"📁 ROI Config:    {ROI_JSON_PATH}")
    print(f"🎯 ROI Range:     ROI_{ROI_START} - ROI_{ROI_END}")
    
    # 检查源目录
    if not SOURCE_IMAGES_DIR.exists():
        print(f"\n❌ Source images directory not found!")
        print(f"   Please update SOURCE_IMAGES_DIR in this script.")
        return
    
    # 加载ROI坐标
    roi_map = load_rois(ROI_JSON_PATH)
    if not roi_map:
        print("\n❌ Could not load ROI coordinates!")
        return
    
    # ROI列表
    roi_ids = list(range(ROI_START, ROI_END + 1))
    print(f"📊 Will generate {len(roi_ids)} ROIs per image")
    
    # 只从 CSV_INPUT_DIR (Cut_preprocesseddata) 读取CSV
    print(f"\n📁 CSV Input Dir: {CSV_INPUT_DIR}")
    
    if not CSV_INPUT_DIR.exists():
        print(f"❌ CSV input directory not found: {CSV_INPUT_DIR}")
        return
    
    csv_files = list(CSV_INPUT_DIR.glob("*.csv"))
    
    if not csv_files:
        print("\n❌ No CSV files found!")
        return
    
    print(f"\n🔍 Found {len(csv_files)} CSV files to process")
    
    # 创建输出目录
    DEBUG_CROPS_BASE.mkdir(parents=True, exist_ok=True)
    
    # 处理所有CSV
    grand_total_images = 0
    grand_total_crops = 0
    grand_total_existed = 0
    
    for csv_file in csv_files:
        images, crops, _, existed = process_csv_file(csv_file, roi_map, roi_ids)
        grand_total_images += images
        grand_total_crops += crops
        grand_total_existed += existed
    
    # 结果
    print("\n" + "="*60)
    print("🎉 RECOVERY COMPLETE")
    print("="*60)
    print(f"   Images with missing ROIs: {grand_total_images}")
    print(f"   New crops generated:      {grand_total_crops}")
    print(f"   Already existed (skipped): {grand_total_existed}")
    print(f"\n📂 Output: {DEBUG_CROPS_BASE}")
    print("="*60)

if __name__ == "__main__":
    main()
