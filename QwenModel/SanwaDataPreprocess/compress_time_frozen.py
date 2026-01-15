"""
压缩时间冻结冗余行 / Compress Time Frozen Redundant Rows
移除连续的 Time Static + Redundant 行，只保留第一行

用途：进一步压缩 Stage 6 输出数据
"""

import pandas as pd
from pathlib import Path
from config_pipeline import STAGE_6_FINAL, PROJECT_ROOT

# 输出目录
OUTPUT_DIR = STAGE_6_FINAL / "compressed"

def compress_time_frozen(df):
    """
    压缩时间冻结冗余行
    逻辑：
    1. 连续的 "Time Static" 行 -> 只保留第一行
    2. 连续的 "Redundant" 行 -> 只保留第一行
    3. "Time Frozen" 行 -> 只保留第一行
    """
    if df.empty:
        return df, []
    
    df = df.copy()
    df.sort_values(by='Filename', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    rows_to_keep = []
    deletion_log = []
    
    i = 0
    while i < len(df):
        current_row = df.iloc[i]
        current_time_status = str(current_row.get('Time_Status', ''))
        current_redundancy = str(current_row.get('Data_Redundancy', ''))
        
        # 总是保留第一行
        rows_to_keep.append(i)
        
        # 如果当前行是 "Time Static" 或有 "Redundant" 标记
        is_frozen_or_redundant = (
            'Time Static' in current_time_status or 
            'Time Frozen' in current_time_status or
            'Redundant' in current_redundancy
        )
        
        if is_frozen_or_redundant:
            # 查找连续的同类行
            j = i + 1
            while j < len(df):
                next_row = df.iloc[j]
                next_time_status = str(next_row.get('Time_Status', ''))
                next_redundancy = str(next_row.get('Data_Redundancy', ''))
                
                # 检查是否也是冻结/冗余行
                next_is_frozen = (
                    'Time Static' in next_time_status or 
                    'Time Frozen' in next_time_status or
                    'Redundant' in next_redundancy
                )
                
                # 检查是否是同一时间状态序列（ROI_52时间戳相同）
                same_plc_time = (
                    str(current_row.get('ROI_52', '')) == str(next_row.get('ROI_52', ''))
                )
                
                if next_is_frozen and same_plc_time:
                    # 标记为删除
                    deletion_log.append({
                        'Deleted_Filename': next_row['Filename'],
                        'Time_Status': next_time_status,
                        'Data_Redundancy': next_redundancy,
                        'ROI_52': next_row.get('ROI_52', ''),
                        'Reason': 'Time Frozen Compression'
                    })
                    j += 1
                else:
                    break
            
            # 跳到非冗余行
            i = j
        else:
            i += 1
    
    # 提取保留的行
    df_compressed = df.iloc[rows_to_keep].copy()
    
    return df_compressed, deletion_log

def process_final_csv(csv_path):
    """处理单个最终CSV文件"""
    print(f"\n📂 Processing: {csv_path.name}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return
    
    original_count = len(df)
    print(f"   Original rows: {original_count}")
    
    # 检查必要的列
    required_cols = ['Filename', 'Time_Status', 'Data_Redundancy']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"   ⚠️ Missing columns: {missing_cols}")
        # 如果缺少列，复制原始文件
        df.to_csv(OUTPUT_DIR / csv_path.name, index=False)
        return
    
    # 压缩
    df_compressed, deletion_log = compress_time_frozen(df)
    
    compressed_count = len(df_compressed)
    removed_count = original_count - compressed_count
    
    print(f"   Compressed rows: {compressed_count}")
    print(f"   Removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
    
    # 保存压缩后的CSV
    output_name = csv_path.stem + "_Compressed.csv"
    df_compressed.to_csv(OUTPUT_DIR / output_name, index=False)
    print(f"   ✅ Saved: {output_name}")
    
    # 保存删除日志
    if deletion_log:
        log_name = csv_path.stem + "_Compression_Log.csv"
        pd.DataFrame(deletion_log).to_csv(OUTPUT_DIR / log_name, index=False)
        print(f"   📋 Log: {log_name}")

def main():
    print("\n" + "="*60)
    print("🗜️  TIME FROZEN COMPRESSION TOOL")
    print("="*60)
    
    print(f"\n📁 Input Dir:  {STAGE_6_FINAL}")
    print(f"📁 Output Dir: {OUTPUT_DIR}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找最终CSV文件
    final_csvs = list(STAGE_6_FINAL.glob("*_Final.csv"))
    
    if not final_csvs:
        print("\n❌ No *_Final.csv files found!")
        return
    
    print(f"\n🔍 Found {len(final_csvs)} files to process")
    
    total_original = 0
    total_compressed = 0
    
    for csv_file in final_csvs:
        try:
            df_orig = pd.read_csv(csv_file)
            total_original += len(df_orig)
        except:
            pass
        process_final_csv(csv_file)
    
    # 统计压缩后的总行数
    for csv_file in OUTPUT_DIR.glob("*_Compressed.csv"):
        try:
            df_comp = pd.read_csv(csv_file)
            total_compressed += len(df_comp)
        except:
            pass
    
    # 结果
    print("\n" + "="*60)
    print("🎉 COMPRESSION COMPLETE")
    print("="*60)
    if total_original > 0:
        print(f"   Total Original:   {total_original}")
        print(f"   Total Compressed: {total_compressed}")
        print(f"   Total Removed:    {total_original - total_compressed}")
        print(f"   Compression Rate: {(total_original - total_compressed)/total_original*100:.1f}%")
    print(f"\n📂 Output: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
