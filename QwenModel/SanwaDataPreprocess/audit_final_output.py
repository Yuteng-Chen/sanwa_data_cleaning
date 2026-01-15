"""
最终输出审计 / Final Output Audit
检查数据质量问题：
1. 超过3位小数的数值
2. 包含HTML代码的字段
3. 异常值检测
4. 复制有问题的ROI裁剪图像到手动检查目录

用途：Stage 6压缩输出的质量检查
"""

import pandas as pd
import re
import shutil
from pathlib import Path
from config_pipeline import STAGE_6_FINAL, DEBUG_CROPS_BASE, get_roi_type

# 审计输出目录
AUDIT_OUTPUT = STAGE_6_FINAL / "audit_report"
MANUAL_CHECK_OUTPUT = AUDIT_OUTPUT / "manual_check_crops"

def detect_html(value):
    """检测HTML代码和模型控制tokens"""
    if pd.isna(value):
        return False
    val_str = str(value)
    # 检测常见HTML模式和模型控制tokens
    html_patterns = [
        r'<[^>]+>',          # HTML标签
        r'&[a-z]+;',         # HTML实体
        r'&#\d+;',           # 数字HTML实体
        r'```',              # Markdown代码块
        r'\*\*',             # Markdown加粗
        r'__',               # Markdown下划线
        r'<\|im_start\|>',   # 模型控制token
        r'<\|im_end\|>',     # 模型控制token
        r'<\|endoftext\|>',  # 模型控制token
        r'<\|pad\|>',        # 模型控制token
        r'<\|assistant\|>',  # 模型控制token
        r'<\|user\|>',       # 模型控制token
        r'<\|system\|>',     # 模型控制token
    ]
    for pattern in html_patterns:
        if re.search(pattern, val_str, re.IGNORECASE):
            return True
    return False

def detect_excess_decimals(value, max_decimals=3):
    """检测超过指定小数位数的数值"""
    if pd.isna(value):
        return False
    val_str = str(value).strip()
    
    # 检查是否是数字
    try:
        float(val_str)
    except:
        return False
    
    # 检查小数位数
    if '.' in val_str:
        decimal_part = val_str.split('.')[-1]
        # 移除尾随零后检查
        decimal_part_stripped = decimal_part.rstrip('0')
        if len(decimal_part) > max_decimals:
            return True
    
    return False

def detect_multiple_decimals(value):
    """检测多个小数点"""
    if pd.isna(value):
        return False
    val_str = str(value).strip()
    return val_str.count('.') > 1

def detect_repeat_pattern(value):
    """检测重复模式 (如 9.1289.128)"""
    if pd.isna(value):
        return False
    val_str = str(value).strip()
    
    # 检查常见重复模式
    # 例如: 1.881.88, 9.1289.128
    if len(val_str) >= 4:
        half = len(val_str) // 2
        if val_str[:half] == val_str[half:half*2]:
            return True
    
    return False

def calculate_roi_medians(df):
    """计算每个ROI列的中位数"""
    medians = {}
    roi_cols = [c for c in df.columns if c.startswith('ROI_')]
    
    for col in roi_cols:
        roi_type = get_roi_type(col)
        if roi_type in ['INTEGER', 'FLOAT']:
            try:
                # 转换为数值，忽略非数值
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                # 过滤掉0和NaN
                valid_vals = numeric_vals[(numeric_vals != 0) & (numeric_vals.notna())]
                if len(valid_vals) >= 5:
                    medians[col] = {
                        'median': valid_vals.median(),
                        'median_digits': len(str(int(abs(valid_vals.median()))))
                    }
            except:
                pass
    return medians

def should_flag_integer(value, median_info):
    """判断INTEGER值是否应该标记为问题"""
    if median_info is None:
        return False
    
    try:
        val = abs(float(value))
        median_val = median_info['median']
        median_digits = median_info['median_digits']
        
        # 计算当前值的位数
        val_digits = len(str(int(val))) if val > 0 else 1
        
        # 只有当值超过中位数3倍 且 位数也比中位数多时才标记
        is_3x_more = val > abs(median_val) * 3
        has_more_digits = val_digits > median_digits
        
        return is_3x_more and has_more_digits
    except:
        return False

def audit_single_csv(csv_path):
    """审计单个CSV文件"""
    print(f"\n📂 Auditing: {csv_path.name}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return None
    
    issues = []
    
    # 计算每个ROI的中位数
    roi_medians = calculate_roi_medians(df)
    
    # 检查所有ROI列
    roi_cols = [c for c in df.columns if c.startswith('ROI_')]
    
    for col in roi_cols:
        # 获取ROI类型
        roi_type = get_roi_type(col)
        median_info = roi_medians.get(col)
        
        for idx, value in df[col].items():
            filename = df.at[idx, 'Filename'] if 'Filename' in df.columns else f"Row_{idx}"
            
            # 检测HTML/特殊tokens（所有类型都检测）
            if detect_html(value):
                issues.append({
                    'Filename': filename,
                    'ROI': col,
                    'Value': str(value)[:100],
                    'Issue': 'Contains HTML/Markdown',
                    'Severity': 'HIGH'
                })
            
            # 检测多个小数点（FLOAT检测，INTEGER只在超过中位数3倍+位数更多时检测）
            if detect_multiple_decimals(value):
                if roi_type == 'FLOAT':
                    issues.append({
                        'Filename': filename,
                        'ROI': col,
                        'Value': str(value)[:50],
                        'Issue': 'Multiple decimal points',
                        'Severity': 'HIGH'
                    })
                elif roi_type == 'INTEGER' and should_flag_integer(value, median_info):
                    issues.append({
                        'Filename': filename,
                        'ROI': col,
                        'Value': str(value)[:50],
                        'Issue': 'Multiple decimal points (INTEGER >3x median)',
                        'Severity': 'HIGH'
                    })
            
            # 检测超过3位小数（仅FLOAT类型检测）
            if roi_type == 'FLOAT' and detect_excess_decimals(value, 3):
                issues.append({
                    'Filename': filename,
                    'ROI': col,
                    'Value': str(value)[:50],
                    'Issue': 'More than 3 decimal places',
                    'Severity': 'MEDIUM'
                })
            
            # 检测重复模式
            if detect_repeat_pattern(value):
                if roi_type == 'FLOAT':
                    issues.append({
                        'Filename': filename,
                        'ROI': col,
                        'Value': str(value)[:50],
                        'Issue': 'Repeat pattern detected',
                        'Severity': 'MEDIUM'
                    })
                elif roi_type == 'INTEGER' and should_flag_integer(value, median_info):
                    issues.append({
                        'Filename': filename,
                        'ROI': col,
                        'Value': str(value)[:50],
                        'Issue': 'Repeat pattern (INTEGER >3x median)',
                        'Severity': 'MEDIUM'
                    })
    
    # 统计
    html_count = sum(1 for i in issues if 'HTML' in i['Issue'])
    decimal_count = sum(1 for i in issues if 'decimal' in i['Issue'].lower())
    repeat_count = sum(1 for i in issues if 'Repeat' in i['Issue'])
    
    print(f"   📊 Total rows: {len(df)}")
    print(f"   ⚠️  HTML/Markdown issues: {html_count}")
    print(f"   ⚠️  Decimal issues: {decimal_count}")
    print(f"   ⚠️  Repeat patterns: {repeat_count}")
    
    return issues

def copy_issue_crops(issues):
    """
    复制有问题的ROI裁剪图像到手动检查目录
    """
    if not issues:
        return 0
    
    # 创建输出目录
    MANUAL_CHECK_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    unique_crops = set()  # 避免重复复制
    
    for issue in issues:
        filename = issue.get('Filename', '')
        roi = issue.get('ROI', '')
        issue_type = issue.get('Issue', 'Unknown')
        
        if not filename or not roi:
            continue
        
        # 构建源路径
        folder_name = Path(filename).stem
        
        # 尝试不同的ROI文件名格式
        for ext in ['jpg', 'png']:
            src_path = DEBUG_CROPS_BASE / folder_name / f"{roi}.{ext}"
            if src_path.exists():
                break
        else:
            continue
        
        # 创建唯一标识避免重复
        crop_key = f"{folder_name}_{roi}"
        if crop_key in unique_crops:
            continue
        unique_crops.add(crop_key)
        
        # 创建目标目录（按问题类型分类）
        issue_folder = issue_type.replace(' ', '_').replace('/', '_')
        dest_dir = MANUAL_CHECK_OUTPUT / issue_folder / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        dest_path = dest_dir / src_path.name
        try:
            shutil.copy(src_path, dest_path)
            copied_count += 1
        except Exception as e:
            pass
    
    return copied_count

def main():
    print("\n" + "="*60)
    print("🔍 FINAL OUTPUT AUDIT TOOL (with Manual Check Crops)")
    print("="*60)
    
    # 创建审计输出目录
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # 优先查找最终CSV，如果没有则查找压缩CSV
    csv_files = list(STAGE_6_FINAL.glob("*_Final.csv"))
    if not csv_files:
        csv_files = list(STAGE_6_FINAL.glob("compressed/*_Compressed.csv"))
    
    if not csv_files:
        print("\n❌ No final/compressed CSV files found!")
        return
    
    print(f"\n🔍 Found {len(csv_files)} files to audit")
    print(f"📁 Crops source: {DEBUG_CROPS_BASE}")
    
    all_issues = []
    
    for csv_file in csv_files:
        issues = audit_single_csv(csv_file)
        if issues:
            for issue in issues:
                issue['Source_File'] = csv_file.name
            all_issues.extend(issues)
    
    # 复制有问题的ROI裁剪图像
    if all_issues:
        print(f"\n📋 Copying issue crops for manual check...")
        copied = copy_issue_crops(all_issues)
        print(f"   ✅ Copied {copied} unique crops to {MANUAL_CHECK_OUTPUT}")
    
    # 保存审计报告
    if all_issues:
        df_issues = pd.DataFrame(all_issues)
        
        # 按严重程度和问题类型排序
        df_issues['Severity_Order'] = df_issues['Severity'].map({'HIGH': 0, 'MEDIUM': 1, 'LOW': 2})
        df_issues.sort_values(['Severity_Order', 'Issue', 'ROI'], inplace=True)
        df_issues.drop('Severity_Order', axis=1, inplace=True)
        
        report_path = AUDIT_OUTPUT / "audit_report.csv"
        df_issues.to_csv(report_path, index=False)
        
        # 生成摘要
        summary = df_issues.groupby(['Issue', 'Severity']).size().reset_index(name='Count')
        summary_path = AUDIT_OUTPUT / "audit_summary.csv"
        summary.to_csv(summary_path, index=False)
        
        print(f"\n📋 Saved audit report: {report_path}")
        print(f"📋 Saved summary: {summary_path}")
    
    # 结果
    print("\n" + "="*60)
    print("🎉 AUDIT COMPLETE")
    print("="*60)
    print(f"   Total issues found: {len(all_issues)}")
    
    if all_issues:
        print("\n   Issue Summary:")
        for issue_type in set(i['Issue'] for i in all_issues):
            count = sum(1 for i in all_issues if i['Issue'] == issue_type)
            print(f"     • {issue_type}: {count}")
    
    print(f"\n📂 Reports saved to: {AUDIT_OUTPUT}")
    print("="*60)

if __name__ == "__main__":
    main()
