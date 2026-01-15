"""
数据处理管道的统一配置文件
Unified Configuration for Data Processing Pipeline
"""
from pathlib import Path

# ================= 目录配置 / Directory Configuration =================
# 项目根目录 / Project Root Directory
PROJECT_ROOT = Path("/scratch/prj0000000262/ocr_data/QwenModel/SanwaDataPreprocess")

# 服务器根目录 / Server Root Directory
# 使用项目根目录作为服务器根目录
SERVER_ROOT = PROJECT_ROOT

# 输入数据路径 / Input Data Paths
CSV_INPUT_DIR = PROJECT_ROOT / "input_images/2025-12-19cslot"
DEBUG_CROPS_INPUT = PROJECT_ROOT / "debug_crops"

# 输入输出目录 / Input/Output Directories
SOURCE_DIR = PROJECT_ROOT / "input_images/2025-12-19cslot"  # 原始图像输入（Stage 0使用）   # 原始图像输入（Stage 0使用）

# ⚠️ 修改此变量可更改所有输出目录的根路径
# Change this variable to redirect all output to a new folder
PREPROCESS_ROOT = PROJECT_ROOT / "pipeline_output"
OUTPUT_BASE = PREPROCESS_ROOT 
# 各阶段输出目录 / Stage Output Directories
# Stage 1: 模拟的OCR输出结构（使用现有数据）/ Simulated OCR output structure (using existing data)
STAGE_1_OCR = OUTPUT_BASE / "stage1_ocr_results/2025-12-19cslot"
STAGE_2_CLEANED = OUTPUT_BASE / "stage2_cleaned_data"
STAGE_3_3B_CORRECTED = OUTPUT_BASE / "stage3_3b_corrected"
STAGE_4_LABELED = OUTPUT_BASE / "stage4_labeled"
STAGE_5_7B_VERIFIED = OUTPUT_BASE / "stage5_7b_verified"
STAGE_6_FINAL = OUTPUT_BASE / "stage6_final_dataset"

# 调试和检查目录 / Debug and Review Directories
DEBUG_CROPS_BASE = OUTPUT_BASE / "stage1_ocr_results/debug_crops/2025-12-19cslot"
ABNORMAL_CROPS_BASE = OUTPUT_BASE / "abnormal_crops_review"
REDUNDANCY_CROPS_BASE = OUTPUT_BASE / "redundancy_crops_review"

# 人工检查目录 / Manual Check Directories
MANUAL_CHECK_BASE_Abnormal = OUTPUT_BASE / "Abnormal"
MANUAL_CHECK_BASE_Mismatch = OUTPUT_BASE / "Mismatch"

# ================= 模型配置 / Model Configuration =================
OLLAMA_MODEL_3B = "qwen2.5vl:3b"
OLLAMA_MODEL_7B = "qwen2.5vl:7b"

# GPU并行处理配置 / GPU Parallel Processing
# 配置说明：您有 4x V100 GPUs (32GB each)
# 
# V100性能参考：
# - 3B模型：~2-3GB显存/实例，每块GPU可同时运行8-10个实例
# - 7B模型：~6-8GB显存/实例，每块GPU可同时运行3-4个实例
#
# 推荐配置（4块V100）：
MAX_WORKERS_3B = 64  # 4 GPUs * 4 workers = 16 (保守配置)
                     # 可以尝试 20-24 如果显存充足
MAX_WORKERS_7B = 32   # 4 GPUs * 2 workers = 8 (保守配置)
                     # 可以尝试 12 如果显存充足

# 性能调优建议：
# - 监控GPU使用率：nvidia-smi -l 1
# - 如果GPU利用率 < 80%，可以增加workers
# - 如果出现OOM错误，减少workers
# - 3B模型处理速度快，可以设置更多workers
# - 7B模型显存需求大，workers数量要保守

# ================= ROI配置 / ROI Configuration =================
ROI_JSON = Path("roi.json")
ROI_PAD = 2
UPSCALE = 2.0
DARKNESS_THRESHOLD = 15

# ROI数据类型映射 / ROI Data Type Mapping
ROI_CONFIGS = [
    {
        'Trigger_Col': 'ROI_12',
        'CSV_Name': 'cam 6 snap1 Latchresult.csv',
        'Columns': {
            'ROI_12': 'STATUS', 'ROI_14': 'STATUS', 'ROI_15': 'STATUS', 
            'ROI_17': 'STATUS', 'ROI_19': 'STATUS',
            'ROI_13': 'INTEGER', 'ROI_16': 'FLOAT', 'ROI_18': 'FLOAT', 
            'ROI_52': 'TIME'
        }
    },
    {
        'Trigger_Col': 'ROI_20',
        'CSV_Name': 'cam 6 snap2 nozzleresult.csv',
        'Columns': {
            'ROI_20': 'STATUS', 'ROI_22': 'STATUS', 'ROI_24': 'STATUS', 
            'ROI_25': 'STATUS', 'ROI_26': 'STATUS', 'ROI_27': 'STATUS', 
            'ROI_28': 'STATUS', 'ROI_29': 'STATUS', 'ROI_30': 'STATUS',
            'ROI_21': 'INTEGER', 'ROI_23': 'FLOAT', 
            'ROI_52': 'TIME'
        }
    },
    {
        'Trigger_Col': 'ROI_1',
        'CSV_Name': 'CslotCam4result.csv',
        'Columns': {
            'ROI_1': 'STATUS', 'ROI_3': 'STATUS', 'ROI_5': 'STATUS', 
            'ROI_7': 'STATUS', 'ROI_9': 'STATUS', 'ROI_10': 'STATUS', 
            'ROI_11': 'STATUS',
            'ROI_2': 'INTEGER', 'ROI_4': 'FLOAT', 'ROI_6': 'FLOAT', 
            'ROI_8': 'FLOAT', 
            'ROI_52': 'TIME'
        }
    },
    {
        'Trigger_Col': 'ROI_31',
        'CSV_Name': 'terminal result.csv',
        'Columns': {
            'ROI_31': 'STATUS', 'ROI_33': 'STATUS', 'ROI_34': 'STATUS', 
            'ROI_36': 'STATUS', 'ROI_38': 'STATUS', 'ROI_40': 'STATUS', 
            'ROI_42': 'STATUS', 'ROI_44': 'STATUS', 'ROI_46': 'STATUS', 
            'ROI_48': 'STATUS', 'ROI_50': 'STATUS',
            'ROI_32': 'INTEGER', 'ROI_35': 'INTEGER', 'ROI_37': 'INTEGER', 
            'ROI_39': 'INTEGER', 'ROI_41': 'INTEGER', 'ROI_43': 'INTEGER', 
            'ROI_45': 'INTEGER', 'ROI_47': 'INTEGER', 'ROI_49': 'INTEGER',
            'ROI_52': 'TIME'
        }
    }
]

# CSV分组配置 / CSV Grouping Configuration
CSV_GROUPS = {
    "CslotCam4result.csv": list(range(1, 12)),
    "cam 6 snap1 Latchresult.csv": list(range(12, 20)),
    "cam 6 snap2 nozzleresult.csv": list(range(20, 31)),
    "terminal result.csv": list(range(31, 51))
}

# 扁平化ROI类型映射 / Flatten ROI Type Map
ROI_TYPE_MAP = {}
for cfg in ROI_CONFIGS:
    ROI_TYPE_MAP.update(cfg['Columns'])

# ================= 数据验证配置 / Data Validation Configuration =================
MAX_DECIMALS = 3
OUTLIER_THRESHOLD = 5.0       # Ratio-based: 检测 >5x 或 <0.2x median 的值 (缺少小数点)
Z_SCORE_THRESHOLD = 3.0       # Z-Score: 检测偏离正常范围的值 (>3 标准差)
                               # Z > 2.0: ~5% 异常 (95% 置信区间)
                               # Z > 2.5: ~1.2% 异常 
                               # Z > 3.0: ~0.3% 异常 (99.7% 置信区间) [推荐]
                               # Z > 3.5: ~0.05% 异常 (更保守)
FROZEN_THRESHOLD_SECONDS = 10.0

# 自适应阈值配置（针对不同数据集）/ Adaptive Threshold Configuration
SIMILARITY_THRESHOLDS = {
    "CslotCam4result.csv": 0.85,          # C-slot较敏感
    "cam 6 snap1 Latchresult.csv": 0.80,  # Latch默认
    "cam 6 snap2 nozzleresult.csv": 0.95, # Nozzle默认
    "terminal result.csv": 0.90           # Terminal数据点多，阈值更高
}

# 默认相似度阈值 / Default Similarity Threshold
DEFAULT_SIMILARITY_THRESHOLD = 0.80

# ================= Prompt模板 / Prompt Templates =================

# 通用噪声过滤规则（应用于所有prompt）
NOISE_FILTER_RULES = """
⚠️ NOISE FILTERING RULES (CRITICAL):
1. IGNORE half-cut numbers at edges (only partial digits visible)
2. IGNORE text with different colored background that's cut off
3. IGNORE background patterns or decorative elements
4. ONLY read complete, fully visible numbers in the main display area
5. If a digit is only 50% visible or less → DO NOT guess, skip it
6. Focus on the primary number display, not peripheral text

🚫🚫🚫 STRICTLY FORBIDDEN OUTPUT (WILL BE REJECTED):
- <|im_start|>, <|im_end|>, <|endoftext|>, <|pad|> - these are model tokens, NOT data!
- Any text starting with <| or ending with |>
- HTML tags: <br>, <p>, <div>, <span>, etc.
- Markdown: **, __, ```, #, etc.
- Output ONLY: the raw number, 'OK', 'NG', or timestamp (HH:MM:SS)
"""

# 数字格式验证规则（应用于FLOAT和INTEGER）
NUMBER_VALIDATION_RULES = """
🔢 NUMBER FORMAT VALIDATION (MUST CHECK):
1. ONLY ONE decimal point allowed (e.g., '5.7.726' is INVALID → probably '5.726')
2. Maximum 3 digits after decimal point (e.g., '1.8888' → truncate to '1.888')
3. Watch for REPEAT PATTERNS that indicate OCR errors:
   - '5.7.726' → likely should be '5.726' (duplicate pattern)
   - '1.881.88' → likely should be '1.88' (repeated number)
   - '9.1289.128' → likely should be '9.128' (repeated number)
4. If you see multiple decimal points → remove duplicates, keep FIRST valid pattern
5. If the number seems 5x or more different from reference:
   - STOP and look MORE carefully at the image
   - Check for missing/extra decimal points
   - Check for digit repetition errors
   - Report what you ACTUALLY see after careful review
"""

PROMPTS = {
    'STATUS': {
        'initial': (
            "Task: Classify the status indicator in this image.\n"
            "\n"
            "🎨 COLOR IDENTIFICATION (CRITICAL):\n"
            "  - GREEN text/color = 'OK' (pass/good)\n"
            "  - RED text/color = 'NG' (fail/bad)\n"
            "  - Trust the COLOR more than the text shape!\n"
            "\n"
            "📋 OUTPUT FORMAT: Must be exactly 'OK' or 'NG'\n"
            "\n"
            "Rules:\n"
            "1. Output ONLY 'OK' or 'NG' (nothing else, no NA)\n"
            "2. GREEN color or starts with O → Output: OK\n"
            "3. RED color or starts with N → Output: NG\n"
            "4. If image is blank or unreadable → Output: NG (default to fail-safe)\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown\n"
            "\n" + NOISE_FILTER_RULES
        ),
        'correction': (
            "Task: Classify as 'OK' or 'NG'.\n"
            "📋 OUTPUT FORMAT: Exactly 'OK' or 'NG' (nothing else)\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown"
        )
    },
    'INTEGER': {
        'initial': (
            "Task: Extract the integer number from this digital display.\n"
            "\n"
            "📋 OUTPUT FORMAT: Integer only (e.g., 95, 100, -5)\n"
            "\n"
            "⚠️ IMPORTANT: Output EXACTLY what you see in the image!\n"
            "   Reference value ~{median_context} is just a GUIDELINE.\n"
            "   If the image shows a different number, OUTPUT THAT NUMBER.\n"
            "\n"
            "Rules:\n"
            "1. Output ONLY the integer you actually see\n"
            "2. NO decimal point allowed for integer fields\n"
            "3. If negative, include the '-' sign\n"
            "4. If blank → Output: 0\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown\n"
            "\n" + NOISE_FILTER_RULES
        ),
        'correction': (
            "Task: Extract the integer from the image.\n"
            "\n"
            "📋 OUTPUT FORMAT: Integer only (e.g., 95, 100)\n"
            "\n"
            "⚠️ CRITICAL: Output EXACTLY what you SEE, not what you expect!\n"
            "   Reference ~{median_context} is just a guideline for context.\n"
            "   Your job is to report the ACTUAL number in the image.\n"
            "\n"
            "Rules:\n"
            "1. Output ONLY the integer number (no decimal point)\n"
            "2. If blank, output '0'\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown"
        )
    },
    'FLOAT': {
        'initial': (
            "Task: Extract the floating-point number from this sensor reading.\n"
            "\n"
            "📋 OUTPUT FORMAT: Decimal number with max 3 decimal places\n"
            "   Examples: 1.188, 16.069, 0.5\n"
            "\n"
            "⚠️ IMPORTANT: Output EXACTLY what you see in the image!\n"
            "   Reference value ~{median_context} is just a GUIDELINE.\n"
            "   If the image shows a different number, OUTPUT THAT NUMBER.\n"
            "\n"
            "Rules:\n"
            "1. Output the ACTUAL number you see\n"
            "2. Maximum 3 decimal places\n"
            "3. ONLY ONE decimal point allowed\n"
            "4. If blank → Output: 0\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown\n"
            "\n" + NOISE_FILTER_RULES
        ),
        'correction': (
            "Task: Extract the DECIMAL NUMBER from the image.\n"
            "\n"
            "📋 OUTPUT FORMAT: X.XXX (e.g., 1.823, 16.069, 0.001)\n"
            "\n"
            "⚠️ CRITICAL: Output EXACTLY what you SEE, not what you expect!\n"
            "   Reference ~{median_context} is just a guideline for context.\n"
            "   Your job is to report the ACTUAL number in the image.\n"
            "\n"
            "   If you see 2.03, output 2.03 (even if reference is 1.18)\n"
            "   If you see 16.5, output 16.5 (even if reference is 16.06)\n"
            "\n"
            "Format Rules:\n"
            "1. ONLY ONE decimal point allowed\n"
            "2. MAXIMUM 3 digits after decimal\n"
            "3. If you see '9.1289.128' → fix to '9.128' (remove duplicate)\n"
            "4. If you see '1.7.798' → fix to '1.798' (fix multiple decimals)\n"
            "5. If blank, output '0'\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown"
        )
    },
    'TIME': {
        'initial': (
            "Task: Read the timestamp from this display.\n"
            "\n"
            "📋 OUTPUT FORMAT: HH:MM:SS (24-hour format)\n"
            "   Examples: 17:06:42, 09:15:30, 23:59:59\n"
            "\n"
            "Rules:\n"
            "1. Output ONLY in format HH:MM:SS\n"
            "2. Use 24-hour format\n"
            "3. If blank → Output: NA\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown"
        ),
        'correction': (
            "Task: Read Timestamp from the image.\n"
            "📋 OUTPUT FORMAT: HH:MM:SS (e.g., 17:06:42)\n"
            "Output ONLY the timestamp you see.\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown"
        ),
        'mismatch': (
            "Task: Read the timestamp from this image.\n"
            "Context: Previous was '{compared_value}', OCR read '{current_value}'.\n"
            "📋 OUTPUT FORMAT: HH:MM:SS\n"
            "Output ONLY the timestamp (HH:MM:SS). If blank → NA.\n"
            "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, <>, HTML, markdown"
        )
    }
}

# ================= 字段特定提示 / Field-Specific Prompts =================
# 为每个具体ROI字段定义预期范围和格式
FIELD_SPECIFIC_HINTS = {
    # CslotCam4result fields
    '1': {'type': 'STATUS', 'hint': 'C-Slot Status', 'format': 'OK or NG'},
    '2': {'type': 'INTEGER', 'hint': 'Counter/Count value', 'format': 'Integer (e.g., 95)', 'typical_range': '90-100'},
    '3': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '4': {'type': 'FLOAT', 'hint': 'Measurement value', 'format': 'X.XXX (e.g., 1.188)', 'typical_range': '1.1-1.3'},
    '5': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '6': {'type': 'FLOAT', 'hint': 'Large measurement', 'format': 'XX.XXX (e.g., 16.069)', 'typical_range': '15.8-16.2'},
    '7': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '8': {'type': 'FLOAT', 'hint': 'Measurement value', 'format': 'X.XXX (e.g., 1.165)', 'typical_range': '1.1-1.3'},
    '9': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '10': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '11': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    
    # cam 6 snap1 Latchresult fields
    '12': {'type': 'STATUS', 'hint': 'Latch Status', 'format': 'OK or NG'},
    '13': {'type': 'INTEGER', 'hint': 'Confidence/Count', 'format': 'Integer (e.g., 97)'},
    '14': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '15': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '16': {'type': 'FLOAT', 'hint': 'Measurement', 'format': 'X.XXX', 'typical_range': '1.5-2.5'},
    '17': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '18': {'type': 'FLOAT', 'hint': 'Measurement', 'format': 'X.XXX', 'typical_range': '1.5-2.5'},
    '19': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    
    # cam 6 snap2 nozzleresult fields
    '20': {'type': 'STATUS', 'hint': 'Nozzle Status', 'format': 'OK or NG'},
    '21': {'type': 'INTEGER', 'hint': 'Count value', 'format': 'Integer'},
    '22': {'type': 'STATUS', 'hint': 'Status indicator', 'format': 'OK or NG'},
    '23': {'type': 'FLOAT', 'hint': 'Measurement', 'format': 'X.XXX'},
    
    # Terminal result fields (31-50)
    '31': {'type': 'STATUS', 'hint': 'Terminal Status', 'format': 'OK or NG'},
    '32': {'type': 'INTEGER', 'hint': 'Terminal Count', 'format': 'Integer'},
    
    # Timestamp fields
    '51': {'type': 'DATE', 'hint': 'Date display', 'format': 'MM/DD/YY'},
    '52': {'type': 'TIME', 'hint': 'Time display', 'format': 'HH:MM:SS'},
}

# Mismatch Correction Prompts (7B Verification)
MISMATCH_PROMPTS = {
    'STATUS': (
        "Task: Read the text in this image strictly.\n"
        "Options: Usually 'OK' or 'NG'.\n"
        "Context: The previous row was '{compared_value}', but OCR read '{current_value}'.\n"
        "Output ONLY the text visible in the image.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <>, HTML, markdown"
    ),
    'INTEGER': (
        "Task: Extract the INTEGER from this image.\n"
        "Context: Previous was '{compared_value}'. Current OCR says '{current_value}'.\n"
        "STRICT RULES:\n"
        "1. Output ONLY the integer number (no decimal point).\n"
        "2. If empty or black, output '0'.\n"
        "3. NO special tokens, NO HTML, NO <|...|> tags.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <>, HTML, markdown"
    ),
    'FLOAT': (
        "Task: Extract the DECIMAL NUMBER from this image.\n"
        "Context: Previous was '{compared_value}'. Current OCR says '{current_value}'.\n"
        "⚠️ CRITICAL FORMAT RULES:\n"
        "1. Output ONLY ONE number with ONLY ONE decimal point.\n"
        "2. MAXIMUM 3 digits after decimal (e.g., 9.128 not 9.12845).\n"
        "3. If you see '9.1289.128' → output '9.128' (remove duplicate).\n"
        "4. If you see '1.7.7988' → output '1.798' (fix multiple decimals).\n"
        "5. If empty or black, output '0'.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <>, HTML, markdown\n"
        "Output format: X.XXX (e.g., 1.823, 9.128, 0.001)"
    ),
    'TIME': (
        "Task: Read the timestamp from this image.\n"
        "Context: Previous was '{compared_value}'. Current OCR says '{current_value}'.\n"
        "Output ONLY the timestamp (HH:MM:SS).\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <>, HTML, markdown"
    )
}

# ================= 日志配置 / Logging Configuration =================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def get_similarity_threshold(csv_name: str) -> float:
    """根据CSV文件名获取相应的相似度阈值"""
    for key in SIMILARITY_THRESHOLDS:
        if key in csv_name:
            return SIMILARITY_THRESHOLDS[key]
    return DEFAULT_SIMILARITY_THRESHOLD

def get_roi_type(roi_id: str) -> str:
    """获取ROI的数据类型（增强版：自动匹配 ROI_ 前缀）"""
    # 1. 尝试直接查找（例如 key是 "ROI_12"）
    if roi_id in ROI_TYPE_MAP:
        return ROI_TYPE_MAP[roi_id]
    
    # 2. 尝试加上前缀查找（例如传入 "12" -> 查找 "ROI_12"）
    if f"ROI_{roi_id}" in ROI_TYPE_MAP:
        return ROI_TYPE_MAP[f"ROI_{roi_id}"]
        
    # 3. 尝试去掉前缀查找（例如传入 "ROI_12" -> 查找 "12"）
    clean_id = str(roi_id).replace("ROI_", "")
    if clean_id in ROI_TYPE_MAP:
        return ROI_TYPE_MAP[clean_id]

    # 4. 实在找不到，才默认是 STATUS
    return 'STATUS'

def get_field_hint(roi_id: str) -> str:
    """获取字段特定的提示信息"""
    # 清理ROI_前缀
    clean_id = str(roi_id).replace('ROI_', '')
    
    if clean_id in FIELD_SPECIFIC_HINTS:
        hint = FIELD_SPECIFIC_HINTS[clean_id]
        parts = [f"Field: ROI_{clean_id} ({hint.get('hint', 'Unknown')})"]
        if 'format' in hint:
            parts.append(f"Expected Format: {hint['format']}")
        if 'typical_range' in hint:
            parts.append(f"Typical Range: {hint['typical_range']} (GUIDELINE ONLY)")
        return "\n".join(parts)
    return ""

def get_prompt(roi_id: str, prompt_type: str = 'initial', 
               ocr_value: str = '', median_value: float = None,
               compared_value: str = '', current_value: str = '',
               prev_filename: str = '', curr_filename: str = '') -> str:
    """
    根据ROI类型和上下文生成prompt
    
    Args:
        roi_id: ROI标识符（如 'ROI_13' 或 '13'）
        prompt_type: 'initial', 'correction', 或 'mismatch'
        ocr_value: 之前的OCR结果（用于correction）
        median_value: 该ROI的中位数值（仅作为参考，不是目标值）
        compared_value: 比较值（用于mismatch）
        current_value: 当前值（用于mismatch）
        prev_filename: 前一张图像文件名（用于mismatch dual comparison）
        curr_filename: 当前图像文件名（用于mismatch dual comparison）
    """
    # 清理ROI_前缀
    clean_id = str(roi_id).replace('ROI_', '')
    roi_type = get_roi_type(clean_id)
    
    # 选择prompt模板
    if prompt_type == 'mismatch':
        template = MISMATCH_PROMPTS.get(roi_type, MISMATCH_PROMPTS['STATUS'])
    else:
        template = PROMPTS.get(roi_type, {}).get(prompt_type, PROMPTS['STATUS']['initial'])
    
    # 格式化median上下文 - 强调这只是参考值，不是目标值
    median_context = "N/A"
    if median_value is not None:
        if roi_type == 'STATUS':
            median_context = f"(typically {median_value})"
        elif roi_type == 'INTEGER':
            median_context = f"{int(median_value)} (reference only - output what you SEE)"
        elif roi_type == 'FLOAT':
            median_context = f"{median_value:.3f} (reference only - output what you SEE)"
        elif roi_type == 'TIME':
            median_context = "(varies)"
    
    # 获取字段特定提示
    field_hint = get_field_hint(roi_id)
    
    # 替换所有占位符
    prompt = template.replace('{ocr_value}', str(ocr_value))
    prompt = prompt.replace('{median_context}', str(median_context))
    prompt = prompt.replace('{compared_value}', str(compared_value))
    prompt = prompt.replace('{current_value}', str(current_value))
    prompt = prompt.replace('{prev_filename}', str(prev_filename))
    prompt = prompt.replace('{curr_filename}', str(curr_filename))
    prompt = prompt.replace('{roi_id}', clean_id)
    
    # 添加字段特定提示（如果有）
    if field_hint:
        prompt = f"📌 {field_hint}\n\n{prompt}"
    
    return prompt

def create_directories():
    """创建所有必要的目录"""
    dirs = [
        SOURCE_DIR, OUTPUT_BASE, STAGE_1_OCR, STAGE_2_CLEANED,
        STAGE_3_3B_CORRECTED, STAGE_4_LABELED, STAGE_5_7B_VERIFIED,
        STAGE_6_FINAL, DEBUG_CROPS_BASE, ABNORMAL_CROPS_BASE,
        REDUNDANCY_CROPS_BASE
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ All directories created successfully")

if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"Server Root: {SERVER_ROOT}")
    print(f"OCR Model 3B: {OLLAMA_MODEL_3B}")
    print(f"OCR Model 7B: {OLLAMA_MODEL_7B}")
    print(f"\nROI Type Map: {len(ROI_TYPE_MAP)} ROIs configured")
    print(f"CSV Groups: {len(CSV_GROUPS)} groups configured")
    create_directories()

