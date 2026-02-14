"""
数据处理管道的统一配置文件
Unified Configuration for Data Processing Pipeline
针对 2025-12-22 批次定制 (修复路径匹配)
"""
from pathlib import Path

# ================= 目录配置 / Directory Configuration =================
# 项目根目录 / Project Root Directory
PROJECT_ROOT = Path("/scratch/prj0000000262/ocr_data/QwenModel/SanwaDataPreprocess")

# 服务器根目录
SERVER_ROOT = PROJECT_ROOT

# --- 关键修改：定义批次变量 (完全匹配你的路径) ---
BATCH_NAME = "12_17_2025cslot/2025-12-17"     # 对应 CSV 文件夹
CROP_DIR_NAME = "12_17_2025cslot/2025-12-17"  # 对应 debug_crops 文件夹

# 输出根目录
PREPROCESS_ROOT = PROJECT_ROOT / "pipeline_output"
OUTPUT_BASE = PREPROCESS_ROOT 

# 输入数据路径 / Input Data Paths
# 1. CSV 输入：指向你提供的 stage1_ocr_results/2025-12-22
CSV_INPUT_DIR = OUTPUT_BASE / "stage1_ocr_results" / BATCH_NAME

# 2. 截图输入：指向共享的 debug_crops 目录下对应批次
DEBUG_CROPS_INPUT = OUTPUT_BASE / "stage1_ocr_results" / "debug_crops" / CROP_DIR_NAME

# 原始图像输入（Stage 0使用，保持对应日期）
SOURCE_DIR = PROJECT_ROOT / "input_images" / "12_18_2025" / "2025-12-18"

# 各阶段输出目录 / Stage Output Directories
# 统一加上 BATCH_NAME 后缀，防止文件混淆
STAGE_1_OCR = OUTPUT_BASE / "stage1_ocr_results" / BATCH_NAME
STAGE_2_CLEANED = OUTPUT_BASE / "stage2_cleaned_data" / BATCH_NAME
STAGE_3_3B_CORRECTED = OUTPUT_BASE / "stage3_3b_corrected" / BATCH_NAME
STAGE_4_LABELED = OUTPUT_BASE / "stage4_labeled" / BATCH_NAME
STAGE_5_7B_VERIFIED = OUTPUT_BASE / "stage5_7b_verified" / BATCH_NAME
STAGE_6_FINAL = OUTPUT_BASE / "stage6_final_dataset" / BATCH_NAME

# 调试和检查目录
# 关键：DEBUG_CROPS_BASE 必须指向存在的截图目录，否则会报 Image Not Found
DEBUG_CROPS_BASE = DEBUG_CROPS_INPUT  # 直接复用上面定义好的路径
ABNORMAL_CROPS_BASE = OUTPUT_BASE / "abnormal_crops_review" / BATCH_NAME
REDUNDANCY_CROPS_BASE = OUTPUT_BASE / "redundancy_crops_review" / BATCH_NAME

# 人工检查目录
MANUAL_CHECK_BASE_Abnormal = OUTPUT_BASE / "Abnormal" / BATCH_NAME
MANUAL_CHECK_BASE_Mismatch = OUTPUT_BASE / "Mismatch" / BATCH_NAME

# ================= 模型配置 / Model Configuration =================
# 这里的名称必须与 `ollama list` 显示的完全一致 (qwen2.5vl 无横杠)
OLLAMA_MODEL_3B = "qwen2.5vl:3b"
OLLAMA_MODEL_7B = "qwen2.5vl:7b"

# GPU并行处理配置 / GPU Parallel Processing
# 针对 8x H200 (140GB) 的强力配置
MAX_WORKERS_3B = 64  # 3B模型极快，且显存足够，拉高并发
MAX_WORKERS_7B = 32   # 7B模型并发

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
OUTLIER_THRESHOLD = 5.0
Z_SCORE_THRESHOLD = 3.0
FROZEN_THRESHOLD_SECONDS = 10.0

# 自适应阈值配置
SIMILARITY_THRESHOLDS = {
    "CslotCam4result.csv": 0.85,
    "cam 6 snap1 Latchresult.csv": 0.80,
    "cam 6 snap2 nozzleresult.csv": 0.95,
    "terminal result.csv": 0.90
}
DEFAULT_SIMILARITY_THRESHOLD = 0.80

# ================= Prompt模板 / Prompt Templates =================
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

# ================= Mismatch验证专用Prompt / Mismatch Verification Prompts =================
MISMATCH_PROMPTS = {
    'STATUS': (
        "Task: Verify status value from TWO consecutive images.\n"
        "Context: Previous frame shows '{compared_value}', current OCR reads '{current_value}'.\n"
        "📋 OUTPUT FORMAT: Exactly 'OK' or 'NG'\n"
        "Compare both images and determine the CORRECT status for the SECOND image.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, HTML, markdown"
    ),
    'INTEGER': (
        "Task: Verify integer value by comparing TWO consecutive images.\n"
        "Context: Previous frame shows '{compared_value}', current OCR reads '{current_value}'.\n"
        "Reference median: ~{median_context}\n"
        "📋 OUTPUT FORMAT: Integer only (e.g., 95, 1911)\n"
        "Compare both images. Output the CORRECT integer for the SECOND image.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, HTML, markdown"
    ),
    'FLOAT': (
        "Task: Verify decimal value by comparing TWO consecutive images.\n"
        "Context: Previous frame shows '{compared_value}', current OCR reads '{current_value}'.\n"
        "Reference median: ~{median_context}\n"
        "📋 OUTPUT FORMAT: Decimal with max 3 places (e.g., 1.188, 16.069)\n"
        "Compare both images. Output the CORRECT decimal for the SECOND image.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, HTML, markdown"
    ),
    'TIME': (
        "Task: Read the timestamp from this image.\n"
        "Context: Previous was '{compared_value}', OCR read '{current_value}'.\n"
        "📋 OUTPUT FORMAT: HH:MM:SS\n"
        "Output ONLY the timestamp (HH:MM:SS). If blank → NA.\n"
        "🚫 FORBIDDEN: <|im_start|>, <|endoftext|>, <|im_end|>, HTML, markdown"
    )
}

# ================= 字段特定提示 / Field-Specific Prompts =================
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
    """获取ROI的数据类型"""
    if roi_id in ROI_TYPE_MAP:
        return ROI_TYPE_MAP[roi_id]
    if f"ROI_{roi_id}" in ROI_TYPE_MAP:
        return ROI_TYPE_MAP[f"ROI_{roi_id}"]
    clean_id = str(roi_id).replace("ROI_", "")
    if clean_id in ROI_TYPE_MAP:
        return ROI_TYPE_MAP[clean_id]
    return 'STATUS'

def get_field_hint(roi_id: str) -> str:
    """获取字段特定的提示信息"""
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
    """根据ROI类型和上下文生成prompt"""
    clean_id = str(roi_id).replace('ROI_', '')
    roi_type = get_roi_type(clean_id)
    
    if prompt_type == 'mismatch':
        template = MISMATCH_PROMPTS.get(roi_type, MISMATCH_PROMPTS['STATUS'])
    else:
        template = PROMPTS.get(roi_type, {}).get(prompt_type, PROMPTS['STATUS']['initial'])
    
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
    
    field_hint = get_field_hint(roi_id)
    
    prompt = template.replace('{ocr_value}', str(ocr_value))
    prompt = prompt.replace('{median_context}', str(median_context))
    prompt = prompt.replace('{compared_value}', str(compared_value))
    prompt = prompt.replace('{current_value}', str(current_value))
    prompt = prompt.replace('{prev_filename}', str(prev_filename))
    prompt = prompt.replace('{curr_filename}', str(curr_filename))
    prompt = prompt.replace('{roi_id}', clean_id)
    
    if field_hint:
        prompt = f"📌 {field_hint}\n\n{prompt}"
    
    return prompt

def create_directories():
    """创建所有必要的目录"""
    dirs = [
        SOURCE_DIR, OUTPUT_BASE, STAGE_1_OCR, STAGE_2_CLEANED,
        STAGE_3_3B_CORRECTED, STAGE_4_LABELED, STAGE_5_7B_VERIFIED,
        STAGE_6_FINAL, DEBUG_CROPS_BASE, ABNORMAL_CROPS_BASE,
        REDUNDANCY_CROPS_BASE, MANUAL_CHECK_BASE_Abnormal, MANUAL_CHECK_BASE_Mismatch
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories for batch {BATCH_NAME} created successfully")

if __name__ == "__main__":
    print("Configuration loaded successfully")
    print(f"Server Root: {SERVER_ROOT}")
    print(f"Target Batch: {BATCH_NAME}")
    print(f"Crop Source: {CROP_DIR_NAME}")
    print(f"\nROI Type Map: {len(ROI_TYPE_MAP)} ROIs configured")
    create_directories()