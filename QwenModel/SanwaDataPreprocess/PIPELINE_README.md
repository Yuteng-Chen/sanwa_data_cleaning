# 自动化OCR数据处理管道
# Automated OCR Data Processing Pipeline

完整的工业级OCR数据处理系统，集成3B和7B模型进行多级验证和修正。

## 📋 目录

- [系统概述](#系统概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [安装配置](#安装配置)
- [快速开始](#快速开始)
- [详细使用指南](#详细使用指南)
- [配置说明](#配置说明)
- [故障排查](#故障排查)

---

## 系统概述

本系统是一个完整的自动化数据处理管道，用于处理工业视觉检测系统的OCR数据。系统分为6个阶段，使用3B和7B视觉语言模型进行多级验证和修正。

### 处理流程

```
原始图像 → [Stage 0: OCR识别] → 
[Stage 1: 数据验证] → [Stage 2: 3B修正] → [Stage 3: 合并] → 
[Stage 4: 标记分析] → [Stage 5: 7B验证] → [Stage 6: 最终整合] → 
清洁数据集
```

---

## 核心特性

### ✨ 智能识别
- **动态Prompt生成**：根据ROI类型（STATUS/INTEGER/FLOAT/TIME）自动生成优化的prompt
- **实时Median追踪**：动态计算并使用median值作为上下文，提高识别准确度
- **自适应精度控制**：根据数据类型自动调整输出格式和精度

### 🔍 多级验证
- **3B模型初步修正**：快速识别和修正明显的OCR错误
- **7B模型高精度验证**：对冗余不匹配进行深度验证
- **统计异常检测**：基于median和阈值的outlier检测

### 🎯 数据质量控制
- **时间状态分析**：检测时间冻结和静态状态
- **模糊冗余检测**：使用自适应阈值识别冗余数据
- **配对压缩**：智能消除冗余行，保留关键数据

### 📊 完整追踪
- **调试输出**：每个阶段生成详细的中间结果
- **异常图像复制**：自动复制问题图像供人工review
- **修正日志记录**：完整记录所有修正和删除操作

---

## 系统架构

### 文件结构

```
deploy_version/
├── config_pipeline.py          # 统一配置文件
├── ocrserver_enhanced.py       # Stage 0: 增强OCR服务器
├── data_pipeline_3b.py         # Stage 1-3: 3B处理管道
├── data_pipeline_7b.py         # Stage 4-6: 7B验证管道
├── run_pipeline.py             # 自动化运行器
├── roi.json                    # ROI配置文件
└── requirements.txt            # Python依赖

pipeline_output/                # 输出目录（自动创建）
├── stage1_ocr_results/         # OCR原始结果
├── stage2_cleaned_data/        # 清理后数据
├── stage3_3b_corrected/        # 3B修正结果
├── stage4_labeled/             # 标记后数据
├── stage5_7b_verified/         # 7B验证结果
└── stage6_final_dataset/       # 最终数据集 ⭐
```

### 各阶段说明

#### Stage 0: 增强OCR服务器（ocrserver_enhanced.py）
- **输入**：原始图像文件
- **处理**：
  - 根据ROI配置裁剪图像
  - 使用3B模型进行OCR识别
  - 实时计算median值
  - 根据数据类型生成动态prompt
- **输出**：CSV文件（每个CSV组一个文件）

#### Stage 1: 数据验证（data_pipeline_3b.py）
- **输入**：Stage 0的CSV结果
- **处理**：
  - 逐行验证数据格式
  - 统计异常值检测
  - 生成异常日志
- **输出**：_Cleaned.csv, _Abnormal_Log.csv

#### Stage 2: 3B模型修正（data_pipeline_3b.py）
- **输入**：Stage 1的异常日志
- **处理**：
  - 使用3B模型重新识别异常值
  - 动态调整median上下文
  - 生成修正建议
- **输出**：_AI_3B_Fixed.csv

#### Stage 3: 合并修正（data_pipeline_3b.py）
- **输入**：Stage 1清洁数据 + Stage 2修正日志
- **处理**：
  - 将3B修正结果合并回原数据集
  - 更新相应的单元格
- **输出**：_3B_Corrected.csv

#### Stage 4: 数据标记（data_pipeline_7b.py）
- **输入**：Stage 3的3B修正数据
- **处理**：
  - 时间状态分析（静态/冻结/变化）
  - 模糊冗余检测（使用自适应阈值）
  - 识别冗余不匹配
- **输出**：_Labeled.csv, _Redundancy_Mismatch_Log.csv

#### Stage 5: 7B模型验证（data_pipeline_7b.py）
- **输入**：Stage 4的冗余不匹配日志
- **处理**：
  - 使用7B模型进行高精度验证
  - 判定真实变化 vs OCR错误
  - 生成最终判决
- **输出**：_AI_7B_Verified.csv

#### Stage 6: 最终整合（data_pipeline_7b.py）
- **输入**：Stage 4标记数据 + Stage 5验证日志
- **处理**：
  - 应用7B修正
  - 配对消除冗余行
  - 计算真实时间间隔
- **输出**：_Final.csv, _Deletion_Log.csv ⭐

---

## 安装配置

### 1. 系统要求

- **操作系统**：Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.8+
- **GPU**: NVIDIA V100 或更好（用于Ollama）
- **内存**: 16GB+ RAM
- **磁盘**: 50GB+ 可用空间

### 2. 安装依赖

```bash
# 1. 激活虚拟环境
source py313_env/bin/activate

# 2. 安装Python依赖
pip install -r requirements.txt

# 3. 安装Ollama（如果尚未安装）
curl https://ollama.ai/install.sh | sh

# 4. 拉取模型
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b
```

### 3. 配置系统

编辑 `config_pipeline.py`：

```python
# 修改服务器根目录
SERVER_ROOT = Path("/your/path/to/sanwa_project")

# 调整GPU工作线程（根据GPU内存）
MAX_WORKERS_3B = 4  # V100可用4-8
MAX_WORKERS_7B = 2  # 7B模型较大，建议2

# 调整相似度阈值（根据数据集特性）
SIMILARITY_THRESHOLDS = {
    "CslotCam4result.csv": 0.85,
    "cam 6 snap1 Latchresult.csv": 0.80,
    "cam 6 snap2 nozzleresult.csv": 0.80,
    "terminal result.csv": 0.90
}
```

### 4. 准备ROI配置

确保 `roi.json` 文件存在并包含正确的ROI配置：

```json
[
  {"name": "1", "x": 100, "y": 200, "w": 50, "h": 30},
  {"name": "2", "x": 150, "y": 200, "w": 50, "h": 30},
  ...
]
```

---

## 快速开始

### 方法1: 完整自动运行

```bash
# 运行完整管道（包括OCR）
python run_pipeline.py --full

# 跳过OCR阶段（使用已有的OCR结果）
python run_pipeline.py --full --skip-ocr
```

### 方法2: 分阶段运行

```bash
# 1. 运行OCR服务器（Stage 0）
python ocrserver_enhanced.py

# 2. 运行3B处理管道（Stages 1-3）
python data_pipeline_3b.py

# 3. 运行7B验证管道（Stages 4-6）
python data_pipeline_7b.py
```

### 方法3: 交互模式

```bash
python run_pipeline.py
# 然后根据提示选择要执行的操作
```

---

## 详细使用指南

### Scenario 1: 首次处理新数据

1. **准备数据**
   ```bash
   # 将图像放入输入目录
   cp /path/to/images/* /home/ubuntu/sanwa_project/input_images/
   ```

2. **运行完整管道**
   ```bash
   python run_pipeline.py --full
   ```

3. **检查结果**
   ```bash
   # 最终数据集
   ls /home/ubuntu/sanwa_project/pipeline_output/stage6_final_dataset/
   
   # 查看某个最终CSV
   head /home/ubuntu/sanwa_project/pipeline_output/stage6_final_dataset/CslotCam4result_Final.csv
   ```

### Scenario 2: 已有OCR结果，仅运行清理流程

1. **将OCR结果放入Stage 1目录**
   ```bash
   cp /path/to/ocr_results/*.csv \
      /home/ubuntu/sanwa_project/pipeline_output/stage1_ocr_results/CSV_Results/
   ```

2. **运行3B和7B管道**
   ```bash
   python data_pipeline_3b.py
   python data_pipeline_7b.py
   ```

### Scenario 3: 仅重新运行7B验证

如果3B结果满意，只需重新运行7B阶段：

```bash
python data_pipeline_7b.py
```

### Scenario 4: 调整阈值后重新标记

1. **编辑配置**
   ```python
   # 在 config_pipeline.py 中修改
   SIMILARITY_THRESHOLDS = {
       "CslotCam4result.csv": 0.90,  # 提高阈值
       ...
   }
   ```

2. **重新运行Stage 4-6**
   ```bash
   python data_pipeline_7b.py
   ```

---

## 配置说明

### 核心配置参数

#### 目录配置
```python
SERVER_ROOT = Path("/home/ubuntu/sanwa_project")  # 服务器根目录
SOURCE_DIR = SERVER_ROOT / "input_images"         # 原始图像输入
```

#### 模型配置
```python
OLLAMA_MODEL_3B = "qwen2.5vl:3b"  # 3B模型名称
OLLAMA_MODEL_7B = "qwen2.5vl:7b"  # 7B模型名称
MAX_WORKERS_3B = 4                 # 3B并行线程数
MAX_WORKERS_7B = 2                 # 7B并行线程数
```

#### 图像处理配置
```python
ROI_PAD = 2             # ROI边界扩展像素
UPSCALE = 2.0           # 图像上采样倍数
DARKNESS_THRESHOLD = 15 # 暗度阈值（低于此值视为太暗）
```

#### 数据验证配置
```python
MAX_DECIMALS = 3                # 浮点数最大小数位
OUTLIER_THRESHOLD = 5.0         # Outlier检测倍数
FROZEN_THRESHOLD_SECONDS = 10.0 # 时间冻结阈值
```

#### 自适应相似度阈值
```python
SIMILARITY_THRESHOLDS = {
    "CslotCam4result.csv": 0.85,          
    "cam 6 snap1 Latchresult.csv": 0.80,  
    "cam 6 snap2 nozzleresult.csv": 0.80, 
    "terminal result.csv": 0.90           
}
```

**调整建议**：
- **高阈值（0.85-0.95）**：适用于数据点多、变化少的场景
- **低阈值（0.70-0.80）**：适用于数据点少、变化频繁的场景
- **观察 _Redundancy_Mismatch_Log.csv**：如果误报多，提高阈值；如果漏报多，降低阈值

### ROI配置

每个ROI需要定义数据类型，系统会据此生成不同的prompt：

```python
'ROI_12': 'STATUS',   # OK/NG状态
'ROI_13': 'INTEGER',  # 整数计数
'ROI_16': 'FLOAT',    # 浮点数测量值
'ROI_52': 'TIME'      # 时间戳 HH:MM:SS
```

---

## 故障排查

### 问题1: Ollama连接失败

**症状**：
```
[OCR Error ROI_1]: Connection refused
```

**解决方案**：
```bash
# 检查Ollama服务状态
systemctl status ollama

# 重启Ollama
systemctl restart ollama

# 检查模型是否存在
ollama list
```

### 问题2: GPU内存不足

**症状**：
```
CUDA out of memory
```

**解决方案**：
```python
# 在 config_pipeline.py 中降低并行数
MAX_WORKERS_3B = 2  # 从4降到2
MAX_WORKERS_7B = 1  # 从2降到1
```

### 问题3: 冗余检测过于敏感

**症状**：大量非冗余数据被标记为冗余

**解决方案**：
```python
# 在 config_pipeline.py 中提高相似度阈值
SIMILARITY_THRESHOLDS = {
    "CslotCam4result.csv": 0.90,  # 从0.85提高到0.90
    ...
}
```

### 问题4: ROI识别不准确

**症状**：特定ROI的识别率低

**解决方案**：

1. **检查裁剪图像**：
   ```bash
   # 查看调试裁剪
   ls pipeline_output/stage1_ocr_results/debug_crops/
   ```

2. **调整ROI配置**：
   ```json
   // 在 roi.json 中调整坐标和大小
   {"name": "13", "x": 100, "y": 200, "w": 60, "h": 40}
   ```

3. **调整图像处理参数**：
   ```python
   ROI_PAD = 5      # 增加边界扩展
   UPSCALE = 3.0    # 增加上采样倍数
   ```

### 问题5: 3B修正效果不佳

**症状**：3B模型的修正准确率低

**解决方案**：

1. **切换到7B模型**：
   ```python
   # 在 ocrserver_enhanced.py 中
   OLLAMA_MODEL_3B = "qwen2.5vl:7b"  # 使用7B替代3B
   ```

2. **调整prompt**：
   编辑 `config_pipeline.py` 中的 `PROMPTS` 字典，添加更多上下文

### 问题6: 处理速度太慢

**优化建议**：

1. **增加并行度**：
   ```python
   MAX_WORKERS_3B = 8  # 如果GPU内存充足
   ```

2. **跳过已处理的文件**：
   手动移除 `input_images` 中已处理的图像

3. **分批处理**：
   将大量图像分成小批次，分别处理

---

## 输出文件说明

### 最终数据集（_Final.csv）

包含以下关键列：

| 列名 | 说明 |
|------|------|
| `Filename` | 原始图像文件名 |
| `File_UTC` | 文件时间戳（UTC） |
| `Machine_Text` | 机器时间戳（原始文本） |
| `Machine_UTC` | 机器时间戳（UTC转换） |
| `ROI_1` ~ `ROI_52` | 各ROI的识别结果 |
| `Time_Status` | 时间状态（New/Static/Frozen） |
| `Data_Redundancy` | 冗余状态（Unique/Redundant） |
| `Matched_File` | 匹配的冗余文件 |
| `Duration_Since_Change` | 自上次变化的时长（秒） |
| `Real_Freeze_Duration_Sec` | 真实冻结时长（秒） |

### 删除日志（_Deletion_Log.csv）

记录被消除的冗余行：

| 列名 | 说明 |
|------|------|
| `Deleted_Filename` | 被删除的文件名 |
| `Reason` | 删除原因 |
| `Original_Status` | 原始冗余状态 |

### 异常日志（_Abnormal_Log.csv）

记录所有异常值：

| 列名 | 说明 |
|------|------|
| `Filename` | 图像文件名 |
| `Timestamp` | 时间戳 |
| `ROI_ID` | 异常的ROI |
| `Value` | 异常值 |
| `Reason` | 异常原因 |
| `AI_3B_Corrected` | 3B修正结果 |

---

## 性能参考

基于V100 GPU的性能数据：

| 阶段 | 处理速度 | 备注 |
|------|---------|------|
| Stage 0 (OCR) | ~5-10 images/min | 取决于ROI数量 |
| Stage 1 (验证) | ~1000 rows/s | 纯Python处理 |
| Stage 2 (3B修正) | ~20-30 images/min | 仅处理异常值 |
| Stage 3 (合并) | ~500 rows/s | 纯Pandas操作 |
| Stage 4 (标记) | ~300 rows/s | 包含相似度计算 |
| Stage 5 (7B验证) | ~10-15 images/min | 仅处理不匹配 |
| Stage 6 (整合) | ~500 rows/s | 纯Pandas操作 |

**1000张图像的预计总时长**：2-4小时（取决于异常率）

---

## 高级技巧

### 1. 自定义Prompt模板

编辑 `config_pipeline.py` 中的 `PROMPTS` 字典来优化识别效果：

```python
PROMPTS = {
    'FLOAT': {
        'initial': (
            "你的自定义prompt..."
        ),
        ...
    }
}
```

### 2. 动态调整Median更新策略

在 `ocrserver_enhanced.py` 中修改 `MedianTracker` 类：

```python
# 调整加权平均系数
roi_medians[roi_id] = (curr_median * 0.8) + (val_num * 0.2)  # 更信任新值
```

### 3. 批量处理多个数据集

```bash
#!/bin/bash
for dataset in dataset1 dataset2 dataset3; do
    echo "Processing $dataset..."
    # 修改配置指向不同数据集
    sed -i "s|SERVER_ROOT = .*|SERVER_ROOT = Path(\"/data/$dataset\")|" config_pipeline.py
    python run_pipeline.py --full --skip-ocr
done
```

### 4. 并行处理多个CSV

修改 `data_pipeline_3b.py` 和 `data_pipeline_7b.py`，使用 `concurrent.futures` 并行处理多个CSV文件。

---

## 贡献与支持

如有问题或建议，请联系开发团队或提交Issue。

---

## 更新日志

### v1.0.0 (2026-01-04)
- ✨ 初始版本发布
- ✨ 集成3B和7B模型的完整管道
- ✨ 动态Prompt生成和实时Median追踪
- ✨ 自适应相似度阈值
- ✨ 完整的自动化运行器

---

**祝使用愉快！🎉**

