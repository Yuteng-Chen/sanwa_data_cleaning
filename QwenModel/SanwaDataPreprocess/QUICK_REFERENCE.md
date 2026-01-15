# 快速参考卡 Quick Reference Card

## 🚀 快速开始

```bash
# 1. 首次部署
bash setup.sh

# 2. 运行完整管道
python run_pipeline.py --full

# 3. 查看结果
ls pipeline_output/stage6_final_dataset/
```

## 📂 关键文件

| 文件 | 用途 |
|------|------|
| `config_pipeline.py` | 统一配置（目录、模型、阈值） |
| `roi.json` | ROI坐标配置 |
| `ocrserver_enhanced.py` | OCR服务器（Stage 0） |
| `data_pipeline_3b.py` | 3B处理管道（Stage 1-3） |
| `data_pipeline_7b.py` | 7B验证管道（Stage 4-6） |
| `run_pipeline.py` | 自动化运行器 |
| `PIPELINE_README.md` | 完整文档 |

## 🎯 常用命令

```bash
# 完整运行（包括OCR）
python run_pipeline.py --full

# 跳过OCR（使用已有结果）
python run_pipeline.py --full --skip-ocr

# 只运行3B管道
python data_pipeline_3b.py

# 只运行7B管道
python data_pipeline_7b.py

# 只运行OCR服务器
python ocrserver_enhanced.py

# 交互模式
python run_pipeline.py
```

## ⚙️ 关键配置参数

### 目录配置
```python
SERVER_ROOT = Path("/home/ubuntu/sanwa_project")
```

### 模型配置
```python
OLLAMA_MODEL_3B = "qwen2.5vl:3b"
OLLAMA_MODEL_7B = "qwen2.5vl:7b"
MAX_WORKERS_3B = 4  # GPU并行数
MAX_WORKERS_7B = 2
```

### 相似度阈值（每个数据集可不同）
```python
SIMILARITY_THRESHOLDS = {
    "CslotCam4result.csv": 0.85,
    "cam 6 snap1 Latchresult.csv": 0.80,
    "cam 6 snap2 nozzleresult.csv": 0.80,
    "terminal result.csv": 0.90
}
```

## 📊 处理流程

```
图像 → OCR(3B) → 验证 → 3B修正 → 合并 → 
标记 → 7B验证 → 最终整合 → 清洁数据集
```

## 🔍 输出位置

| 阶段 | 输出位置 |
|------|---------|
| Stage 1 | `pipeline_output/stage1_ocr_results/` |
| Stage 2 | `pipeline_output/stage2_cleaned_data/` |
| Stage 3 | `pipeline_output/stage3_3b_corrected/` |
| Stage 4 | `pipeline_output/stage4_labeled/` |
| Stage 5 | `pipeline_output/stage5_7b_verified/` |
| **Stage 6** | **`pipeline_output/stage6_final_dataset/`** ⭐ |

## 🐛 常见问题快速修复

### Ollama连接失败
```bash
systemctl restart ollama
```

### GPU内存不足
```python
# 在 config_pipeline.py 中
MAX_WORKERS_3B = 2  # 降低并行数
```

### 冗余检测太敏感
```python
# 在 config_pipeline.py 中提高阈值
SIMILARITY_THRESHOLDS = {
    "your_csv.csv": 0.90,  # 从0.80提高
}
```

### ROI识别不准
```python
# 在 config_pipeline.py 中
ROI_PAD = 5      # 增加边界
UPSCALE = 3.0    # 增加缩放
```

## 📈 性能参考

- **OCR速度**: ~5-10 images/min (V100)
- **1000张图像**: ~2-4小时完整处理
- **建议批次**: 500-1000张/批

## 🔧 调优建议

### 提高准确率
1. 增加 `UPSCALE` (图像质量)
2. 增加 `ROI_PAD` (上下文信息)
3. 使用7B替代3B模型
4. 调整相似度阈值

### 提高速度
1. 增加 `MAX_WORKERS` (如果GPU内存充足)
2. 降低 `UPSCALE` (如果图像已清晰)
3. 分批处理

### 降低误报
1. 提高 `SIMILARITY_THRESHOLDS`
2. 增加 `OUTLIER_THRESHOLD`
3. 调整 `FROZEN_THRESHOLD_SECONDS`

## 📚 ROI数据类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `STATUS` | OK/NG状态 | OK, NG |
| `INTEGER` | 整数计数 | 123, -5 |
| `FLOAT` | 浮点测量值 | 1.88, -0.52 |
| `TIME` | 时间戳 | 14:35:22 |

## 💡 最佳实践

1. **首次运行**: 先处理小批量（10-50张）测试配置
2. **检查中间结果**: 确认Stage 1-3无误再运行Stage 4-6
3. **保存原始数据**: 永远不要覆盖原始图像和CSV
4. **定期备份**: 备份 `pipeline_output` 目录
5. **监控GPU**: 使用 `nvidia-smi` 监控内存使用
6. **日志记录**: 重定向输出到日志文件

```bash
# 运行并保存日志
python run_pipeline.py --full 2>&1 | tee pipeline_$(date +%Y%m%d_%H%M%S).log
```

## 🆘 获取帮助

```bash
# 查看详细用法
python run_pipeline.py --help-usage

# 查看完整文档
cat PIPELINE_README.md

# 检查配置
python -c "from config_pipeline import *; print(f'SERVER_ROOT: {SERVER_ROOT}')"
```

---

**更多详情请参阅 `PIPELINE_README.md`**

