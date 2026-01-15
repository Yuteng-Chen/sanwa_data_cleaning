# Sanwa Data Preprocessing Pipeline

A comprehensive OCR and data processing pipeline for industrial equipment monitoring, powered by Ollama and Qwen vision models.

## 🚀 Features

- **Intelligent OCR Processing**: Enhanced OCR server with dynamic prompts based on data field types
- **Dual-Model Architecture**: 
  - Qwen2.5VL:3B for fast initial processing and data cleaning
  - Qwen2.5VL:7B for high-accuracy verification and mismatch correction
- **Real-time Median Tracking**: Dynamic context generation based on historical data patterns
- **Dark Background Detection**: Color-based text recognition (red/green status, white numbers)
- **Dual-Image Comparison**: Advanced mismatch correction using previous and current frame comparison
- **Automated Pipeline**: End-to-end processing from raw images to cleaned, labeled datasets
- **GPU Optimization**: Multi-GPU support with configurable parallel processing

## 📋 System Requirements

### Hardware
- **GPU**: 4x NVIDIA V100 (32GB) or equivalent
- **RAM**: 64GB+ recommended
- **Storage**: 100GB+ for models and data

### Software
- Python 3.8+
- CUDA 11.8+ / CUDA 12.x
- Ollama 0.1.0+

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/fangyuanwan/SanwaDataPreprocess.git
cd SanwaDataPreprocess
```

### 2. Quick Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b
```

## 📁 Project Structure

```
├── config_pipeline.py          # Central configuration
├── ocrserver_enhanced.py       # Enhanced OCR server with color detection
├── data_pipeline_3b.py         # 3B model data cleaning pipeline
├── data_pipeline_7b.py         # 7B model verification pipeline
├── run_pipeline.py             # Main pipeline orchestrator
├── get_roi.py                  # ROI extraction utilities
├── setup.sh                    # Automated setup script
├── requirements.txt            # Python dependencies
├── PIPELINE_README.md          # Detailed documentation
├── QUICK_REFERENCE.md          # Quick reference guide
└── Archive/                    # Original scripts (reference only)
```

## ⚙️ Configuration

Edit `config_pipeline.py` to customize:

```python
# GPU Configuration (for 4x V100)
MAX_WORKERS_3B = 16  # 3B model parallel workers
MAX_WORKERS_7B = 8   # 7B model parallel workers

# Data paths
INPUT_CSV = "./input/data.csv"
OUTPUT_DIR = "./output"

# Processing thresholds
OUTLIER_THRESHOLD = 3.0  # Standard deviations
SIMILARITY_THRESHOLD = 0.95  # Redundancy detection
```

## 🚀 Usage

### Quick Start
```bash
# Run the complete pipeline
python run_pipeline.py
```

### Individual Components

**1. Enhanced OCR Server (with color detection)**
```bash
python ocrserver_enhanced.py
```

**2. 3B Model Data Cleaning**
```bash
python data_pipeline_3b.py
```

**3. 7B Model Verification**
```bash
python data_pipeline_7b.py
```

## 📊 Pipeline Workflow

```
Raw Images → OCR Server (3B) → Data Cleaning (3B) → Verification (7B) → Final Dataset
                ↓                      ↓                    ↓
        Color Detection      Outlier Detection    Dual-Image Comparison
        Median Tracking      Type Validation      Mismatch Correction
        Dynamic Prompts      Initial Labeling     Redundancy Removal
```

## 🎯 Key Features Explained

### 1. Dynamic Prompt Generation
- **STATUS fields**: Detects red (NG) / green (OK) text colors
- **INTEGER fields**: Enforces whole number validation
- **FLOAT fields**: Handles decimal precision with median context
- **TIME fields**: Validates time format consistency

### 2. Dual-Image Comparison
For mismatch correction, the system compares:
- Previous frame ROI (expected value)
- Current frame ROI (detected value)
- Applies error detection rules to determine correct value

### 3. Color-based Detection
On dark backgrounds:
- **Red text** → Status: NG
- **Green text** → Status: OK
- **White text** → Numerical values

### 4. Real-time Median Tracking
- Calculates running median for each ROI field
- Provides context to LLM for outlier detection
- Improves accuracy for time-series data

## 📈 Performance Optimization

### GPU Monitoring
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Detailed memory usage
watch -n 1 nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### Tuning Workers
Start with default configuration and adjust based on GPU utilization:
- **GPU util < 80%**: Increase workers (3B: +4, 7B: +2)
- **OOM errors**: Decrease workers (-2 to -4)
- **Optimal range**: 16-24 for 3B, 8-12 for 7B (4x V100)

## 📝 Documentation

- **[PIPELINE_README.md](./PIPELINE_README.md)**: Comprehensive pipeline documentation
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)**: Quick command reference
- **[DARK_BACKGROUND_ENHANCEMENT.md](./DARK_BACKGROUND_ENHANCEMENT.md)**: Color detection details
- **[DUAL_IMAGE_ENHANCEMENT.md](./DUAL_IMAGE_ENHANCEMENT.md)**: Dual-image comparison logic
- **[FINAL_SUMMARY.md](./FINAL_SUMMARY.md)**: Complete feature summary

## 🐛 Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
sudo systemctl restart ollama
```

**2. GPU Memory Issues**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce workers in config_pipeline.py
```

**3. Model Not Found**
```bash
# Re-pull models
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Fangyuan Wan** - Initial work

## 🙏 Acknowledgments

- Ollama team for the excellent local LLM framework
- Qwen team for the powerful vision-language models
- OpenCV community for image processing tools

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation in the `/docs` folder

---

**Last Updated**: January 2026  
**Version**: 1.0.0

