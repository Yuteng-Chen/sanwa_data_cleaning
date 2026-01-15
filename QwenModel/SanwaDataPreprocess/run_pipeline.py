#!/usr/bin/env python3
"""
完整数据处理自动化运行器
Full Data Processing Automation Runner

此脚本协调所有管道阶段的执行
This script coordinates execution of all pipeline stages
"""

import sys
import time
from pathlib import Path

# 导入配置和各阶段
from config_pipeline import *

def print_banner(text):
    """打印横幅"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_prerequisites():
    """检查前置条件"""
    print_banner("🔍 CHECKING PREREQUISITES")
    
    # 检查配置文件
    if not ROI_JSON.exists():
        print(f"❌ Error: {ROI_JSON} not found!")
        print("   Please create roi.json with your ROI configuration")
        return False
    
    # 检查Ollama模型
    try:
        import ollama
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        
        if OLLAMA_MODEL_3B not in model_names:
            print(f"⚠️  Warning: {OLLAMA_MODEL_3B} not found in Ollama")
            print(f"   Run: ollama pull {OLLAMA_MODEL_3B}")
        
        if OLLAMA_MODEL_7B not in model_names:
            print(f"⚠️  Warning: {OLLAMA_MODEL_7B} not found in Ollama")
            print(f"   Run: ollama pull {OLLAMA_MODEL_7B}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not check Ollama models: {e}")
    
    # 创建目录
    create_directories()
    
    print("✅ Prerequisites check complete\n")
    return True

def run_stage(stage_name, stage_function):
    """运行单个阶段"""
    print_banner(f"▶️  {stage_name}")
    start_time = time.time()
    
    try:
        stage_function()
        duration = time.time() - start_time
        print(f"\n✅ {stage_name} completed in {duration:.1f}s")
        return True
    except KeyboardInterrupt:
        print(f"\n🛑 {stage_name} interrupted by user")
        raise
    except Exception as e:
        print(f"\n❌ {stage_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_pipeline(skip_ocr=False):
    """运行完整管道"""
    print_banner("🚀 STARTING FULL DATA PROCESSING PIPELINE")
    
    overall_start = time.time()
    
    # 阶段0: OCR服务器（可选）
    if not skip_ocr:
        print_banner("STAGE 0: Enhanced OCR Server (3B)")
        print("ℹ️  This stage processes raw images and generates initial OCR results")
        print("ℹ️  If you already have OCR results in Stage 1, you can skip this")
        
        response = input("\n⏸️  Run OCR Server? (y/n, default=n): ").strip().lower()
        
        if response == 'y':
            print("\n🔄 Starting OCR Server...")
            print("   Press Ctrl+C when all images are processed")
            try:
                import ocrserver_enhanced
                ocrserver_enhanced.main()
            except KeyboardInterrupt:
                print("\n✅ OCR Server stopped")
        else:
            print("⏭️  Skipping OCR Server")
    
    # 阶段1-3: 3B管道
    print_banner("🤖 3B MODEL PIPELINE (Stages 1-3)")
    
    try:
        import data_pipeline_3b
        if not run_stage("3B Pipeline", data_pipeline_3b.main):
            print("\n❌ 3B Pipeline failed. Cannot continue.")
            return False
    except Exception as e:
        print(f"\n❌ Error importing 3B pipeline: {e}")
        return False
    
    # 阶段4-6: 7B管道
    print_banner("🤖 7B MODEL PIPELINE (Stages 4-6)")
    
    try:
        import data_pipeline_7b
        if not run_stage("7B Pipeline", data_pipeline_7b.main):
            print("\n❌ 7B Pipeline failed.")
            return False
    except Exception as e:
        print(f"\n❌ Error importing 7B pipeline: {e}")
        return False
    
    # 完成
    overall_duration = time.time() - overall_start
    
    print_banner("🎉 PIPELINE COMPLETE!")
    print(f"⏱️  Total Time: {overall_duration/60:.1f} minutes")
    print(f"\n📂 Final Output Location:")
    print(f"   {STAGE_6_FINAL}")
    print(f"\n📊 Processing Summary:")
    print(f"   - Stage 1 OCR Results: {STAGE_1_OCR}")
    print(f"   - Stage 2 Cleaned Data: {STAGE_2_CLEANED}")
    print(f"   - Stage 3 3B Corrected: {STAGE_3_3B_CORRECTED}")
    print(f"   - Stage 4 Labeled Data: {STAGE_4_LABELED}")
    print(f"   - Stage 5 7B Verified: {STAGE_5_7B_VERIFIED}")
    print(f"   - Stage 6 Final Dataset: {STAGE_6_FINAL}")
    
    return True

def run_specific_stage(stage_number):
    """运行特定阶段"""
    print_banner(f"▶️  RUNNING SPECIFIC STAGE: {stage_number}")
    
    if stage_number == 0:
        import ocrserver_enhanced
        ocrserver_enhanced.main()
    elif stage_number in [1, 2, 3]:
        import data_pipeline_3b
        data_pipeline_3b.main()
    elif stage_number in [4, 5, 6]:
        import data_pipeline_7b
        data_pipeline_7b.main()
    else:
        print(f"❌ Invalid stage number: {stage_number}")
        return False
    
    return True

def show_usage():
    """显示使用说明"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    DATA PROCESSING PIPELINE RUNNER                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python run_pipeline.py [options]

Options:
    --full              Run full pipeline (all stages)
    --skip-ocr          Skip OCR stage (use existing results)
    --stage N           Run specific stage (0-6)
    --help              Show this help message

Stages:
    Stage 0: Enhanced OCR Server (3B) - Process raw images
    Stage 1: Data Validation & Cleaning
    Stage 2: 3B Model Correction
    Stage 3: Merge 3B Corrections
    Stage 4: Data Labeling (Time/Redundancy)
    Stage 5: 7B Model Verification
    Stage 6: Final Consolidation

Examples:
    # Run full pipeline (interactive)
    python run_pipeline.py --full
    
    # Run full pipeline, skip OCR (use existing results)
    python run_pipeline.py --full --skip-ocr
    
    # Run only 3B pipeline (stages 1-3)
    python run_pipeline.py --stage 1
    
    # Run only 7B pipeline (stages 4-6)
    python run_pipeline.py --stage 4

Configuration:
    Edit config_pipeline.py to customize:
    - Directory paths
    - Model names
    - ROI configurations
    - Processing thresholds

""")

def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Processing Pipeline Runner')
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--skip-ocr', action='store_true', help='Skip OCR stage')
    parser.add_argument('--stage', type=int, metavar='N', help='Run specific stage (0-6)')
    parser.add_argument('--help-usage', action='store_true', help='Show detailed usage')
    
    args = parser.parse_args()
    
    if args.help_usage:
        show_usage()
        return
    
    # 检查前置条件
    if not check_prerequisites():
        sys.exit(1)
    
    try:
        if args.full:
            success = run_full_pipeline(skip_ocr=args.skip_ocr)
            sys.exit(0 if success else 1)
        elif args.stage is not None:
            success = run_specific_stage(args.stage)
            sys.exit(0 if success else 1)
        else:
            # 交互模式
            print_banner("🤖 INTERACTIVE MODE")
            print("1. Run Full Pipeline")
            print("2. Run 3B Pipeline Only (Stages 1-3)")
            print("3. Run 7B Pipeline Only (Stages 4-6)")
            print("4. Run OCR Server Only (Stage 0)")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                run_full_pipeline()
            elif choice == '2':
                import data_pipeline_3b
                data_pipeline_3b.main()
            elif choice == '3':
                import data_pipeline_7b
                data_pipeline_7b.main()
            elif choice == '4':
                import ocrserver_enhanced
                ocrserver_enhanced.main()
            elif choice == '5':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

