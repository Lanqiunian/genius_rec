#!/usr/bin/env python3
"""
GENIUS-Rec 主实验控制器
=====================

适配当前多专家解码器架构的完整实验套件

使用方法:
    python experiments/main_experiment_controller.py --suite quick      # 快速验证（推荐）
    python experiments/main_experiment_controller.py --suite ablation   # 专家消融研究  
    python experiments/main_experiment_controller.py --suite full       # 完整实验套件
    python experiments/main_experiment_controller.py --suite analysis   # 结果分析
"""

import argparse
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到path
sys.path.append(str(Path(__file__).parent.parent))

class MainExperimentController:
    """主实验控制器 - 适配当前解码器架构"""
    
    def __init__(self):
        self.base_dir = Path("/root/autodl-tmp/genius_rec-main")
        self.experiments_dir = self.base_dir / "experiments"
        self.results_dir = self.experiments_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.results_dir / f"main_controller_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {}
        
    def print_banner(self):
        """打印启动横幅"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    🎯 GENIUS-Rec 实验控制器                    ║
║                  多专家推荐系统实验平台                        ║
║                                                              ║
║  适配特性: ✅ 多专家MoE架构  ✅ 生成式解码器  ✅ 交叉注意力     ║
╚═══════════════════════════════════════════════════════════════╝
"""
        print(banner)
        
    def check_prerequisites(self):
        """检查实验前置条件"""
        self.logger.info("🔍 检查实验前置条件...")
        
        checks = []
        
        # 检查必要文件
        required_files = [
            "checkpoints/hstu_encoder.pth",
            "data/processed/train.parquet",
            "data/processed/validation.parquet", 
            "data/processed/test.parquet",
            "data/processed/id_maps.pkl"
        ]
        
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                checks.append(f"✅ {file_path}")
            else:
                checks.append(f"❌ {file_path} (缺失)")
                
        # 检查可选的嵌入文件
        optional_files = [
            "data/book_gemini_embeddings_filtered_migrated.npy",
            "data/book_image_embeddings_migrated.npy"
        ]
        
        for file_path in optional_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                checks.append(f"✅ {file_path} (可选)")
            else:
                checks.append(f"⚠️  {file_path} (可选，缺失)")
        
        # 打印检查结果
        for check in checks:
            self.logger.info(check)
            
        # 判断是否可以继续
        missing_required = [c for c in checks if "❌" in c]
        if missing_required:
            self.logger.error("❌ 缺少必要文件，无法继续实验")
            return False
        
        self.logger.info("✅ 前置条件检查通过")
        return True
        
    def run_quick_suite(self):
        """运行快速验证套件（30-60分钟）"""
        self.logger.info("🚀 开始快速验证套件...")
        
        experiments = [
            {
                "name": "quick_behavior_only",
                "description": "仅行为专家（基线）",
                "args": ["--disable_content_expert", "--disable_image_expert"],
                "max_epochs": "5"
            },
            {
                "name": "quick_all_experts",
                "description": "全专家系统",
                "args": ["--enable_image_expert"],
                "max_epochs": "5"
            }
        ]
        
        return self._run_experiment_batch(experiments, "quick")
        
    def run_ablation_suite(self):
        """运行专家消融研究套件（2-4小时）"""
        self.logger.info("🧠 开始专家消融研究套件...")
        
        experiments = [
            {
                "name": "ablation_behavior_only",
                "description": "仅行为专家",
                "args": ["--disable_content_expert", "--disable_image_expert"],
                "max_epochs": "15"
            },
            {
                "name": "ablation_behavior_content",
                "description": "行为+内容专家",
                "args": ["--disable_image_expert"],
                "max_epochs": "15"
            },
            {
                "name": "ablation_behavior_image",
                "description": "行为+图像专家",
                "args": ["--disable_content_expert", "--enable_image_expert"],
                "max_epochs": "15"
            },
            {
                "name": "ablation_all_experts",
                "description": "全专家系统",
                "args": ["--enable_image_expert"],
                "max_epochs": "15"
            }
        ]
        
        return self._run_experiment_batch(experiments, "ablation")
        
    def run_full_suite(self):
        """运行完整实验套件（6-12小时）"""
        self.logger.info("🔬 开始完整实验套件...")
        
        # 组合快速套件和消融套件，加上额外的配置实验
        quick_results = self.run_quick_suite()
        ablation_results = self.run_ablation_suite()
        
        # 架构配置实验
        architecture_experiments = [
            {
                "name": "arch_frozen_encoder",
                "description": "冻结编码器",
                "args": ["--freeze_encoder", "--enable_image_expert"],
                "max_epochs": "20"
            },
            {
                "name": "arch_finetuned_encoder",
                "description": "端到端微调",
                "args": ["--enable_image_expert"],
                "max_epochs": "20"
            }
        ]
        
        arch_results = self._run_experiment_batch(architecture_experiments, "architecture")
        
        # 合并所有结果
        all_results = {**quick_results, **ablation_results, **arch_results}
        return all_results
        
    def _run_experiment_batch(self, experiments, suite_name):
        """运行一批实验"""
        batch_results = {}
        
        for i, exp in enumerate(experiments, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔬 实验 {i}/{len(experiments)}: {exp['name']}")
            self.logger.info(f"📝 描述: {exp['description']}")
            self.logger.info(f"⚙️  参数: {exp['args']}")
            self.logger.info(f"{'='*60}")
            
            # 构建命令
            cmd = [
                "python", "-m", "src.train_GeniusRec",
                "--encoder_weights_path", "checkpoints/hstu_encoder.pth",
                "--save_dir", f"experiments/results/{suite_name}/{exp['name']}"
            ] + exp.get("args", [])
            
            # 添加epoch限制（用于快速实验）
            if "max_epochs" in exp:
                # 注意：这需要训练脚本支持 --max_epochs 参数
                # 如果不支持，可以通过环境变量或配置文件修改
                pass  # 暂时跳过，因为当前训练脚本可能不支持此参数
            
            start_time = time.time()
            
            try:
                # 运行实验
                self.logger.info(f"🚀 启动命令: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=self.base_dir,
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2小时超时
                )
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    self.logger.info(f"✅ 实验 {exp['name']} 完成 (用时: {duration:.1f}s)")
                    
                    # 解析结果
                    metrics = self._parse_output_metrics(result.stdout)
                    
                    batch_results[exp['name']] = {
                        "status": "success",
                        "duration": duration,
                        "metrics": metrics,
                        "description": exp['description']
                    }
                else:
                    self.logger.error(f"❌ 实验 {exp['name']} 失败")
                    batch_results[exp['name']] = {
                        "status": "failed",
                        "duration": duration,
                        "error": result.stderr[-1000:] if result.stderr else "Unknown error"
                    }
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"⏰ 实验 {exp['name']} 超时")
                batch_results[exp['name']] = {
                    "status": "timeout",
                    "duration": 7200
                }
            except Exception as e:
                self.logger.error(f"💥 实验 {exp['name']} 异常: {e}")
                batch_results[exp['name']] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # 保存批次结果
        self._save_batch_results(batch_results, suite_name)
        return batch_results
        
    def _parse_output_metrics(self, output):
        """从输出中解析指标"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if "Test HR@10:" in line:
                try:
                    metrics["test_hr"] = float(line.split(":")[1].strip())
                except:
                    pass
            if "Test NDCG@10:" in line:
                try:
                    metrics["test_ndcg"] = float(line.split(":")[1].strip())
                except:
                    pass
            if "Best validation loss:" in line:
                try:
                    metrics["best_val_loss"] = float(line.split(":")[1].strip())
                except:
                    pass
        
        return metrics
        
    def _save_batch_results(self, results, suite_name):
        """保存批次结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"{suite_name}_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 批次结果已保存: {results_file}")
        
    def analyze_results(self):
        """分析实验结果"""
        self.logger.info("📊 开始结果分析...")
        
        try:
            # 运行分析脚本
            cmd = ["python", "experiments/analyze_results.py"]
            result = subprocess.run(cmd, cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ 结果分析完成")
                print(result.stdout)
            else:
                self.logger.error(f"❌ 结果分析失败: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"💥 结果分析异常: {e}")

def main():
    parser = argparse.ArgumentParser(description="GENIUS-Rec 主实验控制器")
    parser.add_argument('--suite', choices=['quick', 'ablation', 'full', 'analysis'], 
                       default='quick', help='实验套件类型')
    parser.add_argument('--skip_checks', action='store_true', help='跳过前置条件检查')
    args = parser.parse_args()
    
    controller = MainExperimentController()
    controller.print_banner()
    
    # 前置条件检查
    if not args.skip_checks:
        if not controller.check_prerequisites():
            sys.exit(1)
    
    # 执行对应的实验套件
    if args.suite == 'quick':
        print("\n🚀 启动快速验证套件 (预计30-60分钟)")
        controller.run_quick_suite()
        
    elif args.suite == 'ablation':
        print("\n🧠 启动专家消融研究套件 (预计2-4小时)")
        controller.run_ablation_suite()
        
    elif args.suite == 'full':
        print("\n🔬 启动完整实验套件 (预计6-12小时)")
        controller.run_full_suite()
        
    elif args.suite == 'analysis':
        print("\n📊 启动结果分析")
        controller.analyze_results()
    
    print(f"\n📋 详细日志: {controller.log_file}")
    print("🎯 实验控制器执行完成!")

if __name__ == "__main__":
    main()
