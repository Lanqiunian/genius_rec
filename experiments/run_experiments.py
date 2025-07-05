#!/usr/bin/env python3
"""
GENIUS-Rec 实验管理脚本
======================

这个脚本设计了全面的实验来验证GENIUS-Rec模型的各个组件和假设：

1. 专家系统对比实验
2. 架构配置实验  
3. 超参数敏感性分析
4. 数据增强效果验证
5. 与传统方法对比

使用方法:
    python experiments/run_experiments.py --experiment_suite all
    python experiments/run_experiments.py --experiment_suite expert_ablation
    python experiments/run_experiments.py --experiment_suite hyperparameter_search


    # 1. 首次运行：快速验证（30-60分钟）
    python start_experiments.py --mode quick

    # 2. 深入研究：专家系统分析（2-4小时）
    python start_experiments.py --mode expert

    # 3. 完整评估：所有实验（6-12小时）  
    python start_experiments.py --mode full

    # 4. 结果分析
    python experiments/analyze_results.py
"""

import argparse
import subprocess
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import itertools

class ExperimentRunner:
    def __init__(self, base_dir: str = "/root/autodl-tmp/genius_rec-main"):
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / "experiments"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 创建实验日志
        self.experiment_log = self.experiment_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.setup_logging()
        
        # 基础训练命令
        self.base_cmd = ["python", "-m", "src.train_GeniusRec"]
        self.encoder_weights = "checkpoints/hstu_encoder.pth"
        
        self.results = {}
        
    def setup_logging(self):
        """设置实验日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiment_log),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_single_experiment(self, exp_name: str, args: List[str], save_dir: str = None) -> Dict[str, Any]:
        """运行单个实验"""
        if save_dir is None:
            save_dir = f"experiments/checkpoints/{exp_name}"
            
        # 构建完整命令
        cmd = self.base_cmd + [
            "--encoder_weights_path", self.encoder_weights,
            "--save_dir", save_dir
        ] + args
        
        self.logger.info(f"🚀 开始实验: {exp_name}")
        self.logger.info(f"📋 命令: {' '.join(cmd)}")
        self.logger.info("📈 训练进度将实时显示...")
        
        start_time = time.time()
        captured_output = []
        
        try:
            # 🔧 使用Popen实现实时输出+捕获
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取并显示输出，同时保存
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # 实时显示
                    captured_output.append(output.strip())  # 同时捕获
            
            # 等待进程结束
            return_code = process.wait(timeout=7200)  # 2小时超时
            duration = time.time() - start_time
            
            # 合并捕获的输出
            full_output = '\n'.join(captured_output)
            
            if return_code == 0:
                self.logger.info(f"✅ 实验 {exp_name} 成功完成 (用时: {duration:.1f}s)")
                
                # 尝试解析最终结果
                metrics = self.parse_metrics_from_output(full_output)
                
                return {
                    "status": "success",
                    "duration": duration,
                    "metrics": metrics,
                    "save_dir": save_dir,
                    "args": args
                }
            else:
                self.logger.error(f"❌ 实验 {exp_name} 失败")
                return {
                    "status": "failed",
                    "duration": duration,
                    "return_code": return_code,
                    "args": args
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ 实验 {exp_name} 超时")
            process.kill()
            return {
                "status": "timeout",
                "duration": 7200,
                "args": args
            }
        except Exception as e:
            self.logger.error(f"💥 实验 {exp_name} 异常: {e}")
            return {
                "status": "error",
                "error": str(e),
                "args": args
            }
    
    def parse_metrics_from_output(self, output: str) -> Dict[str, float]:
        """从训练输出中解析最终指标"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            # 解析测试结果 - 适配当前输出格式
            if "Test HR@10:" in line:
                try:
                    hr_value = float(line.split(":")[1].strip())
                    metrics["test_hr"] = hr_value
                except:
                    pass
                    
            if "Test NDCG@10:" in line:
                try:
                    ndcg_value = float(line.split(":")[1].strip())
                    metrics["test_ndcg"] = ndcg_value
                except:
                    pass
                    
            # 解析验证结果
            if "Best validation loss:" in line:
                try:
                    val_loss = float(line.split(":")[1].strip())
                    metrics["best_val_loss"] = val_loss
                except:
                    pass
                    
            # 解析训练完成标志
            if "training finished" in line.lower():
                metrics["training_completed"] = True
        
        return metrics

    def expert_ablation_experiments(self):
        """专家系统消融实验"""
        self.logger.info("🧠 开始专家系统消融实验")
        
        expert_configs = [
            # 单专家实验
            ("behavior_only", ["--disable_content_expert", "--disable_image_expert"]),
            ("content_only", ["--disable_behavior_expert", "--disable_image_expert"]), 
            ("image_only", ["--disable_behavior_expert", "--disable_content_expert", "--enable_image_expert"]),
            
            # 双专家实验
            ("behavior_content", ["--disable_image_expert"]),  # 行为+内容专家
            ("behavior_image", ["--disable_content_expert", "--enable_image_expert"]),
            ("content_image", ["--disable_behavior_expert", "--enable_image_expert"]),
            
            # 全专家实验
            ("all_experts", ["--enable_image_expert"]),
            
            # 传统基线（仅行为专家）
            ("baseline_traditional", ["--disable_content_expert", "--disable_image_expert"]),
        ]
        
        results = {}
        for exp_name, args in expert_configs:
            full_exp_name = f"expert_ablation_{exp_name}"
            result = self.run_single_experiment(
                full_exp_name, 
                args,
                f"experiments/checkpoints/expert_ablation/{exp_name}"
            )
            results[full_exp_name] = result
            
        self.results.update(results)
        return results
    
    def architecture_experiments(self):
        """架构配置实验"""
        self.logger.info("🏗️ 开始架构配置实验")
        
        # 这些需要修改配置文件或创建临时配置
        arch_configs = [
            # 编码器冻结 vs 微调
            ("finetuned_encoder", []),  # 默认微调编码器
            ("frozen_encoder", ["--freeze_encoder"]),  # 冻结编码器
            
            # 不同的解码器层数（需要通过环境变量或配置文件修改）
            ("deep_decoder", []),  # 可以通过配置调整
            ("shallow_decoder", []),
        ]
        
        results = {}
        for exp_name, args in arch_configs:
            full_exp_name = f"architecture_{exp_name}"
            result = self.run_single_experiment(
                full_exp_name,
                args,
                f"experiments/checkpoints/architecture/{exp_name}"
            )
            results[full_exp_name] = result
            
        self.results.update(results)
        return results
    
    def hyperparameter_search(self):
        """超参数搜索实验"""
        self.logger.info("🎛️ 开始超参数搜索实验")
        
        # 由于超参数在配置文件中，这里主要测试可通过命令行控制的参数
        # 可以创建不同的配置文件或使用环境变量
        
        hyperparameter_configs = [
            # 不同的保存目录以避免冲突
            ("baseline_hp", []),
            
            # 这里可以扩展更多超参数
            # 比如通过环境变量设置学习率、batch size等
        ]
        
        results = {}
        for exp_name, args in hyperparameter_configs:
            full_exp_name = f"hyperparameter_{exp_name}"
            result = self.run_single_experiment(
                full_exp_name,
                args,
                f"experiments/checkpoints/hyperparameter/{exp_name}"
            )
            results[full_exp_name] = result
            
        self.results.update(results)
        return results
    
    def data_augmentation_experiments(self):
        """数据增强效果实验"""
        self.logger.info("📚 开始数据增强效果实验")
        
        # 测试图像嵌入的效果 - 使用正确的路径
        data_configs = [
            ("no_image_embeddings", ["--disable_image_expert"]),  # 禁用图像专家
            ("with_image_embeddings", ["--enable_image_expert"]),  # 启用图像专家，使用配置中的默认路径
        ]
        
        results = {}
        for exp_name, args in data_configs:
            full_exp_name = f"data_augmentation_{exp_name}"
            result = self.run_single_experiment(
                full_exp_name,
                args,
                f"experiments/checkpoints/data_augmentation/{exp_name}"
            )
            results[full_exp_name] = result
            
        self.results.update(results)
        return results
    
    def baseline_comparison_experiments(self):
        """与基线方法对比实验"""
        self.logger.info("📊 开始基线对比实验")
        
        # 运行传统基线模型
        baseline_cmd = ["python", "-m", "baseline.train_baseline"]
        
        self.logger.info("🔄 运行传统Transformer基线...")
        
        try:
            result = subprocess.run(
                baseline_cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode == 0:
                baseline_metrics = self.parse_metrics_from_output(result.stdout)
                self.results["baseline_transformer"] = {
                    "status": "success",
                    "metrics": baseline_metrics,
                    "type": "baseline"
                }
                self.logger.info("✅ 基线实验完成")
            else:
                self.logger.error("❌ 基线实验失败")
                self.results["baseline_transformer"] = {
                    "status": "failed",
                    "error": result.stderr,
                    "type": "baseline"
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error("⏰ 基线实验超时")
            self.results["baseline_transformer"] = {
                "status": "timeout",
                "type": "baseline"
            }
    
    def save_results(self):
        """保存实验结果"""
        results_file = self.experiment_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"📄 实验结果已保存到: {results_file}")
        
        # 生成简化的结果报告
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """生成实验结果摘要报告"""
        report_file = self.experiment_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# GENIUS-Rec 实验结果报告\n\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 专家系统消融实验结果
            f.write("## 🧠 专家系统消融实验\n\n")
            f.write("| 实验配置 | 状态 | Test HR@10 | Test NDCG@10 | 最佳验证损失 |\n")
            f.write("|---------|------|------------|--------------|-------------|\n")
            
            for exp_name, result in self.results.items():
                if exp_name.startswith("expert_ablation"):
                    status = result.get("status", "unknown")
                    metrics = result.get("metrics", {})
                    hr = metrics.get("test_hr", "N/A")
                    ndcg = metrics.get("test_ndcg", "N/A") 
                    val_loss = metrics.get("best_val_loss", "N/A")
                    
                    config_name = exp_name.replace("expert_ablation_", "")
                    f.write(f"| {config_name} | {status} | {hr} | {ndcg} | {val_loss} |\n")
            
            f.write("\n## 🏗️ 架构配置实验\n\n")
            f.write("| 架构配置 | 状态 | Test HR@10 | Test NDCG@10 | 最佳验证损失 |\n")
            f.write("|---------|------|------------|--------------|-------------|\n")
            
            for exp_name, result in self.results.items():
                if exp_name.startswith("architecture"):
                    status = result.get("status", "unknown")
                    metrics = result.get("metrics", {})
                    hr = metrics.get("test_hr", "N/A")
                    ndcg = metrics.get("test_ndcg", "N/A")
                    val_loss = metrics.get("best_val_loss", "N/A")
                    
                    config_name = exp_name.replace("architecture_", "")
                    f.write(f"| {config_name} | {status} | {hr} | {ndcg} | {val_loss} |\n")
            
            f.write("\n## 📈 关键发现\n\n")
            f.write("### 最佳配置\n")
            
            # 找到最佳HR@10结果
            best_hr = 0
            best_hr_config = None
            
            for exp_name, result in self.results.items():
                if result.get("status") == "success":
                    metrics = result.get("metrics", {})
                    hr = metrics.get("test_hr", 0)
                    if hr > best_hr:
                        best_hr = hr
                        best_hr_config = exp_name
            
            if best_hr_config:
                f.write(f"- **最佳HR@10**: {best_hr:.4f} (配置: {best_hr_config})\n")
            
            f.write("\n### 专家系统效果分析\n")
            f.write("- [ ] 分析不同专家组合的性能差异\n")
            f.write("- [ ] 验证多模态信息的价值\n")
            f.write("- [ ] 评估计算复杂度 vs 性能提升的权衡\n")
            
        self.logger.info(f"📋 实验摘要报告已生成: {report_file}")

    def quick_validation_experiments(self):
        """快速验证实验（用于调试和快速迭代）"""
        self.logger.info("⚡ 开始快速验证实验")
        
        quick_configs = [
            ("quick_behavior_only", ["--disable_content_expert"]),
            ("quick_all_experts", ["--enable_image_expert"]),
        ]
        
        results = {}
        for exp_name, args in quick_configs:
            # 可以通过环境变量设置更少的epochs用于快速验证
            result = self.run_single_experiment(
                exp_name,
                args,
                f"experiments/checkpoints/quick_validation/{exp_name}"
            )
            results[exp_name] = result
            
        self.results.update(results)
        return results

def main():
    parser = argparse.ArgumentParser(description="GENIUS-Rec 实验管理脚本")
    parser.add_argument(
        "--experiment_suite", 
        choices=["all", "expert_ablation", "architecture", "hyperparameter", "data_augmentation", "baseline_comparison", "quick_validation"],
        default="expert_ablation",
        help="选择要运行的实验套件"
    )
    parser.add_argument("--base_dir", type=str, default="/root/autodl-tmp/genius_rec-main", help="项目根目录")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.base_dir)
    
    runner.logger.info("🎯 GENIUS-Rec 实验开始")
    runner.logger.info(f"📂 项目目录: {args.base_dir}")
    runner.logger.info(f"🧪 实验套件: {args.experiment_suite}")
    
    if args.experiment_suite == "all":
        runner.expert_ablation_experiments()
        runner.architecture_experiments()
        runner.hyperparameter_search()
        runner.data_augmentation_experiments()
        runner.baseline_comparison_experiments()
    elif args.experiment_suite == "expert_ablation":
        runner.expert_ablation_experiments()
    elif args.experiment_suite == "architecture":
        runner.architecture_experiments()
    elif args.experiment_suite == "hyperparameter":
        runner.hyperparameter_search()
    elif args.experiment_suite == "data_augmentation":
        runner.data_augmentation_experiments()
    elif args.experiment_suite == "baseline_comparison":
        runner.baseline_comparison_experiments()
    elif args.experiment_suite == "quick_validation":
        runner.quick_validation_experiments()
    
    runner.save_results()
    runner.logger.info("🎉 所有实验完成!")

if __name__ == "__main__":
    main()
