#!/usr/bin/env python3
"""
GENIUS-Rec 快速实验验证脚本
==========================

这个脚本专注于验证核心研究假设，快速得出结论：

1. 多专家系统是否真的有效？
2. 图像专家的贡献有多大？
3. 内容专家 vs 行为专家哪个更重要？
4. 与传统方法的性能对比如何？

快速实验策略：
- 使用较少的epochs（5-10轮）进行快速验证
- 专注于最关键的配置对比
- 实时报告进度和初步结果
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import logging

class QuickExperimentRunner:
    def __init__(self):
        self.base_dir = Path("/root/autodl-tmp/genius_rec-main")
        self.results_dir = self.base_dir / "experiments" / "quick_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        log_file = self.results_dir / f"quick_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.results = {}
        
    def run_quick_experiment(self, name: str, args: list, max_time: int = 1800):
        """运行单个快速实验（30分钟超时）"""
        
        cmd = [
            "python", "-m", "src.train_GeniusRec",
            "--encoder_weights_path", "checkpoints/hstu_encoder.pth",
            "--save_dir", f"experiments/quick_checkpoints/{name}"
            ] + args
        
        self.logger.info(f"🚀 开始快速实验: {name}")
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
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True
            )
            
            # 实时读取并显示输出，同时保存到captured_output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # 实时显示到终端
                    captured_output.append(output.strip())  # 同时捕获
                    
                    # 🔧 修复：只记录关键信息，避免进度条刷屏
                    if any(keyword in output for keyword in [
                        "Starting Training", "开始训练", "训练完成", "training finished",
                        "Best Val Loss", "HR@", "NDCG@", "实验", "loading", "加载",
                        "✅", "❌", "⚠️"
                    ]):
                        self.logger.info(f"[训练输出] {output.strip()}")
                        
                    # 🔧 修复：只记录每10个epoch的完整进度，避免每个batch都记录
                    if "training finished" in output and "Average Loss" in output:
                        self.logger.info(f"[Epoch完成] {output.strip()}")
            
        if return_code == 0:
            self.logger.info(f"✅ 快速实验 {name} 成功完成 (用时: {duration:.1f}s)")
            
            # 解析结果
            metrics = self.parse_metrics_from_output(full_output)
            
            return {
                "status": "success",
                "duration": duration,
                "metrics": metrics,
                "args": args
            }
        else:
            self.logger.error(f"❌ 快速实验 {name} 失败")
            return {
                "status": "failed", 
                "duration": duration,
                "return_code": return_code,
                "output": full_output[-1000:] if full_output else "No output captured"
            }
            
            if return_code == 0:
                self.logger.info(f"✅ {name} 完成 (用时: {duration:.1f}秒)")
                
                # 从捕获的输出中解析指标
                metrics = self._parse_final_metrics(full_output)
                
                # 如果输出中没有指标，尝试从checkpoint读取
                if not metrics:
                    checkpoint_metrics = self._parse_metrics_from_checkpoint(f"experiments/quick_checkpoints/{name}")
                    metrics.update(checkpoint_metrics)
                
                return {
                    "status": "success",
                    "duration": duration,
                    "metrics": metrics
                }
            else:
                self.logger.error(f"❌ {name} 失败，返回码: {return_code}")
                return {"status": "failed", "return_code": return_code}
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"⏰ {name} 超时，终止进程")
            process.kill()
            return {"status": "timeout"}
        except Exception as e:
            self.logger.error(f"💥 {name} 异常: {e}")
            return {"status": "error", "error": str(e)}
    
    def _parse_final_metrics(self, output: str) -> dict:
        """解析最终指标"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if "Test HR@10:" in line:
                try:
                    metrics['hr'] = float(line.split(':')[1].strip())
                except:
                    pass
            if "Test NDCG@10:" in line:
                try:
                    metrics['ndcg'] = float(line.split(':')[1].strip())
                except:
                    pass
            if "Best Val Loss:" in line:
                try:
                    metrics['val_loss'] = float(line.split(':')[1].strip())
                except:
                    pass
        
        return metrics
    
    def parse_metrics_from_output(self, output: str) -> dict:
        """从训练输出解析关键指标"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            # 解析HR和NDCG指标
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
            # 解析验证损失
            if "Best validation loss:" in line:
                try:
                    metrics["best_val_loss"] = float(line.split(":")[1].strip())
                except:
                    pass
        
        return metrics

    def _parse_metrics_from_checkpoint(self, checkpoint_dir: str) -> dict:
        """从保存的checkpoint文件中读取指标"""
        import torch
        
        metrics = {}
        checkpoint_path = Path(checkpoint_dir) / "genius_rec_best.pth"
        
        try:
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # 尝试从checkpoint中提取指标
                if isinstance(checkpoint, dict):
                    metrics['val_loss'] = checkpoint.get('best_val_loss', 'N/A')
                    metrics['val_ppl'] = checkpoint.get('val_ppl', 'N/A')
                    
                    # 其他可能的指标
                    for key in checkpoint.keys():
                        if 'hr' in key.lower():
                            metrics['hr'] = checkpoint[key]
                        elif 'ndcg' in key.lower():
                            metrics['ndcg'] = checkpoint[key]
                
                self.logger.info(f"📊 从checkpoint读取指标: {metrics}")
            else:
                self.logger.warning(f"⚠️  checkpoint文件未找到: {checkpoint_path}")
                
        except Exception as e:
            self.logger.error(f"❌ 无法读取checkpoint指标: {e}")
            
        return metrics

    def core_ablation_study(self):
        """核心消融研究 - 验证主要假设"""
        
        self.logger.info("\n" + "🎯" + "="*50)
        self.logger.info("🎯 开始核心消融研究")
        self.logger.info("🎯 目标: 验证多专家系统的核心假设")
        self.logger.info("🎯" + "="*50)
        
        experiments = [
            # 基线：仅行为专家（传统方法）
            ("baseline_behavior_only", ["--disable_content_expert", "--disable_image_expert"]),
            
            # 加入内容专家
            ("behavior_plus_content", ["--disable_image_expert"]),  # 行为+内容
            
            # 加入图像专家  
            ("behavior_plus_image", ["--disable_content_expert", "--enable_image_expert"]),  # 行为+图像
            
            # 全专家配置
            ("all_experts", ["--enable_image_expert"]),  # 启用所有专家
        ]
        
        for name, args in experiments:
            self.logger.info(f"\n" + "="*60)
            self.logger.info(f"📊 当前实验: {name}")
            self.logger.info(f"🎯 实验目标: {self._get_experiment_description(name)}")
            self.logger.info(f"⚙️  参数配置: {args if args else '默认配置'}")
            self.logger.info("="*60)
            
            result = self.run_quick_experiment(name, args)
            self.results[name] = result
            
            # 实时报告
            if result.get("status") == "success":
                metrics = result.get("metrics", {})
                hr = metrics.get('hr', 'N/A')
                ndcg = metrics.get('ndcg', 'N/A')
                
                if isinstance(hr, float):
                    hr = f"{hr:.4f}"
                if isinstance(ndcg, float):
                    ndcg = f"{ndcg:.4f}"
                    
                self.logger.info(f"📊 {name} 结果: HR@10={hr}, NDCG@10={ndcg}")
            else:
                self.logger.error(f"❌ {name} 实验失败: {result.get('status', 'unknown error')}")
    def _get_experiment_description(self, name: str) -> str:
        """获取实验描述"""
        descriptions = {
            "baseline_behavior_only": "仅使用行为专家(传统推荐方法基线)",
            "behavior_plus_content": "行为专家 + 内容专家(文本嵌入)", 
            "behavior_plus_image": "行为专家 + 图像专家(视觉嵌入)",
            "all_experts": "全专家系统(行为+内容+图像)",
            "single_only_behavior": "单独测试行为专家",
            "single_only_content": "单独测试内容专家", 
            "single_only_image": "单独测试图像专家"
        }
        return descriptions.get(name, "未知实验")
    
    def expert_importance_ranking(self):
        """专家重要性排序实验"""
        
        self.logger.info("\n" + "🏆" + "="*50)
        self.logger.info("🏆 开始专家重要性排序实验")
        self.logger.info("🏆" + "="*50)
        
        single_expert_experiments = [
            ("only_behavior", ["--disable_content_expert", "--disable_image_expert"]),
            ("only_content", ["--disable_behavior_expert", "--disable_image_expert"]),
            ("only_image", ["--disable_behavior_expert", "--disable_content_expert"]),
        ]
        
        for name, args in single_expert_experiments:
            experiment_name = f"single_{name}"
            self.logger.info(f"\n" + "="*60)
            self.logger.info(f"📊 当前实验: {experiment_name}")
            self.logger.info(f"🎯 实验目标: {self._get_experiment_description(experiment_name)}")
            self.logger.info(f"⚙️  参数配置: {args}")
            self.logger.info("="*60)
            
            result = self.run_quick_experiment(experiment_name, args)
            self.results[experiment_name] = result
            
            # 实时报告
            if result.get("status") == "success":
                metrics = result.get("metrics", {})
                hr = metrics.get('hr', 'N/A')
                ndcg = metrics.get('ndcg', 'N/A')
                
                if isinstance(hr, float):
                    hr = f"{hr:.4f}"
                if isinstance(ndcg, float):
                    ndcg = f"{ndcg:.4f}"
                    
                self.logger.info(f"📊 {experiment_name} 结果: HR@10={hr}, NDCG@10={ndcg}")
            else:
                self.logger.error(f"❌ {experiment_name} 实验失败: {result.get('status', 'unknown error')}")
    
    def generate_quick_report(self):
        """生成快速报告"""
        
        report_file = self.results_dir / f"quick_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# GENIUS-Rec 快速实验报告\n\n")
            f.write(f"实验时间: {datetime.now()}\n\n")
            
            f.write("## 🎯 核心假设验证\n\n")
            f.write("| 配置 | 状态 | HR@10 | NDCG@10 | 备注 |\n")
            f.write("|------|------|-------|---------|------|\n")
            
            # 核心实验结果
            core_experiments = ["baseline_behavior_only", "behavior_plus_content", "behavior_plus_image", "all_experts"]
            
            for exp_name in core_experiments:
                if exp_name in self.results:
                    result = self.results[exp_name]
                    status = result.get("status", "unknown")
                    metrics = result.get("metrics", {})
                    hr = metrics.get("hr", "N/A")
                    ndcg = metrics.get("ndcg", "N/A")
                    
                    if isinstance(hr, float):
                        hr = f"{hr:.4f}"
                    if isinstance(ndcg, float):
                        ndcg = f"{ndcg:.4f}"
                    
                    note = ""
                    if exp_name == "baseline_behavior_only":
                        note = "传统方法基线"
                    elif exp_name == "all_experts":
                        note = "完整GENIUS-Rec"
                    
                    f.write(f"| {exp_name} | {status} | {hr} | {ndcg} | {note} |\n")
            
            f.write("1. **深入超参数优化**: 针对表现最好的配置进行精细调优\n")
            f.write("2. **更长时间训练**: 将有希望的配置训练更多epochs\n")
            f.write("3. **数据增强**: 探索更多数据增强策略\n")
            f.write("4. **错误分析**: 分析模型在哪些情况下表现不佳\n")
        
        self.logger.info(f"📋 快速报告已生成: {report_file}")
        
        # 保存原始结果数据
        results_file = self.results_dir / f"quick_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"💾 原始结果已保存: {results_file}")
    
    def run_all_quick_experiments(self):
        """运行所有快速实验"""
        
        self.logger.info("\n" + "🏁" + "="*60)
        self.logger.info("🏁 GENIUS-Rec 快速验证实验套件启动")
        self.logger.info("🏁 实验包括: 核心消融研究 + 专家重要性排序")
        self.logger.info("🏁 预计总时间: 30-60分钟")
        self.logger.info("🏁" + "="*60)
        
        start_time = time.time()
        
        # 1. 核心消融研究
        self.core_ablation_study()
        
        # 2. 专家重要性排序  
        self.expert_importance_ranking()
        
        total_time = time.time() - start_time
        self.logger.info(f"⏱️ 所有快速实验完成，总用时: {total_time/60:.1f} 分钟")
        
        # 3. 生成报告
        self.generate_quick_report()
        
        # 4. 实时总结
        self._print_quick_summary()
    
    def _print_quick_summary(self):
        """打印快速总结"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 GENIUS-Rec 快速实验总结")
        self.logger.info("="*60)
        
        successful_experiments = [name for name, result in self.results.items() 
                                if result.get("status") == "success"]
        
        self.logger.info(f"✅ 成功完成实验: {len(successful_experiments)}/{len(self.results)}")
        
        if successful_experiments:
            self.logger.info("\n📊 主要结果:")
            
            for exp_name in ["baseline_behavior_only", "all_experts"]:
                if exp_name in self.results and self.results[exp_name].get("status") == "success":
                    metrics = self.results[exp_name].get("metrics", {})
                    hr = metrics.get("hr", "N/A")
                    ndcg = metrics.get("ndcg", "N/A")
                    
                    if isinstance(hr, float):
                        hr = f"{hr:.4f}"
                    if isinstance(ndcg, float):
                        ndcg = f"{ndcg:.4f}"
                    
                    self.logger.info(f"   {exp_name}: HR@10={hr}, NDCG@10={ndcg}")
        
        self.logger.info("\n📋 详细报告已保存到 experiments/quick_results/ 目录")
        self.logger.info("💡 建议查看生成的Markdown报告了解详细分析")

def main():
    runner = QuickExperimentRunner()
    runner.run_all_quick_experiments()

if __name__ == "__main__":
    main()
