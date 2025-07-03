#!/usr/bin/env python3
"""
GENIUS-Rec 实验结果分析器
========================

这个脚本用于分析和可视化实验结果，生成科研报告
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_dir: str = "experiments"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_experiment_results(self, results_file: str = None):
        """加载实验结果"""
        if results_file is None:
            # 自动找到最新的结果文件
            result_files = list(self.results_dir.glob("experiment_results_*.json"))
            if not result_files:
                print("❌ 未找到实验结果文件")
                return None
            results_file = str(max(result_files, key=lambda x: x.stat().st_mtime))
        
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_expert_ablation(self, results: dict):
        """分析专家系统消融实验结果"""
        print("🧠 分析专家系统消融实验...")
        
        expert_results = {}
        for exp_name, result in results.items():
            if exp_name.startswith("expert_ablation") and result.get("status") == "success":
                config_name = exp_name.replace("expert_ablation_", "")
                metrics = result.get("metrics", {})
                expert_results[config_name] = {
                    "hr": metrics.get("test_hr", 0),
                    "ndcg": metrics.get("test_ndcg", 0),
                    "val_loss": metrics.get("best_val_loss", float('inf'))
                }
        
        if not expert_results:
            print("⚠️ 没有找到成功的专家系统消融实验结果")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(expert_results).T
        
        # 绘制专家系统对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # HR@10对比
        df['hr'].plot(kind='bar', ax=axes[0], title='HR@10 by Expert Configuration')
        axes[0].set_ylabel('HR@10')
        axes[0].tick_params(axis='x', rotation=45)
        
        # NDCG@10对比
        df['ndcg'].plot(kind='bar', ax=axes[1], title='NDCG@10 by Expert Configuration')
        axes[1].set_ylabel('NDCG@10')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 验证损失对比
        df['val_loss'].plot(kind='bar', ax=axes[2], title='Validation Loss by Expert Configuration')
        axes[2].set_ylabel('Validation Loss')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_ablation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细分析
        analysis_file = self.output_dir / 'expert_ablation_analysis.txt'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("# 专家系统消融实验分析\n\n")
            f.write(f"分析时间: {datetime.now()}\n\n")
            
            # 排序结果
            hr_ranking = df.sort_values('hr', ascending=False)
            ndcg_ranking = df.sort_values('ndcg', ascending=False)
            
            f.write("## HR@10 排名:\n")
            for i, (config, row) in enumerate(hr_ranking.iterrows(), 1):
                f.write(f"{i}. {config}: {row['hr']:.4f}\n")
            
            f.write("\n## NDCG@10 排名:\n")
            for i, (config, row) in enumerate(ndcg_ranking.iterrows(), 1):
                f.write(f"{i}. {config}: {row['ndcg']:.4f}\n")
            
            # 计算提升
            if 'behavior_only' in df.index and 'all_experts' in df.index:
                baseline_hr = df.loc['behavior_only', 'hr']
                best_hr = df.loc['all_experts', 'hr']
                improvement = ((best_hr - baseline_hr) / baseline_hr) * 100
                f.write(f"\n## 性能提升:\n")
                f.write(f"全专家系统相对仅行为专家提升: {improvement:.2f}%\n")
        
        print(f"✅ 专家系统分析完成，结果保存到: {analysis_file}")
        return df
    
    def analyze_architecture_experiments(self, results: dict):
        """分析架构配置实验结果"""
        print("🏗️ 分析架构配置实验...")
        
        arch_results = {}
        for exp_name, result in results.items():
            if exp_name.startswith("architecture") and result.get("status") == "success":
                config_name = exp_name.replace("architecture_", "")
                metrics = result.get("metrics", {})
                arch_results[config_name] = {
                    "hr": metrics.get("test_hr", 0),
                    "ndcg": metrics.get("test_ndcg", 0),
                    "val_loss": metrics.get("best_val_loss", float('inf'))
                }
        
        if not arch_results:
            print("⚠️ 没有找到成功的架构配置实验结果")
            return
        
        df = pd.DataFrame(arch_results).T
        
        # 绘制架构对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        df[['hr', 'ndcg']].plot(kind='bar', ax=axes[0], title='Performance by Architecture')
        axes[0].set_ylabel('Score')
        axes[0].legend(['HR@10', 'NDCG@10'])
        axes[0].tick_params(axis='x', rotation=45)
        
        df['val_loss'].plot(kind='bar', ax=axes[1], title='Validation Loss by Architecture')
        axes[1].set_ylabel('Validation Loss')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 架构分析完成")
        return df
    
    def generate_comprehensive_report(self, results: dict):
        """生成综合报告"""
        print("📋 生成综合实验报告...")
        
        report_file = self.output_dir / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# GENIUS-Rec 综合实验报告\n\n")
            f.write(f"报告生成时间: {datetime.now()}\n\n")
            
            f.write("## 📊 实验概览\n\n")
            
            # 统计实验数量
            total_experiments = len(results)
            successful_experiments = sum(1 for r in results.values() if r.get("status") == "success")
            failed_experiments = total_experiments - successful_experiments
            
            f.write(f"- 总实验数: {total_experiments}\n")
            f.write(f"- 成功实验: {successful_experiments}\n")
            f.write(f"- 失败实验: {failed_experiments}\n")
            f.write(f"- 成功率: {(successful_experiments/total_experiments)*100:.1f}%\n\n")
            
            # 最佳结果
            best_hr = 0
            best_hr_config = None
            best_ndcg = 0
            best_ndcg_config = None
            
            for exp_name, result in results.items():
                if result.get("status") == "success":
                    metrics = result.get("metrics", {})
                    hr = metrics.get("test_hr", 0)
                    ndcg = metrics.get("test_ndcg", 0)
                    
                    if hr > best_hr:
                        best_hr = hr
                        best_hr_config = exp_name
                    
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_ndcg_config = exp_name
            
            f.write("## 🏆 最佳结果\n\n")
            f.write(f"- **最佳HR@10**: {best_hr:.4f} (配置: {best_hr_config})\n")
            f.write(f"- **最佳NDCG@10**: {best_ndcg:.4f} (配置: {best_ndcg_config})\n\n")
            
            f.write("## 🔍 详细分析\n\n")
            f.write("详细的实验分析请参考以下文件：\n")
            f.write("- `expert_ablation_analysis.txt` - 专家系统消融分析\n")
            f.write("- `expert_ablation_analysis.png` - 专家系统可视化结果\n")
            f.write("- `architecture_analysis.png` - 架构配置可视化结果\n\n")
            
            f.write("## 💡 结论与建议\n\n")
            f.write("基于实验结果，我们得出以下结论：\n\n")
            
            # 分析专家系统效果
            expert_success = any(exp_name.startswith("expert_ablation") and result.get("status") == "success" 
                               for exp_name, result in results.items())
            
            if expert_success:
                f.write("1. **专家系统验证**: 多专家系统实验成功运行，证明了架构的可行性\n")
            else:
                f.write("1. **专家系统**: 需要进一步调试和优化\n")
            
            f.write("2. **性能基准**: 建立了GENIUS-Rec在Amazon Books数据集上的性能基准\n")
            f.write("3. **后续研究**: 建议进一步优化表现最好的配置\n\n")
            
            f.write("## 📈 研究贡献\n\n")
            f.write("本实验验证了以下研究假设：\n")
            f.write("- [ ] 多专家系统在推荐任务中的有效性\n")
            f.write("- [ ] 生成式推荐相对传统方法的优势\n")
            f.write("- [ ] 多模态信息（文本、图像）的价值\n")
            f.write("- [ ] Encoder-Decoder架构在推荐中的适用性\n\n")
        
        print(f"✅ 综合报告已生成: {report_file}")
        return report_file

    def run_full_analysis(self, results_file: str = None):
        """运行完整的结果分析"""
        print("🔬 开始完整的实验结果分析...")
        
        # 加载结果
        results = self.load_experiment_results(results_file)
        if not results:
            return
        
        print(f"📁 加载了 {len(results)} 个实验结果")
        
        # 分析专家系统
        expert_df = self.analyze_expert_ablation(results)
        
        # 分析架构配置
        arch_df = self.analyze_architecture_experiments(results)
        
        # 生成综合报告
        report_file = self.generate_comprehensive_report(results)
        
        print("\n🎉 实验结果分析完成!")
        print(f"📄 查看综合报告: {report_file}")
        print(f"📁 所有分析文件保存在: {self.output_dir}")

def main():
    analyzer = ExperimentAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
