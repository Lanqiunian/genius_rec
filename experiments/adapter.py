#!/usr/bin/env python3
"""
GENIUS-Rec 实验适配脚本
=====================

此脚本专门用于适配当前的多专家解码器架构，确保实验配置与代码状态一致。

主要功能：
1. 验证当前解码器状态
2. 检查专家系统配置
3. 运行适配后的实验
4. 生成兼容性报告
"""

import sys
import torch
import logging
from pathlib import Path
import subprocess
import json
from datetime import datetime

# 添加项目根目录到path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.decoder.decoder import GenerativeDecoder

class ExperimentAdapter:
    """实验适配器 - 确保实验与当前代码状态兼容"""
    
    def __init__(self):
        self.config = get_config()
        self.base_dir = Path("/root/autodl-tmp/genius_rec-main")
        self.adapter_log = self.base_dir / "logs" / f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.adapter_log),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def verify_decoder_compatibility(self):
        """验证解码器兼容性"""
        self.logger.info("🔍 验证解码器兼容性...")
        
        try:
            expert_config = self.config.get('expert_system', {})
            
            # 测试解码器初始化
            decoder = GenerativeDecoder(
                num_items=1000,  # 测试值
                embedding_dim=64,
                num_layers=4,
                num_heads=4,
                ffn_hidden_dim=256,
                max_seq_len=50,
                expert_config=expert_config
            )
            
            self.logger.info("✅ 解码器初始化成功")
            
            # 检查启用的专家
            enabled_experts = decoder.enabled_experts
            self.logger.info(f"📋 当前启用的专家: {enabled_experts}")
            
            # 测试前向传播
            batch_size, seq_len = 2, 10
            target_ids = torch.randint(0, 1000, (batch_size, seq_len))
            encoder_output = torch.randn(batch_size, seq_len, 64)
            memory_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            
            with torch.no_grad():
                logits, weights, balancing_loss = decoder(
                    target_ids, encoder_output, memory_padding_mask,
                    return_weights=True
                )
            
            self.logger.info(f"✅ 前向传播测试成功 - logits shape: {logits.shape}")
            self.logger.info(f"✅ 专家权重 shape: {weights.shape if weights is not None else 'None'}")
            self.logger.info(f"✅ 平衡损失: {balancing_loss.item()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 解码器兼容性验证失败: {e}")
            return False
    
    def check_expert_system_config(self):
        """检查专家系统配置"""
        self.logger.info("🧠 检查专家系统配置...")
        
        expert_config = self.config.get('expert_system', {})
        experts = expert_config.get('experts', {})
        
        self.logger.info("📋 专家配置状态:")
        for expert_name, enabled in experts.items():
            status = "✅ 启用" if enabled else "❌ 禁用"
            self.logger.info(f"  - {expert_name}: {status}")
        
        # 检查必要的嵌入路径
        data_dir = self.config['data']['data_dir']
        
        if experts.get('content_expert', False):
            text_embedding_path = data_dir / "book_gemini_embeddings_filtered_migrated.npy"
            if text_embedding_path.exists():
                self.logger.info("✅ 文本嵌入文件存在")
            else:
                self.logger.warning(f"⚠️  文本嵌入文件缺失: {text_embedding_path}")
        
        if experts.get('image_expert', False):
            image_embedding_path = data_dir / "book_image_embeddings_migrated.npy"
            if image_embedding_path.exists():
                self.logger.info("✅ 图像嵌入文件存在")
            else:
                self.logger.warning(f"⚠️  图像嵌入文件缺失: {image_embedding_path}")
        
        return expert_config
    
    def generate_compatible_experiment_configs(self):
        """生成兼容的实验配置"""
        self.logger.info("⚙️  生成兼容的实验配置...")
        
        # 基于当前解码器状态的实验配置
        experiment_configs = {
            "core_ablation": [
                {
                    "name": "baseline_behavior_only",
                    "description": "仅行为专家（传统推荐基线）",
                    "args": ["--disable_content_expert", "--disable_image_expert"],
                    "expected_experts": ["behavior_expert"]
                },
                {
                    "name": "behavior_plus_content", 
                    "description": "行为专家 + 内容专家",
                    "args": ["--disable_image_expert"],
                    "expected_experts": ["behavior_expert", "content_expert"]
                },
                {
                    "name": "behavior_plus_image",
                    "description": "行为专家 + 图像专家",
                    "args": ["--disable_content_expert", "--enable_image_expert"],
                    "expected_experts": ["behavior_expert", "image_expert"]
                },
                {
                    "name": "all_experts",
                    "description": "全专家系统",
                    "args": ["--enable_image_expert"],
                    "expected_experts": ["behavior_expert", "content_expert", "image_expert"]
                }
            ],
            
            "architecture_tests": [
                {
                    "name": "frozen_encoder",
                    "description": "冻结编码器测试",
                    "args": ["--freeze_encoder", "--enable_image_expert"],
                    "expected_experts": ["behavior_expert", "content_expert", "image_expert"]
                },
                {
                    "name": "finetuned_encoder",
                    "description": "端到端微调测试",
                    "args": ["--enable_image_expert"],
                    "expected_experts": ["behavior_expert", "content_expert", "image_expert"]
                }
            ]
        }
        
        # 保存配置
        config_file = self.base_dir / "experiments" / "adapted_configs.json"
        with open(config_file, 'w') as f:
            json.dump(experiment_configs, f, indent=2)
        
        self.logger.info(f"✅ 实验配置已保存至: {config_file}")
        return experiment_configs
    
    def run_compatibility_check(self):
        """运行完整的兼容性检查"""
        self.logger.info("🔧 开始兼容性检查...")
        
        checks = [
            ("解码器兼容性", self.verify_decoder_compatibility),
            ("专家系统配置", lambda: self.check_expert_system_config() is not None),
            ("实验配置生成", lambda: self.generate_compatible_experiment_configs() is not None)
        ]
        
        results = {}
        for check_name, check_func in checks:
            try:
                result = check_func()
                results[check_name] = "✅ 通过" if result else "❌ 失败"
                self.logger.info(f"{check_name}: {results[check_name]}")
            except Exception as e:
                results[check_name] = f"❌ 异常: {e}"
                self.logger.error(f"{check_name}: {results[check_name]}")
        
        # 生成兼容性报告
        report_path = self.base_dir / "experiments" / "compatibility_report.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "decoder_status": "compatible" if results.get("解码器兼容性") == "✅ 通过" else "incompatible",
            "recommendations": self.generate_recommendations(results)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 兼容性报告已保存至: {report_path}")
        return report
    
    def generate_recommendations(self, check_results):
        """基于检查结果生成建议"""
        recommendations = []
        
        if check_results.get("解码器兼容性") != "✅ 通过":
            recommendations.append("需要修复解码器兼容性问题")
        
        if check_results.get("专家系统配置") != "✅ 通过":
            recommendations.append("需要检查专家系统配置和数据文件")
        
        if not recommendations:
            recommendations.append("系统兼容性良好，可以开始实验")
        
        return recommendations

def main():
    print("🔧 GENIUS-Rec 实验适配器启动")
    print("="*50)
    
    adapter = ExperimentAdapter()
    report = adapter.run_compatibility_check()
    
    print("\n" + "="*50)
    print("📊 兼容性检查完成")
    
    if report["decoder_status"] == "compatible":
        print("✅ 系统兼容性良好！")
        print("\n推荐的下一步操作:")
        print("1. 运行快速验证: python experiments/quick_validation.py")
        print("2. 运行专家消融: python experiments/run_experiments.py --experiment_suite expert_ablation")
        print("3. 查看结果分析: python experiments/analyze_results.py")
    else:
        print("❌ 发现兼容性问题，请查看日志解决后重试")
    
    print(f"\n📋 详细报告: {adapter.base_dir}/experiments/compatibility_report.json")
    print(f"📋 适配日志: {adapter.adapter_log}")

if __name__ == "__main__":
    main()
