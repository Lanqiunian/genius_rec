#!/usr/bin/env python3
"""
GENIUS-Rec 高级实验配置管理器
=============================

这个脚本允许动态修改配置文件来进行更细粒度的超参数实验
"""

import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List
import itertools

class ConfigManager:
    """配置管理器，用于动态生成实验配置"""
    
    def __init__(self, base_config_path: str = "src/config.py"):
        self.base_config_path = Path(base_config_path)
        self.experiment_configs_dir = Path("experiments/configs")
        self.experiment_configs_dir.mkdir(parents=True, exist_ok=True)
        
    def create_config_variant(self, variant_name: str, modifications: Dict[str, Any]) -> str:
        """创建配置变体"""
        # 读取基础配置
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 创建修改后的配置
        modified_content = self._apply_modifications(config_content, modifications)
        
        # 保存新配置
        variant_path = self.experiment_configs_dir / f"config_{variant_name}.py"
        with open(variant_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
            
        return str(variant_path)
    
    def _apply_modifications(self, content: str, modifications: Dict[str, Any]) -> str:
        """应用配置修改"""
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_line = line
            
            # 查找需要修改的配置项
            for key_path, new_value in modifications.items():
                if self._line_matches_config_path(line, key_path):
                    # 替换配置值
                    modified_line = self._replace_config_value(line, new_value)
                    break
                    
            modified_lines.append(modified_line)
        
        return '\n'.join(modified_lines)
    
    def _line_matches_config_path(self, line: str, key_path: str) -> bool:
        """检查行是否匹配配置路径"""
        # 简化的匹配逻辑，可以根据需要扩展
        key = key_path.split('.')[-1]
        return f'"{key}":' in line or f"'{key}':" in line
    
    def _replace_config_value(self, line: str, new_value: Any) -> str:
        """替换配置值"""
        # 找到冒号位置
        colon_pos = line.find(':')
        if colon_pos == -1:
            return line
            
        # 保留缩进和键名
        prefix = line[:colon_pos + 1]
        
        # 根据值类型格式化
        if isinstance(new_value, str):
            new_line = f'{prefix} "{new_value}",'
        elif isinstance(new_value, bool):
            new_line = f'{prefix} {str(new_value)},'
        elif isinstance(new_value, (int, float)):
            new_line = f'{prefix} {new_value},'
        else:
            new_line = f'{prefix} {new_value},'
            
        return new_line

class AdvancedExperimentGenerator:
    """高级实验生成器"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        
    def generate_hyperparameter_grid(self) -> List[Dict[str, Any]]:
        """生成超参数网格搜索配置"""
        
        hyperparameter_grid = {
            "finetune.learning_rate.decoder_lr": [1e-3, 5e-4, 2e-3],
            "finetune.batch_size": [32, 64, 128],
            "decoder_model.num_layers": [2, 4, 6],
            "decoder_model.num_heads": [4, 8],
            "expert_system.fusion_strategy.expert_dropout": [0.0, 0.1, 0.2],
        }
        
        # 生成所有组合（注意：这可能产生很多组合）
        keys = list(hyperparameter_grid.keys())
        values = list(hyperparameter_grid.values())
        
        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)
            
        return configs[:10]  # 限制前10个配置以避免过多实验
    
    def generate_expert_system_configs(self) -> List[Dict[str, Any]]:
        """生成专家系统配置变体"""
        
        expert_configs = [
            # 基础专家组合
            {
                "expert_system.experts.behavior_expert": True,
                "expert_system.experts.content_expert": False,
                "expert_system.experts.image_expert": False,
            },
            {
                "expert_system.experts.behavior_expert": True,
                "expert_system.experts.content_expert": True,
                "expert_system.experts.image_expert": False,
            },
            {
                "expert_system.experts.behavior_expert": True,
                "expert_system.experts.content_expert": True,
                "expert_system.experts.image_expert": True,
            },
            # 不同的门控配置
            {
                "expert_system.experts.behavior_expert": True,
                "expert_system.experts.content_expert": True,
                "expert_system.experts.image_expert": True,
                "expert_system.gate_config.gate_type": "mlp",
                "expert_system.gate_config.gate_hidden_dim": 128,
            },
            # 不同的融合策略
            {
                "expert_system.experts.behavior_expert": True,
                "expert_system.experts.content_expert": True,
                "expert_system.experts.image_expert": True,
                "expert_system.fusion_strategy.method": "attention_fusion",
            },
        ]
        
        return expert_configs
    
    def generate_architecture_configs(self) -> List[Dict[str, Any]]:
        """生成架构配置变体"""
        
        arch_configs = [
            # 不同的解码器深度
            {
                "decoder_model.num_layers": 2,
                "decoder_model.ffn_hidden_dim": 128,
            },
            {
                "decoder_model.num_layers": 4,
                "decoder_model.ffn_hidden_dim": 256,
            },
            {
                "decoder_model.num_layers": 6,
                "decoder_model.ffn_hidden_dim": 512,
            },
            # 不同的注意力头数
            {
                "decoder_model.num_heads": 2,
                "expert_system.content_expert.attention_heads": 2,
                "expert_system.image_expert.attention_heads": 2,
            },
            {
                "decoder_model.num_heads": 8,
                "expert_system.content_expert.attention_heads": 8,
                "expert_system.image_expert.attention_heads": 8,
            },
        ]
        
        return arch_configs

def create_experiment_configs():
    """创建所有实验配置文件"""
    generator = AdvancedExperimentGenerator()
    
    print("🔧 生成实验配置文件...")
    
    all_configs = {}
    
    # 1. 专家系统配置
    expert_configs = generator.generate_expert_system_configs()
    for i, config in enumerate(expert_configs):
        variant_name = f"expert_variant_{i+1}"
        config_path = generator.config_manager.create_config_variant(variant_name, config)
        all_configs[variant_name] = {"path": config_path, "type": "expert_system"}
        print(f"   ✅ 创建专家配置: {variant_name}")
    
    # 2. 架构配置
    arch_configs = generator.generate_architecture_configs()
    for i, config in enumerate(arch_configs):
        variant_name = f"arch_variant_{i+1}"
        config_path = generator.config_manager.create_config_variant(variant_name, config)
        all_configs[variant_name] = {"path": config_path, "type": "architecture"}
        print(f"   ✅ 创建架构配置: {variant_name}")
    
    # 3. 超参数配置
    hp_configs = generator.generate_hyperparameter_grid()
    for i, config in enumerate(hp_configs):
        variant_name = f"hp_variant_{i+1}"
        config_path = generator.config_manager.create_config_variant(variant_name, config)
        all_configs[variant_name] = {"path": config_path, "type": "hyperparameter"}
        print(f"   ✅ 创建超参数配置: {variant_name}")
    
    # 保存配置索引
    config_index_path = Path("experiments/config_index.json")
    with open(config_index_path, 'w', encoding='utf-8') as f:
        json.dump(all_configs, f, indent=2)
    
    print(f"\n📋 总共生成 {len(all_configs)} 个配置文件")
    print(f"📄 配置索引保存到: {config_index_path}")
    
    return all_configs

if __name__ == "__main__":
    create_experiment_configs()
