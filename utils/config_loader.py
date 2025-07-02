import yaml
import os
from pathlib import Path

class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def get(self, key_path, default=None):
        """通过点分隔的路径获取配置值
        
        Args:
            key_path: 配置路径，如 'data.mimic_path'
            default: 默认值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_paths(self):
        """获取数据路径配置"""
        return {
            'mimic_path': self.get('data.mimic_path'),
            'processed_path': self.get('data.processed_data_path'),
            'train_split': self.get('data.train_split'),
            'val_split': self.get('data.val_split'),
            'test_split': self.get('data.test_split')
        }
    
    def get_training_config(self, agent_type='single'):
        """获取训练配置"""
        if agent_type == 'single':
            return self.get('single_agent')
        elif agent_type == 'multi':
            return self.get('multi_agent')
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")

# 全局配置实例
config = ConfigLoader() 