"""配置加载工具"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件
    
    Args:
        config_path: YAML文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_all_configs(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """加载configs目录下所有配置文件
    
    Args:
        config_dir: 配置目录，默认为项目根目录的configs文件夹
        
    Returns:
        合并后的配置字典
    """
    if config_dir is None:
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs"
    
    config_files = {
        'model': 'model_config.yaml',
        'env': 'env_config.yaml',
        'train': 'train_config.yaml',
    }
    
    merged_config = {}
    for key, filename in config_files.items():
        filepath = os.path.join(config_dir, filename)
        if os.path.exists(filepath):
            merged_config[key] = load_config(filepath)
        else:
            print(f"Warning: Config file not found: {filepath}")
    
    return merged_config


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """递归合并两个字典，override优先级更高
    
    Args:
        base: 基础字典
        override: 覆盖字典
        
    Returns:
        合并后的字典
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent
