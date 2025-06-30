#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理器
Configuration Manager

负责加载、验证和管理系统配置文件

作者: Wetland Research Team
"""

import os
import yaml
import json
import toml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """运行时配置"""
    device: str = "auto"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    memory_limit: str = "16GB"
    cache_size: str = "4GB"
    batch_size: int = 32
    random_seed: int = 42
    deterministic: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    root_dir: str = "data/"
    raw_dir: str = "data/raw/"
    processed_dir: str = "data/processed/"
    samples_dir: str = "data/samples/"
    
    # 高光谱数据配置
    hyperspectral_format: str = "ENVI"
    bands: int = 400
    wavelength_range: List[int] = field(default_factory=lambda: [400, 2500])
    spatial_resolution: int = 30
    nodata_value: int = -9999
    
    # 训练样本配置
    samples_format: str = "shapefile"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    min_samples_per_class: int = 50


@dataclass
class ModelConfig:
    """模型配置"""
    default_classifier: str = "hybrid_cnn"
    classes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # 训练配置
    epochs: int = 100
    patience: int = 15
    min_delta: float = 0.001
    
    # 优化器配置
    optimizer_name: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.01


class Config:
    """配置管理器
    
    负责加载、验证、合并和管理系统的所有配置信息
    支持YAML、JSON、TOML格式的配置文件
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """初始化配置管理器
        
        Args:
            config_dict: 配置字典，如果为None则使用默认配置
        """
        self._config = config_dict or {}
        self._default_config = self._get_default_config()
        self._config_cache = {}
        
        # 合并默认配置
        self._config = self._deep_merge(self._default_config, self._config)
        
        # 验证配置
        self._validate_config()
        
        logger.info("Configuration manager initialized")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """从配置文件创建配置管理器
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Config: 配置管理器实例
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式不支持或内容无效
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # 根据文件扩展名选择加载器
        suffix = config_path.suffix.lower()
        loaders = {
            '.yaml': yaml.safe_load,
            '.yml': yaml.safe_load,
            '.json': json.load,
            '.toml': toml.load
        }
        
        if suffix not in loaders:
            raise ValueError(f"Unsupported configuration file format: {suffix}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = loaders[suffix](f)
            
            logger.info(f"Configuration loaded from: {config_path}")
            return cls(config_dict)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置管理器
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Config: 配置管理器实例
        """
        return cls(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "WETLAND_") -> 'Config':
        """从环境变量创建配置管理器
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            Config: 配置管理器实例
        """
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower()
                
                # 尝试转换数据类型
                try:
                    # 尝试解析为JSON
                    config_dict[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    # 如果不是JSON，保持字符串
                    config_dict[config_key] = value
        
        logger.info(f"Configuration loaded from environment variables with prefix: {prefix}")
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        支持点分隔的嵌套键，如 'data.hyperspectral.bands'
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        # 使用缓存提高性能
        if key in self._config_cache:
            return self._config_cache[key]
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            
            # 缓存结果
            self._config_cache[key] = value
            return value
            
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        # 导航到父级字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
        
        # 清除缓存
        self._config_cache.clear()
        
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            config_dict: 配置字典
        """
        self._config = self._deep_merge(self._config, config_dict)
        self._config_cache.clear()
        
        # 重新验证配置
        self._validate_config()
        
        logger.info("Configuration updated")
    
    def merge_from_file(self, config_path: Union[str, Path]) -> None:
        """从文件合并配置
        
        Args:
            config_path: 配置文件路径
        """
        other_config = Config.from_file(config_path)
        self.update(other_config._config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return deepcopy(self._config)
    
    def save(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """保存配置到文件
        
        Args:
            file_path: 保存路径
            format: 文件格式 ('yaml', 'json', 'toml')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        savers = {
            'yaml': lambda f: yaml.dump(self._config, f, default_flow_style=False, 
                                      allow_unicode=True, indent=2),
            'json': lambda f: json.dump(self._config, f, indent=2, ensure_ascii=False),
            'toml': lambda f: toml.dump(self._config, f)
        }
        
        if format not in savers:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            savers[format](f)
        
        logger.info(f"Configuration saved to: {file_path}")
    
    def get_runtime_config(self) -> RuntimeConfig:
        """获取运行时配置
        
        Returns:
            RuntimeConfig: 运行时配置对象
        """
        runtime_dict = self.get('runtime', {})
        return RuntimeConfig(**runtime_dict)
    
    def get_data_config(self) -> DataConfig:
        """获取数据配置
        
        Returns:
            DataConfig: 数据配置对象
        """
        data_dict = self.get('data', {})
        return DataConfig(**data_dict)
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置
        
        Returns:
            ModelConfig: 模型配置对象
        """
        model_dict = self.get('classification', {})
        training_dict = self.get('training', {})
        
        # 合并模型和训练配置
        combined_dict = {**model_dict, **training_dict}
        return ModelConfig(**combined_dict)
    
    def validate_paths(self) -> bool:
        """验证路径配置
        
        Returns:
            bool: 验证是否通过
        """
        data_config = self.get_data_config()
        
        # 检查数据目录
        paths_to_check = [
            data_config.root_dir,
            data_config.raw_dir,
            data_config.processed_dir,
            data_config.samples_dir
        ]
        
        for path in paths_to_check:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"Path does not exist: {path}")
                # 尝试创建目录
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {path}: {e}")
                    return False
        
        return True
    
    def get_classes(self) -> Dict[int, Dict[str, Any]]:
        """获取分类类别配置
        
        Returns:
            Dict[int, Dict[str, Any]]: 类别配置字典
        """
        return self.get('classification.classes', {})
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表
        
        Returns:
            List[str]: 类别名称列表
        """
        classes = self.get_classes()
        return [class_info.get('name', f'Class_{class_id}') 
                for class_id, class_info in sorted(classes.items())]
    
    def get_class_colors(self) -> List[List[int]]:
        """获取类别颜色列表
        
        Returns:
            List[List[int]]: 类别颜色列表
        """
        classes = self.get_classes()
        return [class_info.get('color', [128, 128, 128]) 
                for class_id, class_info in sorted(classes.items())]
    
    def get_num_classes(self) -> int:
        """获取类别数量
        
        Returns:
            int: 类别数量
        """
        return len(self.get_classes())
    
    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典
        
        Args:
            dict1: 基础字典
            dict2: 要合并的字典
            
        Returns:
            Dict[str, Any]: 合并后的字典
        """
        result = deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        return {
            'project': {
                'name': 'wetland_hyperspectral_classification',
                'version': '1.0.0',
                'description': '湿地高光谱分类系统'
            },
            'runtime': {
                'device': 'auto',
                'gpu_ids': [0],
                'num_workers': 4,
                'memory_limit': '16GB',
                'cache_size': '4GB',
                'batch_size': 32,
                'random_seed': 42,
                'deterministic': True
            },
            'data': {
                'root_dir': 'data/',
                'raw_dir': 'data/raw/',
                'processed_dir': 'data/processed/',
                'samples_dir': 'data/samples/',
                'hyperspectral': {
                    'format': 'ENVI',
                    'bands': 400,
                    'wavelength_range': [400, 2500],
                    'spatial_resolution': 30,
                    'nodata_value': -9999
                },
                'samples': {
                    'format': 'shapefile',
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15,
                    'stratify': True,
                    'min_samples_per_class': 50
                }
            },
            'classification': {
                'default_classifier': 'hybrid_cnn',
                'classes': {
                    1: {'name': '水体', 'color': [0, 0, 255]},
                    2: {'name': '挺水植物', 'color': [0, 255, 0]},
                    3: {'name': '浮叶植物', 'color': [128, 255, 0]},
                    4: {'name': '沉水植物', 'color': [0, 255, 128]},
                    5: {'name': '湿生草本', 'color': [255, 255, 0]},
                    6: {'name': '有机质土壤', 'color': [139, 69, 19]},
                    7: {'name': '矿物质土壤', 'color': [205, 133, 63]},
                    8: {'name': '建筑物', 'color': [255, 0, 0]},
                    9: {'name': '道路', 'color': [128, 128, 128]},
                    10: {'name': '农田', 'color': [255, 165, 0]}
                }
            },
            'training': {
                'epochs': 100,
                'patience': 15,
                'min_delta': 0.001,
                'optimizer': {
                    'name': 'AdamW',
                    'lr': 0.001,
                    'weight_decay': 0.01
                }
            },
            'output': {
                'base_dir': 'output/',
                'create_timestamp_dir': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/wetland_classification.log'
            }
        }
    
    def _validate_config(self) -> None:
        """验证配置的有效性
        
        Raises:
            ValueError: 配置无效
        """
        # 验证数据分割比例
        data_config = self.get('data.samples', {})
        train_ratio = data_config.get('train_ratio', 0.7)
        val_ratio = data_config.get('val_ratio', 0.15)
        test_ratio = data_config.get('test_ratio', 0.15)
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got: {train_ratio + val_ratio + test_ratio}")
        
        # 验证类别配置
        classes = self.get_classes()
        if not classes:
            logger.warning("No classes defined in configuration")
        
        # 验证类别ID连续性
        class_ids = sorted(classes.keys())
        if class_ids and class_ids != list(range(1, len(class_ids) + 1)):
            logger.warning("Class IDs are not continuous starting from 1")
        
        logger.debug("Configuration validation passed")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Config(num_classes={self.get_num_classes()}, device={self.get('runtime.device')})"
    
    def __getitem__(self, key: str) -> Any:
        """支持字典风格的访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典风格的设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return self.get(key) is not None