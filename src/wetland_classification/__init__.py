#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
湿地高光谱分类系统
Wetland Hyperspectral Classification System

基于深度学习与机器学习的高光谱遥感湿地生态系统精细化分类与景观格局分析系统

作者: Wetland Research Team
版本: 1.0.0
许可证: MIT License
"""

import sys
import warnings
from pathlib import Path

# 版本信息
__version__ = "1.0.0"
__author__ = "Wetland Research Team"
__email__ = "wetland.research@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024, Wetland Research Team"

# 项目信息
__title__ = "wetland-hyperspectral-classification"
__description__ = "基于深度学习与机器学习的高光谱遥感湿地生态系统精细化分类与景观格局分析系统"
__url__ = "https://github.com/yourusername/wetland-hyperspectral-classification"

# 最低Python版本要求
MINIMUM_PYTHON_VERSION = (3, 8)

# 检查Python版本
if sys.version_info < MINIMUM_PYTHON_VERSION:
    raise RuntimeError(
        f"Python {'.'.join(map(str, MINIMUM_PYTHON_VERSION))} or higher is required. "
        f"You are using Python {'.'.join(map(str, sys.version_info[:3]))}."
    )

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="gdal")

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

# 配置文件目录
CONFIG_DIR = PROJECT_ROOT / "config"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "output"

# 日志目录
LOGS_DIR = PROJECT_ROOT / "logs"

# 创建必要的目录
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR,
                  MODELS_DIR, PRETRAINED_MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 导入核心模块
try:
    from .config import Config
    from .pipeline import Pipeline
    
    # 数据模块
    from .data import DataLoader, DataValidator, DataAugmentation
    
    # 预处理模块
    from .preprocessing import (
        RadiometricCorrector,
        AtmosphericCorrector,
        GeometricCorrector,
        NoiseReducer
    )
    
    # 特征提取模块
    from .features import (
        SpectralFeatureExtractor,
        VegetationIndexCalculator,
        TextureFeatureExtractor,
        SpatialFeatureExtractor
    )
    
    # 分类模块
    from .classification import (
        TraditionalClassifier,
        DeepLearningClassifier,
        EnsembleClassifier
    )
    
    # 后处理模块
    from .postprocessing import (
        SpatialFilter,
        MorphologyProcessor,
        ConsistencyChecker
    )
    
    # 景观分析模块
    from .landscape import (
        LandscapeMetrics,
        ConnectivityAnalyzer
    )
    
    # 评估模块
    from .evaluation import (
        MetricsCalculator,
        CrossValidator,
        UncertaintyAnalyzer
    )
    
    # 工具模块
    from .utils import (
        IOUtils,
        VisualizationUtils,
        Logger
    )

except ImportError as e:
    warnings.warn(
        f"Failed to import some modules: {e}. "
        "Some functionality may not be available.",
        ImportWarning
    )

# 公开API
__all__ = [
    # 版本和元信息
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__title__",
    "__description__",
    "__url__",
    
    # 目录路径
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SAMPLES_DIR",
    "CONFIG_DIR",
    "MODELS_DIR",
    "PRETRAINED_MODELS_DIR",
    "OUTPUT_DIR",
    "LOGS_DIR",
    
    # 核心类
    "Config",
    "Pipeline",
    
    # 数据处理
    "DataLoader",
    "DataValidator",
    "DataAugmentation",
    
    # 预处理
    "RadiometricCorrector",
    "AtmosphericCorrector",
    "GeometricCorrector",
    "NoiseReducer",
    
    # 特征提取
    "SpectralFeatureExtractor",
    "VegetationIndexCalculator",
    "TextureFeatureExtractor",
    "SpatialFeatureExtractor",
    
    # 分类
    "TraditionalClassifier",
    "DeepLearningClassifier",
    "EnsembleClassifier",
    
    # 后处理
    "SpatialFilter",
    "MorphologyProcessor",
    "ConsistencyChecker",
    
    # 景观分析
    "LandscapeMetrics",
    "ConnectivityAnalyzer",
    
    # 评估
    "MetricsCalculator",
    "CrossValidator",
    "UncertaintyAnalyzer",
    
    # 工具
    "IOUtils",
    "VisualizationUtils",
    "Logger",
]

# 湿地分类系统常量
WETLAND_CLASSES = {
    1: {"name": "水体", "color": [0, 0, 255], "description": "开放水面、浅水区域、季节性水体"},
    2: {"name": "挺水植物", "color": [0, 255, 0], "description": "芦苇、香蒲等挺水植物群落"},
    3: {"name": "浮叶植物", "color": [128, 255, 0], "description": "荷花、睡莲等浮叶植物群落"},
    4: {"name": "沉水植物", "color": [0, 255, 128], "description": "眼子菜、苦草等沉水植物群落"},
    5: {"name": "湿生草本", "color": [255, 255, 0], "description": "湿生草本植物群落"},
    6: {"name": "有机质土壤", "color": [139, 69, 19], "description": "富含有机质的湿地土壤"},
    7: {"name": "矿物质土壤", "color": [205, 133, 63], "description": "以矿物质为主的土壤"},
    8: {"name": "建筑物", "color": [255, 0, 0], "description": "建筑物和人工结构"},
    9: {"name": "道路", "color": [128, 128, 128], "description": "道路和交通设施"},
    10: {"name": "农田", "color": [255, 165, 0], "description": "农田和农业用地"},
}

# 支持的高光谱传感器
SUPPORTED_SENSORS = {
    "AVIRIS": {
        "spectral_range": [400, 2500],
        "spectral_bands": 224,
        "spectral_resolution": 10,
        "data_format": "ENVI"
    },
    "CASI": {
        "spectral_range": [400, 1000],
        "spectral_bands": 288,
        "spectral_resolution": 2.5,
        "data_format": "ENVI"
    },
    "HyMap": {
        "spectral_range": [450, 2480],
        "spectral_bands": 126,
        "spectral_resolution": 15,
        "data_format": "ENVI"
    },
    "Hyperion": {
        "spectral_range": [400, 2500],
        "spectral_bands": 242,
        "spectral_resolution": 10,
        "data_format": "GeoTIFF"
    }
}

# 支持的分类算法
SUPPORTED_CLASSIFIERS = {
    "traditional": ["svm", "random_forest", "xgboost", "knn"],
    "deep_learning": ["cnn_3d", "hybrid_cnn", "vision_transformer", "resnet_hs"],
    "ensemble": ["voting_ensemble", "stacking_ensemble", "weighted_ensemble"]
}

# 植被指数定义
VEGETATION_INDICES = {
    "NDVI": {
        "name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - Red) / (NIR + Red)",
        "bands": ["Red", "NIR"]
    },
    "EVI": {
        "name": "Enhanced Vegetation Index",
        "formula": "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
        "bands": ["Blue", "Red", "NIR"]
    },
    "NDWI": {
        "name": "Normalized Difference Water Index",
        "formula": "(Green - NIR) / (Green + NIR)",
        "bands": ["Green", "NIR"]
    },
    "MNDWI": {
        "name": "Modified Normalized Difference Water Index",
        "formula": "(Green - SWIR) / (Green + SWIR)",
        "bands": ["Green", "SWIR"]
    }
}

def get_version() -> str:
    """获取版本信息"""
    return __version__

def get_info() -> dict:
    """获取包信息"""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

def check_dependencies() -> dict:
    """检查依赖包是否正确安装"""
    dependencies = {}
    
    # 核心科学计算库
    core_packages = [
        "numpy", "scipy", "pandas", "scikit-learn", "scikit-image"
    ]
    
    # 深度学习框架
    dl_packages = [
        "torch", "torchvision", "tensorflow"
    ]
    
    # 地理空间处理库
    geo_packages = [
        "gdal", "rasterio", "geopandas", "shapely", "fiona"
    ]
    
    # 可视化库
    viz_packages = [
        "matplotlib", "seaborn", "plotly"
    ]
    
    all_packages = core_packages + dl_packages + geo_packages + viz_packages
    
    for package in all_packages:
        try:
            __import__(package)
            dependencies[package] = "✓ 已安装"
        except ImportError:
            dependencies[package] = "✗ 未安装"
    
    return dependencies

def setup_environment():
    """设置环境变量和初始化"""
    import os
    import numpy as np
    
    # 设置随机种子
    np.random.seed(42)
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = '42'
    
    # 设置GDAL环境变量
    if 'GDAL_DATA' not in os.environ:
        try:
            from osgeo import gdal
            gdal_data = gdal.GetConfigOption('GDAL_DATA')
            if gdal_data:
                os.environ['GDAL_DATA'] = gdal_data
        except ImportError:
            pass
    
    # 设置日志级别
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# 初始化环境
setup_environment()

# 欢迎信息
def print_welcome():
    """打印欢迎信息"""
    print(f"""
    🌿 湿地高光谱分类系统 v{__version__} 🌿
    
    基于深度学习与机器学习的高光谱遥感湿地生态系统精细化分类与景观格局分析系统
    
    ✨ 特性:
    • 🔧 完整数据流水线: 从原始数据到最终分类结果
    • 🤖 多算法融合: 传统机器学习 + 深度学习
    • 🌿 专业湿地分类: 针对湿地生态系统优化
    • 📊 景观格局分析: 全面的景观生态学指标
    • ⚡ 高性能计算: GPU加速 + 分布式处理
    
    📚 快速开始:
    from wetland_classification import Pipeline, Config
    config = Config.from_file('config/config.yaml')
    pipeline = Pipeline(config)
    results = pipeline.run('data/raw/hyperspectral_data.tif')
    
    🆘 获取帮助: https://github.com/yourusername/wetland-hyperspectral-classification
    """)

# 如果直接运行此模块，显示包信息
if __name__ == "__main__":
    print_welcome()
    print("\n📦 包信息:")
    info = get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 依赖检查:")
    deps = check_dependencies()
    for package, status in deps.items():
        print(f"  {package}: {status}")