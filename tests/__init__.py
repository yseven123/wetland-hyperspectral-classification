"""
湿地高光谱分类系统测试包

本包包含了系统所有模块的单元测试和集成测试。
测试使用pytest框架，支持并行测试和覆盖率分析。

测试模块说明：
- test_data.py: 数据加载和验证模块测试
- test_preprocessing.py: 数据预处理模块测试  
- test_features.py: 特征提取模块测试
- test_classification.py: 分类算法模块测试
- test_integration.py: 系统集成测试

运行测试：
    pytest tests/                    # 运行所有测试
    pytest tests/test_data.py        # 运行单个模块测试
    pytest tests/ -v                 # 详细输出
    pytest tests/ --cov=src          # 生成覆盖率报告

Author: 湿地高光谱分类系统开发团队
Date: 2024
"""

import sys
import os
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# 添加源码路径到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 测试配置常量
TEST_DATA_SHAPE = (100, 100, 200)  # 测试用高光谱数据形状 (H, W, Bands)
TEST_NUM_CLASSES = 8               # 湿地分类类别数
TEST_SAMPLE_SIZE = 1000            # 测试样本数量
TEMP_TEST_DIR = None               # 临时测试目录

def setup_module():
    """模块级别的设置函数，在所有测试开始前运行"""
    global TEMP_TEST_DIR
    TEMP_TEST_DIR = tempfile.mkdtemp(prefix='wetland_test_')
    print(f"创建临时测试目录: {TEMP_TEST_DIR}")

def teardown_module():
    """模块级别的清理函数，在所有测试结束后运行"""
    global TEMP_TEST_DIR
    if TEMP_TEST_DIR and os.path.exists(TEMP_TEST_DIR):
        shutil.rmtree(TEMP_TEST_DIR)
        print(f"清理临时测试目录: {TEMP_TEST_DIR}")

@pytest.fixture
def sample_hyperspectral_data():
    """生成模拟高光谱数据的fixture"""
    # 生成符合高光谱特征的模拟数据
    np.random.seed(42)
    data = np.random.random(TEST_DATA_SHAPE).astype(np.float32)
    
    # 添加一些光谱特征模拟（如红边、水分吸收带等）
    # 模拟植被红边效应（700-750nm附近）
    red_edge_bands = slice(70, 80)
    data[:, :, red_edge_bands] *= 1.5
    
    # 模拟水分吸收带（950nm, 1200nm, 1400nm等）
    water_bands = [95, 120, 140]
    for band in water_bands:
        if band < TEST_DATA_SHAPE[2]:
            data[:, :, band] *= 0.3
    
    return data

@pytest.fixture  
def sample_classification_labels():
    """生成模拟分类标签的fixture"""
    np.random.seed(42)
    labels = np.random.randint(0, TEST_NUM_CLASSES, 
                              size=(TEST_DATA_SHAPE[0], TEST_DATA_SHAPE[1]))
    return labels

@pytest.fixture
def sample_training_samples():
    """生成模拟训练样本的fixture"""
    np.random.seed(42)
    # 生成随机坐标
    rows = np.random.randint(0, TEST_DATA_SHAPE[0], TEST_SAMPLE_SIZE)
    cols = np.random.randint(0, TEST_DATA_SHAPE[1], TEST_SAMPLE_SIZE)
    
    # 生成对应的类别标签
    labels = np.random.randint(0, TEST_NUM_CLASSES, TEST_SAMPLE_SIZE)
    
    return {
        'coordinates': list(zip(rows, cols)),
        'labels': labels.tolist(),
        'class_names': [f'湿地类型_{i}' for i in range(TEST_NUM_CLASSES)]
    }

@pytest.fixture
def temp_test_dir():
    """提供临时测试目录的fixture"""
    return TEMP_TEST_DIR

@pytest.fixture
def mock_config():
    """提供模拟配置的fixture"""
    return {
        'data': {
            'input_path': 'test_data.tif',
            'bands_range': [0, 200],
            'nodata_value': -9999,
            'scale_factor': 0.0001
        },
        'preprocessing': {
            'radiometric_correction': True,
            'atmospheric_correction': True,
            'noise_reduction': True,
            'spatial_filtering': True
        },
        'features': {
            'spectral_indices': True,
            'texture_features': True,
            'spatial_features': True,
            'window_size': 3
        },
        'classification': {
            'train_ratio': 0.7,
            'validation_ratio': 0.15,
            'test_ratio': 0.15,
            'random_state': 42
        },
        'landscape': {
            'patch_metrics': True,
            'class_metrics': True,
            'landscape_metrics': True
        }
    }

# 湿地分类类别映射
WETLAND_CLASSES = {
    0: '开放水面',
    1: '浅水区域', 
    2: '挺水植物',
    3: '浮叶植物',
    4: '沉水植物',
    5: '湿生草本',
    6: '有机质土壤',
    7: '建筑物'
}

# 测试用的光谱波段信息（模拟400-2500nm范围）
TEST_WAVELENGTHS = np.linspace(400, 2500, TEST_DATA_SHAPE[2])

# 常用的植被指数波段索引（基于模拟波段）
VEGETATION_INDICES = {
    'NDVI': {'red': 60, 'nir': 80},      # 归一化植被指数
    'EVI': {'blue': 20, 'red': 60, 'nir': 80},  # 增强植被指数
    'NDWI': {'green': 40, 'nir': 80},    # 归一化水体指数
    'MNDWI': {'green': 40, 'swir': 120}, # 修正归一化水体指数
}

def assert_array_properties(array, expected_shape=None, expected_dtype=None, 
                          expected_range=None):
    """断言数组属性的辅助函数"""
    assert isinstance(array, np.ndarray), "输入必须是numpy数组"
    
    if expected_shape:
        assert array.shape == expected_shape, f"形状不匹配: 期望{expected_shape}, 实际{array.shape}"
    
    if expected_dtype:
        assert array.dtype == expected_dtype, f"数据类型不匹配: 期望{expected_dtype}, 实际{array.dtype}"
    
    if expected_range:
        min_val, max_val = expected_range
        assert np.min(array) >= min_val, f"最小值{np.min(array)}小于期望范围{min_val}"
        assert np.max(array) <= max_val, f"最大值{np.max(array)}大于期望范围{max_val}"

def create_mock_geotiff(filepath, data, crs='EPSG:4326', transform=None):
    """创建模拟GeoTIFF文件的辅助函数"""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        
        if transform is None:
            # 假设覆盖1度x1度的区域
            transform = from_bounds(116, 39, 117, 40, data.shape[1], data.shape[0])
        
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1], 
            count=data.shape[2] if len(data.shape) == 3 else 1,
            dtype=data.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            if len(data.shape) == 3:
                for i in range(data.shape[2]):
                    dst.write(data[:, :, i], i + 1)
            else:
                dst.write(data, 1)
                
    except ImportError:
        # 如果rasterio不可用，创建一个简单的二进制文件
        with open(filepath, 'wb') as f:
            f.write(data.tobytes())

print("湿地高光谱分类系统测试包已初始化")