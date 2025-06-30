#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
湿地高光谱分类系统 - 批量处理示例
Wetland Hyperspectral Classification System - Batch Processing Example

这个示例展示了如何使用湿地高光谱分类系统进行批量处理任务，包括：
- 多场景批量分类
- 并行处理优化
- 大数据集处理策略
- 时序数据批量分析
- 结果统计和汇总
- 质量控制和异常检测

作者: 研究团队
日期: 2024-06-30
版本: 1.0.0
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
from queue import Queue
import psutil

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入湿地分类系统模块
try:
    from wetland_classification import Pipeline
    from wetland_classification.config import Config
    from wetland_classification.data import DataLoader
    from wetland_classification.preprocessing import Preprocessor
    from wetland_classification.features import FeatureExtractor
    from wetland_classification.classification import Classifier
    from wetland_classification.evaluation import ModelEvaluator
    from wetland_classification.utils.visualization import Visualizer
    from wetland_classification.utils.logger import get_logger
    from wetland_classification.utils.io_utils import IOUtils
    from wetland_classification.batch import BatchProcessor
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已正确安装湿地分类系统")
    sys.exit(1)

# 设置日志
logger = get_logger(__name__)

class BatchProcessingManager:
    """
    批量处理管理器
    """
    
    def __init__(self, config_path=None, max_workers=None):
        """
        初始化批量处理管理器
        
        Args:
            config_path: 配置文件路径
            max_workers: 最大并行工作进程数
        """
        self.config = self.load_config(config_path)
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.processed_count = 0
        self.failed_count = 0
        self.processing_stats = {}
        self.start_time = None
        
        # 设置目录
        self.setup_directories()
        
        logger.info(f"批量处理管理器初始化完成，最大并行数: {self.max_workers}")
    
    def load_config(self, config_path):
        """加载配置"""
        if config_path and os.path.exists(config_path):
            return Config.from_file(config_path)
        else:
            return self.create_default_batch_config()
    
    def create_default_batch_config(self):
        """创建默认批量处理配置"""
        config = {
            'batch_processing': {
                'max_workers': self.max_workers,
                'chunk_size': 1000,  # 分块大小
                'memory_limit_gb': 16,  # 内存限制
                'enable_parallel': True,
                'save_intermediate': True,
                'quality_check': True
            },
            'data': {
                'input_directory': 'data/batch_input',
                'output_directory': 'output/batch_processing',
                'file_patterns': ['*.tif', '*.img', '*.hdf5'],
                'recursive_search': True
            },
            'preprocessing': {
                'enable_preprocessing': True,
                'normalization': True,
                'noise_reduction': False,
                'atmospheric_correction': False
            },
            'classification': {
                'model_path': 'models/best_model.pkl',
                'model_type': 'ensemble',
                'batch_prediction': True,
                'confidence_threshold': 0.7
            },
            'output': {
                'save_probabilities': True,
                'save_confidence': True,
                'save_statistics': True,
                'create_overview': True,
                'generate_report': True
            }
        }
        return config
    
    def setup_directories(self):
        """设置批量处理目录"""
        directories = [
            'output/batch_processing',
            'output/batch_processing/results',
            'output/batch_processing/statistics',
            'output/batch_processing/reports',
            'output/batch_processing/failed',
            'output/batch_processing/intermediate',
            'data/batch_input',
            'logs/batch_processing'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

def setup_batch_directories():
    """
    设置批量处理的目录结构
    """
    directories = [
        'output/batch_processing',
        'output/batch_processing/results',
        'output/batch_processing/statistics', 
        'output/batch_processing/reports',
        'output/batch_processing/quality_control',
        'output/batch_processing/temporal_analysis',
        'data/batch_input',
        'data/batch_input/multi_temporal',
        'data/batch_input/different_sensors',
        'logs/batch_processing'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def create_batch_demo_data():
    """
    创建批量处理的演示数据
    """
    logger.info("创建批量处理演示数据...")
    
    batch_input_dir = Path('data/batch_input')
    
    # 创建多个场景的模拟数据
    scenarios = [
        {'name': 'scene_2023_spring', 'season': 'spring', 'year': 2023},
        {'name': 'scene_2023_summer', 'season': 'summer', 'year': 2023},
        {'name': 'scene_2023_autumn', 'season': 'autumn', 'year': 2023},
        {'name': 'scene_2023_winter', 'season': 'winter', 'year': 2023},
        {'name': 'scene_2024_spring', 'season': 'spring', 'year': 2024},
        {'name': 'scene_2024_summer', 'season': 'summer', 'year': 2024},
    ]
    
    # 定义类别信息
    class_info = {
        0: {'name': '背景', 'color': '#000000'},
        1: {'name': '开放水面', 'color': '#0000FF'},
        2: {'name': '浅水区域', 'color': '#4169E1'},
        3: {'name': '挺水植物', 'color': '#228B22'},
        4: {'name': '浮叶植物', 'color': '#32CD32'},
        5: {'name': '湿生草本', 'color': '#9ACD32'},
        6: {'name': '土壤', 'color': '#8B4513'},
        7: {'name': '建筑物', 'color': '#FF0000'}
    }
    
    np.random.seed(42)
    
    for scenario in scenarios:
        scene_dir = batch_input_dir / scenario['name']
        scene_dir.mkdir(exist_ok=True)
        
        # 根据季节调整数据特征
        height, width, bands = 150, 150, 120
        season_factor = get_season_factor(scenario['season'])
        
        # 生成高光谱数据
        hyperspectral_data = generate_seasonal_hyperspectral_data(
            height, width, bands, season_factor
        )
        
        # 生成对应的标签
        labels = generate_seasonal_labels(height, width, season_factor)
        
        # 保存数据
        np.save(scene_dir / 'hyperspectral_data.npy', hyperspectral_data)
        np.save(scene_dir / 'ground_truth.npy', labels)
        
        # 保存元数据
        metadata = {
            'scene_name': scenario['name'],
            'acquisition_date': f"{scenario['year']}-{get_season_month(scenario['season'])}-15",
            'season': scenario['season'],
            'year': scenario['year'],
            'sensor': 'Simulated_Sensor',
            'spatial_resolution': 30,  # 米
            'spectral_bands': bands,
            'data_shape': [height, width, bands],
            'class_info': class_info
        }
        
        with open(scene_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"创建场景数据: {scenario['name']} ({height}x{width}x{bands})")
    
    # 创建波长信息（所有场景共用）
    wavelengths = np.linspace(400, 2400, bands)
    np.save(batch_input_dir / 'wavelengths.npy', wavelengths)
    
    # 保存类别信息
    with open(batch_input_dir / 'class_info.json', 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"批量处理演示数据创建完成，共 {len(scenarios)} 个场景")
    return scenarios, class_info

def get_season_factor(season):
    """获取季节因子，用于调整数据特征"""
    season_factors = {
        'spring': {'vegetation': 0.6, 'water': 0.8, 'temperature': 0.3},
        'summer': {'vegetation': 1.0, 'water': 0.6, 'temperature': 1.0},
        'autumn': {'vegetation': 0.4, 'water': 0.7, 'temperature': 0.6},
        'winter': {'vegetation': 0.2, 'water': 0.9, 'temperature': 0.1}
    }
    return season_factors.get(season, season_factors['summer'])

def get_season_month(season):
    """获取季节对应的典型月份"""
    season_months = {
        'spring': '04',
        'summer': '07', 
        'autumn': '10',
        'winter': '01'
    }
    return season_months.get(season, '07')

def generate_seasonal_hyperspectral_data(height, width, bands, season_factor):
    """
    生成季节性高光谱数据
    """
    data = np.zeros((height, width, bands))
    wavelengths = np.linspace(400, 2400, bands)
    
    # 根据季节调整生成策略
    vegetation_strength = season_factor['vegetation']
    water_level = season_factor['water']
    temperature_effect = season_factor['temperature']
    
    for i in range(height):
        for j in range(width):
            # 根据位置确定基础地物类型
            dist_center = np.sqrt((i - height//2)**2 + (j - width//2)**2)
            
            if dist_center < 30:  # 中心水域
                spectrum = generate_water_spectrum(wavelengths, water_level)
            elif dist_center < 60:  # 植被区域
                spectrum = generate_vegetation_spectrum(wavelengths, vegetation_strength)
            elif dist_center < 70:  # 土壤区域
                spectrum = generate_soil_spectrum(wavelengths, temperature_effect)
            else:  # 其他
                spectrum = generate_mixed_spectrum(wavelengths, season_factor)
            
            # 添加季节性噪声
            seasonal_noise = np.random.normal(0, 0.02 * (1 + temperature_effect), bands)
            spectrum += seasonal_noise
            
            data[i, j, :] = np.clip(spectrum, 0, 1)
    
    return data

def generate_water_spectrum(wavelengths, water_level):
    """生成水体光谱"""
    spectrum = np.ones_like(wavelengths) * 0.05
    
    # 水体在近红外强吸收
    nir_mask = wavelengths > 700
    spectrum[nir_mask] *= 0.1 * water_level
    
    # 蓝绿光有一定反射
    blue_green_mask = (wavelengths >= 400) & (wavelengths <= 600)
    spectrum[blue_green_mask] *= 2
    
    return spectrum

def generate_vegetation_spectrum(wavelengths, vegetation_strength):
    """生成植被光谱"""
    spectrum = np.ones_like(wavelengths) * 0.3
    
    # 植被特征
    red_mask = (wavelengths >= 620) & (wavelengths <= 700)
    nir_mask = (wavelengths >= 750) & (wavelengths <= 1300)
    
    spectrum[red_mask] *= 0.3 * vegetation_strength  # 红光吸收
    spectrum[nir_mask] *= 2.0 * vegetation_strength  # 近红外高反射
    
    # 水分吸收带
    water_bands = [(1400, 1500), (1900, 2000)]
    for start, end in water_bands:
        mask = (wavelengths >= start) & (wavelengths <= end)
        spectrum[mask] *= 0.5
    
    return spectrum

def generate_soil_spectrum(wavelengths, temperature_effect):
    """生成土壤光谱"""
    base_reflectance = 0.2 + 0.2 * temperature_effect
    spectrum = np.ones_like(wavelengths) * base_reflectance
    
    # 土壤在长波方向反射率增加
    long_wave_mask = wavelengths > 1000
    spectrum[long_wave_mask] *= 1.5
    
    return spectrum

def generate_mixed_spectrum(wavelengths, season_factor):
    """生成混合光谱"""
    vegetation_spectrum = generate_vegetation_spectrum(wavelengths, season_factor['vegetation'])
    soil_spectrum = generate_soil_spectrum(wavelengths, season_factor['temperature'])
    
    # 混合比例
    veg_ratio = 0.3 * season_factor['vegetation']
    soil_ratio = 1 - veg_ratio
    
    mixed_spectrum = veg_ratio * vegetation_spectrum + soil_ratio * soil_spectrum
    return mixed_spectrum

def generate_seasonal_labels(height, width, season_factor):
    """
    生成季节性标签数据
    """
    labels = np.zeros((height, width), dtype=int)
    
    vegetation_prob = season_factor['vegetation']
    water_prob = season_factor['water']
    
    for i in range(height):
        for j in range(width):
            dist_center = np.sqrt((i - height//2)**2 + (j - width//2)**2)
            
            if dist_center < 25:  # 核心水域
                if np.random.random() < water_prob:
                    labels[i, j] = 1  # 开放水面
                else:
                    labels[i, j] = 2  # 浅水区域
            elif dist_center < 45:  # 植被过渡带
                if np.random.random() < vegetation_prob:
                    labels[i, j] = np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2])
                else:
                    labels[i, j] = 6  # 土壤
            elif dist_center < 65:  # 外围区域
                if np.random.random() < 0.7:
                    labels[i, j] = 6  # 土壤
                else:
                    labels[i, j] = np.random.choice([3, 5], p=[0.6, 0.4])  # 稀疏植被
            elif dist_center < 70:  # 边缘
                if np.random.random() < 0.1:
                    labels[i, j] = 7  # 建筑物
                else:
                    labels[i, j] = 6  # 土壤
            # else: 保持为0 (背景)
    
    return labels

def batch_processing_workflow():
    """
    批量处理主工作流程
    """
    logger.info("="*60)
    logger.info("开始湿地高光谱批量处理示例")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 步骤1: 准备批量数据
        logger.info("步骤1: 准备批量数据")
        scenarios, class_info = prepare_batch_data()
        
        # 步骤2: 训练或加载模型
        logger.info("步骤2: 准备分类模型")
        model = prepare_classification_model()
        
        # 步骤3: 批量处理配置
        logger.info("步骤3: 配置批量处理")
        batch_config = setup_batch_configuration()
        
        # 步骤4: 串行批量处理
        logger.info("步骤4: 串行批量处理")
        serial_results = serial_batch_processing(scenarios, model, class_info)
        
        # 步骤5: 并行批量处理
        logger.info("步骤5: 并行批量处理")
        parallel_results = parallel_batch_processing(scenarios, model, class_info)
        
        # 步骤6: 性能对比分析
        logger.info("步骤6: 性能对比分析")
        performance_comparison = compare_processing_performance(serial_results, parallel_results)
        
        # 步骤7: 时序分析
        logger.info("步骤7: 时序批量分析")
        temporal_results = temporal_batch_analysis(scenarios, parallel_results, class_info)
        
        # 步骤8: 质量控制
        logger.info("步骤8: 批量质量控制")
        quality_results = batch_quality_control(parallel_results, class_info)
        
        # 步骤9: 统计分析
        logger.info("步骤9: 统计分析与汇总")
        statistical_results = batch_statistical_analysis(parallel_results, temporal_results, class_info)
        
        # 步骤10: 结果可视化
        logger.info("步骤10: 批量结果可视化")
        create_batch_visualizations(
            parallel_results, temporal_results, quality_results, 
            statistical_results, performance_comparison, class_info
        )
        
        # 步骤11: 生成批量报告
        logger.info("步骤11: 生成批量处理报告")
        generate_batch_report(
            parallel_results, temporal_results, quality_results,
            statistical_results, performance_comparison, class_info
        )
        
        total_time = time.time() - start_time
        display_batch_results(statistical_results, performance_comparison, total_time)
        
    except Exception as e:
        logger.error(f"批量处理流程执行失败: {e}")
        raise

def prepare_batch_data():
    """
    准备批量处理数据
    """
    batch_input_dir = Path('data/batch_input')
    
    if not batch_input_dir.exists() or len(list(batch_input_dir.glob('scene_*'))) == 0:
        logger.info("批量演示数据不存在，正在创建...")
        scenarios, class_info = create_batch_demo_data()
    else:
        logger.info("加载已有的批量演示数据...")
        
        # 扫描现有场景
        scenarios = []
        scene_dirs = sorted(batch_input_dir.glob('scene_*'))
        
        for scene_dir in scene_dirs:
            metadata_file = scene_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                scenarios.append({
                    'name': scene_dir.name,
                    'season': metadata.get('season', 'unknown'),
                    'year': metadata.get('year', 2024),
                    'path': scene_dir
                })
        
        # 加载类别信息
        class_info_file = batch_input_dir / 'class_info.json'
        if class_info_file.exists():
            with open(class_info_file, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
            class_info = {int(k): v for k, v in class_info.items()}
        else:
            class_info = {}
    
    logger.info(f"准备完成，共 {len(scenarios)} 个场景")
    return scenarios, class_info

def prepare_classification_model():
    """
    准备分类模型
    """
    model_path = Path('models/batch_processing_model.pkl')
    
    if model_path.exists():
        logger.info("加载预训练模型...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        logger.info("训练新的分类模型...")
        model = train_batch_model()
        
        # 保存模型
        model_path.parent.mkdir(exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return model

def train_batch_model():
    """
    训练批量处理模型
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # 使用第一个场景的数据训练模型
    scene_dir = Path('data/batch_input/scene_2023_summer')
    
    # 加载数据
    data = np.load(scene_dir / 'hyperspectral_data.npy')
    labels = np.load(scene_dir / 'ground_truth.npy')
    
    # 准备训练数据
    height, width, bands = data.shape
    reshaped_data = data.reshape(-1, bands)
    reshaped_labels = labels.ravel()
    
    # 获取有效样本
    valid_mask = reshaped_labels > 0
    X = reshaped_data[valid_mask]
    y = reshaped_labels[valid_mask]
    
    # 特征提取 (简化版)
    features = extract_simple_features(X)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # 保存标准化器
    model.scaler = scaler
    
    logger.info(f"模型训练完成，训练样本数: {len(X)}")
    return model

def extract_simple_features(data):
    """
    提取简化特征
    """
    features = []
    
    # 1. 下采样光谱特征
    step = max(1, data.shape[1] // 20)
    spectral_features = data[:, ::step]
    features.append(spectral_features)
    
    # 2. 统计特征
    mean_values = np.mean(data, axis=1).reshape(-1, 1)
    std_values = np.std(data, axis=1).reshape(-1, 1)
    max_values = np.max(data, axis=1).reshape(-1, 1)
    min_values = np.min(data, axis=1).reshape(-1, 1)
    
    features.extend([mean_values, std_values, max_values, min_values])
    
    # 3. 简化植被指数
    if data.shape[1] >= 100:  # 确保有足够的波段
        # 假设的波段位置
        red_idx = data.shape[1] // 3
        nir_idx = 2 * data.shape[1] // 3
        
        red = data[:, red_idx]
        nir = data[:, nir_idx]
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        features.append(ndvi.reshape(-1, 1))
    
    return np.column_stack(features)

def setup_batch_configuration():
    """
    设置批量处理配置
    """
    config = {
        'parallel': {
            'enable': True,
            'max_workers': min(cpu_count(), 6),
            'chunk_size': 1000
        },
        'memory': {
            'limit_gb': 8,
            'monitor': True,
            'gc_threshold': 0.8
        },
        'processing': {
            'save_intermediate': True,
            'enable_cache': True,
            'quality_check': True
        },
        'output': {
            'compression': True,
            'precision': 'float32',
            'save_confidence': True
        }
    }
    
    logger.info(f"批量处理配置: {config}")
    return config

def serial_batch_processing(scenarios, model, class_info):
    """
    串行批量处理
    """
    logger.info("开始串行批量处理...")
    
    results = {}
    start_time = time.time()
    
    for i, scenario in enumerate(scenarios):
        logger.info(f"处理场景 {i+1}/{len(scenarios)}: {scenario['name']}")
        
        scene_start_time = time.time()
        
        try:
            # 处理单个场景
            result = process_single_scene(scenario, model, class_info)
            
            processing_time = time.time() - scene_start_time
            result['processing_time'] = processing_time
            result['method'] = 'serial'
            
            results[scenario['name']] = result
            
            logger.info(f"场景 {scenario['name']} 处理完成，用时: {processing_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"场景 {scenario['name']} 处理失败: {e}")
            results[scenario['name']] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    logger.info(f"串行批量处理完成，总用时: {total_time:.2f}秒")
    
    return {
        'results': results,
        'total_time': total_time,
        'method': 'serial',
        'success_count': len([r for r in results.values() if 'error' not in r]),
        'failure_count': len([r for r in results.values() if 'error' in r])
    }

def parallel_batch_processing(scenarios, model, class_info):
    """
    并行批量处理
    """
    logger.info("开始并行批量处理...")
    
    results = {}
    start_time = time.time()
    max_workers = min(cpu_count(), len(scenarios), 4)  # 限制并行数
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_scenario = {
            executor.submit(process_single_scene_wrapper, scenario, model, class_info): scenario
            for scenario in scenarios
        }
        
        # 收集结果
        for future in as_completed(future_to_scenario):
            scenario = future_to_scenario[future]
            
            try:
                result = future.result()
                result['method'] = 'parallel'
                results[scenario['name']] = result
                
                logger.info(f"场景 {scenario['name']} 处理完成")
                
            except Exception as e:
                logger.error(f"场景 {scenario['name']} 处理失败: {e}")
                results[scenario['name']] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    logger.info(f"并行批量处理完成，总用时: {total_time:.2f}秒")
    
    return {
        'results': results,
        'total_time': total_time,
        'method': 'parallel',
        'success_count': len([r for r in results.values() if 'error' not in r]),
        'failure_count': len([r for r in results.values() if 'error' in r])
    }

def process_single_scene_wrapper(scenario, model, class_info):
    """
    单场景处理包装器（用于并行处理）
    """
    scene_start_time = time.time()
    result = process_single_scene(scenario, model, class_info)
    processing_time = time.time() - scene_start_time
    result['processing_time'] = processing_time
    return result

def process_single_scene(scenario, model, class_info):
    """
    处理单个场景
    """
    # 确定场景路径
    if 'path' in scenario:
        scene_path = scenario['path']
    else:
        scene_path = Path('data/batch_input') / scenario['name']
    
    # 加载数据
    data = np.load(scene_path / 'hyperspectral_data.npy')
    
    # 预处理（简化）
    processed_data = simple_preprocessing(data)
    
    # 特征提取
    height, width, bands = processed_data.shape
    reshaped_data = processed_data.reshape(-1, bands)
    features = extract_simple_features(reshaped_data)
    
    # 标准化
    if hasattr(model, 'scaler'):
        features_scaled = model.scaler.transform(features)
    else:
        features_scaled = features
    
    # 预测
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    # 重塑结果
    classification_map = predictions.reshape(height, width)
    confidence_map = np.max(probabilities, axis=1).reshape(height, width)
    
    # 计算统计信息
    stats = calculate_scene_statistics(classification_map, confidence_map, class_info)
    
    # 保存结果
    output_dir = Path('output/batch_processing/results') / scenario['name']
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'classification_map.npy', classification_map)
    np.save(output_dir / 'confidence_map.npy', confidence_map)
    
    with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    result = {
        'scene_name': scenario['name'],
        'classification_map': classification_map,
        'confidence_map': confidence_map,
        'statistics': stats,
        'output_dir': str(output_dir)
    }
    
    return result

def simple_preprocessing(data):
    """
    简单预处理
    """
    # 归一化
    processed = np.copy(data)
    
    for band in range(data.shape[2]):
        band_data = data[:, :, band]
        band_min = np.min(band_data)
        band_max = np.max(band_data)
        
        if band_max > band_min:
            processed[:, :, band] = (band_data - band_min) / (band_max - band_min)
        else:
            processed[:, :, band] = 0
    
    return processed

def calculate_scene_statistics(classification_map, confidence_map, class_info):
    """
    计算场景统计信息
    """
    stats = {}
    
    # 类别统计
    unique_classes, counts = np.unique(classification_map, return_counts=True)
    total_pixels = np.prod(classification_map.shape)
    
    class_stats = {}
    for class_id, count in zip(unique_classes, counts):
        if class_id in class_info:
            class_stats[int(class_id)] = {
                'name': class_info[class_id]['name'],
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
    
    stats['class_distribution'] = class_stats
    
    # 置信度统计
    stats['confidence_stats'] = {
        'mean_confidence': float(np.mean(confidence_map)),
        'std_confidence': float(np.std(confidence_map)),
        'min_confidence': float(np.min(confidence_map)),
        'max_confidence': float(np.max(confidence_map)),
        'low_confidence_ratio': float(np.sum(confidence_map < 0.7) / total_pixels)
    }
    
    # 空间统计
    stats['spatial_stats'] = {
        'total_pixels': total_pixels,
        'height': classification_map.shape[0],
        'width': classification_map.shape[1],
        'unique_classes': len(unique_classes)
    }
    
    return stats

def compare_processing_performance(serial_results, parallel_results):
    """
    比较串行和并行处理性能
    """
    logger.info("比较处理性能...")
    
    comparison = {
        'serial': {
            'total_time': serial_results['total_time'],
            'success_count': serial_results['success_count'],
            'failure_count': serial_results['failure_count'],
            'avg_time_per_scene': serial_results['total_time'] / len(serial_results['results']) if serial_results['results'] else 0
        },
        'parallel': {
            'total_time': parallel_results['total_time'],
            'success_count': parallel_results['success_count'],
            'failure_count': parallel_results['failure_count'],
            'avg_time_per_scene': parallel_results['total_time'] / len(parallel_results['results']) if parallel_results['results'] else 0
        }
    }
    
    # 计算加速比
    if serial_results['total_time'] > 0:
        speedup = serial_results['total_time'] / parallel_results['total_time']
        comparison['speedup'] = speedup
        comparison['efficiency'] = speedup / cpu_count()
    
    # 系统资源使用情况
    comparison['system_info'] = {
        'cpu_count': cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    logger.info(f"性能对比完成 - 加速比: {comparison.get('speedup', 'N/A'):.2f}")
    
    return comparison

def temporal_batch_analysis(scenarios, batch_results, class_info):
    """
    时序批量分析
    """
    logger.info("开始时序批量分析...")
    
    # 按年份和季节组织数据
    temporal_data = {}
    
    for scenario in scenarios:
        scene_name = scenario['name']
        year = scenario['year']
        season = scenario['season']
        
        if scene_name in batch_results['results'] and 'error' not in batch_results['results'][scene_name]:
            result = batch_results['results'][scene_name]
            
            if year not in temporal_data:
                temporal_data[year] = {}
            
            temporal_data[year][season] = {
                'scene_name': scene_name,
                'statistics': result['statistics'],
                'classification_map': result['classification_map']
            }
    
    # 时序变化分析
    temporal_analysis = {}
    
    # 1. 季节变化分析
    seasonal_changes = analyze_seasonal_changes(temporal_data, class_info)
    temporal_analysis['seasonal_changes'] = seasonal_changes
    
    # 2. 年际变化分析
    annual_changes = analyze_annual_changes(temporal_data, class_info)
    temporal_analysis['annual_changes'] = annual_changes
    
    # 3. 趋势分析
    trend_analysis = analyze_temporal_trends(temporal_data, class_info)
    temporal_analysis['trend_analysis'] = trend_analysis
    
    # 保存时序分析结果
    output_dir = Path('output/batch_processing/temporal_analysis')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'temporal_analysis.json', 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        serializable_analysis = convert_numpy_to_list(temporal_analysis)
        json.dump(serializable_analysis, f, ensure_ascii=False, indent=2)
    
    logger.info("时序批量分析完成")
    
    return temporal_analysis

def analyze_seasonal_changes(temporal_data, class_info):
    """
    分析季节变化
    """
    seasonal_changes = {}
    
    for year, year_data in temporal_data.items():
        year_changes = {}
        
        # 计算各季节的类别比例
        for season, season_data in year_data.items():
            stats = season_data['statistics']
            class_percentages = {}
            
            for class_id, class_stat in stats['class_distribution'].items():
                class_percentages[class_id] = class_stat['percentage']
            
            year_changes[season] = class_percentages
        
        seasonal_changes[year] = year_changes
    
    return seasonal_changes

def analyze_annual_changes(temporal_data, class_info):
    """
    分析年际变化
    """
    annual_changes = {}
    
    # 按季节比较不同年份
    seasons = ['spring', 'summer', 'autumn', 'winter']
    
    for season in seasons:
        season_annual_data = {}
        
        for year, year_data in temporal_data.items():
            if season in year_data:
                stats = year_data[season]['statistics']
                class_percentages = {}
                
                for class_id, class_stat in stats['class_distribution'].items():
                    class_percentages[class_id] = class_stat['percentage']
                
                season_annual_data[year] = class_percentages
        
        annual_changes[season] = season_annual_data
    
    return annual_changes

def analyze_temporal_trends(temporal_data, class_info):
    """
    分析时序趋势
    """
    trends = {}
    
    # 提取时间序列数据
    time_series_data = []
    
    for year in sorted(temporal_data.keys()):
        for season in ['spring', 'summer', 'autumn', 'winter']:
            if season in temporal_data[year]:
                data_point = {
                    'timestamp': f"{year}_{season}",
                    'year': year,
                    'season': season,
                    'class_distribution': temporal_data[year][season]['statistics']['class_distribution']
                }
                time_series_data.append(data_point)
    
    # 计算趋势
    for class_id in class_info.keys():
        if class_id == 0:  # 跳过背景
            continue
        
        class_percentages = []
        timestamps = []
        
        for data_point in time_series_data:
            class_dist = data_point['class_distribution']
            if str(class_id) in class_dist:
                class_percentages.append(class_dist[str(class_id)]['percentage'])
                timestamps.append(data_point['timestamp'])
        
        if len(class_percentages) > 1:
            # 简单线性趋势计算
            trend_slope = calculate_simple_trend(class_percentages)
            
            trends[class_id] = {
                'class_name': class_info[class_id]['name'],
                'trend_slope': trend_slope,
                'values': class_percentages,
                'timestamps': timestamps,
                'trend_direction': 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable'
            }
    
    return trends

def calculate_simple_trend(values):
    """
    计算简单线性趋势
    """
    if len(values) < 2:
        return 0
    
    n = len(values)
    x = np.arange(n)
    y = np.array(values)
    
    # 简单线性回归斜率
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0
    
    slope = numerator / denominator
    return slope

def batch_quality_control(batch_results, class_info):
    """
    批量质量控制
    """
    logger.info("开始批量质量控制...")
    
    quality_results = {
        'overall_quality': {},
        'scene_quality': {},
        'quality_flags': {},
        'recommendations': []
    }
    
    successful_results = {k: v for k, v in batch_results['results'].items() 
                         if 'error' not in v}
    
    if not successful_results:
        logger.warning("没有成功处理的场景用于质量控制")
        return quality_results
    
    # 1. 置信度质量检查
    confidence_scores = []
    low_confidence_scenes = []
    
    for scene_name, result in successful_results.items():
        confidence_stats = result['statistics']['confidence_stats']
        mean_confidence = confidence_stats['mean_confidence']
        low_confidence_ratio = confidence_stats['low_confidence_ratio']
        
        confidence_scores.append(mean_confidence)
        
        if mean_confidence < 0.6 or low_confidence_ratio > 0.3:
            low_confidence_scenes.append(scene_name)
            quality_results['quality_flags'][scene_name] = 'low_confidence'
    
    # 2. 类别分布合理性检查
    unrealistic_scenes = []
    
    for scene_name, result in successful_results.items():
        class_distribution = result['statistics']['class_distribution']
        
        # 检查是否有异常的类别分布
        water_percentage = sum(stats['percentage'] for class_id, stats in class_distribution.items() 
                              if int(class_id) in [1, 2])  # 水体类别
        
        vegetation_percentage = sum(stats['percentage'] for class_id, stats in class_distribution.items() 
                                   if int(class_id) in [3, 4, 5])  # 植被类别
        
        # 湿地场景应该有合理的水体和植被比例
        if water_percentage < 5 or vegetation_percentage < 10:
            unrealistic_scenes.append(scene_name)
            quality_results['quality_flags'][scene_name] = 'unrealistic_distribution'
    
    # 3. 一致性检查
    consistency_issues = []
    
    if len(successful_results) > 1:
        # 比较相似场景的结果一致性
        scene_similarities = calculate_scene_similarities(successful_results)
        
        for scene_pair, similarity in scene_similarities.items():
            if similarity < 0.5:  # 相似场景但结果差异较大
                consistency_issues.append(scene_pair)
    
    # 4. 空间连续性检查
    spatial_issues = []
    
    for scene_name, result in successful_results.items():
        classification_map = result['classification_map']
        spatial_consistency = check_spatial_consistency(classification_map)
        
        if spatial_consistency < 0.7:
            spatial_issues.append(scene_name)
            quality_results['quality_flags'][scene_name] = 'spatial_inconsistency'
    
    # 汇总质量评估
    quality_results['overall_quality'] = {
        'total_scenes': len(batch_results['results']),
        'successful_scenes': len(successful_results),
        'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'low_confidence_scenes': len(low_confidence_scenes),
        'unrealistic_scenes': len(unrealistic_scenes),
        'spatial_issues': len(spatial_issues),
        'overall_quality_score': calculate_overall_quality_score(
            len(successful_results), len(low_confidence_scenes), 
            len(unrealistic_scenes), len(spatial_issues)
        )
    }
    
    # 各场景质量评分
    for scene_name, result in successful_results.items():
        scene_quality_score = calculate_scene_quality_score(scene_name, result, quality_results['quality_flags'])
        quality_results['scene_quality'][scene_name] = scene_quality_score
    
    # 质量改进建议
    quality_results['recommendations'] = generate_quality_recommendations(
        low_confidence_scenes, unrealistic_scenes, spatial_issues
    )
    
    # 保存质量控制结果
    output_dir = Path('output/batch_processing/quality_control')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(quality_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"批量质量控制完成，整体质量评分: {quality_results['overall_quality']['overall_quality_score']:.2f}")
    
    return quality_results

def calculate_scene_similarities(results):
    """
    计算场景相似度
    """
    similarities = {}
    scene_names = list(results.keys())
    
    for i in range(len(scene_names)):
        for j in range(i + 1, len(scene_names)):
            scene1 = scene_names[i]
            scene2 = scene_names[j]
            
            # 基于类别分布计算相似度
            dist1 = results[scene1]['statistics']['class_distribution']
            dist2 = results[scene2]['statistics']['class_distribution']
            
            similarity = calculate_distribution_similarity(dist1, dist2)
            similarities[f"{scene1}_vs_{scene2}"] = similarity
    
    return similarities

def calculate_distribution_similarity(dist1, dist2):
    """
    计算分布相似度
    """
    # 获取所有类别
    all_classes = set(dist1.keys()) | set(dist2.keys())
    
    # 构建比例向量
    vec1 = []
    vec2 = []
    
    for class_id in all_classes:
        percentage1 = dist1.get(class_id, {}).get('percentage', 0)
        percentage2 = dist2.get(class_id, {}).get('percentage', 0)
        vec1.append(percentage1)
        vec2.append(percentage2)
    
    # 计算余弦相似度
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

def check_spatial_consistency(classification_map):
    """
    检查空间一致性
    """
    # 计算邻域一致性
    height, width = classification_map.shape
    consistent_pixels = 0
    total_pixels = 0
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center_class = classification_map[i, j]
            
            # 检查8邻域
            neighbors = [
                classification_map[i-1, j-1], classification_map[i-1, j], classification_map[i-1, j+1],
                classification_map[i, j-1],                              classification_map[i, j+1],
                classification_map[i+1, j-1], classification_map[i+1, j], classification_map[i+1, j+1]
            ]
            
            # 计算与邻域的一致性
            same_class_neighbors = sum(1 for neighbor in neighbors if neighbor == center_class)
            
            if same_class_neighbors >= 4:  # 至少一半邻居是同类
                consistent_pixels += 1
            
            total_pixels += 1
    
    consistency = consistent_pixels / total_pixels if total_pixels > 0 else 0
    return consistency

def calculate_overall_quality_score(total_successful, low_confidence_count, 
                                  unrealistic_count, spatial_issues_count):
    """
    计算整体质量评分
    """
    if total_successful == 0:
        return 0
    
    # 基础分数
    base_score = 100
    
    # 扣分项
    low_confidence_penalty = (low_confidence_count / total_successful) * 30
    unrealistic_penalty = (unrealistic_count / total_successful) * 40
    spatial_penalty = (spatial_issues_count / total_successful) * 30
    
    final_score = base_score - low_confidence_penalty - unrealistic_penalty - spatial_penalty
    return max(0, final_score)

def calculate_scene_quality_score(scene_name, result, quality_flags):
    """
    计算单场景质量评分
    """
    base_score = 100
    
    # 置信度评分
    confidence_stats = result['statistics']['confidence_stats']
    confidence_score = confidence_stats['mean_confidence'] * 40
    
    # 质量标志扣分
    flag_penalty = 0
    if scene_name in quality_flags:
        flag = quality_flags[scene_name]
        if flag == 'low_confidence':
            flag_penalty += 20
        elif flag == 'unrealistic_distribution':
            flag_penalty += 30
        elif flag == 'spatial_inconsistency':
            flag_penalty += 25
    
    final_score = confidence_score + 60 - flag_penalty
    return max(0, min(100, final_score))

def generate_quality_recommendations(low_confidence_scenes, unrealistic_scenes, spatial_issues):
    """
    生成质量改进建议
    """
    recommendations = []
    
    if low_confidence_scenes:
        recommendations.append({
            'issue': 'low_confidence',
            'affected_scenes': low_confidence_scenes,
            'recommendation': '对低置信度场景进行人工复核，考虑增加训练样本或调整模型参数'
        })
    
    if unrealistic_scenes:
        recommendations.append({
            'issue': 'unrealistic_distribution',
            'affected_scenes': unrealistic_scenes,
            'recommendation': '检查数据质量和预处理流程，确认场景是否为典型湿地环境'
        })
    
    if spatial_issues:
        recommendations.append({
            'issue': 'spatial_inconsistency',
            'affected_scenes': spatial_issues,
            'recommendation': '考虑应用空间后处理算法改善分类结果的空间连续性'
        })
    
    return recommendations

def batch_statistical_analysis(batch_results, temporal_results, class_info):
    """
    批量统计分析
    """
    logger.info("开始批量统计分析...")
    
    successful_results = {k: v for k, v in batch_results['results'].items() 
                         if 'error' not in v}
    
    statistical_results = {
        'summary_statistics': {},
        'class_statistics': {},
        'confidence_analysis': {},
        'processing_statistics': {},
        'temporal_statistics': {}
    }
    
    if not successful_results:
        logger.warning("没有成功处理的场景用于统计分析")
        return statistical_results
    
    # 1. 总体统计
    total_scenes = len(batch_results['results'])
    successful_scenes = len(successful_results)
    processing_times = [result.get('processing_time', 0) for result in successful_results.values()]
    
    statistical_results['summary_statistics'] = {
        'total_scenes': total_scenes,
        'successful_scenes': successful_scenes,
        'success_rate': successful_scenes / total_scenes * 100,
        'mean_processing_time': np.mean(processing_times) if processing_times else 0,
        'total_processing_time': sum(processing_times)
    }
    
    # 2. 类别统计汇总
    class_aggregation = {}
    
    for class_id in class_info.keys():
        if class_id == 0:  # 跳过背景
            continue
        
        class_percentages = []
        class_pixel_counts = []
        
        for result in successful_results.values():
            class_dist = result['statistics']['class_distribution']
            if str(class_id) in class_dist:
                class_percentages.append(class_dist[str(class_id)]['percentage'])
                class_pixel_counts.append(class_dist[str(class_id)]['pixel_count'])
        
        if class_percentages:
            class_aggregation[class_id] = {
                'class_name': class_info[class_id]['name'],
                'mean_percentage': np.mean(class_percentages),
                'std_percentage': np.std(class_percentages),
                'min_percentage': np.min(class_percentages),
                'max_percentage': np.max(class_percentages),
                'total_pixels': sum(class_pixel_counts),
                'scene_count': len(class_percentages)
            }
    
    statistical_results['class_statistics'] = class_aggregation
    
    # 3. 置信度分析
    all_confidences = []
    low_confidence_ratios = []
    
    for result in successful_results.values():
        confidence_stats = result['statistics']['confidence_stats']
        all_confidences.append(confidence_stats['mean_confidence'])
        low_confidence_ratios.append(confidence_stats['low_confidence_ratio'])
    
    statistical_results['confidence_analysis'] = {
        'overall_mean_confidence': np.mean(all_confidences),
        'confidence_std': np.std(all_confidences),
        'min_confidence': np.min(all_confidences),
        'max_confidence': np.max(all_confidences),
        'mean_low_confidence_ratio': np.mean(low_confidence_ratios),
        'high_confidence_scenes': len([c for c in all_confidences if c > 0.8]),
        'low_confidence_scenes': len([c for c in all_confidences if c < 0.6])
    }
    
    # 4. 处理性能统计
    statistical_results['processing_statistics'] = {
        'mean_processing_time': np.mean(processing_times) if processing_times else 0,
        'median_processing_time': np.median(processing_times) if processing_times else 0,
        'min_processing_time': np.min(processing_times) if processing_times else 0,
        'max_processing_time': np.max(processing_times) if processing_times else 0,
        'total_processing_time': sum(processing_times),
        'processing_efficiency': successful_scenes / sum(processing_times) if sum(processing_times) > 0 else 0
    }
    
    # 5. 时序统计
    if temporal_results and 'trend_analysis' in temporal_results:
        trend_stats = {}
        trend_analysis = temporal_results['trend_analysis']
        
        for class_id, trend_data in trend_analysis.items():
            trend_direction = trend_data['trend_direction']
            trend_stats[trend_direction] = trend_stats.get(trend_direction, 0) + 1
        
        statistical_results['temporal_statistics'] = {
            'trend_distribution': trend_stats,
            'stable_classes': trend_stats.get('stable', 0),
            'increasing_classes': trend_stats.get('increasing', 0),
            'decreasing_classes': trend_stats.get('decreasing', 0)
        }
    
    # 保存统计结果
    output_dir = Path('output/batch_processing/statistics')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'statistical_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(statistical_results, f, ensure_ascii=False, indent=2)
    
    logger.info("批量统计分析完成")
    
    return statistical_results

def convert_numpy_to_list(obj):
    """
    递归转换numpy数组为列表，以便JSON序列化
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def create_batch_visualizations(batch_results, temporal_results, quality_results,
                              statistical_results, performance_comparison, class_info):
    """
    创建批量处理可视化
    """
    logger.info("创建批量处理可视化...")
    
    # 1. 处理性能对比图
    create_performance_comparison_chart(performance_comparison)
    
    # 2. 类别分布统计图
    create_class_distribution_charts(statistical_results, class_info)
    
    # 3. 置信度分析图
    create_confidence_analysis_charts(statistical_results, batch_results)
    
    # 4. 时序变化图
    if temporal_results:
        create_temporal_analysis_charts(temporal_results, class_info)
    
    # 5. 质量控制图
    create_quality_control_charts(quality_results)
    
    # 6. 批量处理总览图
    create_batch_overview_charts(batch_results, statistical_results, class_info)
    
    logger.info("批量处理可视化创建完成")

def create_performance_comparison_chart(performance_comparison):
    """
    创建性能对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 处理时间对比
    methods = ['串行处理', '并行处理']
    times = [
        performance_comparison['serial']['total_time'],
        performance_comparison['parallel']['total_time']
    ]
    
    bars = axes[0].bar(methods, times, color=['orange', 'skyblue'], alpha=0.7)
    axes[0].set_ylabel('处理时间 (秒)')
    axes[0].set_title('串行 vs 并行处理时间对比')
    axes[0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, time in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{time:.1f}s', ha='center', va='bottom')
    
    # 系统资源利用图
    cpu_count = performance_comparison['system_info']['cpu_count']
    speedup = performance_comparison.get('speedup', 1)
    efficiency = performance_comparison.get('efficiency', 0)
    
    metrics = ['CPU核心数', '加速比', '并行效率']
    values = [cpu_count, speedup, efficiency * 100]  # 效率转换为百分比
    
    bars = axes[1].bar(metrics, values, color=['green', 'blue', 'red'], alpha=0.7)
    axes[1].set_ylabel('数值')
    axes[1].set_title('并行处理性能指标')
    axes[1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/batch_processing/figures/performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_class_distribution_charts(statistical_results, class_info):
    """
    创建类别分布图表
    """
    class_stats = statistical_results['class_statistics']
    
    if not class_stats:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 平均百分比条形图
    class_names = [stats['class_name'] for stats in class_stats.values()]
    mean_percentages = [stats['mean_percentage'] for stats in class_stats.values()]
    std_percentages = [stats['std_percentage'] for stats in class_stats.values()]
    
    colors = [class_info.get(int(class_id), {}).get('color', 'gray') 
              for class_id in class_stats.keys()]
    
    bars = axes[0, 0].bar(class_names, mean_percentages, yerr=std_percentages,
                         color=colors, alpha=0.7, capsize=5)
    axes[0, 0].set_ylabel('平均百分比 (%)')
    axes[0, 0].set_title('各类别平均覆盖率')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 总像素数饼图
    total_pixels = [stats['total_pixels'] for stats in class_stats.values()]
    
    axes[0, 1].pie(total_pixels, labels=class_names, colors=colors, autopct='%1.1f%%')
    axes[0, 1].set_title('总像素分布')
    
    # 3. 变异系数图
    cv_values = [stats['std_percentage'] / stats['mean_percentage'] * 100 
                 if stats['mean_percentage'] > 0 else 0 
                 for stats in class_stats.values()]
    
    bars = axes[1, 0].bar(class_names, cv_values, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('变异系数 (%)')
    axes[1, 0].set_title('各类别变异程度')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 分布范围图
    min_percentages = [stats['min_percentage'] for stats in class_stats.values()]
    max_percentages = [stats['max_percentage'] for stats in class_stats.values()]
    
    x_pos = np.arange(len(class_names))
    axes[1, 1].fill_between(x_pos, min_percentages, max_percentages, 
                           color='lightblue', alpha=0.5, label='分布范围')
    axes[1, 1].plot(x_pos, mean_percentages, 'ro-', label='平均值')
    
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(class_names, rotation=45)
    axes[1, 1].set_ylabel('百分比 (%)')
    axes[1, 1].set_title('各类别分布范围')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/batch_processing/figures/class_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_analysis_charts(statistical_results, batch_results):
    """
    创建置信度分析图表
    """
    confidence_stats = statistical_results['confidence_analysis']
    successful_results = {k: v for k, v in batch_results['results'].items() 
                         if 'error' not in v}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 置信度分布直方图
    all_mean_confidences = [result['statistics']['confidence_stats']['mean_confidence'] 
                           for result in successful_results.values()]
    
    axes[0, 0].hist(all_mean_confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(confidence_stats['overall_mean_confidence'], color='red', 
                      linestyle='--', label=f"平均值: {confidence_stats['overall_mean_confidence']:.3f}")
    axes[0, 0].set_xlabel('平均置信度')
    axes[0, 0].set_ylabel('场景数量')
    axes[0, 0].set_title('场景置信度分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 置信度等级统计
    confidence_levels = ['低(<0.6)', '中(0.6-0.8)', '高(>0.8)']
    confidence_counts = [
        confidence_stats['low_confidence_scenes'],
        len(all_mean_confidences) - confidence_stats['low_confidence_scenes'] - confidence_stats['high_confidence_scenes'],
        confidence_stats['high_confidence_scenes']
    ]
    
    colors = ['red', 'orange', 'green']
    bars = axes[0, 1].bar(confidence_levels, confidence_counts, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('场景数量')
    axes[0, 1].set_title('置信度等级分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, confidence_counts):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom')
    
    # 3. 低置信度比例散点图
    scene_names = list(successful_results.keys())
    low_confidence_ratios = [result['statistics']['confidence_stats']['low_confidence_ratio'] 
                            for result in successful_results.values()]
    
    scatter = axes[1, 0].scatter(range(len(scene_names)), low_confidence_ratios, 
                                c=all_mean_confidences, cmap='RdYlGn', alpha=0.7, s=60)
    axes[1, 0].axhline(0.3, color='red', linestyle='--', label='警戒线 (30%)')
    axes[1, 0].set_xlabel('场景编号')
    axes[1, 0].set_ylabel('低置信度像素比例')
    axes[1, 0].set_title('各场景低置信度比例')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加颜色条
    plt.colorbar(scatter, ax=axes[1, 0], label='平均置信度')
    
    # 4. 置信度统计汇总
    stats_labels = ['平均置信度', '标准差', '最小值', '最大值']
    stats_values = [
        confidence_stats['overall_mean_confidence'],
        confidence_stats['confidence_std'],
        confidence_stats['min_confidence'],
        confidence_stats['max_confidence']
    ]
    
    bars = axes[1, 1].bar(stats_labels, stats_values, color='lightcoral', alpha=0.7)
    axes[1, 1].set_ylabel('置信度值')
    axes[1, 1].set_title('置信度统计汇总')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, stats_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/batch_processing/figures/confidence_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_analysis_charts(temporal_results, class_info):
    """
    创建时序分析图表
    """
    if 'trend_analysis' not in temporal_results:
        return
    
    trend_analysis = temporal_results['trend_analysis']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 趋势方向统计
    trend_directions = {}
    for trend_data in trend_analysis.values():
        direction = trend_data['trend_direction']
        trend_directions[direction] = trend_directions.get(direction, 0) + 1
    
    colors = {'increasing': 'green', 'decreasing': 'red', 'stable': 'blue'}
    directions = list(trend_directions.keys())
    counts = list(trend_directions.values())
    chart_colors = [colors.get(direction, 'gray') for direction in directions]
    
    bars = axes[0, 0].bar(directions, counts, color=chart_colors, alpha=0.7)
    axes[0, 0].set_ylabel('类别数量')
    axes[0, 0].set_title('时序趋势方向统计')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom')
    
    # 2. 趋势斜率分布
    trend_slopes = [trend_data['trend_slope'] for trend_data in trend_analysis.values()]
    class_names = [trend_data['class_name'] for trend_data in trend_analysis.values()]
    
    colors_slope = ['green' if slope > 0.1 else 'red' if slope < -0.1 else 'blue' 
                   for slope in trend_slopes]
    
    bars = axes[0, 1].bar(class_names, trend_slopes, color=colors_slope, alpha=0.7)
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].axhline(0.1, color='green', linestyle='--', alpha=0.5, label='增长阈值')
    axes[0, 1].axhline(-0.1, color='red', linestyle='--', alpha=0.5, label='下降阈值')
    axes[0, 1].set_ylabel('趋势斜率')
    axes[0, 1].set_title('各类别时序趋势斜率')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 时序变化曲线
    for class_id, trend_data in list(trend_analysis.items())[:4]:  # 显示前4个类别
        values = trend_data['values']
        timestamps = trend_data['timestamps']
        class_name = trend_data['class_name']
        
        # 简化时间戳显示
        x_labels = [ts.split('_')[1] for ts in timestamps]  # 只显示季节
        
        axes[1, 0].plot(range(len(values)), values, marker='o', 
                       label=class_name, linewidth=2)
    
    axes[1, 0].set_xlabel('时间点')
    axes[1, 0].set_ylabel('覆盖率 (%)')
    axes[1, 0].set_title('主要类别时序变化')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 季节变化模式
    if 'seasonal_changes' in temporal_results:
        seasonal_data = temporal_results['seasonal_changes']
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        # 计算各季节的平均类别分布
        seasonal_means = {}
        for season in seasons:
            seasonal_means[season] = {}
            
            for year_data in seasonal_data.values():
                if season in year_data:
                    for class_id, percentage in year_data[season].items():
                        if class_id not in seasonal_means[season]:
                            seasonal_means[season][class_id] = []
                        seasonal_means[season][class_id].append(percentage)
        
        # 绘制主要类别的季节变化
        main_classes = list(trend_analysis.keys())[:3]  # 前3个类别
        
        for class_id in main_classes:
            class_name = class_info.get(int(class_id), {}).get('name', f'Class_{class_id}')
            seasonal_values = []
            
            for season in seasons:
                if class_id in seasonal_means[season]:
                    seasonal_values.append(np.mean(seasonal_means[season][class_id]))
                else:
                    seasonal_values.append(0)
            
            axes[1, 1].plot(seasons, seasonal_values, marker='o', 
                           label=class_name, linewidth=2)
        
        axes[1, 1].set_xlabel('季节')
        axes[1, 1].set_ylabel('覆盖率 (%)')
        axes[1, 1].set_title('季节变化模式')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/batch_processing/figures/temporal_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_quality_control_charts(quality_results):
    """
    创建质量控制图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    overall_quality = quality_results['overall_quality']
    
    # 1. 整体质量评分仪表盘
    quality_score = overall_quality['overall_quality_score']
    
    # 创建半圆形仪表盘
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # 背景扇形
    axes[0, 0].fill_between(theta, 0, r, color='lightgray', alpha=0.3)
    
    # 质量等级区域
    poor_end = np.pi * 0.33
    fair_end = np.pi * 0.67
    good_end = np.pi
    
    axes[0, 0].fill_between(theta[theta <= poor_end], 0, r, color='red', alpha=0.3, label='差 (<60)')
    axes[0, 0].fill_between(theta[(theta > poor_end) & (theta <= fair_end)], 0, r, 
                           color='orange', alpha=0.3, label='中 (60-80)')
    axes[0, 0].fill_between(theta[theta > fair_end], 0, r, 
                           color='green', alpha=0.3, label='优 (>80)')
    
    # 质量指针
    pointer_angle = np.pi * (1 - quality_score / 100)
    pointer_x = r * np.cos(pointer_angle)
    pointer_y = r * np.sin(pointer_angle)
    
    axes[0, 0].arrow(0, 0, pointer_x, pointer_y, head_width=0.05, head_length=0.1, 
                    fc='black', ec='black', linewidth=3)
    
    axes[0, 0].text(0, -0.3, f'质量评分\n{quality_score:.1f}', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
    
    axes[0, 0].set_xlim(-1.2, 1.2)
    axes[0, 0].set_ylim(-0.5, 1.2)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].axis('off')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_title('整体质量评分', fontsize=14)
    
    # 2. 质量问题统计
    issue_types = ['低置信度', '分布异常', '空间不一致']
    issue_counts = [
        overall_quality['low_confidence_scenes'],
        overall_quality['unrealistic_scenes'],
        overall_quality['spatial_issues']
    ]
    
    colors = ['orange', 'red', 'purple']
    bars = axes[0, 1].bar(issue_types, issue_counts, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel('问题场景数')
    axes[0, 1].set_title('质量问题分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, issue_counts):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom')
    
    # 3. 成功率统计
    total_scenes = overall_quality['total_scenes']
    successful_scenes = overall_quality['successful_scenes']
    failed_scenes = total_scenes - successful_scenes
    
    sizes = [successful_scenes, failed_scenes]
    labels = ['成功', '失败']
    colors = ['green', 'red']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title(f'处理成功率\n({successful_scenes}/{total_scenes})')
    
    # 4. 场景质量评分分布
    scene_quality = quality_results['scene_quality']
    
    if scene_quality:
        quality_scores = list(scene_quality.values())
        
        axes[1, 1].hist(quality_scores, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(quality_scores), color='red', linestyle='--', 
                          label=f'平均分: {np.mean(quality_scores):.1f}')
        axes[1, 1].set_xlabel('质量评分')
        axes[1, 1].set_ylabel('场景数量')
        axes[1, 1].set_title('场景质量评分分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/batch_processing/figures/quality_control.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_batch_overview_charts(batch_results, statistical_results, class_info):
    """
    创建批量处理总览图
    """
    successful_results = {k: v for k, v in batch_results['results'].items() 
                         if 'error' not in v}
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 分类结果网格展示 (3x2)
    scene_names = list(successful_results.keys())[:6]  # 显示前6个场景
    
    for i, scene_name in enumerate(scene_names):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        classification_map = successful_results[scene_name]['classification_map']
        
        # 创建颜色映射
        colors = ['black'] + [class_info.get(j, {}).get('color', 'gray') 
                             for j in range(1, len(class_info))]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        im = ax.imshow(classification_map, cmap=cmap)
        ax.set_title(f'{scene_name}', fontsize=10)
        ax.axis('off')
    
    # 2. 处理统计总览 (右上)
    ax_stats = fig.add_subplot(gs[0, 2:])
    
    summary_stats = statistical_results['summary_statistics']
    stats_labels = ['总场景数', '成功场景数', '平均处理时间(s)']
    stats_values = [
        summary_stats['total_scenes'],
        summary_stats['successful_scenes'],
        summary_stats['mean_processing_time']
    ]
    
    bars = ax_stats.bar(stats_labels, stats_values, color=['blue', 'green', 'orange'], alpha=0.7)
    ax_stats.set_title('处理统计总览', fontsize=12)
    ax_stats.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, stats_values):
        ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_values) * 0.02,
                     f'{value:.1f}', ha='center', va='bottom')
    
    # 3. 类别覆盖率汇总 (右中)
    ax_class = fig.add_subplot(gs[1, 2:])
    
    class_stats = statistical_results['class_statistics']
    if class_stats:
        class_names = [stats['class_name'] for stats in class_stats.values()]
        mean_percentages = [stats['mean_percentage'] for stats in class_stats.values()]
        
        colors = [class_info.get(int(class_id), {}).get('color', 'gray') 
                  for class_id in class_stats.keys()]
        
        bars = ax_class.bar(class_names, mean_percentages, color=colors, alpha=0.7)
        ax_class.set_ylabel('平均覆盖率 (%)')
        ax_class.set_title('各类别平均覆盖率', fontsize=12)
        ax_class.tick_params(axis='x', rotation=45)
        ax_class.grid(True, alpha=0.3)
    
    # 4. 置信度分布 (右下)
    ax_conf = fig.add_subplot(gs[2, 2:])
    
    confidence_stats = statistical_results['confidence_analysis']
    confidence_levels = ['高置信度\n(>0.8)', '中置信度\n(0.6-0.8)', '低置信度\n(<0.6)']
    
    total_scenes = summary_stats['successful_scenes']
    high_conf = confidence_stats['high_confidence_scenes']
    low_conf = confidence_stats['low_confidence_scenes']
    medium_conf = total_scenes - high_conf - low_conf
    
    confidence_counts = [high_conf, medium_conf, low_conf]
    colors = ['green', 'orange', 'red']
    
    wedges, texts, autotexts = ax_conf.pie(confidence_counts, labels=confidence_levels, 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
    ax_conf.set_title('置信度分布', fontsize=12)
    
    # 总标题
    fig.suptitle('湿地高光谱批量处理总览', fontsize=16, fontweight='bold')
    
    plt.savefig('output/batch_processing/figures/batch_overview.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_batch_report(batch_results, temporal_results, quality_results,
                         statistical_results, performance_comparison, class_info):
    """
    生成批量处理报告
    """
    logger.info("生成批量处理报告...")
    
    report_path = Path('output/batch_processing/reports/batch_processing_report.md')
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 湿地高光谱分类系统 - 批量处理报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 执行摘要
        f.write("## 📋 执行摘要\n\n")
        summary_stats = statistical_results['summary_statistics']
        
        f.write(f"本次批量处理共处理 **{summary_stats['total_scenes']}** 个场景，")
        f.write(f"成功处理 **{summary_stats['successful_scenes']}** 个，")
        f.write(f"成功率达到 **{summary_stats['success_rate']:.1f}%**。\n\n")
        
        # 2. 性能分析
        f.write("## ⚡ 性能分析\n\n")
        f.write("### 处理效率对比\n\n")
        f.write("| 处理方式 | 总耗时(秒) | 平均耗时(秒) | 加速比 |\n")
        f.write("|----------|------------|--------------|--------|\n")
        
        serial_time = performance_comparison['serial']['total_time']
        parallel_time = performance_comparison['parallel']['total_time']
        speedup = performance_comparison.get('speedup', 1)
        
        f.write(f"| 串行处理 | {serial_time:.2f} | "
               f"{performance_comparison['serial']['avg_time_per_scene']:.2f} | 1.00 |\n")
        f.write(f"| 并行处理 | {parallel_time:.2f} | "
               f"{performance_comparison['parallel']['avg_time_per_scene']:.2f} | {speedup:.2f} |\n\n")
        
        # 3. 分类结果统计
        f.write("## 📊 分类结果统计\n\n")
        f.write("### 各类别覆盖率统计\n\n")
        f.write("| 类别 | 平均覆盖率(%) | 标准差(%) | 最小值(%) | 最大值(%) |\n")
        f.write("|------|---------------|-----------|-----------|----------|\n")
        
        class_stats = statistical_results['class_statistics']
        for class_id, stats in class_stats.items():
            f.write(f"| {stats['class_name']} | {stats['mean_percentage']:.2f} | "
                   f"{stats['std_percentage']:.2f} | {stats['min_percentage']:.2f} | "
                   f"{stats['max_percentage']:.2f} |\n")
        
        f.write("\n")
        
        # 4. 质量控制
        f.write("## 🔍 质量控制\n\n")
        overall_quality = quality_results['overall_quality']
        
        f.write(f"- **整体质量评分**: {overall_quality['overall_quality_score']:.2f}/100\n")
        f.write(f"- **低置信度场景**: {overall_quality['low_confidence_scenes']} 个\n")
        f.write(f"- **分布异常场景**: {overall_quality['unrealistic_scenes']} 个\n")
        f.write(f"- **空间不一致场景**: {overall_quality['spatial_issues']} 个\n\n")
        
        # 5. 时序分析
        if temporal_results and 'trend_analysis' in temporal_results:
            f.write("## 📈 时序分析\n\n")
            trend_analysis = temporal_results['trend_analysis']
            
            f.write("### 类别变化趋势\n\n")
            f.write("| 类别 | 趋势方向 | 趋势斜率 | 变化描述 |\n")
            f.write("|------|----------|----------|----------|\n")
            
            for class_id, trend_data in trend_analysis.items():
                direction = trend_data['trend_direction']
                slope = trend_data['trend_slope']
                
                if direction == 'increasing':
                    description = '面积增加趋势'
                elif direction == 'decreasing':
                    description = '面积减少趋势'
                else:
                    description = '面积相对稳定'
                
                f.write(f"| {trend_data['class_name']} | {direction} | "
                       f"{slope:.4f} | {description} |\n")
            
            f.write("\n")
        
        # 6. 置信度分析
        f.write("## 🎯 置信度分析\n\n")
        confidence_stats = statistical_results['confidence_analysis']
        
        f.write(f"- **平均置信度**: {confidence_stats['overall_mean_confidence']:.3f}\n")
        f.write(f"- **置信度标准差**: {confidence_stats['confidence_std']:.3f}\n")
        f.write(f"- **高置信度场景** (>0.8): {confidence_stats['high_confidence_scenes']} 个\n")
        f.write(f"- **低置信度场景** (<0.6): {confidence_stats['low_confidence_scenes']} 个\n")
        f.write(f"- **平均低置信度像素比例**: {confidence_stats['mean_low_confidence_ratio']:.3f}\n\n")
        
        # 7. 建议和改进
        f.write("## 💡 建议和改进\n\n")
        
        if quality_results.get('recommendations'):
            f.write("### 质量改进建议\n\n")
            for rec in quality_results['recommendations']:
                f.write(f"**{rec['issue']}**: {rec['recommendation']}\n\n")
        
        f.write("### 总体建议\n\n")
        f.write("1. **性能优化**: 并行处理显著提升了处理效率，建议在生产环境中使用\n")
        f.write("2. **质量监控**: 建立自动化质量监控机制，及时发现异常结果\n")
        f.write("3. **模型优化**: 针对低置信度区域，考虑增加训练样本或调整模型参数\n")
        f.write("4. **后处理**: 对空间不一致的结果应用空间滤波等后处理方法\n\n")
        
        # 8. 技术参数
        f.write("## 🔧 技术参数\n\n")
        system_info = performance_comparison['system_info']
        f.write(f"- **CPU核心数**: {system_info['cpu_count']}\n")
        f.write(f"- **总内存**: {system_info['memory_gb']:.1f} GB\n")
        f.write(f"- **可用内存**: {system_info['available_memory_gb']:.1f} GB\n")
        f.write(f"- **并行效率**: {performance_comparison.get('efficiency', 0):.2f}\n\n")
        
        # 9. 附录
        f.write("## 📎 附录\n\n")
        f.write("### 文件清单\n\n")
        f.write("- `batch_processing_report.md`: 本报告\n")
        f.write("- `statistical_analysis.json`: 详细统计数据\n")
        f.write("- `quality_report.json`: 质量控制报告\n")
        f.write("- `temporal_analysis.json`: 时序分析结果\n")
        f.write("- `figures/`: 所有可视化图表\n")
        f.write("- `results/`: 各场景分类结果\n\n")
        
        f.write("---\n")
        f.write("*报告生成完成*")
    
    logger.info(f"批量处理报告已保存至: {report_path}")

def display_batch_results(statistical_results, performance_comparison, total_time):
    """
    显示批量处理结果
    """
    logger.info("="*60)
    logger.info("批量处理示例完成!")
    logger.info("="*60)
    
    summary_stats = statistical_results['summary_statistics']
    
    print(f"\n📊 批量处理统计:")
    print(f"   总场景数: {summary_stats['total_scenes']}")
    print(f"   成功场景数: {summary_stats['successful_scenes']}")
    print(f"   成功率: {summary_stats['success_rate']:.1f}%")
    print(f"   平均处理时间: {summary_stats['mean_processing_time']:.2f} 秒/场景")
    
    print(f"\n⚡ 性能对比:")
    print(f"   串行处理: {performance_comparison['serial']['total_time']:.2f} 秒")
    print(f"   并行处理: {performance_comparison['parallel']['total_time']:.2f} 秒")
    print(f"   加速比: {performance_comparison.get('speedup', 1):.2f}x")
    print(f"   并行效率: {performance_comparison.get('efficiency', 0):.2f}")
    
    confidence_stats = statistical_results['confidence_analysis']
    print(f"\n🎯 置信度统计:")
    print(f"   平均置信度: {confidence_stats['overall_mean_confidence']:.3f}")
    print(f"   高置信度场景: {confidence_stats['high_confidence_scenes']} 个")
    print(f"   低置信度场景: {confidence_stats['low_confidence_scenes']} 个")
    
    print(f"\n⏱️  总执行时间: {total_time:.2f} 秒")
    
    print(f"\n📁 输出文件位置:")
    print(f"   - 处理结果: output/batch_processing/results/")
    print(f"   - 统计分析: output/batch_processing/statistics/")
    print(f"   - 质量控制: output/batch_processing/quality_control/")
    print(f"   - 可视化图: output/batch_processing/figures/")
    print(f"   - 综合报告: output/batch_processing/reports/")

def main():
    """
    主函数
    """
    try:
        # 设置目录
        setup_batch_directories()
        
        # 执行批量处理工作流程
        batch_processing_workflow()
        
        print("\n✅ 批量处理示例执行完成!")
        print("🔍 请查看 output/batch_processing/ 目录下的详细结果")
        print("📊 综合报告: output/batch_processing/reports/batch_processing_report.md")
        print("📈 统计分析: output/batch_processing/statistics/")
        print("🎨 可视化图表: output/batch_processing/figures/")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        print("\n⚠️ 程序被用户中断")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"\n❌ 程序执行失败: {e}")
        print("💡 请检查错误信息并重试")
        raise

if __name__ == "__main__":
    main()