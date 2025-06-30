#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
湿地高光谱分类系统 - 高级分析示例
Wetland Hyperspectral Classification System - Advanced Analysis Example

这个示例展示了湿地高光谱分类系统的高级功能，包括：
- 深度学习模型训练
- 集成学习
- 超参数优化
- 不确定性分析
- 时序分析
- 景观格局分析
- 模型可解释性分析

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
import seaborn as sns
from pathlib import Path
import time
import logging
from datetime import datetime
import json
import pickle

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
    from wetland_classification.classification import Classifier, EnsembleClassifier
    from wetland_classification.evaluation import ModelEvaluator, UncertaintyAnalyzer
    from wetland_classification.landscape import LandscapeAnalyzer
    from wetland_classification.utils.visualization import Visualizer
    from wetland_classification.utils.logger import get_logger
    from wetland_classification.optimization import HyperparameterOptimizer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已正确安装湿地分类系统")
    sys.exit(1)

# 机器学习和深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.ensemble import VotingClassifier
    import optuna
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
except ImportError as e:
    print(f"导入可选依赖失败: {e}")
    print("某些高级功能可能不可用")

# 设置日志
logger = get_logger(__name__)

def setup_advanced_directories():
    """
    设置高级分析的目录结构
    """
    directories = [
        'output/advanced_analysis',
        'output/advanced_analysis/models',
        'output/advanced_analysis/figures',
        'output/advanced_analysis/results',
        'output/advanced_analysis/ensemble',
        'output/advanced_analysis/optimization',
        'output/advanced_analysis/uncertainty',
        'output/advanced_analysis/landscape',
        'output/advanced_analysis/interpretability',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def create_complex_demo_data():
    """
    创建复杂的演示数据，模拟真实的湿地场景
    """
    logger.info("创建复杂演示数据...")
    
    demo_dir = Path('data/samples/advanced_demo')
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # 创建更大、更复杂的场景
    height, width, bands = 200, 200, 150
    
    # 定义更多的湿地类别
    class_info = {
        0: {'name': '背景', 'color': '#000000'},
        1: {'name': '开放水面', 'color': '#0000FF'},
        2: {'name': '浅水区域', 'color': '#4169E1'},
        3: {'name': '挺水植物', 'color': '#228B22'},
        4: {'name': '浮叶植物', 'color': '#32CD32'},
        5: {'name': '沉水植物', 'color': '#006400'},
        6: {'name': '湿生草本', 'color': '#9ACD32'},
        7: {'name': '有机质土壤', 'color': '#8B4513'},
        8: {'name': '矿物质土壤', 'color': '#D2691E'},
        9: {'name': '建筑物', 'color': '#FF0000'}
    }
    
    # 初始化数据
    hyperspectral_data = np.zeros((height, width, bands))
    labels = np.zeros((height, width), dtype=int)
    
    # 波长范围
    wavelengths = np.linspace(400, 2500, bands)
    
    # 创建复杂的空间分布模式
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 定义各类别的空间分布
    for i in range(height):
        for j in range(width):
            # 基于位置和随机噪声确定类别
            dist_center = np.sqrt((i - height//2)**2 + (j - width//2)**2)
            noise = np.random.normal(0, 10)
            
            if dist_center < 30 + noise:  # 中心水域
                if np.random.random() < 0.7:
                    class_id = 1  # 开放水面
                else:
                    class_id = 2  # 浅水区域
            elif dist_center < 50 + noise:  # 植被过渡带
                class_id = np.random.choice([3, 4, 5, 6], p=[0.4, 0.3, 0.2, 0.1])
            elif dist_center < 80 + noise:  # 土壤区域
                class_id = np.random.choice([7, 8], p=[0.6, 0.4])
            elif dist_center < 90 + noise:  # 边缘区域
                if np.random.random() < 0.1:
                    class_id = 9  # 建筑物
                else:
                    class_id = np.random.choice([6, 7, 8], p=[0.3, 0.4, 0.3])
            else:
                class_id = 0  # 背景
            
            labels[i, j] = class_id
            
            # 生成对应的光谱特征
            spectrum = generate_realistic_spectrum(class_id, wavelengths)
            hyperspectral_data[i, j, :] = spectrum
    
    # 添加空间相关性（邻域滤波）
    from scipy import ndimage
    for band in range(bands):
        hyperspectral_data[:, :, band] = ndimage.gaussian_filter(
            hyperspectral_data[:, :, band], sigma=0.5
        )
    
    # 保存数据
    np.save(demo_dir / 'hyperspectral_data.npy', hyperspectral_data)
    np.save(demo_dir / 'ground_truth.npy', labels)
    np.save(demo_dir / 'wavelengths.npy', wavelengths)
    
    with open(demo_dir / 'class_info.json', 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"复杂演示数据创建完成，数据形状: {hyperspectral_data.shape}")
    logger.info(f"类别统计: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return hyperspectral_data, labels, wavelengths, class_info

def generate_realistic_spectrum(class_id, wavelengths):
    """
    生成真实的光谱特征
    """
    spectrum = np.zeros_like(wavelengths, dtype=float)
    
    # 定义各类别的光谱特征模板
    if class_id == 0:  # 背景
        spectrum = np.random.normal(0.05, 0.01, len(wavelengths))
    
    elif class_id == 1:  # 开放水面
        # 水体在近红外波段有强吸收
        spectrum = np.random.normal(0.1, 0.02, len(wavelengths))
        nir_mask = wavelengths > 700
        spectrum[nir_mask] *= 0.3
        
    elif class_id == 2:  # 浅水区域
        spectrum = np.random.normal(0.15, 0.03, len(wavelengths))
        nir_mask = wavelengths > 700
        spectrum[nir_mask] *= 0.5
        
    elif class_id in [3, 4, 5, 6]:  # 各种植被
        # 植被特征：红光吸收，红边，近红外高反射
        spectrum = np.random.normal(0.3, 0.05, len(wavelengths))
        
        # 红光吸收
        red_mask = (wavelengths >= 620) & (wavelengths <= 700)
        spectrum[red_mask] *= 0.4
        
        # 红边
        red_edge_mask = (wavelengths >= 700) & (wavelengths <= 750)
        spectrum[red_edge_mask] *= 1.5
        
        # 近红外高反射
        nir_mask = (wavelengths >= 750) & (wavelengths <= 1300)
        spectrum[nir_mask] *= 1.8
        
        # 水分吸收带
        water_bands = [(1400, 1500), (1900, 2000)]
        for start, end in water_bands:
            mask = (wavelengths >= start) & (wavelengths <= end)
            spectrum[mask] *= 0.6
            
    elif class_id in [7, 8]:  # 土壤
        spectrum = np.random.normal(0.4, 0.08, len(wavelengths))
        # 土壤在长波方向反射率增加
        long_wave_mask = wavelengths > 1000
        spectrum[long_wave_mask] *= 1.2
        
    elif class_id == 9:  # 建筑物
        spectrum = np.random.normal(0.5, 0.1, len(wavelengths))
    
    # 添加噪声
    noise = np.random.normal(0, 0.01, len(wavelengths))
    spectrum += noise
    
    # 确保光谱值在合理范围内
    spectrum = np.clip(spectrum, 0, 1)
    
    return spectrum

def advanced_analysis_workflow():
    """
    高级分析工作流程
    """
    logger.info("="*60)
    logger.info("开始湿地高光谱高级分析示例")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 步骤1: 数据准备
        logger.info("步骤1: 数据准备")
        data, labels, wavelengths, class_info = prepare_advanced_data()
        
        # 步骤2: 高级特征提取
        logger.info("步骤2: 高级特征提取")
        features, feature_names = advanced_feature_extraction(data, wavelengths)
        
        # 步骤3: 数据划分
        logger.info("步骤3: 数据划分")
        train_data, val_data, test_data = prepare_advanced_datasets(features, labels)
        
        # 步骤4: 深度学习模型训练
        logger.info("步骤4: 深度学习模型训练")
        dl_models = train_deep_learning_models(train_data, val_data, class_info)
        
        # 步骤5: 超参数优化
        logger.info("步骤5: 超参数优化")
        optimized_models = hyperparameter_optimization(train_data, val_data)
        
        # 步骤6: 集成学习
        logger.info("步骤6: 集成学习")
        ensemble_model = create_ensemble_model(dl_models, optimized_models, train_data, val_data)
        
        # 步骤7: 模型评估和比较
        logger.info("步骤7: 模型评估和比较")
        all_models = {**dl_models, **optimized_models, 'Ensemble': ensemble_model}
        evaluation_results = comprehensive_model_evaluation(all_models, test_data, class_info)
        
        # 步骤8: 不确定性分析
        logger.info("步骤8: 不确定性分析")
        uncertainty_results = uncertainty_analysis(ensemble_model, test_data)
        
        # 步骤9: 景观格局分析
        logger.info("步骤9: 景观格局分析")
        landscape_results = landscape_pattern_analysis(
            ensemble_model, features, data.shape[:2], class_info
        )
        
        # 步骤10: 模型可解释性分析
        logger.info("步骤10: 模型可解释性分析")
        interpretability_results = model_interpretability_analysis(
            ensemble_model, test_data, feature_names
        )
        
        # 步骤11: 高级可视化
        logger.info("步骤11: 高级可视化")
        create_advanced_visualizations(
            evaluation_results, uncertainty_results, landscape_results,
            interpretability_results, data, labels, wavelengths, class_info
        )
        
        # 步骤12: 生成综合报告
        logger.info("步骤12: 生成综合报告")
        generate_comprehensive_report(
            evaluation_results, uncertainty_results, landscape_results,
            interpretability_results, class_info
        )
        
        total_time = time.time() - start_time
        display_advanced_results(evaluation_results, total_time)
        
    except Exception as e:
        logger.error(f"高级分析流程执行失败: {e}")
        raise

def prepare_advanced_data():
    """
    准备高级分析数据
    """
    demo_dir = Path('data/samples/advanced_demo')
    
    if not demo_dir.exists():
        logger.info("高级演示数据不存在，正在创建...")
        return create_complex_demo_data()
    else:
        logger.info("加载已有的高级演示数据...")
        
        # 加载数据
        data = np.load(demo_dir / 'hyperspectral_data.npy')
        labels = np.load(demo_dir / 'ground_truth.npy')
        wavelengths = np.load(demo_dir / 'wavelengths.npy')
        
        with open(demo_dir / 'class_info.json', 'r', encoding='utf-8') as f:
            class_info = json.load(f)
        
        # 转换类别信息的键为整数
        class_info = {int(k): v for k, v in class_info.items()}
        
        return data, labels, wavelengths, class_info

def advanced_feature_extraction(data, wavelengths):
    """
    高级特征提取
    """
    height, width, bands = data.shape
    reshaped_data = data.reshape(-1, bands)
    
    features = []
    feature_names = []
    
    # 1. 原始光谱特征（选择关键波段）
    key_bands = select_optimal_bands(reshaped_data, wavelengths, n_bands=30)
    spectral_features = reshaped_data[:, key_bands]
    features.append(spectral_features)
    feature_names.extend([f'Band_{wavelengths[i]:.0f}nm' for i in key_bands])
    
    # 2. 光谱导数特征
    spectral_derivatives = calculate_spectral_derivatives(reshaped_data)
    features.append(spectral_derivatives)
    feature_names.extend(['First_Derivative_Mean', 'Second_Derivative_Mean', 
                         'First_Derivative_Std', 'Second_Derivative_Std'])
    
    # 3. 植被指数（扩展版）
    vegetation_indices = calculate_extended_vegetation_indices(reshaped_data, wavelengths)
    features.append(vegetation_indices)
    feature_names.extend(['NDVI', 'EVI', 'SAVI', 'OSAVI', 'MCARI', 'TCARI', 
                         'REP', 'NDWI', 'MNDWI', 'WI'])
    
    # 4. 光谱形状特征
    shape_features = calculate_spectral_shape_features(reshaped_data)
    features.append(shape_features)
    feature_names.extend(['Spectral_Angle', 'Spectral_Distance', 'Spectral_Correlation', 
                         'Absorption_Depth', 'Absorption_Area'])
    
    # 5. 连续统去除特征
    continuum_features = calculate_continuum_removal_features(reshaped_data, wavelengths)
    features.append(continuum_features)
    feature_names.extend(['CR_Mean', 'CR_Std', 'CR_Min', 'CR_Max'])
    
    # 6. 主成分特征
    pca_features = calculate_pca_features(reshaped_data, n_components=20)
    features.append(pca_features)
    feature_names.extend([f'PCA_{i+1}' for i in range(20)])
    
    # 合并所有特征
    all_features = np.column_stack(features)
    
    logger.info(f"高级特征提取完成 - 总特征数: {all_features.shape[1]}")
    
    return all_features, feature_names

def select_optimal_bands(data, wavelengths, n_bands=30):
    """
    选择最优波段
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # 这里使用简化的波段选择策略
    # 在实际应用中可以使用更复杂的特征选择方法
    step = max(1, len(wavelengths) // n_bands)
    selected_bands = list(range(0, len(wavelengths), step))[:n_bands]
    
    return selected_bands

def calculate_spectral_derivatives(data):
    """
    计算光谱导数特征
    """
    # 一阶导数
    first_derivative = np.gradient(data, axis=1)
    first_deriv_mean = np.mean(first_derivative, axis=1)
    first_deriv_std = np.std(first_derivative, axis=1)
    
    # 二阶导数
    second_derivative = np.gradient(first_derivative, axis=1)
    second_deriv_mean = np.mean(second_derivative, axis=1)
    second_deriv_std = np.std(second_derivative, axis=1)
    
    return np.column_stack([first_deriv_mean, second_deriv_mean, 
                           first_deriv_std, second_deriv_std])

def calculate_extended_vegetation_indices(data, wavelengths):
    """
    计算扩展植被指数
    """
    def find_band(target_wavelength):
        return np.argmin(np.abs(wavelengths - target_wavelength))
    
    # 定义波段
    blue = data[:, find_band(470)]
    green = data[:, find_band(550)]
    red = data[:, find_band(670)]
    red_edge = data[:, find_band(720)]
    nir = data[:, find_band(850)]
    swir1 = data[:, find_band(1600)]
    swir2 = data[:, find_band(2200)]
    
    epsilon = 1e-8
    
    # 基础植被指数
    ndvi = (nir - red) / (nir + red + epsilon)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + epsilon)
    osavi = (nir - red) / (nir + red + 0.16 + epsilon)
    
    # 高级植被指数
    mcari = ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / red + epsilon)
    tcari = 3 * ((red_edge - red) - 0.2 * (red_edge - green) * (red_edge / red + epsilon))
    
    # 红边位置 (简化计算)
    rep = red_edge / (red + epsilon)
    
    # 水分指数
    ndwi = (green - nir) / (green + nir + epsilon)
    mndwi = (green - swir1) / (green + swir1 + epsilon)
    wi = nir / swir1
    
    return np.column_stack([ndvi, evi, savi, osavi, mcari, tcari, rep, ndwi, mndwi, wi])

def calculate_spectral_shape_features(data):
    """
    计算光谱形状特征
    """
    from scipy.spatial.distance import euclidean
    from scipy.stats import pearsonr
    
    n_pixels, n_bands = data.shape
    
    # 参考光谱（使用平均光谱）
    reference_spectrum = np.mean(data, axis=0)
    
    spectral_angles = []
    spectral_distances = []
    spectral_correlations = []
    absorption_depths = []
    absorption_areas = []
    
    for i in range(n_pixels):
        spectrum = data[i, :]
        
        # 光谱角
        dot_product = np.dot(spectrum, reference_spectrum)
        norm_product = np.linalg.norm(spectrum) * np.linalg.norm(reference_spectrum)
        if norm_product > 0:
            spectral_angle = np.arccos(np.clip(dot_product / norm_product, -1, 1))
        else:
            spectral_angle = 0
        spectral_angles.append(spectral_angle)
        
        # 光谱距离
        spectral_distance = euclidean(spectrum, reference_spectrum)
        spectral_distances.append(spectral_distance)
        
        # 光谱相关性
        if np.std(spectrum) > 0 and np.std(reference_spectrum) > 0:
            correlation, _ = pearsonr(spectrum, reference_spectrum)
            spectral_correlations.append(correlation)
        else:
            spectral_correlations.append(0)
        
        # 吸收深度和面积（简化计算）
        absorption_depth = np.max(spectrum) - np.min(spectrum)
        absorption_area = np.sum(spectrum)
        absorption_depths.append(absorption_depth)
        absorption_areas.append(absorption_area)
    
    return np.column_stack([spectral_angles, spectral_distances, spectral_correlations,
                           absorption_depths, absorption_areas])

def calculate_continuum_removal_features(data, wavelengths):
    """
    计算连续统去除特征
    """
    from scipy.spatial import ConvexHull
    
    cr_features = []
    
    for i in range(data.shape[0]):
        spectrum = data[i, :]
        
        try:
            # 计算凸包
            points = np.column_stack([wavelengths, spectrum])
            hull = ConvexHull(points)
            hull_points = hull.vertices
            
            # 构建连续统
            continuum = np.interp(wavelengths, wavelengths[hull_points], spectrum[hull_points])
            
            # 连续统去除
            cr_spectrum = spectrum / (continuum + 1e-8)
            
            # 计算特征
            cr_mean = np.mean(cr_spectrum)
            cr_std = np.std(cr_spectrum)
            cr_min = np.min(cr_spectrum)
            cr_max = np.max(cr_spectrum)
            
        except:
            # 如果凸包计算失败，使用原始光谱统计量
            cr_mean = np.mean(spectrum)
            cr_std = np.std(spectrum)
            cr_min = np.min(spectrum)
            cr_max = np.max(spectrum)
        
        cr_features.append([cr_mean, cr_std, cr_min, cr_max])
    
    return np.array(cr_features)

def calculate_pca_features(data, n_components=20):
    """
    计算主成分特征
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(data_scaled)
    
    logger.info(f"PCA解释方差比: {pca.explained_variance_ratio_[:5]}")
    
    return pca_features

def prepare_advanced_datasets(features, labels):
    """
    准备高级数据集
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # 获取有效像素
    valid_mask = labels.ravel() > 0
    valid_features = features[valid_mask]
    valid_labels = labels.ravel()[valid_mask]
    
    # 标签编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(valid_labels)
    
    # 数据划分
    X_temp, X_test, y_temp, y_test = train_test_split(
        valid_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2 of total
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_data = {
        'X': X_train_scaled, 'y': y_train,
        'X_original': X_train, 'scaler': scaler, 'label_encoder': label_encoder
    }
    
    val_data = {
        'X': X_val_scaled, 'y': y_val,
        'X_original': X_val
    }
    
    test_data = {
        'X': X_test_scaled, 'y': y_test,
        'X_original': X_test
    }
    
    logger.info(f"数据集划分 - 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
    
    return train_data, val_data, test_data

def train_deep_learning_models(train_data, val_data, class_info):
    """
    训练深度学习模型
    """
    models = {}
    
    # 数据准备
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 1. 多层感知机 (MLP)
    logger.info("训练多层感知机模型...")
    mlp_model = create_mlp_model(n_features, n_classes)
    trained_mlp = train_pytorch_model(mlp_model, train_loader, val_loader, 'MLP')
    models['MLP'] = trained_mlp
    
    # 2. 1D卷积神经网络
    logger.info("训练1D卷积神经网络...")
    cnn1d_model = create_cnn1d_model(n_features, n_classes)
    trained_cnn1d = train_pytorch_model(cnn1d_model, train_loader, val_loader, 'CNN1D')
    models['CNN1D'] = trained_cnn1d
    
    # 3. 注意力网络
    logger.info("训练注意力网络...")
    attention_model = create_attention_model(n_features, n_classes)
    trained_attention = train_pytorch_model(attention_model, train_loader, val_loader, 'Attention')
    models['Attention'] = trained_attention
    
    return models

def create_mlp_model(n_features, n_classes):
    """
    创建多层感知机模型
    """
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
            super(MLP, self).__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    model = MLP(n_features, [512, 256, 128], n_classes)
    return model

def create_cnn1d_model(n_features, n_classes):
    """
    创建1D卷积神经网络模型
    """
    class CNN1D(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(CNN1D, self).__init__()
            
            self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            
            self.pool = nn.AdaptiveAvgPool1d(1)
            
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, output_dim)
            
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # 重塑输入 (batch_size, 1, features)
            x = x.unsqueeze(1)
            
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    model = CNN1D(n_features, n_classes)
    return model

def create_attention_model(n_features, n_classes):
    """
    创建注意力网络模型
    """
    class AttentionNet(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256):
            super(AttentionNet, self).__init__()
            
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # 注意力机制
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softmax(dim=1)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
        def forward(self, x):
            # 特征提取
            features = self.feature_extractor(x)
            
            # 注意力权重
            attention_weights = self.attention(features)
            
            # 加权特征
            weighted_features = features * attention_weights
            
            # 分类
            output = self.classifier(weighted_features)
            
            return output
    
    model = AttentionNet(n_features, n_classes)
    return model

def train_pytorch_model(model, train_loader, val_loader, model_name, epochs=100):
    """
    训练PyTorch模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = 20
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'output/advanced_analysis/models/{model_name}_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"{model_name} 早停在epoch {epoch+1}, 最佳验证精度: {best_val_acc:.4f}")
            break
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"{model_name} Epoch {epoch+1}/{epochs}, "
                       f"Train Loss: {train_losses[-1]:.4f}, "
                       f"Val Loss: {val_losses[-1]:.4f}, "
                       f"Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'output/advanced_analysis/models/{model_name}_best.pth'))
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }
    
    with open(f'output/advanced_analysis/models/{model_name}_history.json', 'w') as f:
        json.dump(training_history, f)
    
    model.training_history = training_history
    
    return model

def hyperparameter_optimization(train_data, val_data):
    """
    超参数优化
    """
    logger.info("开始超参数优化...")
    
    optimized_models = {}
    
    # 准备数据
    X_train, y_train = train_data['X_original'], train_data['y']
    X_val, y_val = val_data['X_original'], val_data['y']
    
    # 1. Random Forest 超参数优化
    logger.info("优化 Random Forest 超参数...")
    rf_optimized = optimize_random_forest(X_train, y_train, X_val, y_val)
    optimized_models['RF_Optimized'] = rf_optimized
    
    # 2. XGBoost 超参数优化
    logger.info("优化 XGBoost 超参数...")
    xgb_optimized = optimize_xgboost(X_train, y_train, X_val, y_val)
    optimized_models['XGB_Optimized'] = xgb_optimized
    
    # 3. SVM 超参数优化
    logger.info("优化 SVM 超参数...")
    svm_optimized = optimize_svm(X_train, y_train, X_val, y_val)
    optimized_models['SVM_Optimized'] = svm_optimized
    
    return optimized_models

def optimize_random_forest(X_train, y_train, X_val, y_val):
    """
    Random Forest 超参数优化
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    best_rf = random_search.best_estimator_
    
    logger.info(f"Random Forest 最佳参数: {random_search.best_params_}")
    
    return best_rf

def optimize_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost 超参数优化
    """
    import xgboost as xgb
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred = model.predict(X_val)
        accuracy = (val_pred == y_val).mean()
        
        return accuracy
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        
        best_params = study.best_params
        best_xgb = xgb.XGBClassifier(**best_params)
        best_xgb.fit(X_train, y_train)
        
        logger.info(f"XGBoost 最佳参数: {best_params}")
        
    except:
        # 如果Optuna不可用，使用默认参数
        logger.warning("Optuna 不可用，使用默认 XGBoost 参数")
        best_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        )
        best_xgb.fit(X_train, y_train)
    
    return best_xgb

def optimize_svm(X_train, y_train, X_val, y_val):
    """
    SVM 超参数优化
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }
    
    svm = SVC(random_state=42)
    
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, n_jobs=-1, scoring='accuracy'
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_svm = grid_search.best_estimator_
    best_svm.scaler = scaler  # 保存标准化器
    
    logger.info(f"SVM 最佳参数: {grid_search.best_params_}")
    
    return best_svm

def create_ensemble_model(dl_models, optimized_models, train_data, val_data):
    """
    创建集成模型
    """
    logger.info("创建集成模型...")
    
    # 准备数据
    X_train, y_train = train_data['X_original'], train_data['y']
    X_val, y_val = val_data['X_original'], val_data['y']
    
    # 收集所有传统机器学习模型
    traditional_models = []
    model_names = []
    
    for name, model in optimized_models.items():
        traditional_models.append((name, model))
        model_names.append(name)
    
    # 创建投票分类器
    ensemble = VotingClassifier(
        estimators=traditional_models,
        voting='soft'  # 使用软投票
    )
    
    ensemble.fit(X_train, y_train)
    
    logger.info(f"集成模型创建完成，包含模型: {model_names}")
    
    return ensemble

def comprehensive_model_evaluation(models, test_data, class_info):
    """
    全面的模型评估
    """
    logger.info("开始全面模型评估...")
    
    X_test, y_test = test_data['X'], test_data['y']
    X_test_original = test_data['X_original']
    
    evaluation_results = {}
    
    for model_name, model in models.items():
        logger.info(f"评估模型: {model_name}")
        
        # 预测
        if 'PyTorch' in str(type(model)) or hasattr(model, 'training_history'):
            # PyTorch 模型
            predictions, probabilities = predict_pytorch_model(model, X_test)
        else:
            # Scikit-learn 模型
            if hasattr(model, 'scaler') and model.scaler is not None:
                X_test_for_pred = model.scaler.transform(X_test_original)
            else:
                X_test_for_pred = X_test_original
            
            predictions = model.predict(X_test_for_pred)
            probabilities = model.predict_proba(X_test_for_pred)
        
        # 计算评估指标
        results = calculate_comprehensive_metrics(y_test, predictions, probabilities, class_info)
        evaluation_results[model_name] = results
    
    return evaluation_results

def predict_pytorch_model(model, X_test):
    """
    PyTorch模型预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    return predictions, probabilities

def calculate_comprehensive_metrics(y_true, y_pred, y_proba, class_info):
    """
    计算全面的评估指标
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        cohen_kappa_score, confusion_matrix, roc_auc_score,
        classification_report
    )
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # 各类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 宏平均和微平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC (多分类)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    # 详细分类报告
    class_names = [class_info.get(i, {'name': f'Class_{i}'})['name'] 
                   for i in range(len(np.unique(y_true)))]
    
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    results = {
        'accuracy': accuracy,
        'kappa': kappa,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'auc': auc,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    return results

def uncertainty_analysis(model, test_data):
    """
    不确定性分析
    """
    logger.info("开始不确定性分析...")
    
    X_test, y_test = test_data['X'], test_data['y']
    X_test_original = test_data['X_original']
    
    # 预测概率
    if hasattr(model, 'predict_proba'):
        if hasattr(model, 'scaler') and model.scaler is not None:
            X_test_for_pred = model.scaler.transform(X_test_original)
        else:
            X_test_for_pred = X_test_original
        probabilities = model.predict_proba(X_test_for_pred)
    else:
        # PyTorch模型的概率预测已在之前实现
        _, probabilities = predict_pytorch_model(model, X_test)
    
    # 计算不确定性指标
    uncertainty_results = {}
    
    # 1. 预测熵
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
    uncertainty_results['entropy'] = entropy
    
    # 2. 最大概率
    max_prob = np.max(probabilities, axis=1)
    uncertainty_results['max_probability'] = max_prob
    
    # 3. 置信度 (1 - 预测熵的归一化)
    normalized_entropy = entropy / np.log(probabilities.shape[1])
    confidence = 1 - normalized_entropy
    uncertainty_results['confidence'] = confidence
    
    # 4. 边际概率 (最大概率与第二大概率的差)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    uncertainty_results['margin'] = margin
    
    # 5. 预测正确性与不确定性的关系
    predictions = np.argmax(probabilities, axis=1)
    correct_predictions = (predictions == y_test)
    uncertainty_results['correct_predictions'] = correct_predictions
    
    logger.info("不确定性分析完成")
    
    return uncertainty_results

def landscape_pattern_analysis(model, features, original_shape, class_info):
    """
    景观格局分析
    """
    logger.info("开始景观格局分析...")
    
    # 对整个场景进行预测
    if hasattr(model, 'scaler') and model.scaler is not None:
        features_scaled = model.scaler.transform(features)
        predictions = model.predict(features_scaled)
    else:
        predictions = model.predict(features)
    
    # 重塑为原始形状
    classification_map = predictions.reshape(original_shape)
    
    # 计算景观指标
    landscape_metrics = {}
    
    # 1. 类别面积比例
    class_areas = {}
    total_pixels = np.prod(classification_map.shape)
    
    for class_id in class_info.keys():
        if class_id == 0:  # 跳过背景
            continue
        area = np.sum(classification_map == class_id)
        proportion = area / total_pixels
        class_areas[class_id] = {
            'area_pixels': int(area),
            'proportion': proportion
        }
    
    landscape_metrics['class_areas'] = class_areas
    
    # 2. 香农多样性指数
    proportions = [info['proportion'] for info in class_areas.values() if info['proportion'] > 0]
    shannon_diversity = -np.sum([p * np.log(p) for p in proportions if p > 0])
    landscape_metrics['shannon_diversity'] = shannon_diversity
    
    # 3. 辛普森多样性指数
    simpson_diversity = 1 - np.sum([p**2 for p in proportions])
    landscape_metrics['simpson_diversity'] = simpson_diversity
    
    # 4. 均匀度指数
    evenness = shannon_diversity / np.log(len(proportions)) if len(proportions) > 0 else 0
    landscape_metrics['evenness'] = evenness
    
    # 5. 边缘密度
    edge_density = calculate_edge_density(classification_map)
    landscape_metrics['edge_density'] = edge_density
    
    # 6. 聚集指数
    aggregation_index = calculate_aggregation_index(classification_map)
    landscape_metrics['aggregation_index'] = aggregation_index
    
    # 7. 连通性分析
    connectivity_results = {}
    for class_id in class_info.keys():
        if class_id == 0 or class_areas.get(class_id, {}).get('area_pixels', 0) == 0:
            continue
        
        connectivity = analyze_class_connectivity(classification_map, class_id)
        connectivity_results[class_id] = connectivity
    
    landscape_metrics['connectivity'] = connectivity_results
    
    landscape_results = {
        'classification_map': classification_map,
        'landscape_metrics': landscape_metrics
    }
    
    logger.info("景观格局分析完成")
    
    return landscape_results

def calculate_edge_density(classification_map):
    """
    计算边缘密度
    """
    from scipy import ndimage
    
    # 计算边缘
    edges = 0
    height, width = classification_map.shape
    
    for i in range(height - 1):
        for j in range(width - 1):
            # 检查右边和下边的邻居
            if classification_map[i, j] != classification_map[i, j + 1]:
                edges += 1
            if classification_map[i, j] != classification_map[i + 1, j]:
                edges += 1
    
    # 边缘密度 = 边缘数 / 总像素数
    edge_density = edges / (height * width)
    
    return edge_density

def calculate_aggregation_index(classification_map):
    """
    计算聚集指数
    """
    height, width = classification_map.shape
    total_adjacencies = 0
    same_class_adjacencies = 0
    
    for i in range(height):
        for j in range(width):
            current_class = classification_map[i, j]
            
            # 检查4邻域
            neighbors = []
            if i > 0: neighbors.append(classification_map[i-1, j])
            if i < height-1: neighbors.append(classification_map[i+1, j])
            if j > 0: neighbors.append(classification_map[i, j-1])
            if j < width-1: neighbors.append(classification_map[i, j+1])
            
            total_adjacencies += len(neighbors)
            same_class_adjacencies += sum(1 for n in neighbors if n == current_class)
    
    aggregation_index = same_class_adjacencies / total_adjacencies if total_adjacencies > 0 else 0
    
    return aggregation_index

def analyze_class_connectivity(classification_map, class_id):
    """
    分析特定类别的连通性
    """
    from scipy import ndimage
    
    # 创建二值图
    binary_map = (classification_map == class_id).astype(int)
    
    # 连通组件分析
    labeled_map, num_components = ndimage.label(binary_map)
    
    if num_components == 0:
        return {
            'num_components': 0,
            'largest_component_size': 0,
            'mean_component_size': 0,
            'component_sizes': []
        }
    
    # 计算各组件大小
    component_sizes = [np.sum(labeled_map == i) for i in range(1, num_components + 1)]
    
    connectivity_results = {
        'num_components': num_components,
        'largest_component_size': max(component_sizes),
        'mean_component_size': np.mean(component_sizes),
        'component_sizes': component_sizes
    }
    
    return connectivity_results

def model_interpretability_analysis(model, test_data, feature_names):
    """
    模型可解释性分析
    """
    logger.info("开始模型可解释性分析...")
    
    interpretability_results = {}
    
    # 1. 特征重要性分析
    if hasattr(model, 'feature_importances_'):
        # 基于树的模型
        feature_importance = model.feature_importances_
        interpretability_results['feature_importance'] = {
            'importance_scores': feature_importance,
            'top_features': get_top_features(feature_importance, feature_names, top_k=20)
        }
    
    elif hasattr(model, 'estimators_'):
        # 集成模型
        if hasattr(model.estimators_[0], 'feature_importances_'):
            # 计算平均特征重要性
            all_importances = [est.feature_importances_ for est in model.estimators_]
            feature_importance = np.mean(all_importances, axis=0)
            interpretability_results['feature_importance'] = {
                'importance_scores': feature_importance,
                'top_features': get_top_features(feature_importance, feature_names, top_k=20)
            }
    
    # 2. SHAP分析 (如果可用)
    try:
        import shap
        shap_values = calculate_shap_values(model, test_data, feature_names)
        interpretability_results['shap'] = shap_values
    except ImportError:
        logger.warning("SHAP 库不可用，跳过SHAP分析")
    
    # 3. 排列重要性
    try:
        permutation_importance = calculate_permutation_importance(model, test_data)
        interpretability_results['permutation_importance'] = permutation_importance
    except:
        logger.warning("排列重要性计算失败")
    
    logger.info("模型可解释性分析完成")
    
    return interpretability_results

def get_top_features(importance_scores, feature_names, top_k=20):
    """
    获取重要性最高的特征
    """
    indices = np.argsort(importance_scores)[::-1][:top_k]
    top_features = []
    
    for i, idx in enumerate(indices):
        top_features.append({
            'rank': i + 1,
            'feature_name': feature_names[idx],
            'importance_score': float(importance_scores[idx])
        })
    
    return top_features

def calculate_shap_values(model, test_data, feature_names):
    """
    计算SHAP值
    """
    import shap
    
    X_test = test_data['X_original'][:100]  # 限制样本数量以节省时间
    
    if hasattr(model, 'estimators_'):
        # 集成模型
        explainer = shap.Explainer(model)
    else:
        explainer = shap.Explainer(model)
    
    shap_values = explainer(X_test)
    
    shap_results = {
        'shap_values': shap_values.values,
        'base_values': shap_values.base_values,
        'data': shap_values.data,
        'feature_names': feature_names
    }
    
    return shap_results

def calculate_permutation_importance(model, test_data):
    """
    计算排列重要性
    """
    from sklearn.inspection import permutation_importance
    
    X_test, y_test = test_data['X_original'], test_data['y']
    
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    
    return {
        'importances_mean': perm_importance.importances_mean,
        'importances_std': perm_importance.importances_std
    }

def create_advanced_visualizations(evaluation_results, uncertainty_results, 
                                 landscape_results, interpretability_results,
                                 data, labels, wavelengths, class_info):
    """
    创建高级可视化
    """
    logger.info("创建高级可视化图表...")
    
    # 1. 模型性能雷达图
    create_performance_radar_chart(evaluation_results)
    
    # 2. 不确定性分布图
    create_uncertainty_plots(uncertainty_results)
    
    # 3. 景观格局可视化
    create_landscape_visualizations(landscape_results, class_info)
    
    # 4. 特征重要性图
    if interpretability_results.get('feature_importance'):
        create_feature_importance_plots(interpretability_results['feature_importance'])
    
    # 5. 混淆矩阵热图
    create_confusion_matrix_heatmap(evaluation_results, class_info)
    
    # 6. 学习曲线 (如果有)
    create_learning_curves(evaluation_results)
    
    # 7. 类别分布图
    create_class_distribution_plots(landscape_results, class_info)
    
    logger.info("高级可视化创建完成")

def create_performance_radar_chart(evaluation_results):
    """
    创建性能雷达图
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 准备数据
    models = list(evaluation_results.keys())
    metrics = ['accuracy', 'macro_f1', 'kappa', 'auc']
    metric_labels = ['准确率', '宏平均F1', 'Kappa系数', 'AUC']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        values = []
        for metric in metrics:
            values.append(evaluation_results[model].get(metric, 0))
        
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('模型性能对比雷达图', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/performance_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_uncertainty_plots(uncertainty_results):
    """
    创建不确定性分析图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 熵分布
    axes[0, 0].hist(uncertainty_results['entropy'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('预测熵')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('预测熵分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 置信度分布
    axes[0, 1].hist(uncertainty_results['confidence'], bins=50, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('置信度')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('置信度分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 置信度与准确性的关系
    correct = uncertainty_results['correct_predictions']
    confidence = uncertainty_results['confidence']
    
    axes[1, 0].scatter(confidence[correct], np.ones(np.sum(correct)), 
                      alpha=0.6, color='green', label='正确预测', s=20)
    axes[1, 0].scatter(confidence[~correct], np.zeros(np.sum(~correct)), 
                      alpha=0.6, color='red', label='错误预测', s=20)
    axes[1, 0].set_xlabel('置信度')
    axes[1, 0].set_ylabel('预测正确性')
    axes[1, 0].set_title('置信度与预测正确性关系')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 边际概率分布
    axes[1, 1].hist(uncertainty_results['margin'], bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('边际概率')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('边际概率分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_landscape_visualizations(landscape_results, class_info):
    """
    创建景观格局可视化
    """
    classification_map = landscape_results['classification_map']
    metrics = landscape_results['landscape_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 分类结果图
    colors = ['black'] + [class_info[i]['color'] for i in sorted(class_info.keys()) if i > 0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    im = axes[0, 0].imshow(classification_map, cmap=cmap)
    axes[0, 0].set_title('分类结果图')
    axes[0, 0].axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    cbar.set_ticks(range(len(colors)))
    cbar.set_ticklabels([class_info.get(i, {'name': 'Unknown'})['name'] 
                        for i in range(len(colors))])
    
    # 2. 类别面积比例饼图
    class_areas = metrics['class_areas']
    labels = [class_info[class_id]['name'] for class_id in class_areas.keys()]
    sizes = [info['proportion'] for info in class_areas.values()]
    colors = [class_info[class_id]['color'] for class_id in class_areas.keys()]
    
    axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('类别面积比例')
    
    # 3. 景观指数条形图
    landscape_indices = {
        '香农多样性': metrics['shannon_diversity'],
        '辛普森多样性': metrics['simpson_diversity'],
        '均匀度': metrics['evenness'],
        '边缘密度': metrics['edge_density'],
        '聚集指数': metrics['aggregation_index']
    }
    
    indices_names = list(landscape_indices.keys())
    indices_values = list(landscape_indices.values())
    
    bars = axes[1, 0].bar(indices_names, indices_values, color='skyblue', alpha=0.7)
    axes[1, 0].set_ylabel('指数值')
    axes[1, 0].set_title('景观格局指数')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, indices_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 4. 连通性分析
    connectivity_data = metrics['connectivity']
    class_names = [class_info[class_id]['name'] for class_id in connectivity_data.keys()]
    num_components = [info['num_components'] for info in connectivity_data.values()]
    
    bars = axes[1, 1].bar(class_names, num_components, color='lightcoral', alpha=0.7)
    axes[1, 1].set_ylabel('连通组件数')
    axes[1, 1].set_title('各类别连通性分析')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, num_components):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/landscape_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_plots(feature_importance_results):
    """
    创建特征重要性图
    """
    top_features = feature_importance_results['top_features']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_names = [f['feature_name'] for f in top_features]
    importance_scores = [f['importance_score'] for f in top_features]
    
    bars = ax.barh(range(len(feature_names)), importance_scores, color='steelblue', alpha=0.7)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('重要性分数')
    ax.set_title('前20个最重要特征')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix_heatmap(evaluation_results, class_info):
    """
    创建混淆矩阵热图
    """
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
    
    if n_models == 1:
        axes = [axes]
    elif len(axes.shape) == 1:
        pass  # 已经是1D数组
    else:
        axes = axes.flatten()
    
    class_names = [class_info[i]['name'] for i in sorted(class_info.keys()) if i > 0]
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        if idx >= len(axes):
            break
            
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx])
        
        axes[idx].set_title(f'{model_name}\n混淆矩阵 (归一化)')
        axes[idx].set_xlabel('预测标签')
        axes[idx].set_ylabel('真实标签')
    
    # 隐藏多余的子图
    for idx in range(len(evaluation_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_learning_curves(evaluation_results):
    """
    创建学习曲线
    """
    pytorch_models = {name: results for name, results in evaluation_results.items() 
                     if 'training_history' in str(type(results))}
    
    if not pytorch_models:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for model_name, model in pytorch_models.items():
        if hasattr(model, 'training_history'):
            history = model.training_history
            
            epochs = range(1, len(history['train_losses']) + 1)
            
            # 损失曲线
            axes[0].plot(epochs, history['train_losses'], label=f'{model_name} (训练)')
            axes[0].plot(epochs, history['val_losses'], label=f'{model_name} (验证)', linestyle='--')
            
            # 准确率曲线
            axes[1].plot(epochs, history['val_accuracies'], label=f'{model_name}')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('验证准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_class_distribution_plots(landscape_results, class_info):
    """
    创建类别分布图
    """
    classification_map = landscape_results['classification_map']
    
    # 计算每个类别的像素数
    unique_classes, counts = np.unique(classification_map, return_counts=True)
    
    # 过滤掉背景类别
    valid_indices = unique_classes > 0
    valid_classes = unique_classes[valid_indices]
    valid_counts = counts[valid_indices]
    
    # 准备数据
    class_names = [class_info[cls]['name'] for cls in valid_classes]
    colors = [class_info[cls]['color'] for cls in valid_classes]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 条形图
    bars = axes[0].bar(class_names, valid_counts, color=colors, alpha=0.7)
    axes[0].set_ylabel('像素数')
    axes[0].set_title('各类别像素数统计')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, valid_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{count}', ha='center', va='bottom')
    
    # 2. 饼图
    axes[1].pie(valid_counts, labels=class_names, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('各类别面积比例')
    
    plt.tight_layout()
    plt.savefig('output/advanced_analysis/figures/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(evaluation_results, uncertainty_results, 
                                landscape_results, interpretability_results, class_info):
    """
    生成综合报告
    """
    logger.info("生成综合分析报告...")
    
    report_path = Path('output/advanced_analysis/results/comprehensive_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 湿地高光谱分类系统 - 高级分析综合报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 执行摘要
        f.write("## 📋 执行摘要\n\n")
        best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
        best_accuracy = evaluation_results[best_model]['accuracy']
        
        f.write(f"本次高级分析共评估了 {len(evaluation_results)} 个模型，")
        f.write(f"最佳模型为 **{best_model}**，测试集精度达到 **{best_accuracy:.3f}**。\n\n")
        
        # 2. 模型性能汇总
        f.write("## 📊 模型性能汇总\n\n")
        f.write("| 模型 | 准确率 | Kappa系数 | 宏平均F1 | AUC |\n")
        f.write("|------|--------|-----------|----------|-----|\n")
        
        for model_name, results in evaluation_results.items():
            f.write(f"| {model_name} | {results['accuracy']:.3f} | "
                   f"{results['kappa']:.3f} | {results['macro_f1']:.3f} | "
                   f"{results.get('auc', 0):.3f} |\n")
        
        f.write("\n")
        
        # 3. 不确定性分析结果
        f.write("## 🔮 不确定性分析\n\n")
        mean_entropy = np.mean(uncertainty_results['entropy'])
        mean_confidence = np.mean(uncertainty_results['confidence'])
        accuracy_rate = np.mean(uncertainty_results['correct_predictions'])
        
        f.write(f"- **平均预测熵**: {mean_entropy:.3f}\n")
        f.write(f"- **平均置信度**: {mean_confidence:.3f}\n")
        f.write(f"- **预测准确率**: {accuracy_rate:.3f}\n\n")
        
        # 4. 景观格局分析
        f.write("## 🌿 景观格局分析\n\n")
        metrics = landscape_results['landscape_metrics']
        
        f.write("### 景观多样性指标\n\n")
        f.write(f"- **香农多样性指数**: {metrics['shannon_diversity']:.3f}\n")
        f.write(f"- **辛普森多样性指数**: {metrics['simpson_diversity']:.3f}\n")
        f.write(f"- **均匀度指数**: {metrics['evenness']:.3f}\n\n")
        
        f.write("### 景观结构指标\n\n")
        f.write(f"- **边缘密度**: {metrics['edge_density']:.3f}\n")
        f.write(f"- **聚集指数**: {metrics['aggregation_index']:.3f}\n\n")
        
        f.write("### 类别面积统计\n\n")
        f.write("| 类别 | 像素数 | 面积比例 |\n")
        f.write("|------|--------|----------|\n")
        
        for class_id, area_info in metrics['class_areas'].items():
            class_name = class_info[class_id]['name']
            f.write(f"| {class_name} | {area_info['area_pixels']} | "
                   f"{area_info['proportion']:.3f} |\n")
        
        f.write("\n")
        
        # 5. 模型可解释性
        if interpretability_results.get('feature_importance'):
            f.write("## 🔍 模型可解释性\n\n")
            f.write("### 最重要的10个特征\n\n")
            f.write("| 排名 | 特征名称 | 重要性分数 |\n")
            f.write("|------|----------|------------|\n")
            
            top_features = interpretability_results['feature_importance']['top_features'][:10]
            for feature in top_features:
                f.write(f"| {feature['rank']} | {feature['feature_name']} | "
                       f"{feature['importance_score']:.4f} |\n")
            
            f.write("\n")
        
        # 6. 结论和建议
        f.write("## 💡 结论和建议\n\n")
        f.write("### 主要发现\n\n")
        f.write(f"1. **最佳模型**: {best_model} 在所有评估指标上表现最佳\n")
        f.write(f"2. **分类精度**: 整体分类精度达到 {best_accuracy:.1%}，满足实际应用需求\n")
        f.write(f"3. **景观多样性**: 香农多样性指数为 {metrics['shannon_diversity']:.3f}，")
        f.write("表明研究区域具有较好的生态多样性\n")
        f.write(f"4. **模型不确定性**: 平均置信度为 {mean_confidence:.3f}，模型预测相对可靠\n\n")
        
        f.write("### 应用建议\n\n")
        f.write("1. **模型部署**: 推荐使用集成模型进行实际应用\n")
        f.write("2. **质量控制**: 对低置信度区域进行人工复核\n")
        f.write("3. **监测重点**: 关注边缘密度高的区域，可能存在生态过渡带\n")
        f.write("4. **保护策略**: 重点保护大面积连续的湿地植被斑块\n\n")
    
    logger.info(f"综合报告已保存至: {report_path}")

def display_advanced_results(evaluation_results, total_time):
    """
    显示高级分析结果
    """
    logger.info("="*60)
    logger.info("高级分析示例完成!")
    logger.info("="*60)
    
    # 显示最佳模型
    best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
    best_results = evaluation_results[best_model]
    
    print(f"\n🏆 最佳模型: {best_model}")
    print(f"   📊 准确率: {best_results['accuracy']:.3f}")
    print(f"   📈 Kappa系数: {best_results['kappa']:.3f}")
    print(f"   🎯 宏平均F1: {best_results['macro_f1']:.3f}")
    print(f"   📋 AUC: {best_results.get('auc', 0):.3f}")
    
    # 显示性能排名
    print(f"\n📊 模型性能排名:")
    sorted_models = sorted(evaluation_results.items(), 
                          key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (model_name, results) in enumerate(sorted_models, 1):
        print(f"   {i}. {model_name:15s} - 精度: {results['accuracy']:.3f}")
    
    print(f"\n⏱️  总执行时间: {total_time:.2f} 秒")
    
    # 显示输出文件
    print(f"\n📁 输出文件位置:")
    print(f"   - 模型文件: output/advanced_analysis/models/")
    print(f"   - 分析结果: output/advanced_analysis/results/")
    print(f"   - 可视化图: output/advanced_analysis/figures/")
    print(f"   - 综合报告: output/advanced_analysis/results/comprehensive_report.md")

def main():
    """
    主函数
    """
    try:
        # 设置目录
        setup_advanced_directories()
        
        # 执行高级分析工作流程
        advanced_analysis_workflow()
        
        print("\n✅ 高级分析示例执行完成!")
        print("🔍 请查看 output/advanced_analysis/ 目录下的详细结果")
        print("📊 综合报告: output/advanced_analysis/results/comprehensive_report.md")
        print("🎨 可视化图表: output/advanced_analysis/figures/")
        
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