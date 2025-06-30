#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
湿地高光谱分类系统 - 基础分类示例
Wetland Hyperspectral Classification System - Basic Classification Example

这个示例展示了如何使用湿地高光谱分类系统进行基础的分类任务，
包括数据加载、预处理、特征提取、模型训练和结果评估的完整流程。

作者: 研究团队
日期: 2024-06-30
版本: 1.0.0
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging

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
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已正确安装湿地分类系统")
    sys.exit(1)

# 设置日志
logger = get_logger(__name__)

def setup_directories():
    """
    设置必要的目录结构
    """
    directories = [
        'output/basic_classification',
        'output/basic_classification/results',
        'output/basic_classification/figures',
        'output/basic_classification/models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def check_data_availability():
    """
    检查演示数据是否可用
    """
    demo_data_path = 'data/samples/demo_scene'
    
    if not os.path.exists(demo_data_path):
        logger.warning(f"演示数据目录不存在: {demo_data_path}")
        logger.info("正在创建模拟演示数据...")
        create_demo_data()
    else:
        logger.info(f"找到演示数据: {demo_data_path}")

def create_demo_data():
    """
    创建模拟的演示数据用于测试
    """
    logger.info("开始创建模拟演示数据...")
    
    # 创建演示数据目录
    demo_dir = Path('data/samples/demo_scene')
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成模拟高光谱数据
    np.random.seed(42)
    height, width, bands = 128, 128, 100
    
    # 模拟不同的地物类型
    hyperspectral_data = np.zeros((height, width, bands))
    labels = np.zeros((height, width), dtype=int)
    
    # 创建不同区域的光谱特征
    for i in range(height):
        for j in range(width):
            # 根据位置确定类别
            if i < height // 3:  # 水体区域
                class_id = 1
                base_spectrum = np.random.normal(0.1, 0.02, bands)
                base_spectrum[40:60] = np.random.normal(0.05, 0.01, 20)  # 水体低反射
            elif i < 2 * height // 3:  # 植被区域
                class_id = 2
                base_spectrum = np.random.normal(0.3, 0.05, bands)
                base_spectrum[60:80] = np.random.normal(0.7, 0.1, 20)  # 植被红边
            else:  # 土壤区域
                class_id = 3
                base_spectrum = np.random.normal(0.4, 0.08, bands)
            
            # 添加噪声
            noise = np.random.normal(0, 0.01, bands)
            hyperspectral_data[i, j, :] = np.clip(base_spectrum + noise, 0, 1)
            labels[i, j] = class_id
    
    # 保存模拟数据
    np.save(demo_dir / 'hyperspectral_data.npy', hyperspectral_data)
    np.save(demo_dir / 'ground_truth.npy', labels)
    
    # 创建波长信息
    wavelengths = np.linspace(400, 2500, bands)
    np.save(demo_dir / 'wavelengths.npy', wavelengths)
    
    # 创建类别信息
    class_info = {
        1: {'name': '水体', 'color': '#0000FF'},
        2: {'name': '植被', 'color': '#00FF00'},
        3: {'name': '土壤', 'color': '#8B4513'}
    }
    
    import json
    with open(demo_dir / 'class_info.json', 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"模拟演示数据创建完成，数据形状: {hyperspectral_data.shape}")

def basic_classification_workflow():
    """
    基础分类工作流程示例
    """
    logger.info("="*60)
    logger.info("开始湿地高光谱基础分类示例")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 步骤1: 创建默认配置
        logger.info("步骤1: 创建配置")
        config = create_basic_config()
        logger.info("配置创建完成")
        
        # 步骤2: 数据加载
        logger.info("步骤2: 加载数据")
        hyperspectral_data, labels, wavelengths, class_info = load_demo_data()
        logger.info(f"数据加载完成 - 数据形状: {hyperspectral_data.shape}, 类别数: {len(np.unique(labels[labels>0]))}")
        
        # 步骤3: 数据预处理
        logger.info("步骤3: 数据预处理")
        processed_data = preprocess_data(hyperspectral_data)
        logger.info("数据预处理完成")
        
        # 步骤4: 特征提取
        logger.info("步骤4: 特征提取")
        features, feature_names = extract_features(processed_data, wavelengths)
        logger.info(f"特征提取完成 - 特征维度: {features.shape[1]}")
        
        # 步骤5: 准备训练数据
        logger.info("步骤5: 准备训练数据")
        X_train, X_test, y_train, y_test, train_indices, test_indices = prepare_training_data(
            features, labels
        )
        logger.info(f"数据划分完成 - 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        
        # 步骤6: 模型训练
        logger.info("步骤6: 模型训练")
        trained_models = train_multiple_models(X_train, y_train, X_test, y_test)
        logger.info(f"模型训练完成 - 共训练 {len(trained_models)} 个模型")
        
        # 步骤7: 模型评估
        logger.info("步骤7: 模型评估")
        evaluation_results = evaluate_models(trained_models, X_test, y_test, class_info)
        
        # 步骤8: 选择最佳模型进行场景分类
        logger.info("步骤8: 场景分类")
        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
        best_model = trained_models[best_model_name]
        logger.info(f"选择最佳模型: {best_model_name}")
        
        classification_map = classify_full_scene(best_model, features, hyperspectral_data.shape[:2])
        
        # 步骤9: 结果可视化
        logger.info("步骤9: 结果可视化")
        create_visualizations(
            hyperspectral_data, labels, classification_map, 
            evaluation_results, wavelengths, class_info
        )
        
        # 步骤10: 保存结果
        logger.info("步骤10: 保存结果")
        save_results(
            classification_map, evaluation_results, trained_models, 
            config, class_info
        )
        
        # 显示最终结果
        total_time = time.time() - start_time
        display_final_results(evaluation_results, total_time)
        
    except Exception as e:
        logger.error(f"基础分类流程执行失败: {e}")
        raise

def create_basic_config():
    """
    创建基础配置
    """
    config = {
        'data': {
            'input_path': 'data/samples/demo_scene',
            'output_path': 'output/basic_classification',
            'file_format': 'npy'
        },
        'preprocessing': {
            'normalization': True,
            'noise_reduction': False,
            'bad_bands_removal': False
        },
        'features': {
            'spectral_features': True,
            'vegetation_indices': ['NDVI', 'NDWI', 'EVI'],
            'texture_features': False,  # 简化示例，不使用纹理特征
            'spatial_features': False
        },
        'classification': {
            'models': ['svm', 'random_forest', 'xgboost'],
            'test_ratio': 0.3,
            'random_state': 42
        },
        'evaluation': {
            'metrics': ['accuracy', 'kappa', 'f1_score'],
            'cross_validation': False  # 简化示例
        }
    }
    
    return config

def load_demo_data():
    """
    加载演示数据
    """
    demo_dir = Path('data/samples/demo_scene')
    
    # 加载高光谱数据
    hyperspectral_data = np.load(demo_dir / 'hyperspectral_data.npy')
    
    # 加载标签
    labels = np.load(demo_dir / 'ground_truth.npy')
    
    # 加载波长信息
    wavelengths = np.load(demo_dir / 'wavelengths.npy')
    
    # 加载类别信息
    import json
    with open(demo_dir / 'class_info.json', 'r', encoding='utf-8') as f:
        class_info = json.load(f)
    
    # 转换类别信息的键为整数
    class_info = {int(k): v for k, v in class_info.items()}
    
    return hyperspectral_data, labels, wavelengths, class_info

def preprocess_data(hyperspectral_data):
    """
    数据预处理
    """
    # 简单的归一化预处理
    processed_data = np.copy(hyperspectral_data)
    
    # 按波段进行归一化
    for band in range(processed_data.shape[2]):
        band_data = processed_data[:, :, band]
        band_min = np.min(band_data)
        band_max = np.max(band_data)
        
        if band_max > band_min:
            processed_data[:, :, band] = (band_data - band_min) / (band_max - band_min)
        else:
            processed_data[:, :, band] = 0
    
    logger.info("数据归一化完成")
    return processed_data

def extract_features(hyperspectral_data, wavelengths):
    """
    特征提取
    """
    height, width, bands = hyperspectral_data.shape
    
    # 重塑数据为2D格式 (n_pixels, n_bands)
    reshaped_data = hyperspectral_data.reshape(-1, bands)
    
    features = []
    feature_names = []
    
    # 1. 原始光谱特征（下采样）
    spectral_step = max(1, bands // 20)  # 选择约20个光谱波段
    selected_bands = range(0, bands, spectral_step)
    spectral_features = reshaped_data[:, selected_bands]
    features.append(spectral_features)
    feature_names.extend([f'Band_{wavelengths[i]:.0f}nm' for i in selected_bands])
    
    # 2. 植被指数
    vegetation_indices = calculate_vegetation_indices(reshaped_data, wavelengths)
    features.append(vegetation_indices)
    feature_names.extend(['NDVI', 'NDWI', 'EVI', 'SAVI'])
    
    # 3. 光谱统计特征
    spectral_stats = calculate_spectral_statistics(reshaped_data)
    features.append(spectral_stats)
    feature_names.extend(['Mean', 'Std', 'Skewness', 'Kurtosis'])
    
    # 合并所有特征
    all_features = np.column_stack(features)
    
    logger.info(f"特征提取完成 - 总特征数: {all_features.shape[1]}")
    
    return all_features, feature_names

def calculate_vegetation_indices(data, wavelengths):
    """
    计算植被指数
    """
    # 找到最接近的波段
    def find_nearest_band(target_wavelength):
        return np.argmin(np.abs(wavelengths - target_wavelength))
    
    # 定义关键波段
    red_band = find_nearest_band(670)     # 红光
    nir_band = find_nearest_band(850)     # 近红外
    blue_band = find_nearest_band(470)    # 蓝光
    swir_band = find_nearest_band(1600)   # 短波红外
    
    red = data[:, red_band]
    nir = data[:, nir_band]
    blue = data[:, blue_band]
    swir = data[:, swir_band]
    
    # 避免除零错误
    epsilon = 1e-8
    
    # NDVI
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # NDWI
    ndwi = (nir - swir) / (nir + swir + epsilon)
    
    # EVI
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)
    
    # SAVI
    L = 0.5  # 土壤调节因子
    savi = (1 + L) * (nir - red) / (nir + red + L + epsilon)
    
    return np.column_stack([ndvi, ndwi, evi, savi])

def calculate_spectral_statistics(data):
    """
    计算光谱统计特征
    """
    from scipy import stats
    
    # 计算各种统计量
    mean_values = np.mean(data, axis=1)
    std_values = np.std(data, axis=1)
    skewness_values = stats.skew(data, axis=1)
    kurtosis_values = stats.kurtosis(data, axis=1)
    
    return np.column_stack([mean_values, std_values, skewness_values, kurtosis_values])

def prepare_training_data(features, labels):
    """
    准备训练数据
    """
    from sklearn.model_selection import train_test_split
    
    # 获取有效像素（非零标签）
    valid_mask = labels.ravel() > 0
    valid_features = features[valid_mask]
    valid_labels = labels.ravel()[valid_mask]
    
    # 数据划分
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        valid_features, valid_labels, 
        np.where(valid_mask)[0],  # 原始索引
        test_size=0.3, 
        random_state=42, 
        stratify=valid_labels
    )
    
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, train_idx, test_idx

def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    训练多个模型
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    
    models = {}
    scalers = {}
    
    # 1. SVM 模型
    logger.info("训练 SVM 模型...")
    scaler_svm = StandardScaler()
    X_train_scaled = scaler_svm.fit_transform(X_train)
    X_test_scaled = scaler_svm.transform(X_test)
    
    svm_model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    models['SVM'] = svm_model
    scalers['SVM'] = scaler_svm
    
    # 2. Random Forest 模型
    logger.info("训练 Random Forest 模型...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    models['Random Forest'] = rf_model
    scalers['Random Forest'] = None  # RF不需要标准化
    
    # 3. XGBoost 模型
    logger.info("训练 XGBoost 模型...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    models['XGBoost'] = xgb_model
    scalers['XGBoost'] = None  # XGBoost不需要标准化
    
    # 将标准化器存储在模型中以便后续使用
    for model_name in models:
        if hasattr(models[model_name], 'scaler'):
            models[model_name].scaler = scalers[model_name]
        else:
            # 为模型对象添加scaler属性
            setattr(models[model_name], 'scaler', scalers[model_name])
    
    return models

def evaluate_models(models, X_test, y_test, class_info):
    """
    评估模型性能
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import cohen_kappa_score, f1_score
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"评估 {model_name} 模型...")
        
        # 预测
        if hasattr(model, 'scaler') and model.scaler is not None:
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 分类报告
        class_names = [class_info[i]['name'] for i in sorted(class_info.keys())]
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        results[model_name] = {
            'accuracy': accuracy,
            'kappa': kappa,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} - 精度: {accuracy:.3f}, Kappa: {kappa:.3f}, F1: {f1:.3f}")
    
    return results

def classify_full_scene(model, features, original_shape):
    """
    对整个场景进行分类
    """
    logger.info("开始场景分类...")
    
    # 预测
    if hasattr(model, 'scaler') and model.scaler is not None:
        features_scaled = model.scaler.transform(features)
        predictions = model.predict(features_scaled)
    else:
        predictions = model.predict(features)
    
    # 重塑为原始形状
    classification_map = predictions.reshape(original_shape)
    
    logger.info("场景分类完成")
    return classification_map

def create_visualizations(hyperspectral_data, true_labels, predicted_labels, 
                         evaluation_results, wavelengths, class_info):
    """
    创建可视化结果
    """
    logger.info("创建可视化图表...")
    
    # 设置图表样式
    plt.style.use('default')
    
    # 1. 分类结果对比图
    create_classification_comparison(true_labels, predicted_labels, class_info)
    
    # 2. 模型性能对比图
    create_performance_comparison(evaluation_results)
    
    # 3. 混淆矩阵图
    create_confusion_matrices(evaluation_results, class_info)
    
    # 4. 光谱特征图
    create_spectral_signatures(hyperspectral_data, true_labels, wavelengths, class_info)
    
    logger.info("可视化图表创建完成")

def create_classification_comparison(true_labels, predicted_labels, class_info):
    """
    创建分类结果对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 创建颜色映射
    colors = ['black'] + [class_info[i]['color'] for i in sorted(class_info.keys())]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # 真实标签
    im1 = axes[0].imshow(true_labels, cmap=cmap, vmin=0, vmax=len(class_info))
    axes[0].set_title('真实标签', fontsize=14)
    axes[0].axis('off')
    
    # 预测标签
    im2 = axes[1].imshow(predicted_labels, cmap=cmap, vmin=0, vmax=len(class_info))
    axes[1].set_title('预测结果', fontsize=14)
    axes[1].axis('off')
    
    # 添加图例
    handles = []
    labels = []
    for class_id in sorted(class_info.keys()):
        from matplotlib.patches import Patch
        handles.append(Patch(color=class_info[class_id]['color']))
        labels.append(class_info[class_id]['name'])
    
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=len(class_info))
    
    plt.tight_layout()
    plt.savefig('output/basic_classification/figures/classification_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_comparison(evaluation_results):
    """
    创建模型性能对比图
    """
    models = list(evaluation_results.keys())
    metrics = ['accuracy', 'kappa', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [evaluation_results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.upper())
    
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('模型性能对比')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, metric in enumerate(metrics):
        values = [evaluation_results[model][metric] for model in models]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/basic_classification/figures/performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrices(evaluation_results, class_info):
    """
    创建混淆矩阵图
    """
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    class_names = [class_info[i]['name'] for i in sorted(class_info.keys())]
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        axes[idx].set_title(f'{model_name}\n混淆矩阵')
        
        # 添加文本标注
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                             ha="center", va="center",
                             color="white" if cm_normalized[i, j] > thresh else "black")
        
        axes[idx].set_ylabel('真实标签')
        axes[idx].set_xlabel('预测标签')
        axes[idx].set_xticks(range(len(class_names)))
        axes[idx].set_yticks(range(len(class_names)))
        axes[idx].set_xticklabels(class_names, rotation=45)
        axes[idx].set_yticklabels(class_names)
    
    plt.tight_layout()
    plt.savefig('output/basic_classification/figures/confusion_matrices.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_spectral_signatures(hyperspectral_data, labels, wavelengths, class_info):
    """
    创建光谱特征图
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for class_id in sorted(class_info.keys()):
        # 获取该类别的像素
        class_mask = labels == class_id
        if np.sum(class_mask) == 0:
            continue
        
        class_pixels = hyperspectral_data[class_mask]
        
        # 计算平均光谱和标准差
        mean_spectrum = np.mean(class_pixels, axis=0)
        std_spectrum = np.std(class_pixels, axis=0)
        
        # 绘制光谱曲线
        color = class_info[class_id]['color']
        ax.plot(wavelengths, mean_spectrum, color=color, 
               label=class_info[class_id]['name'], linewidth=2)
        
        # 添加标准差阴影
        ax.fill_between(wavelengths, 
                       mean_spectrum - std_spectrum,
                       mean_spectrum + std_spectrum,
                       color=color, alpha=0.2)
    
    ax.set_xlabel('波长 (nm)')
    ax.set_ylabel('反射率')
    ax.set_title('各类别光谱特征曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/basic_classification/figures/spectral_signatures.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_results(classification_map, evaluation_results, models, config, class_info):
    """
    保存结果
    """
    logger.info("保存分析结果...")
    
    output_dir = Path('output/basic_classification')
    
    # 1. 保存分类结果图
    np.save(output_dir / 'results' / 'classification_map.npy', classification_map)
    
    # 2. 保存评估结果
    import json
    
    # 准备可序列化的评估结果
    serializable_results = {}
    for model_name, results in evaluation_results.items():
        serializable_results[model_name] = {
            'accuracy': float(results['accuracy']),
            'kappa': float(results['kappa']),
            'f1_score': float(results['f1_score']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
        }
    
    with open(output_dir / 'results' / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 3. 保存模型
    import pickle
    for model_name, model in models.items():
        model_file = output_dir / 'models' / f'{model_name.lower().replace(" ", "_")}_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
    
    # 4. 保存配置
    with open(output_dir / 'results' / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 5. 生成文本报告
    generate_text_report(evaluation_results, class_info, output_dir)
    
    logger.info(f"所有结果已保存到: {output_dir}")

def generate_text_report(evaluation_results, class_info, output_dir):
    """
    生成文本报告
    """
    report_file = output_dir / 'results' / 'classification_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("湿地高光谱分类系统 - 基础分类报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("类别信息:\n")
        for class_id, info in class_info.items():
            f.write(f"  {class_id}: {info['name']}\n")
        f.write("\n")
        
        f.write("模型性能汇总:\n")
        f.write("-"*50 + "\n")
        
        for model_name, results in evaluation_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  总体精度: {results['accuracy']:.4f}\n")
            f.write(f"  Kappa系数: {results['kappa']:.4f}\n")
            f.write(f"  F1分数:   {results['f1_score']:.4f}\n")
        
        f.write("\n\n最佳模型: ")
        best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
        f.write(f"{best_model} (精度: {evaluation_results[best_model]['accuracy']:.4f})\n")

def display_final_results(evaluation_results, total_time):
    """
    显示最终结果
    """
    logger.info("="*60)
    logger.info("基础分类示例完成!")
    logger.info("="*60)
    
    # 显示性能汇总
    print("\n📊 模型性能汇总:")
    print("-" * 50)
    for model_name, results in evaluation_results.items():
        print(f"{model_name:15s} | 精度: {results['accuracy']:.3f} | "
              f"Kappa: {results['kappa']:.3f} | F1: {results['f1_score']:.3f}")
    
    # 显示最佳模型
    best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
    best_accuracy = evaluation_results[best_model]['accuracy']
    
    print(f"\n🏆 最佳模型: {best_model} (精度: {best_accuracy:.3f})")
    print(f"⏱️  总用时: {total_time:.2f} 秒")
    
    # 显示输出文件
    print(f"\n📁 输出文件位置:")
    print(f"   - 分类结果: output/basic_classification/results/")
    print(f"   - 可视化图: output/basic_classification/figures/")
    print(f"   - 训练模型: output/basic_classification/models/")
    
    logger.info("基础分类示例执行成功!")

def main():
    """
    主函数
    """
    try:
        # 设置目录
        setup_directories()
        
        # 检查数据
        check_data_availability()
        
        # 执行基础分类工作流程
        basic_classification_workflow()
        
        print("\n✅ 基础分类示例执行完成!")
        print("🔍 请查看 output/basic_classification/ 目录下的结果文件")
        print("📊 可视化图表已保存在 figures/ 子目录中")
        
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