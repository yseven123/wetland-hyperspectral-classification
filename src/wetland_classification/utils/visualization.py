"""
可视化工具模块
Visualization Utils Module

提供各种可视化功能，包括：
- 光谱曲线和签名图
- 分类结果地图
- 统计图表和混淆矩阵
- 训练过程可视化
- 景观分析可视化
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import rasterio.plot
from sklearn.metrics import confusion_matrix
import geopandas as gpd

from .logger import get_logger

logger = get_logger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 湿地分类配色方案
WETLAND_COLORS = {
    'water': '#0077BE',           # 蓝色 - 水体
    'emergent_vegetation': '#228B22', # 森林绿 - 挺水植物
    'floating_vegetation': '#90EE90', # 浅绿 - 浮叶植物
    'submerged_vegetation': '#006400', # 深绿 - 沉水植物
    'wet_soil': '#8B4513',        # 褐色 - 湿润土壤
    'dry_soil': '#DEB887',        # 浅褐 - 干燥土壤
    'built_up': '#696969',        # 灰色 - 建筑物
    'background': '#F5F5DC'       # 米色 - 背景
}


def plot_spectral_signature(
    spectral_data: np.ndarray,
    wavelengths: List[float],
    class_labels: Optional[List[str]] = None,
    title: str = "光谱特征曲线",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    绘制光谱特征曲线
    
    Args:
        spectral_data: 光谱数据 (n_samples, n_bands)
        wavelengths: 波长列表
        class_labels: 类别标签
        title: 图表标题
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制光谱特征曲线")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确保数据是2D的
    if len(spectral_data.shape) == 1:
        spectral_data = spectral_data.reshape(1, -1)
    
    # 绘制每个样本的光谱曲线
    for i, spectrum in enumerate(spectral_data):
        label = class_labels[i] if class_labels and i < len(class_labels) else f"样本 {i+1}"
        ax.plot(wavelengths, spectrum, label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('波长 (nm)', fontsize=12)
    ax.set_ylabel('反射率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加常见植被特征波段标记
    vegetation_bands = {
        '红边': 700,
        '近红外': 800,
        '短波红外1': 1600,
        '短波红外2': 2200
    }
    
    for band_name, wavelength in vegetation_bands.items():
        if min(wavelengths) <= wavelength <= max(wavelengths):
            ax.axvline(x=wavelength, color='red', linestyle='--', alpha=0.5)
            ax.text(wavelength, ax.get_ylim()[1]*0.9, band_name, 
                   rotation=90, ha='right', va='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"光谱特征曲线已保存至: {save_path}")
    
    return fig


def plot_spectral_curves(
    hyperspectral_data: np.ndarray,
    ground_truth: np.ndarray,
    wavelengths: List[float],
    class_names: List[str],
    n_samples_per_class: int = 10,
    title: str = "各类别平均光谱曲线",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    绘制各类别的平均光谱曲线及标准差
    
    Args:
        hyperspectral_data: 高光谱数据 (height, width, bands)
        ground_truth: 地面真实标签 (height, width)
        wavelengths: 波长列表
        class_names: 类别名称
        n_samples_per_class: 每类采样数量
        title: 图表标题
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制各类别光谱曲线")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    # 重塑数据
    data_2d = hyperspectral_data.reshape(-1, hyperspectral_data.shape[-1])
    labels_1d = ground_truth.flatten()
    
    class_spectra = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = labels_1d == i
        if np.sum(class_mask) == 0:
            continue
            
        class_data = data_2d[class_mask]
        
        # 随机采样
        if len(class_data) > n_samples_per_class:
            indices = np.random.choice(len(class_data), n_samples_per_class, replace=False)
            class_data = class_data[indices]
        
        class_spectra[class_name] = class_data
        
        # 计算均值和标准差
        mean_spectrum = np.mean(class_data, axis=0)
        std_spectrum = np.std(class_data, axis=0)
        
        # 绘制均值曲线
        ax1.plot(wavelengths, mean_spectrum, color=colors[i], 
                label=f"{class_name} (n={len(class_data)})", linewidth=2)
        
        # 绘制标准差带
        ax1.fill_between(wavelengths, 
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        color=colors[i], alpha=0.2)
        
        # 绘制原始曲线（子图2）
        for spectrum in class_data[:5]:  # 最多显示5条原始曲线
            ax2.plot(wavelengths, spectrum, color=colors[i], alpha=0.3, linewidth=0.8)
    
    # 设置第一个子图（均值曲线）
    ax1.set_xlabel('波长 (nm)', fontsize=12)
    ax1.set_ylabel('反射率', fontsize=12)
    ax1.set_title(f'{title} - 均值±标准差', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 设置第二个子图（原始曲线）
    ax2.set_xlabel('波长 (nm)', fontsize=12)
    ax2.set_ylabel('反射率', fontsize=12)
    ax2.set_title('原始光谱曲线样本', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"类别光谱曲线已保存至: {save_path}")
    
    return fig


def plot_classification_map(
    classification_result: np.ndarray,
    class_names: List[str],
    title: str = "分类结果图",
    colors: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    show_legend: bool = True
) -> plt.Figure:
    """
    绘制分类结果地图
    
    Args:
        classification_result: 分类结果 (height, width)
        class_names: 类别名称列表
        title: 图表标题
        colors: 颜色列表
        save_path: 保存路径
        figsize: 图片大小
        show_legend: 是否显示图例
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制分类结果地图")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_classes = len(class_names)
    
    # 设置颜色
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # 创建颜色映射
    cmap = ListedColormap(colors[:n_classes])
    norm = BoundaryNorm(range(n_classes + 1), n_classes)
    
    # 绘制分类图
    im = ax.imshow(classification_result, cmap=cmap, norm=norm, aspect='equal')
    
    # 添加图例
    if show_legend:
        patches_list = [patches.Patch(color=colors[i], label=class_names[i]) 
                       for i in range(n_classes)]
        ax.legend(handles=patches_list, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('像素列', fontsize=12)
    ax.set_ylabel('像素行', fontsize=12)
    
    # 移除刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"分类结果地图已保存至: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "混淆矩阵",
    normalize: str = 'true',  # 'true', 'pred', 'all', None
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        title: 图表标题
        normalize: 标准化方式
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制混淆矩阵")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存至: {save_path}")
    
    return fig


def plot_accuracy_curves(
    train_accuracies: List[float],
    val_accuracies: List[float],
    title: str = "模型训练精度曲线",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制训练和验证精度曲线
    
    Args:
        train_accuracies: 训练精度列表
        val_accuracies: 验证精度列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制精度曲线")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_accuracies) + 1)
    
    ax.plot(epochs, train_accuracies, 'b-', label='训练精度', linewidth=2)
    ax.plot(epochs, val_accuracies, 'r-', label='验证精度', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('精度', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 标注最佳验证精度
    best_val_epoch = np.argmax(val_accuracies) + 1
    best_val_acc = max(val_accuracies)
    ax.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.5)
    ax.text(best_val_epoch, best_val_acc, 
           f'最佳: Epoch {best_val_epoch}\n精度: {best_val_acc:.3f}',
           ha='center', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"精度曲线已保存至: {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "特征重要性",
    top_n: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名称列表
        importances: 特征重要性值
        title: 图表标题
        top_n: 显示前N个重要特征
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制特征重要性图")
    
    # 排序并选择前N个特征
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建水平条形图
    bars = ax.barh(range(len(sorted_names)), sorted_importances, color='skyblue')
    
    # 设置标签
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 在条形上添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(sorted_importances) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    # 反转y轴，使最重要的特征在顶部
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征重要性图已保存至: {save_path}")
    
    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "类别分布",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制类别分布图
    
    Args:
        labels: 标签数组
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制类别分布图")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 计算每个类别的数量
    unique, counts = np.unique(labels, return_counts=True)
    
    # 确保所有类别都包含在内
    class_counts = np.zeros(len(class_names))
    for i, count in zip(unique, counts):
        if i < len(class_names):
            class_counts[i] = count
    
    # 条形图
    bars = ax1.bar(range(len(class_names)), class_counts, color='lightblue')
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('类别样本数量', fontsize=12)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    
    # 在条形上添加数值标签
    for bar, count in zip(bars, class_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    # 饼图
    non_zero_indices = class_counts > 0
    if np.any(non_zero_indices):
        pie_counts = class_counts[non_zero_indices]
        pie_names = [class_names[i] for i in range(len(class_names)) if non_zero_indices[i]]
        
        ax2.pie(pie_counts, labels=pie_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('类别比例', fontsize=12)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"类别分布图已保存至: {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "训练历史",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    绘制训练历史曲线
    
    Args:
        history: 包含训练指标的字典
        title: 图表标题
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制训练历史曲线")
    
    metrics = list(history.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        epochs = range(1, len(history[metric]) + 1)
        axes[i].plot(epochs, history[metric], 'b-', linewidth=2)
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].set_title(f'{metric}变化', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练历史曲线已保存至: {save_path}")
    
    return fig


def plot_landscape_metrics(
    metrics_df: pd.DataFrame,
    title: str = "景观格局指标",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    绘制景观格局指标图
    
    Args:
        metrics_df: 景观指标数据框
        title: 图表标题
        save_path: 保存路径
        figsize: 图片大小
        
    Returns:
        matplotlib Figure对象
    """
    logger.info("正在绘制景观格局指标图")
    
    n_metrics = len(metrics_df.columns) - 1  # 假设第一列是类别名称
    n_rows = (n_metrics + 2) // 3  # 每行3个子图
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    class_names = metrics_df.iloc[:, 0]  # 第一列为类别名称
    
    for i, metric in enumerate(metrics_df.columns[1:]):
        if i < len(axes):
            values = metrics_df[metric]
            axes[i].bar(range(len(class_names)), values, color='lightgreen')
            axes[i].set_xlabel('类别', fontsize=10)
            axes[i].set_ylabel(metric, fontsize=10)
            axes[i].set_title(metric, fontsize=11)
            axes[i].set_xticks(range(len(class_names)))
            axes[i].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"景观格局指标图已保存至: {save_path}")
    
    return fig


def create_interactive_classification_map(
    classification_result: np.ndarray,
    class_names: List[str],
    title: str = "交互式分类结果地图"
) -> go.Figure:
    """
    创建交互式分类结果地图
    
    Args:
        classification_result: 分类结果
        class_names: 类别名称
        title: 图表标题
        
    Returns:
        Plotly Figure对象
    """
    logger.info("正在创建交互式分类结果地图")
    
    fig = px.imshow(
        classification_result,
        color_continuous_scale='tab10',
        title=title,
        labels={'color': '类别'}
    )
    
    # 更新颜色条标签
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="类别",
            tickvals=list(range(len(class_names))),
            ticktext=class_names
        )
    )
    
    return fig


def save_publication_figure(
    fig: plt.Figure,
    save_path: Union[str, Path],
    dpi: int = 300,
    format: str = 'png',
    transparent: bool = False
) -> None:
    """
    保存适合发表的高质量图片
    
    Args:
        fig: matplotlib图形对象
        save_path: 保存路径
        dpi: 分辨率
        format: 保存格式
        transparent: 是否透明背景
    """
    logger.info(f"正在保存高质量图片至: {save_path}")
    
    fig.savefig(
        save_path,
        dpi=dpi,
        format=format,
        bbox_inches='tight',
        transparent=transparent,
        facecolor='white' if not transparent else 'none'
    )
    
    logger.info("高质量图片保存完成")


def create_report_figures(
    classification_result: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str],
    metrics: Dict[str, float],
    output_dir: Union[str, Path],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    创建完整的报告图片集
    
    Args:
        classification_result: 分类结果
        ground_truth: 地面真实值
        class_names: 类别名称
        metrics: 评估指标
        output_dir: 输出目录
        figsize: 图片大小
    """
    logger.info("正在创建完整报告图片集")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 分类结果地图
    fig1 = plot_classification_map(
        classification_result, class_names, 
        title="分类结果图", figsize=figsize
    )
    save_publication_figure(fig1, output_dir / "classification_map.png")
    plt.close(fig1)
    
    # 2. 混淆矩阵
    y_true = ground_truth.flatten()
    y_pred = classification_result.flatten()
    fig2 = plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="混淆矩阵", figsize=figsize
    )
    save_publication_figure(fig2, output_dir / "confusion_matrix.png")
    plt.close(fig2)
    
    # 3. 类别分布
    fig3 = plot_class_distribution(
        y_true, class_names,
        title="类别分布", figsize=figsize
    )
    save_publication_figure(fig3, output_dir / "class_distribution.png")
    plt.close(fig3)
    
    # 4. 评估指标摘要图
    fig4, ax = plt.subplots(figsize=(8, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color='lightcoral')
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('分类性能指标', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # 在条形上添加数值标签
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_publication_figure(fig4, output_dir / "performance_metrics.png")
    plt.close(fig4)
    
    logger.info(f"报告图片集已保存至: {output_dir}")