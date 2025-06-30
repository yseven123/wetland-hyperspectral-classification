"""
辅助函数模块
Helpers Module

提供各种通用的辅助功能，包括：
- 数据处理和转换
- 内存管理
- 文件操作
- 数学计算
- 验证工具
"""

import os
import gc
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psutil
from scipy import ndimage
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from .logger import get_logger

logger = get_logger(__name__)


# ==================== 数组操作函数 ====================

def reshape_for_sklearn(data: np.ndarray) -> np.ndarray:
    """
    将数据重塑为适合sklearn的格式 (n_samples, n_features)
    
    Args:
        data: 输入数据，形状为 (height, width, bands) 或 (n_samples, n_features)
        
    Returns:
        重塑后的数据 (n_samples, n_features)
    """
    if len(data.shape) == 3:
        # (height, width, bands) -> (n_samples, n_features)
        return data.reshape(-1, data.shape[-1])
    elif len(data.shape) == 2:
        return data
    else:
        raise ValueError(f"不支持的数据形状: {data.shape}")


def reshape_for_pytorch(
    data: np.ndarray,
    target_shape: str = "NCHW"
) -> torch.Tensor:
    """
    将数据重塑为适合PyTorch的格式
    
    Args:
        data: 输入数据
        target_shape: 目标格式 ("NCHW", "NHWC", "NCH", "NHC")
        
    Returns:
        PyTorch张量
    """
    tensor = torch.from_numpy(data.astype(np.float32))
    
    if target_shape == "NCHW" and len(tensor.shape) == 3:
        # (H, W, C) -> (1, C, H, W)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif target_shape == "NHWC" and len(tensor.shape) == 3:
        # (H, W, C) -> (1, H, W, C)
        tensor = tensor.unsqueeze(0)
    elif target_shape == "NCH" and len(tensor.shape) == 2:
        # (W, C) -> (1, C, W)
        tensor = tensor.permute(1, 0).unsqueeze(0)
    
    return tensor


def normalize_data(
    data: np.ndarray,
    method: str = "minmax",
    axis: Optional[int] = None,
    clip_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    数据归一化
    
    Args:
        data: 输入数据
        method: 归一化方法 ("minmax", "zscore", "robust")
        axis: 归一化轴
        clip_range: 裁剪范围
        
    Returns:
        Tuple[归一化后的数据, 归一化参数]
    """
    logger.debug(f"使用{method}方法进行数据归一化")
    
    # 数据裁剪
    if clip_range:
        data = np.clip(data, clip_range[0], clip_range[1])
    
    if method == "minmax":
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        
        # 避免除零
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        
        normalized_data = (data - data_min) / data_range
        params = {"min": data_min, "max": data_max, "method": method}
        
    elif method == "zscore":
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        
        # 避免除零
        data_std[data_std == 0] = 1
        
        normalized_data = (data - data_mean) / data_std
        params = {"mean": data_mean, "std": data_std, "method": method}
        
    elif method == "robust":
        data_median = np.median(data, axis=axis, keepdims=True)
        data_mad = np.median(np.abs(data - data_median), axis=axis, keepdims=True)
        
        # 避免除零
        data_mad[data_mad == 0] = 1
        
        normalized_data = (data - data_median) / data_mad
        params = {"median": data_median, "mad": data_mad, "method": method}
        
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return normalized_data, params


def standardize_data(
    data: np.ndarray,
    scaler: Optional[object] = None,
    fit: bool = True
) -> Tuple[np.ndarray, object]:
    """
    数据标准化（使用sklearn的标准化器）
    
    Args:
        data: 输入数据
        scaler: 已存在的标准化器
        fit: 是否拟合标准化器
        
    Returns:
        Tuple[标准化后的数据, 标准化器]
    """
    original_shape = data.shape
    data_2d = reshape_for_sklearn(data)
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        standardized_data = scaler.fit_transform(data_2d)
        logger.debug("数据标准化完成（拟合+转换）")
    else:
        standardized_data = scaler.transform(data_2d)
        logger.debug("数据标准化完成（仅转换）")
    
    # 恢复原始形状
    if len(original_shape) == 3:
        standardized_data = standardized_data.reshape(original_shape)
    
    return standardized_data, scaler


# ==================== 时间工具函数 ====================

def format_duration(seconds: float) -> str:
    """
    格式化时间持续长度
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}分{remaining_seconds:.2f}秒"
    else:
        hours = int(seconds // 3600)
        remaining_seconds = seconds % 3600
        minutes = int(remaining_seconds // 60)
        seconds = remaining_seconds % 60
        return f"{hours}小时{minutes}分{seconds:.2f}秒"


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取时间戳字符串
    
    Args:
        format_str: 时间格式字符串
        
    Returns:
        时间戳字符串
    """
    return datetime.now().strftime(format_str)


class Timer:
    """简单的计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self.elapsed_time
    
    @property
    def elapsed_time(self) -> float:
        """获取耗时"""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ==================== 文件工具函数 ====================

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
        
    Returns:
        Path对象
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小
    """
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def clean_temp_files(temp_dir: Union[str, Path] = None) -> None:
    """
    清理临时文件
    
    Args:
        temp_dir: 临时目录路径，如果为None则使用系统临时目录
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    temp_dir = Path(temp_dir)
    
    if not temp_dir.exists():
        return
    
    try:
        wetland_temp_files = list(temp_dir.glob("wetland_*"))
        for temp_file in wetland_temp_files:
            if temp_file.is_file():
                temp_file.unlink()
            elif temp_file.is_dir():
                shutil.rmtree(temp_file)
        
        logger.info(f"清理了 {len(wetland_temp_files)} 个临时文件")
        
    except Exception as e:
        logger.warning(f"清理临时文件时出错: {e}")


# ==================== 内存管理函数 ====================

def get_memory_usage() -> Dict[str, float]:
    """
    获取当前内存使用情况
    
    Returns:
        内存使用信息字典
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / (1024 * 1024)
    }


def check_memory_availability(required_mb: float) -> bool:
    """
    检查是否有足够的可用内存
    
    Args:
        required_mb: 所需内存（MB）
        
    Returns:
        是否有足够内存
    """
    memory_info = get_memory_usage()
    available_mb = memory_info['available_mb']
    
    logger.debug(f"检查内存: 需要 {required_mb:.2f}MB, 可用 {available_mb:.2f}MB")
    
    return available_mb >= required_mb


def clear_cache():
    """清理内存缓存"""
    gc.collect()
    
    # 如果使用PyTorch，清理GPU缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    logger.debug("内存缓存已清理")


def estimate_memory_usage(
    data_shape: Tuple[int, ...],
    dtype: np.dtype,
    copies: int = 1
) -> float:
    """
    估算数据的内存使用量
    
    Args:
        data_shape: 数据形状
        dtype: 数据类型
        copies: 数据副本数量
        
    Returns:
        估算的内存使用量（MB）
    """
    total_elements = np.prod(data_shape)
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = total_elements * bytes_per_element * copies
    
    return total_bytes / (1024 * 1024)


# ==================== 数学工具函数 ====================

def calculate_class_weights(
    y: np.ndarray,
    method: str = "balanced"
) -> Dict[int, float]:
    """
    计算类别权重
    
    Args:
        y: 标签数组
        method: 计算方法 ("balanced", "inverse")
        
    Returns:
        类别权重字典
    """
    unique_classes = np.unique(y)
    
    if method == "balanced":
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y
        )
    elif method == "inverse":
        class_counts = np.bincount(y)
        total_samples = len(y)
        weights = total_samples / (len(unique_classes) * class_counts[unique_classes])
    else:
        raise ValueError(f"不支持的权重计算方法: {method}")
    
    weight_dict = dict(zip(unique_classes, weights))
    
    logger.debug(f"计算类别权重: {weight_dict}")
    return weight_dict


def compute_spatial_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    metric: str = "euclidean"
) -> float:
    """
    计算空间距离
    
    Args:
        coords1: 坐标1 (x, y)
        coords2: 坐标2 (x, y)
        metric: 距离度量 ("euclidean", "manhattan", "chebyshev")
        
    Returns:
        距离值
    """
    if metric == "euclidean":
        return np.sqrt(np.sum((coords1 - coords2) ** 2))
    elif metric == "manhattan":
        return np.sum(np.abs(coords1 - coords2))
    elif metric == "chebyshev":
        return np.max(np.abs(coords1 - coords2))
    else:
        raise ValueError(f"不支持的距离度量: {metric}")


def apply_smoothing_filter(
    image: np.ndarray,
    filter_type: str = "gaussian",
    **kwargs
) -> np.ndarray:
    """
    应用平滑滤波器
    
    Args:
        image: 输入图像
        filter_type: 滤波器类型 ("gaussian", "median", "uniform")
        **kwargs: 滤波器参数
        
    Returns:
        滤波后的图像
    """
    if filter_type == "gaussian":
        sigma = kwargs.get('sigma', 1.0)
        return ndimage.gaussian_filter(image, sigma=sigma)
    elif filter_type == "median":
        size = kwargs.get('size', 3)
        return ndimage.median_filter(image, size=size)
    elif filter_type == "uniform":
        size = kwargs.get('size', 3)
        return ndimage.uniform_filter(image, size=size)
    else:
        raise ValueError(f"不支持的滤波器类型: {filter_type}")


def compute_histogram_statistics(
    data: np.ndarray,
    bins: int = 50
) -> Dict[str, Any]:
    """
    计算直方图统计信息
    
    Args:
        data: 输入数据
        bins: 直方图分箱数
        
    Returns:
        统计信息字典
    """
    hist, bin_edges = np.histogram(data.flatten(), bins=bins)
    
    return {
        'histogram': hist,
        'bin_edges': bin_edges,
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'percentiles': {
            '25th': np.percentile(data, 25),
            '75th': np.percentile(data, 75),
            '95th': np.percentile(data, 95),
            '99th': np.percentile(data, 99)
        }
    }


# ==================== 验证工具函数 ====================

def validate_parameters(
    params: Dict[str, Any],
    required_params: List[str],
    param_types: Optional[Dict[str, type]] = None,
    param_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None
) -> bool:
    """
    验证参数的有效性
    
    Args:
        params: 参数字典
        required_params: 必需参数列表
        param_types: 参数类型字典
        param_ranges: 参数范围字典
        
    Returns:
        是否验证通过
    """
    logger.debug("开始参数验证")
    
    # 检查必需参数
    for param in required_params:
        if param not in params:
            logger.error(f"缺少必需参数: {param}")
            return False
    
    # 检查参数类型
    if param_types:
        for param, expected_type in param_types.items():
            if param in params and not isinstance(params[param], expected_type):
                logger.error(f"参数类型错误: {param} 应为 {expected_type}, 实际为 {type(params[param])}")
                return False
    
    # 检查参数范围
    if param_ranges:
        for param, (min_val, max_val) in param_ranges.items():
            if param in params:
                value = params[param]
                if value < min_val or value > max_val:
                    logger.error(f"参数超出范围: {param}={value}, 应在 [{min_val}, {max_val}] 内")
                    return False
    
    logger.debug("参数验证通过")
    return True


def check_data_consistency(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    检查数据一致性
    
    Args:
        data: 数据数组
        labels: 标签数组
        metadata: 元数据
        
    Returns:
        是否一致
    """
    logger.debug("开始数据一致性检查")
    
    # 检查数据基本属性
    if data.size == 0:
        logger.error("数据为空")
        return False
    
    if np.any(np.isnan(data)):
        logger.warning("数据包含NaN值")
    
    if np.any(np.isinf(data)):
        logger.warning("数据包含无穷值")
    
    # 检查标签一致性
    if labels is not None:
        if len(data.shape) == 3:  # 图像数据
            expected_shape = data.shape[:2]
            if labels.shape != expected_shape:
                logger.error(f"标签形状不匹配: 期望 {expected_shape}, 实际 {labels.shape}")
                return False
        elif len(data.shape) == 2:  # 特征数据
            if len(labels) != data.shape[0]:
                logger.error(f"标签数量不匹配: 期望 {data.shape[0]}, 实际 {len(labels)}")
                return False
    
    # 检查元数据一致性
    if metadata:
        if 'shape' in metadata and metadata['shape'] != data.shape:
            logger.warning(f"元数据形状不匹配: 元数据 {metadata['shape']}, 实际 {data.shape}")
    
    logger.debug("数据一致性检查通过")
    return True


def verify_model_compatibility(
    model: Any,
    input_shape: Tuple[int, ...],
    output_classes: int
) -> bool:
    """
    验证模型兼容性
    
    Args:
        model: 模型对象
        input_shape: 输入形状
        output_classes: 输出类别数
        
    Returns:
        是否兼容
    """
    logger.debug("开始模型兼容性验证")
    
    try:
        # 检查是否有predict方法
        if not hasattr(model, 'predict') and not hasattr(model, 'forward'):
            logger.error("模型缺少predict或forward方法")
            return False
        
        # 对于sklearn模型
        if hasattr(model, 'n_features_in_'):
            expected_features = np.prod(input_shape)
            if model.n_features_in_ != expected_features:
                logger.error(f"特征数不匹配: 期望 {expected_features}, 模型 {model.n_features_in_}")
                return False
        
        # 对于深度学习模型
        if hasattr(model, 'forward'):
            try:
                import torch
                # 创建测试输入
                test_input = torch.randn(1, *input_shape)
                with torch.no_grad():
                    output = model(test_input)
                
                if output.shape[-1] != output_classes:
                    logger.error(f"输出类别数不匹配: 期望 {output_classes}, 实际 {output.shape[-1]}")
                    return False
                    
            except Exception as e:
                logger.error(f"模型测试失败: {e}")
                return False
        
        logger.debug("模型兼容性验证通过")
        return True
        
    except Exception as e:
        logger.error(f"模型兼容性验证失败: {e}")
        return False


def create_train_val_split(
    data: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建训练和验证数据集分割
    
    Args:
        data: 输入数据
        labels: 标签
        val_ratio: 验证集比例
        stratify: 是否分层抽样
        random_state: 随机种子
        
    Returns:
        Tuple[训练数据, 验证数据, 训练标签, 验证标签]
    """
    from sklearn.model_selection import train_test_split
    
    stratify_y = labels if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels,
        test_size=val_ratio,
        stratify=stratify_y,
        random_state=random_state
    )
    
    logger.info(f"数据分割完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


# ==================== 通用工具函数 ====================

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
    Returns:
        结果数组
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = default
    
    return result


def ensure_numpy_array(data: Any) -> np.ndarray:
    """
    确保数据是numpy数组
    
    Args:
        data: 输入数据
        
    Returns:
        numpy数组
    """
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'numpy'):  # PyTorch张量
        return data.numpy()
    else:
        return np.array(data)


def print_system_summary():
    """打印系统摘要信息"""
    memory_info = get_memory_usage()
    
    print("=" * 50)
    print("系统状态摘要")
    print("=" * 50)
    print(f"内存使用: {memory_info['rss_mb']:.2f} MB ({memory_info['percent']:.1f}%)")
    print(f"可用内存: {memory_info['available_mb']:.2f} MB")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)


if __name__ == "__main__":
    # 测试函数
    print_system_summary()