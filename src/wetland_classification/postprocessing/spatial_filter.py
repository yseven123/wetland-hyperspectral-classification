"""
空间滤波器
=========

这个模块实现了多种空间滤波算法，用于优化高光谱分类结果。
主要功能包括噪声去除、边界平滑、空间一致性增强等。

主要算法：
- 中值滤波
- 高斯滤波  
- 双边滤波
- 形态学滤波
- 自适应滤波
- 马尔可夫随机场
- 条件随机场

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import medfilt2d
from skimage import filters, morphology, segmentation
from skimage.restoration import denoise_bilateral
from sklearn.cluster import KMeans
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class SpatialFilter:
    """
    空间滤波器基类
    
    所有空间滤波器的基础类，提供通用的滤波接口和工具函数。
    """
    
    def __init__(self, 
                 filter_size: int = 3,
                 preserve_boundaries: bool = True,
                 handle_edge: str = 'reflect',
                 **kwargs):
        """
        初始化空间滤波器
        
        Parameters:
        -----------
        filter_size : int, default=3
            滤波器大小
        preserve_boundaries : bool, default=True
            是否保持边界
        handle_edge : str, default='reflect'
            边界处理方式
        """
        self.filter_size = filter_size
        self.preserve_boundaries = preserve_boundaries
        self.handle_edge = handle_edge
        self.config = kwargs
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用空间滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类结果图像
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        raise NotImplementedError("子类必须实现apply方法")
    
    def _validate_input(self, classification_map: np.ndarray):
        """验证输入数据"""
        if classification_map.ndim != 2:
            raise ValueError("分类图像必须是2D数组")
        
        if classification_map.size == 0:
            raise ValueError("分类图像不能为空")


class MedianFilter(SpatialFilter):
    """
    中值滤波器
    
    使用中值滤波去除分类结果中的椒盐噪声，特别适合处理孤立的错分像素。
    """
    
    def __init__(self, 
                 filter_size: int = 3,
                 iterations: int = 1,
                 **kwargs):
        """
        初始化中值滤波器
        
        Parameters:
        -----------
        filter_size : int, default=3
            滤波器大小（奇数）
        iterations : int, default=1
            迭代次数
        """
        super().__init__(filter_size=filter_size, **kwargs)
        self.iterations = iterations
        
        if filter_size % 2 == 0:
            raise ValueError("滤波器大小必须是奇数")
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用中值滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"应用中值滤波 - 滤波器大小: {self.filter_size}, 迭代次数: {self.iterations}")
        
        filtered_map = classification_map.copy()
        
        for i in range(self.iterations):
            # 使用scipy的中值滤波
            filtered_map = median_filter(
                filtered_map, 
                size=self.filter_size,
                mode=self.handle_edge
            )
            
            logger.debug(f"完成第 {i+1} 次中值滤波")
        
        # 计算变化统计
        changed_pixels = np.sum(filtered_map != classification_map)
        total_pixels = classification_map.size
        change_ratio = changed_pixels / total_pixels
        
        logger.info(f"中值滤波完成 - 改变像素: {changed_pixels}/{total_pixels} ({change_ratio:.2%})")
        
        return filtered_map


class GaussianFilter(SpatialFilter):
    """
    高斯滤波器
    
    使用高斯滤波进行空间平滑，适合处理连续的分类边界。
    """
    
    def __init__(self, 
                 sigma: float = 1.0,
                 truncate: float = 4.0,
                 discretize_levels: Optional[int] = None,
                 **kwargs):
        """
        初始化高斯滤波器
        
        Parameters:
        -----------
        sigma : float, default=1.0
            高斯核标准差
        truncate : float, default=4.0
            截断参数
        discretize_levels : int, optional
            离散化层级数
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.truncate = truncate
        self.discretize_levels = discretize_levels
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用高斯滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"应用高斯滤波 - sigma: {self.sigma}")
        
        # 获取唯一类别
        unique_classes = np.unique(classification_map)
        n_classes = len(unique_classes)
        
        if n_classes <= 2:
            # 二值图像的高斯滤波
            filtered_map = gaussian_filter(
                classification_map.astype(float),
                sigma=self.sigma,
                truncate=self.truncate,
                mode=self.handle_edge
            )
            
            # 二值化
            threshold = (unique_classes[0] + unique_classes[-1]) / 2
            filtered_map = np.where(filtered_map > threshold, 
                                  unique_classes[-1], unique_classes[0])
        else:
            # 多类别的高斯滤波
            filtered_map = self._multiclass_gaussian_filter(
                classification_map, unique_classes
            )
        
        logger.info("高斯滤波完成")
        
        return filtered_map.astype(classification_map.dtype)
    
    def _multiclass_gaussian_filter(self, classification_map: np.ndarray, 
                                   unique_classes: np.ndarray) -> np.ndarray:
        """多类别高斯滤波"""
        height, width = classification_map.shape
        n_classes = len(unique_classes)
        
        # 创建one-hot编码
        class_maps = np.zeros((height, width, n_classes))
        for i, class_id in enumerate(unique_classes):
            class_maps[:, :, i] = (classification_map == class_id).astype(float)
        
        # 对每个类别图应用高斯滤波
        filtered_class_maps = np.zeros_like(class_maps)
        for i in range(n_classes):
            filtered_class_maps[:, :, i] = gaussian_filter(
                class_maps[:, :, i],
                sigma=self.sigma,
                truncate=self.truncate,
                mode=self.handle_edge
            )
        
        # 选择概率最大的类别
        max_indices = np.argmax(filtered_class_maps, axis=2)
        filtered_map = unique_classes[max_indices]
        
        return filtered_map


class BilateralFilter(SpatialFilter):
    """
    双边滤波器
    
    保持边界的同时进行空间平滑，适合处理具有清晰边界的分类结果。
    """
    
    def __init__(self, 
                 sigma_color: float = 0.1,
                 sigma_spatial: float = 1.0,
                 **kwargs):
        """
        初始化双边滤波器
        
        Parameters:
        -----------
        sigma_color : float, default=0.1
            颜色空间标准差
        sigma_spatial : float, default=1.0
            空间标准差
        """
        super().__init__(**kwargs)
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用双边滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"应用双边滤波 - sigma_color: {self.sigma_color}, sigma_spatial: {self.sigma_spatial}")
        
        # 将分类图像归一化到[0,1]
        unique_classes = np.unique(classification_map)
        normalized_map = (classification_map - unique_classes.min()) / (unique_classes.max() - unique_classes.min())
        
        # 应用双边滤波
        filtered_normalized = denoise_bilateral(
            normalized_map,
            sigma_color=self.sigma_color,
            sigma_spatial=self.sigma_spatial,
            channel_axis=None
        )
        
        # 量化到最近的类别
        filtered_map = self._quantize_to_classes(filtered_normalized, unique_classes)
        
        logger.info("双边滤波完成")
        
        return filtered_map
    
    def _quantize_to_classes(self, filtered_map: np.ndarray, 
                            unique_classes: np.ndarray) -> np.ndarray:
        """量化到最近的类别"""
        # 反归一化
        denormalized = filtered_map * (unique_classes.max() - unique_classes.min()) + unique_classes.min()
        
        # 找到最近的类别
        quantized = np.zeros_like(denormalized, dtype=unique_classes.dtype)
        for pixel_val in np.nditer(denormalized, flags=['multi_index']):
            idx = denormalized[pixel_val.multi_index]
            distances = np.abs(unique_classes - idx)
            nearest_class = unique_classes[np.argmin(distances)]
            quantized[pixel_val.multi_index] = nearest_class
        
        return quantized


class MorphologicalFilter(SpatialFilter):
    """
    形态学滤波器
    
    使用形态学操作进行分类结果优化，包括开运算、闭运算等。
    """
    
    def __init__(self, 
                 operation: str = 'opening',
                 structuring_element: str = 'disk',
                 size: int = 3,
                 iterations: int = 1,
                 **kwargs):
        """
        初始化形态学滤波器
        
        Parameters:
        -----------
        operation : str, default='opening'
            形态学操作类型 ('opening', 'closing', 'gradient', 'tophat', 'blackhat')
        structuring_element : str, default='disk'
            结构元素类型 ('disk', 'square', 'diamond')
        size : int, default=3
            结构元素大小
        iterations : int, default=1
            迭代次数
        """
        super().__init__(**kwargs)
        self.operation = operation
        self.structuring_element = structuring_element
        self.size = size
        self.iterations = iterations
        
        # 创建结构元素
        self.selem = self._create_structuring_element()
    
    def _create_structuring_element(self):
        """创建结构元素"""
        if self.structuring_element == 'disk':
            return morphology.disk(self.size)
        elif self.structuring_element == 'square':
            return morphology.square(self.size)
        elif self.structuring_element == 'diamond':
            return morphology.diamond(self.size)
        else:
            raise ValueError(f"不支持的结构元素类型: {self.structuring_element}")
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用形态学滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"应用形态学滤波 - 操作: {self.operation}, 结构元素: {self.structuring_element}")
        
        # 获取唯一类别
        unique_classes = np.unique(classification_map)
        
        if len(unique_classes) == 2:
            # 二值形态学操作
            filtered_map = self._binary_morphology(classification_map, unique_classes)
        else:
            # 多类别形态学操作
            filtered_map = self._multiclass_morphology(classification_map, unique_classes)
        
        logger.info("形态学滤波完成")
        
        return filtered_map
    
    def _binary_morphology(self, classification_map: np.ndarray, 
                          unique_classes: np.ndarray) -> np.ndarray:
        """二值形态学操作"""
        # 转换为二值图像
        binary_map = classification_map == unique_classes[-1]
        
        # 应用形态学操作
        for _ in range(self.iterations):
            if self.operation == 'opening':
                binary_map = morphology.opening(binary_map, self.selem)
            elif self.operation == 'closing':
                binary_map = morphology.closing(binary_map, self.selem)
            elif self.operation == 'gradient':
                binary_map = morphology.dilation(binary_map, self.selem) - \
                           morphology.erosion(binary_map, self.selem)
            elif self.operation == 'tophat':
                binary_map = morphology.white_tophat(binary_map, self.selem)
            elif self.operation == 'blackhat':
                binary_map = morphology.black_tophat(binary_map, self.selem)
        
        # 转换回分类图像
        filtered_map = np.where(binary_map, unique_classes[-1], unique_classes[0])
        
        return filtered_map.astype(classification_map.dtype)
    
    def _multiclass_morphology(self, classification_map: np.ndarray, 
                              unique_classes: np.ndarray) -> np.ndarray:
        """多类别形态学操作"""
        filtered_map = classification_map.copy()
        
        # 对每个类别分别进行形态学操作
        for class_id in unique_classes:
            # 创建二值掩码
            class_mask = classification_map == class_id
            
            # 应用形态学操作
            for _ in range(self.iterations):
                if self.operation == 'opening':
                    class_mask = morphology.opening(class_mask, self.selem)
                elif self.operation == 'closing':
                    class_mask = morphology.closing(class_mask, self.selem)
                # 其他操作...
            
            # 更新分类图像
            if self.operation in ['opening', 'closing']:
                # 对于开运算和闭运算，直接替换
                other_classes_mask = ~(classification_map == class_id)
                filtered_map = np.where(class_mask, class_id, 
                                      np.where(other_classes_mask, filtered_map, class_id))
        
        return filtered_map


class AdaptiveFilter(SpatialFilter):
    """
    自适应滤波器
    
    根据局部图像特征自适应调整滤波参数。
    """
    
    def __init__(self, 
                 window_size: int = 5,
                 threshold_ratio: float = 0.5,
                 min_cluster_size: int = 4,
                 **kwargs):
        """
        初始化自适应滤波器
        
        Parameters:
        -----------
        window_size : int, default=5
            分析窗口大小
        threshold_ratio : float, default=0.5
            阈值比例
        min_cluster_size : int, default=4
            最小聚类大小
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio
        self.min_cluster_size = min_cluster_size
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        应用自适应滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"应用自适应滤波 - 窗口大小: {self.window_size}")
        
        height, width = classification_map.shape
        filtered_map = classification_map.copy()
        half_window = self.window_size // 2
        
        # 滑动窗口处理
        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                # 提取局部窗口
                window = classification_map[
                    i - half_window:i + half_window + 1,
                    j - half_window:j + half_window + 1
                ]
                
                # 分析局部统计
                center_value = classification_map[i, j]
                window_stats = self._analyze_window(window)
                
                # 自适应决策
                if self._should_filter(center_value, window_stats):
                    filtered_map[i, j] = window_stats['dominant_class']
        
        logger.info("自适应滤波完成")
        
        return filtered_map
    
    def _analyze_window(self, window: np.ndarray) -> Dict[str, Any]:
        """分析窗口统计特征"""
        unique_values, counts = np.unique(window, return_counts=True)
        
        # 主导类别
        dominant_idx = np.argmax(counts)
        dominant_class = unique_values[dominant_idx]
        dominant_ratio = counts[dominant_idx] / window.size
        
        # 多样性
        diversity = len(unique_values)
        
        # 熵
        probabilities = counts / window.size
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return {
            'dominant_class': dominant_class,
            'dominant_ratio': dominant_ratio,
            'diversity': diversity,
            'entropy': entropy,
            'unique_classes': unique_values,
            'class_counts': counts
        }
    
    def _should_filter(self, center_value: int, window_stats: Dict[str, Any]) -> bool:
        """判断是否应该滤波"""
        # 如果中心值是主导类别，不滤波
        if center_value == window_stats['dominant_class']:
            return False
        
        # 如果主导类别占比超过阈值，进行滤波
        if window_stats['dominant_ratio'] >= self.threshold_ratio:
            return True
        
        # 如果多样性太高，不滤波（保持边界）
        if window_stats['diversity'] > 3:
            return False
        
        return False


class MarkovRandomFieldFilter(SpatialFilter):
    """
    马尔可夫随机场滤波器
    
    使用MRF模型进行空间上下文感知的分类优化。
    """
    
    def __init__(self, 
                 beta: float = 1.0,
                 iterations: int = 10,
                 convergence_threshold: float = 1e-4,
                 **kwargs):
        """
        初始化MRF滤波器
        
        Parameters:
        -----------
        beta : float, default=1.0
            空间平滑参数
        iterations : int, default=10
            最大迭代次数
        convergence_threshold : float, default=1e-4
            收敛阈值
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.iterations = iterations
        self.convergence_threshold = convergence_threshold
    
    def apply(self, classification_map: np.ndarray, 
             probability_map: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        """
        应用MRF滤波
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
        probability_map : np.ndarray, optional
            类别概率图 (height, width, n_classes)
            
        Returns:
        --------
        filtered_map : np.ndarray
            滤波后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"应用MRF滤波 - beta: {self.beta}, 最大迭代: {self.iterations}")
        
        # 获取类别信息
        unique_classes = np.unique(classification_map)
        n_classes = len(unique_classes)
        height, width = classification_map.shape
        
        # 初始化概率图
        if probability_map is None:
            # 基于分类结果创建硬概率
            probability_map = np.zeros((height, width, n_classes))
            for i, class_id in enumerate(unique_classes):
                probability_map[:, :, i] = (classification_map == class_id).astype(float)
        
        # MRF优化
        optimized_proba = self._mrf_optimization(probability_map, unique_classes)
        
        # 生成最终分类
        final_classification = unique_classes[np.argmax(optimized_proba, axis=2)]
        
        logger.info("MRF滤波完成")
        
        return final_classification.astype(classification_map.dtype)
    
    def _mrf_optimization(self, probability_map: np.ndarray, 
                         unique_classes: np.ndarray) -> np.ndarray:
        """MRF优化"""
        height, width, n_classes = probability_map.shape
        current_proba = probability_map.copy()
        
        # 定义邻域（4连通）
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for iteration in range(self.iterations):
            new_proba = current_proba.copy()
            total_change = 0.0
            
            for i in range(height):
                for j in range(width):
                    # 计算数据项（观测概率）
                    data_term = probability_map[i, j, :]
                    
                    # 计算平滑项（邻域一致性）
                    smooth_term = np.zeros(n_classes)
                    neighbor_count = 0
                    
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_proba = current_proba[ni, nj, :]
                            smooth_term += neighbor_proba
                            neighbor_count += 1
                    
                    if neighbor_count > 0:
                        smooth_term /= neighbor_count
                    
                    # 组合数据项和平滑项
                    combined_energy = data_term + self.beta * smooth_term
                    
                    # 归一化
                    combined_energy = combined_energy / np.sum(combined_energy)
                    
                    # 计算变化
                    change = np.sum(np.abs(combined_energy - current_proba[i, j, :]))
                    total_change += change
                    
                    new_proba[i, j, :] = combined_energy
            
            current_proba = new_proba
            
            # 检查收敛
            avg_change = total_change / (height * width)
            if avg_change < self.convergence_threshold:
                logger.info(f"MRF在第 {iteration + 1} 次迭代后收敛")
                break
        
        return current_proba


class SpatialFilterPipeline:
    """
    空间滤波流水线
    
    组合多个滤波器形成完整的后处理流水线。
    """
    
    def __init__(self, filters: List[SpatialFilter]):
        """
        初始化滤波流水线
        
        Parameters:
        -----------
        filters : list
            滤波器列表
        """
        self.filters = filters
        self.filter_history = []
    
    def apply(self, classification_map: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        应用滤波流水线
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        results : dict
            包含所有中间结果的字典
        """
        logger.info(f"开始应用滤波流水线 - 滤波器数量: {len(self.filters)}")
        
        results = {'original': classification_map.copy()}
        current_map = classification_map.copy()
        
        for i, filter_obj in enumerate(self.filters):
            filter_name = filter_obj.__class__.__name__
            logger.info(f"应用滤波器 {i+1}/{len(self.filters)}: {filter_name}")
            
            # 应用滤波
            filtered_map = filter_obj.apply(current_map, **kwargs)
            
            # 保存结果
            step_name = f"step_{i+1}_{filter_name}"
            results[step_name] = filtered_map.copy()
            
            # 计算变化统计
            changes = np.sum(filtered_map != current_map)
            total_pixels = current_map.size
            change_ratio = changes / total_pixels
            
            # 记录历史
            self.filter_history.append({
                'filter': filter_name,
                'step': i + 1,
                'changes': changes,
                'change_ratio': change_ratio
            })
            
            logger.info(f"{filter_name} 完成 - 改变像素: {changes}/{total_pixels} ({change_ratio:.2%})")
            
            # 更新当前图像
            current_map = filtered_map
        
        results['final'] = current_map
        
        logger.info("滤波流水线处理完成")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        if not self.filter_history:
            return {}
        
        total_changes = sum(step['changes'] for step in self.filter_history)
        total_change_ratio = sum(step['change_ratio'] for step in self.filter_history)
        
        return {
            'total_steps': len(self.filter_history),
            'total_changes': total_changes,
            'average_change_ratio': total_change_ratio / len(self.filter_history),
            'step_details': self.filter_history
        }


def create_wetland_filter_pipeline() -> SpatialFilterPipeline:
    """
    创建针对湿地分类优化的滤波流水线
    
    Returns:
    --------
    pipeline : SpatialFilterPipeline
        湿地优化滤波流水线
    """
    filters = [
        # 1. 中值滤波去除孤立噪声点
        MedianFilter(filter_size=3, iterations=1),
        
        # 2. 形态学开运算去除小的噪声区域
        MorphologicalFilter(operation='opening', structuring_element='disk', size=2),
        
        # 3. 形态学闭运算填补小的空洞
        MorphologicalFilter(operation='closing', structuring_element='disk', size=3),
        
        # 4. 自适应滤波优化边界
        AdaptiveFilter(window_size=5, threshold_ratio=0.6),
        
        # 5. 轻微的高斯平滑
        GaussianFilter(sigma=0.8)
    ]
    
    return SpatialFilterPipeline(filters)


def create_filter(filter_type: str, **kwargs) -> SpatialFilter:
    """
    滤波器工厂函数
    
    Parameters:
    -----------
    filter_type : str
        滤波器类型
    **kwargs : dict
        滤波器参数
        
    Returns:
    --------
    filter_obj : SpatialFilter
        滤波器实例
    """
    filters = {
        'median': MedianFilter,
        'gaussian': GaussianFilter,
        'bilateral': BilateralFilter,
        'morphological': MorphologicalFilter,
        'adaptive': AdaptiveFilter,
        'mrf': MarkovRandomFieldFilter
    }
    
    if filter_type.lower() not in filters:
        available = ', '.join(filters.keys())
        raise ValueError(f"不支持的滤波器类型: {filter_type}. 可用类型: {available}")
    
    return filters[filter_type.lower()](**kwargs)