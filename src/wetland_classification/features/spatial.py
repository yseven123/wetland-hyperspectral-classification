#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
空间特征提取器
Spatial Feature Extractor

提取高光谱数据的空间特征，包括形态学特征、边缘特征、梯度特征等

作者: Wetland Research Team
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import logging

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from sklearn.feature_extraction import image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Some spatial features will be limited.")

try:
    from skimage.feature import peak_local_maxima
    from skimage.segmentation import watershed
    from skimage.morphology import disk, square, remove_small_objects
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available. Some spatial features will be limited.")

from ..config import Config

logger = logging.getLogger(__name__)


class SpatialFeatureExtractor:
    """空间特征提取器
    
    支持的空间特征：
    - 形态学特征：开运算、闭运算、梯度、顶帽、黑帽
    - 边缘特征：Sobel、Canny、Laplacian、Roberts边缘检测
    - 梯度特征：梯度幅值、梯度方向、梯度散度
    - 几何特征：长轴、短轴、偏心率、面积
    - 结构特征：连通组件、边界长度、紧凑度
    - 分水岭特征：分水岭分割的区域特性
    """
    
    def __init__(self, config: Config):
        """初始化空间特征提取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.spatial_config = config.get('features.spatial', {})
        
        # 空间特征参数
        self.morphological = self.spatial_config.get('morphological', True)
        self.edge_detection = self.spatial_config.get('edge_detection', True)
        self.gradient = self.spatial_config.get('gradient', True)
        
        # 结构元素大小
        self.structure_sizes = [3, 5, 7]
        
        logger.info("SpatialFeatureExtractor initialized")
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """提取空间特征
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            
        Returns:
            np.ndarray: 空间特征 (H, W, F)
        """
        logger.info("Extracting spatial features")
        
        height, width, bands = data.shape
        spatial_features = []
        
        # 选择用于空间分析的代表性波段
        selected_bands = self._select_representative_bands(data)
        
        # 1. 形态学特征
        if self.morphological:
            morphological_features = self._extract_morphological_features(data, selected_bands)
            if morphological_features is not None:
                spatial_features.append(morphological_features)
                logger.debug("Extracted morphological features")
        
        # 2. 边缘特征
        if self.edge_detection:
            edge_features = self._extract_edge_features(data, selected_bands)
            if edge_features is not None:
                spatial_features.append(edge_features)
                logger.debug("Extracted edge features")
        
        # 3. 梯度特征
        if self.gradient:
            gradient_features = self._extract_gradient_features(data, selected_bands)
            if gradient_features is not None:
                spatial_features.append(gradient_features)
                logger.debug("Extracted gradient features")
        
        # 4. 结构特征
        structure_features = self._extract_structure_features(data, selected_bands)
        if structure_features is not None:
            spatial_features.append(structure_features)
            logger.debug("Extracted structure features")
        
        # 5. 局部统计特征
        local_stats_features = self._extract_local_statistics(data, selected_bands)
        if local_stats_features is not None:
            spatial_features.append(local_stats_features)
            logger.debug("Extracted local statistics features")
        
        # 合并所有空间特征
        if spatial_features:
            combined_features = np.concatenate(spatial_features, axis=2)
            logger.info(f"Extracted spatial features: {data.shape} -> {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No spatial features extracted")
            return np.zeros((height, width, 1))
    
    def _select_representative_bands(self, data: np.ndarray) -> List[int]:
        """选择代表性波段进行空间分析
        
        Args:
            data: 输入数据
            
        Returns:
            List[int]: 选择的波段索引
        """
        bands = data.shape[2]
        
        if bands <= 10:
            return list(range(bands))
        elif bands <= 50:
            # 选择均匀分布的波段
            return list(range(0, bands, bands // 10))
        else:
            # 对于高光谱数据，选择关键波段
            # 通常包括可见光、近红外、短波红外的代表性波段
            key_indices = [
                0,  # 蓝色
                bands // 8,  # 绿色
                bands // 4,  # 红色
                bands // 2,  # 近红外
                3 * bands // 4,  # 短波红外1
                bands - 1  # 短波红外2
            ]
            return [idx for idx in key_indices if idx < bands]
    
    def _extract_morphological_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取形态学特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 形态学特征
        """
        height, width, bands = data.shape
        
        # 形态学操作类型
        morphological_ops = ['opening', 'closing', 'gradient', 'tophat', 'blackhat']
        
        num_features = len(selected_bands) * len(morphological_ops) * len(self.structure_sizes)
        morphological_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            
            # 归一化到[0, 1]
            normalized_band = self._normalize_band(band_data)
            
            for size in self.structure_sizes:
                # 创建结构元素
                if HAS_SKIMAGE:
                    structure = disk(size // 2)
                else:
                    structure = np.ones((size, size))
                
                for op_name in morphological_ops:
                    try:
                        if op_name == 'opening':
                            result = ndimage.binary_opening(
                                normalized_band > 0.5, structure=structure
                            ).astype(np.float32)
                        elif op_name == 'closing':
                            result = ndimage.binary_closing(
                                normalized_band > 0.5, structure=structure
                            ).astype(np.float32)
                        elif op_name == 'gradient':
                            dilated = ndimage.binary_dilation(
                                normalized_band > 0.5, structure=structure
                            )
                            eroded = ndimage.binary_erosion(
                                normalized_band > 0.5, structure=structure
                            )
                            result = (dilated - eroded).astype(np.float32)
                        elif op_name == 'tophat':
                            opened = ndimage.binary_opening(
                                normalized_band > 0.5, structure=structure
                            )
                            result = ((normalized_band > 0.5) - opened).astype(np.float32)
                        elif op_name == 'blackhat':
                            closed = ndimage.binary_closing(
                                normalized_band > 0.5, structure=structure
                            )
                            result = (closed - (normalized_band > 0.5)).astype(np.float32)
                        
                        morphological_features[:, :, feature_idx] = result
                        feature_idx += 1
                    
                    except Exception as e:
                        logger.warning(f"Morphological operation {op_name} failed: {e}")
                        feature_idx += 1
                        continue
        
        return morphological_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_edge_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取边缘特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 边缘特征
        """
        height, width, bands = data.shape
        
        edge_operators = ['sobel', 'laplacian', 'roberts', 'prewitt']
        
        num_features = len(selected_bands) * len(edge_operators)
        edge_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx].astype(np.float32)
            
            for operator in edge_operators:
                try:
                    if operator == 'sobel':
                        # Sobel边缘检测
                        sobel_x = ndimage.sobel(band_data, axis=1)
                        sobel_y = ndimage.sobel(band_data, axis=0)
                        result = np.sqrt(sobel_x**2 + sobel_y**2)
                    
                    elif operator == 'laplacian':
                        # Laplacian边缘检测
                        result = ndimage.laplace(band_data)
                    
                    elif operator == 'roberts':
                        # Roberts交叉梯度
                        roberts_x = ndimage.convolve(band_data, np.array([[1, 0], [0, -1]]))
                        roberts_y = ndimage.convolve(band_data, np.array([[0, 1], [-1, 0]]))
                        result = np.sqrt(roberts_x**2 + roberts_y**2)
                    
                    elif operator == 'prewitt':
                        # Prewitt边缘检测
                        prewitt_x = ndimage.convolve(band_data, 
                                                   np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
                        prewitt_y = ndimage.convolve(band_data, 
                                                   np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
                        result = np.sqrt(prewitt_x**2 + prewitt_y**2)
                    
                    edge_features[:, :, feature_idx] = result
                    feature_idx += 1
                
                except Exception as e:
                    logger.warning(f"Edge detection {operator} failed: {e}")
                    feature_idx += 1
                    continue
        
        return edge_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_gradient_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取梯度特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 梯度特征
        """
        height, width, bands = data.shape
        
        # 梯度特征：幅值、方向、散度、旋度
        gradient_features_per_band = 4
        num_features = len(selected_bands) * gradient_features_per_band
        gradient_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx].astype(np.float32)
            
            try:
                # 计算梯度
                grad_y, grad_x = np.gradient(band_data)
                
                # 梯度幅值
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                gradient_features[:, :, feature_idx] = gradient_magnitude
                feature_idx += 1
                
                # 梯度方向
                gradient_direction = np.arctan2(grad_y, grad_x)
                gradient_features[:, :, feature_idx] = gradient_direction
                feature_idx += 1
                
                # 梯度散度
                grad_xx = np.gradient(grad_x, axis=1)
                grad_yy = np.gradient(grad_y, axis=0)
                divergence = grad_xx + grad_yy
                gradient_features[:, :, feature_idx] = divergence
                feature_idx += 1
                
                # 梯度旋度 (curl的z分量)
                grad_xy = np.gradient(grad_x, axis=0)
                grad_yx = np.gradient(grad_y, axis=1)
                curl_z = grad_yx - grad_xy
                gradient_features[:, :, feature_idx] = curl_z
                feature_idx += 1
            
            except Exception as e:
                logger.warning(f"Gradient calculation failed for band {band_idx}: {e}")
                feature_idx += gradient_features_per_band
                continue
        
        return gradient_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_structure_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取结构特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 结构特征
        """
        height, width, bands = data.shape
        
        # 结构特征：局部方差、局部对比度、局部均匀性
        structure_features_per_band = 6
        num_features = len(selected_bands) * structure_features_per_band
        structure_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        window_sizes = [3, 5, 7]
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx].astype(np.float32)
            
            for window_size in window_sizes:
                try:
                    # 局部方差
                    local_variance = ndimage.generic_filter(
                        band_data, np.var, size=window_size
                    )
                    structure_features[:, :, feature_idx] = local_variance
                    feature_idx += 1
                    
                    # 局部对比度
                    local_contrast = ndimage.generic_filter(
                        band_data, 
                        lambda x: np.max(x) - np.min(x), 
                        size=window_size
                    )
                    structure_features[:, :, feature_idx] = local_contrast
                    feature_idx += 1
                
                except Exception as e:
                    logger.warning(f"Structure feature calculation failed: {e}")
                    feature_idx += 2
                    continue
        
        return structure_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_local_statistics(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取局部统计特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 局部统计特征
        """
        height, width, bands = data.shape
        
        # 局部统计函数
        local_stats = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'median': np.median,
            'range': lambda x: np.max(x) - np.min(x)
        }
        
        window_size = 5
        num_features = len(selected_bands) * len(local_stats)
        local_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx].astype(np.float32)
            
            for stat_name, stat_func in local_stats.items():
                try:
                    local_stat = ndimage.generic_filter(
                        band_data, stat_func, size=window_size
                    )
                    local_features[:, :, feature_idx] = local_stat
                    feature_idx += 1
                
                except Exception as e:
                    logger.warning(f"Local statistics {stat_name} failed: {e}")
                    feature_idx += 1
                    continue
        
        return local_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_canny_edges(self, image: np.ndarray, low_threshold: float = 0.1, 
                           high_threshold: float = 0.2) -> np.ndarray:
        """提取Canny边缘
        
        Args:
            image: 输入图像
            low_threshold: 低阈值
            high_threshold: 高阈值
            
        Returns:
            np.ndarray: Canny边缘
        """
        if HAS_CV2:
            # 使用OpenCV的Canny边缘检测
            normalized = self._normalize_band(image)
            edges = cv2.Canny(
                (normalized * 255).astype(np.uint8),
                int(low_threshold * 255),
                int(high_threshold * 255)
            )
            return edges.astype(np.float32) / 255.0
        else:
            # 简化的边缘检测
            sobel_x = ndimage.sobel(image, axis=1)
            sobel_y = ndimage.sobel(image, axis=0)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 应用双阈值
            strong_edges = edge_magnitude > high_threshold
            weak_edges = (edge_magnitude >= low_threshold) & (edge_magnitude <= high_threshold)
            
            # 简化的连接过程
            edges = strong_edges.astype(np.float32)
            edges[weak_edges] = 0.5  # 弱边缘标记为0.5
            
            return edges
    
    def _extract_corner_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取角点特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 角点特征
        """
        if not HAS_CV2:
            logger.warning("OpenCV not available for corner detection")
            return None
        
        height, width, bands = data.shape
        num_features = len(selected_bands) * 2  # Harris和Shi-Tomasi角点
        corner_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            normalized_band = self._normalize_band(band_data)
            image_uint8 = (normalized_band * 255).astype(np.uint8)
            
            try:
                # Harris角点检测
                harris_corners = cv2.cornerHarris(image_uint8, 2, 3, 0.04)
                corner_features[:, :, feature_idx] = harris_corners
                feature_idx += 1
                
                # Shi-Tomasi角点检测
                corners = cv2.goodFeaturesToTrack(
                    image_uint8, maxCorners=1000, qualityLevel=0.01, minDistance=10
                )
                
                # 创建角点密度图
                corner_density = np.zeros((height, width))
                if corners is not None:
                    for corner in corners:
                        x, y = corner.ravel()
                        if 0 <= int(y) < height and 0 <= int(x) < width:
                            corner_density[int(y), int(x)] = 1
                
                # 应用高斯滤波来创建密度场
                corner_density = ndimage.gaussian_filter(corner_density, sigma=2)
                corner_features[:, :, feature_idx] = corner_density
                feature_idx += 1
            
            except Exception as e:
                logger.warning(f"Corner detection failed for band {band_idx}: {e}")
                feature_idx += 2
                continue
        
        return corner_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_watershed_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取分水岭分割特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: 分水岭特征
        """
        if not HAS_SKIMAGE:
            logger.warning("scikit-image not available for watershed features")
            return None
        
        height, width, bands = data.shape
        num_features = len(selected_bands) * 3  # 区域数量、平均区域大小、边界密度
        watershed_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            normalized_band = self._normalize_band(band_data)
            
            try:
                # 计算梯度
                gradient = ndimage.sobel(normalized_band)
                
                # 寻找局部最小值作为标记
                local_minima = peak_local_maxima(-normalized_band, min_distance=5)
                markers = np.zeros_like(normalized_band, dtype=int)
                for i, (y, x) in enumerate(zip(local_minima[0], local_minima[1])):
                    markers[y, x] = i + 1
                
                # 分水岭分割
                labels = watershed(gradient, markers)
                
                # 计算区域特征
                num_regions = len(np.unique(labels)) - 1  # 排除背景
                region_sizes = np.bincount(labels.ravel())[1:]  # 排除背景
                
                # 区域数量密度
                region_density = np.full((height, width), num_regions / (height * width))
                watershed_features[:, :, feature_idx] = region_density
                feature_idx += 1
                
                # 平均区域大小
                avg_region_size = np.mean(region_sizes) if len(region_sizes) > 0 else 0
                avg_size_map = np.full((height, width), avg_region_size)
                watershed_features[:, :, feature_idx] = avg_size_map
                feature_idx += 1
                
                # 边界密度
                boundaries = ndimage.laplace(labels.astype(np.float32)) != 0
                boundary_density = ndimage.uniform_filter(boundaries.astype(np.float32), size=5)
                watershed_features[:, :, feature_idx] = boundary_density
                feature_idx += 1
            
            except Exception as e:
                logger.warning(f"Watershed features failed for band {band_idx}: {e}")
                feature_idx += 3
                continue
        
        return watershed_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    # 辅助函数
    def _normalize_band(self, band_data: np.ndarray) -> np.ndarray:
        """归一化波段数据到[0, 1]
        
        Args:
            band_data: 波段数据
            
        Returns:
            np.ndarray: 归一化后的数据
        """
        min_val, max_val = np.min(band_data), np.max(band_data)
        if max_val > min_val:
            return (band_data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(band_data)
    
    def _create_structure_element(self, shape: str, size: int) -> np.ndarray:
        """创建结构元素
        
        Args:
            shape: 形状 ('disk', 'square', 'cross')
            size: 大小
            
        Returns:
            np.ndarray: 结构元素
        """
        if shape == 'disk' and HAS_SKIMAGE:
            return disk(size // 2)
        elif shape == 'square':
            return np.ones((size, size))
        elif shape == 'cross':
            element = np.zeros((size, size))
            center = size // 2
            element[center, :] = 1
            element[:, center] = 1
            return element
        else:
            return np.ones((size, size))
    
    def get_feature_names(self) -> List[str]:
        """获取空间特征名称列表
        
        Returns:
            List[str]: 特征名称列表
        """
        feature_names = []
        
        if self.morphological:
            morphological_ops = ['opening', 'closing', 'gradient', 'tophat', 'blackhat']
            for size in self.structure_sizes:
                for op in morphological_ops:
                    feature_names.append(f'Morph_{op}_size{size}')
        
        if self.edge_detection:
            edge_operators = ['sobel', 'laplacian', 'roberts', 'prewitt']
            feature_names.extend([f'Edge_{op}' for op in edge_operators])
        
        if self.gradient:
            gradient_features = ['magnitude', 'direction', 'divergence', 'curl_z']
            feature_names.extend([f'Grad_{feat}' for feat in gradient_features])
        
        # 结构特征
        structure_features = ['local_variance', 'local_contrast']
        for window_size in [3, 5, 7]:
            for feat in structure_features:
                feature_names.append(f'Struct_{feat}_w{window_size}')
        
        # 局部统计特征
        local_stats = ['mean', 'std', 'min', 'max', 'median', 'range']
        feature_names.extend([f'Local_{stat}' for stat in local_stats])
        
        return feature_names
    
    def validate_spatial_features(self, original_data: np.ndarray,
                                spatial_features: np.ndarray) -> Dict[str, Any]:
        """验证空间特征提取结果
        
        Args:
            original_data: 原始数据
            spatial_features: 空间特征
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'input_shape': original_data.shape,
            'output_shape': spatial_features.shape,
            'num_spatial_features': spatial_features.shape[2],
            'data_quality': {},
            'issues': []
        }
        
        # 检查数据质量
        nan_ratio = np.sum(np.isnan(spatial_features)) / spatial_features.size
        inf_ratio = np.sum(np.isinf(spatial_features)) / spatial_features.size
        
        validation_results['data_quality'] = {
            'nan_ratio': float(nan_ratio),
            'inf_ratio': float(inf_ratio),
            'value_range': (float(np.nanmin(spatial_features)), float(np.nanmax(spatial_features))),
            'mean_value': float(np.nanmean(spatial_features)),
            'std_value': float(np.nanstd(spatial_features))
        }
        
        # 检查潜在问题
        if nan_ratio > 0.01:
            validation_results['issues'].append(f"High NaN ratio: {nan_ratio:.2%}")
        
        if inf_ratio > 0:
            validation_results['issues'].append(f"Infinite values: {inf_ratio:.2%}")
        
        if spatial_features.shape[2] == 0:
            validation_results['issues'].append("No spatial features extracted")
        
        # 检查特征多样性
        feature_variance = np.var(spatial_features, axis=(0, 1))
        low_variance_count = np.sum(feature_variance < 1e-6)
        if low_variance_count > spatial_features.shape[2] * 0.3:
            validation_results['issues'].append("Many features have very low variance")
        
        # 检查空间连续性
        spatial_continuity = self._check_spatial_continuity(spatial_features)
        if spatial_continuity < 0.7:
            validation_results['issues'].append("Low spatial continuity in features")
        
        # 评估特征质量
        if len(validation_results['issues']) == 0:
            validation_results['quality'] = 'Good'
        elif len(validation_results['issues']) <= 2:
            validation_results['quality'] = 'Acceptable'
        else:
            validation_results['quality'] = 'Poor'
        
        validation_results['spatial_continuity'] = float(spatial_continuity)
        
        return validation_results
    
    def _check_spatial_continuity(self, features: np.ndarray) -> float:
        """检查空间特征的连续性
        
        Args:
            features: 空间特征
            
        Returns:
            float: 连续性指标 (0-1)
        """
        if features.shape[2] == 0:
            return 0.0
        
        # 计算相邻像素间的相关性
        correlations = []
        
        for f in range(min(features.shape[2], 10)):  # 检查前10个特征
            feature_map = features[:, :, f]
            
            # 计算水平相邻像素的相关性
            horizontal_corr = np.corrcoef(
                feature_map[:, :-1].flatten(),
                feature_map[:, 1:].flatten()
            )[0, 1]
            
            # 计算垂直相邻像素的相关性
            vertical_corr = np.corrcoef(
                feature_map[:-1, :].flatten(),
                feature_map[1:, :].flatten()
            )[0, 1]
            
            if not np.isnan(horizontal_corr):
                correlations.append(horizontal_corr)
            if not np.isnan(vertical_corr):
                correlations.append(vertical_corr)
        
        return np.mean(correlations) if correlations else 0.0