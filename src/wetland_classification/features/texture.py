#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纹理特征提取器
Texture Feature Extractor

提取高光谱数据的纹理特征，用于识别地物的空间结构特征

作者: Wetland Research Team
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
from scipy import ndimage
from sklearn.feature_extraction import image

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.feature import greycoprops  # 备用导入
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available. Some texture features will be limited.")

from ..config import Config

logger = logging.getLogger(__name__)


class TextureFeatureExtractor:
    """纹理特征提取器
    
    支持的纹理特征：
    - GLCM (Gray Level Co-occurrence Matrix): 对比度、相关性、能量、同质性等
    - LBP (Local Binary Pattern): 局部二值模式
    - GLRLM (Gray Level Run Length Matrix): 灰度游程矩阵
    - Gabor滤波器: 不同尺度和方向的纹理特征
    - 统计纹理: 方差、偏度、峰度等
    - Haralick纹理特征: 基于GLCM的14个Haralick特征
    """
    
    def __init__(self, config: Config):
        """初始化纹理特征提取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.texture_config = config.get('features.texture', {})
        
        # 纹理计算参数
        self.window_size = self.texture_config.get('window_size', [7, 7])
        self.distances = self.texture_config.get('distances', [1, 2])
        self.angles = self.texture_config.get('angles', [0, 45, 90, 135])
        self.methods = self.texture_config.get('methods', ['GLCM', 'LBP'])
        
        # 转换角度为弧度
        self.angles_rad = [np.radians(angle) for angle in self.angles]
        
        logger.info("TextureFeatureExtractor initialized")
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """提取纹理特征
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            
        Returns:
            np.ndarray: 纹理特征 (H, W, F)
        """
        logger.info("Extracting texture features")
        
        height, width, bands = data.shape
        texture_features = []
        
        # 选择用于纹理分析的波段（可以是所有波段或选择的代表性波段）
        selected_bands = self._select_bands_for_texture(data)
        
        for method in self.methods:
            try:
                if method == 'GLCM':
                    glcm_features = self._extract_glcm_features(data, selected_bands)
                    if glcm_features is not None:
                        texture_features.append(glcm_features)
                        logger.debug("Extracted GLCM features")
                
                elif method == 'LBP':
                    lbp_features = self._extract_lbp_features(data, selected_bands)
                    if lbp_features is not None:
                        texture_features.append(lbp_features)
                        logger.debug("Extracted LBP features")
                
                elif method == 'GLRLM':
                    glrlm_features = self._extract_glrlm_features(data, selected_bands)
                    if glrlm_features is not None:
                        texture_features.append(glrlm_features)
                        logger.debug("Extracted GLRLM features")
                
                elif method == 'Gabor':
                    gabor_features = self._extract_gabor_features(data, selected_bands)
                    if gabor_features is not None:
                        texture_features.append(gabor_features)
                        logger.debug("Extracted Gabor features")
                
                elif method == 'Statistical':
                    statistical_features = self._extract_statistical_features(data, selected_bands)
                    if statistical_features is not None:
                        texture_features.append(statistical_features)
                        logger.debug("Extracted statistical texture features")
                
            except Exception as e:
                logger.warning(f"Failed to extract {method} features: {e}")
                continue
        
        if texture_features:
            combined_features = np.concatenate(texture_features, axis=2)
            logger.info(f"Extracted texture features: {data.shape} -> {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No texture features extracted")
            return np.zeros((height, width, 1))
    
    def _select_bands_for_texture(self, data: np.ndarray) -> List[int]:
        """选择用于纹理分析的波段
        
        Args:
            data: 输入数据
            
        Returns:
            List[int]: 选择的波段索引
        """
        bands = data.shape[2]
        
        # 如果波段数太多，选择代表性波段以提高计算效率
        if bands > 50:
            # 选择均匀分布的波段
            selected_indices = np.linspace(0, bands-1, 20, dtype=int)
            return selected_indices.tolist()
        elif bands > 20:
            # 选择部分波段
            selected_indices = np.linspace(0, bands-1, bands//3, dtype=int)
            return selected_indices.tolist()
        else:
            # 使用所有波段
            return list(range(bands))
    
    def _extract_glcm_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取GLCM (灰度共生矩阵) 纹理特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: GLCM特征
        """
        if not HAS_SKIMAGE:
            logger.warning("scikit-image not available for GLCM features")
            return self._extract_simple_glcm_features(data, selected_bands)
        
        height, width, bands = data.shape
        
        # GLCM属性
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 
                     'correlation', 'ASM']  # ASM = Angular Second Moment = energy^2
        
        num_features = len(selected_bands) * len(glcm_props) * len(self.distances) * len(self.angles)
        glcm_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            
            # 量化到合适的灰度级别以提高计算效率
            quantized_band = self._quantize_image(band_data, levels=64)
            
            for distance in self.distances:
                for angle in self.angles_rad:
                    try:
                        # 计算GLCM
                        glcm = graycomatrix(
                            quantized_band, 
                            distances=[distance], 
                            angles=[angle],
                            levels=64,
                            symmetric=True, 
                            normed=True
                        )
                        
                        # 计算GLCM属性
                        for prop in glcm_props:
                            if prop == 'ASM':
                                # ASM = energy^2
                                energy_val = graycoprops(glcm, 'energy')[0, 0]
                                prop_value = energy_val ** 2
                            else:
                                try:
                                    prop_value = graycoprops(glcm, prop)[0, 0]
                                except:
                                    # 如果某个属性不可用，尝试其他方法
                                    prop_value = 0.0
                            
                            # 创建特征图（每个像素周围窗口的GLCM特征）
                            feature_map = self._compute_local_glcm_feature(
                                quantized_band, prop, distance, angle
                            )
                            
                            glcm_features[:, :, feature_idx] = feature_map
                            feature_idx += 1
                    
                    except Exception as e:
                        logger.warning(f"GLCM calculation failed for band {band_idx}, "
                                     f"distance {distance}, angle {np.degrees(angle)}: {e}")
                        feature_idx += len(glcm_props)
                        continue
        
        return glcm_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_simple_glcm_features(self, data: np.ndarray, selected_bands: List[int]) -> np.ndarray:
        """简化的GLCM特征提取（不依赖scikit-image）
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            np.ndarray: 简化的纹理特征
        """
        height, width, bands = data.shape
        num_features = len(selected_bands) * 4  # 4个简化特征
        
        simple_features = np.zeros((height, width, num_features))
        feature_idx = 0
        
        window_size = self.window_size[0]
        half_window = window_size // 2
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            
            # 对比度 (方差)
            contrast = ndimage.generic_filter(band_data, np.var, size=window_size)
            simple_features[:, :, feature_idx] = contrast
            feature_idx += 1
            
            # 均匀性 (标准差的倒数)
            homogeneity = ndimage.generic_filter(
                band_data, 
                lambda x: 1.0 / (1.0 + np.std(x)), 
                size=window_size
            )
            simple_features[:, :, feature_idx] = homogeneity
            feature_idx += 1
            
            # 能量 (均值的平方)
            energy = ndimage.generic_filter(
                band_data, 
                lambda x: np.mean(x) ** 2, 
                size=window_size
            )
            simple_features[:, :, feature_idx] = energy
            feature_idx += 1
            
            # 熵 (简化版本)
            entropy = ndimage.generic_filter(
                band_data, 
                self._local_entropy, 
                size=window_size
            )
            simple_features[:, :, feature_idx] = entropy
            feature_idx += 1
        
        return simple_features
    
    def _extract_lbp_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取LBP (局部二值模式) 特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: LBP特征
        """
        if not HAS_SKIMAGE:
            logger.warning("scikit-image not available for LBP features")
            return self._extract_simple_lbp_features(data, selected_bands)
        
        height, width, bands = data.shape
        
        # LBP参数
        radius = 3
        n_points = 8 * radius
        
        num_features = len(selected_bands) * 2  # LBP值和uniform LBP
        lbp_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            
            # 量化图像
            quantized_band = self._quantize_image(band_data, levels=256)
            
            try:
                # 计算LBP
                lbp = local_binary_pattern(quantized_band, n_points, radius, method='uniform')
                lbp_features[:, :, feature_idx] = lbp
                feature_idx += 1
                
                # 计算旋转不变的LBP
                lbp_ri = local_binary_pattern(quantized_band, n_points, radius, method='ror')
                lbp_features[:, :, feature_idx] = lbp_ri
                feature_idx += 1
                
            except Exception as e:
                logger.warning(f"LBP calculation failed for band {band_idx}: {e}")
                feature_idx += 2
                continue
        
        return lbp_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_simple_lbp_features(self, data: np.ndarray, selected_bands: List[int]) -> np.ndarray:
        """简化的LBP特征提取
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            np.ndarray: 简化的LBP特征
        """
        height, width, bands = data.shape
        num_features = len(selected_bands)
        
        lbp_features = np.zeros((height, width, num_features))
        
        for i, band_idx in enumerate(selected_bands):
            band_data = data[:, :, band_idx]
            
            # 简化的局部二值模式
            simple_lbp = self._compute_simple_lbp(band_data)
            lbp_features[:, :, i] = simple_lbp
        
        return lbp_features
    
    def _extract_glrlm_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取GLRLM (灰度游程长度矩阵) 特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: GLRLM特征
        """
        height, width, bands = data.shape
        
        # GLRLM属性
        glrlm_props = ['SRE', 'LRE', 'GLN', 'RLN', 'RP']  # 简化的GLRLM属性
        
        num_features = len(selected_bands) * len(glrlm_props) * len(self.angles)
        glrlm_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        window_size = self.window_size[0]
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            quantized_band = self._quantize_image(band_data, levels=32)
            
            for angle in self.angles:
                for prop in glrlm_props:
                    try:
                        # 计算局部GLRLM特征
                        feature_map = self._compute_local_glrlm_feature(
                            quantized_band, prop, angle, window_size
                        )
                        glrlm_features[:, :, feature_idx] = feature_map
                        feature_idx += 1
                    
                    except Exception as e:
                        logger.warning(f"GLRLM calculation failed: {e}")
                        feature_idx += 1
                        continue
        
        return glrlm_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_gabor_features(self, data: np.ndarray, selected_bands: List[int]) -> Optional[np.ndarray]:
        """提取Gabor滤波器特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            Optional[np.ndarray]: Gabor特征
        """
        height, width, bands = data.shape
        
        # Gabor参数
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, 45, 90, 135]
        
        num_features = len(selected_bands) * len(frequencies) * len(orientations) * 2  # 实部和虚部
        gabor_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx].astype(np.float32)
            
            for freq in frequencies:
                for orientation in orientations:
                    try:
                        # 创建Gabor滤波器
                        gabor_real, gabor_imag = self._create_gabor_filter(
                            band_data.shape, freq, np.radians(orientation)
                        )
                        
                        # 应用滤波器
                        filtered_real = ndimage.convolve(band_data, gabor_real)
                        filtered_imag = ndimage.convolve(band_data, gabor_imag)
                        
                        gabor_features[:, :, feature_idx] = filtered_real
                        gabor_features[:, :, feature_idx + 1] = filtered_imag
                        feature_idx += 2
                    
                    except Exception as e:
                        logger.warning(f"Gabor filtering failed: {e}")
                        feature_idx += 2
                        continue
        
        return gabor_features[:, :, :feature_idx] if feature_idx > 0 else None
    
    def _extract_statistical_features(self, data: np.ndarray, selected_bands: List[int]) -> np.ndarray:
        """提取统计纹理特征
        
        Args:
            data: 输入数据
            selected_bands: 选择的波段
            
        Returns:
            np.ndarray: 统计纹理特征
        """
        height, width, bands = data.shape
        window_size = self.window_size[0]
        
        # 统计特征
        statistical_funcs = {
            'variance': np.var,
            'skewness': lambda x: self._compute_skewness(x),
            'kurtosis': lambda x: self._compute_kurtosis(x),
            'range': lambda x: np.max(x) - np.min(x),
            'mean_abs_deviation': lambda x: np.mean(np.abs(x - np.mean(x)))
        }
        
        num_features = len(selected_bands) * len(statistical_funcs)
        statistical_features = np.zeros((height, width, num_features))
        
        feature_idx = 0
        
        for band_idx in selected_bands:
            band_data = data[:, :, band_idx]
            
            for feat_name, feat_func in statistical_funcs.items():
                try:
                    feature_map = ndimage.generic_filter(
                        band_data, feat_func, size=window_size
                    )
                    statistical_features[:, :, feature_idx] = feature_map
                    feature_idx += 1
                except Exception as e:
                    logger.warning(f"Statistical feature {feat_name} failed: {e}")
                    feature_idx += 1
                    continue
        
        return statistical_features
    
    # 辅助函数
    def _quantize_image(self, image: np.ndarray, levels: int = 64) -> np.ndarray:
        """量化图像到指定的灰度级别
        
        Args:
            image: 输入图像
            levels: 灰度级别数
            
        Returns:
            np.ndarray: 量化后的图像
        """
        # 归一化到[0, 1]
        img_min, img_max = np.min(image), np.max(image)
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(image)
        
        # 量化到指定级别
        quantized = (normalized * (levels - 1)).astype(np.uint8)
        return quantized
    
    def _compute_local_glcm_feature(self, image: np.ndarray, property_name: str, 
                                  distance: int, angle: float) -> np.ndarray:
        """计算局部GLCM特征
        
        Args:
            image: 输入图像
            property_name: GLCM属性名
            distance: 距离
            angle: 角度
            
        Returns:
            np.ndarray: 特征图
        """
        height, width = image.shape
        window_size = self.window_size[0]
        half_window = window_size // 2
        
        feature_map = np.zeros((height, width))
        
        # 计算偏移量
        dx = int(distance * np.cos(angle))
        dy = int(distance * np.sin(angle))
        
        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                # 提取窗口
                window = image[i-half_window:i+half_window+1, 
                             j-half_window:j+half_window+1]
                
                try:
                    # 计算GLCM
                    glcm = graycomatrix(
                        window, distances=[distance], angles=[angle],
                        levels=64, symmetric=True, normed=True
                    )
                    
                    # 计算属性
                    if property_name == 'ASM':
                        energy_val = graycoprops(glcm, 'energy')[0, 0]
                        feature_value = energy_val ** 2
                    else:
                        feature_value = graycoprops(glcm, property_name)[0, 0]
                    
                    feature_map[i, j] = feature_value
                    
                except:
                    feature_map[i, j] = 0.0
        
        return feature_map
    
    def _compute_simple_lbp(self, image: np.ndarray) -> np.ndarray:
        """计算简化的LBP
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: LBP特征图
        """
        height, width = image.shape
        lbp = np.zeros((height, width))
        
        # 8邻域偏移
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
                  (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = image[i, j]
                lbp_value = 0
                
                for k, (di, dj) in enumerate(offsets):
                    neighbor = image[i + di, j + dj]
                    if neighbor >= center:
                        lbp_value += 2 ** k
                
                lbp[i, j] = lbp_value
        
        return lbp
    
    def _compute_local_glrlm_feature(self, image: np.ndarray, property_name: str,
                                   angle: int, window_size: int) -> np.ndarray:
        """计算局部GLRLM特征
        
        Args:
            image: 输入图像
            property_name: GLRLM属性名
            angle: 角度
            window_size: 窗口大小
            
        Returns:
            np.ndarray: 特征图
        """
        height, width = image.shape
        half_window = window_size // 2
        feature_map = np.zeros((height, width))
        
        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                window = image[i-half_window:i+half_window+1, 
                             j-half_window:j+half_window+1]
                
                try:
                    # 简化的GLRLM计算
                    glrlm_value = self._compute_simple_glrlm(window, angle, property_name)
                    feature_map[i, j] = glrlm_value
                except:
                    feature_map[i, j] = 0.0
        
        return feature_map
    
    def _compute_simple_glrlm(self, window: np.ndarray, angle: int, property_name: str) -> float:
        """计算简化的GLRLM特征
        
        Args:
            window: 图像窗口
            angle: 角度
            property_name: 属性名
            
        Returns:
            float: GLRLM特征值
        """
        # 简化实现：计算指定方向的游程长度统计
        if angle == 0:  # 水平方向
            runs = []
            for row in window:
                run_lengths = self._get_run_lengths(row)
                runs.extend(run_lengths)
        elif angle == 90:  # 垂直方向
            runs = []
            for col in range(window.shape[1]):
                column = window[:, col]
                run_lengths = self._get_run_lengths(column)
                runs.extend(run_lengths)
        else:
            # 对角线方向（简化）
            runs = []
            for diag in range(-window.shape[0]+1, window.shape[1]):
                diagonal = np.diagonal(window, offset=diag)
                if len(diagonal) > 1:
                    run_lengths = self._get_run_lengths(diagonal)
                    runs.extend(run_lengths)
        
        if not runs:
            return 0.0
        
        runs = np.array(runs)
        
        # 计算不同的GLRLM属性
        if property_name == 'SRE':  # Short Run Emphasis
            return np.sum(1.0 / (runs ** 2)) / len(runs) if len(runs) > 0 else 0.0
        elif property_name == 'LRE':  # Long Run Emphasis
            return np.sum(runs ** 2) / len(runs) if len(runs) > 0 else 0.0
        elif property_name == 'GLN':  # Gray Level Nonuniformity
            return np.var(runs)
        elif property_name == 'RLN':  # Run Length Nonuniformity
            return len(np.unique(runs))
        elif property_name == 'RP':  # Run Percentage
            return len(runs) / window.size
        else:
            return np.mean(runs)
    
    def _get_run_lengths(self, sequence: np.ndarray) -> List[int]:
        """获取序列中的游程长度
        
        Args:
            sequence: 输入序列
            
        Returns:
            List[int]: 游程长度列表
        """
        if len(sequence) == 0:
            return []
        
        runs = []
        current_value = sequence[0]
        current_length = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_value:
                current_length += 1
            else:
                runs.append(current_length)
                current_value = sequence[i]
                current_length = 1
        
        runs.append(current_length)
        return runs
    
    def _create_gabor_filter(self, shape: Tuple[int, int], frequency: float, 
                           orientation: float) -> Tuple[np.ndarray, np.ndarray]:
        """创建Gabor滤波器
        
        Args:
            shape: 滤波器形状
            frequency: 频率
            orientation: 方向
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (实部滤波器, 虚部滤波器)
        """
        sigma_x = sigma_y = 1.0 / (2 * np.pi * frequency)
        kernel_size = min(15, min(shape) // 4)  # 限制核大小
        
        y, x = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        
        # 旋转坐标
        x_rot = x * np.cos(orientation) + y * np.sin(orientation)
        y_rot = -x * np.sin(orientation) + y * np.cos(orientation)
        
        # Gabor函数
        gaussian = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
        sinusoid_real = np.cos(2 * np.pi * frequency * x_rot)
        sinusoid_imag = np.sin(2 * np.pi * frequency * x_rot)
        
        gabor_real = gaussian * sinusoid_real
        gabor_imag = gaussian * sinusoid_imag
        
        return gabor_real, gabor_imag
    
    def _local_entropy(self, values: np.ndarray) -> float:
        """计算局部熵
        
        Args:
            values: 输入值
            
        Returns:
            float: 熵值
        """
        if len(values) == 0:
            return 0.0
        
        # 量化值
        quantized = np.round(values * 10).astype(int)
        
        # 计算直方图
        unique_values, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / len(values)
        
        # 计算熵
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _compute_skewness(self, values: np.ndarray) -> float:
        """计算偏度
        
        Args:
            values: 输入值
            
        Returns:
            float: 偏度
        """
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((values - mean) / std) ** 3)
        return skewness
    
    def _compute_kurtosis(self, values: np.ndarray) -> float:
        """计算峰度
        
        Args:
            values: 输入值
            
        Returns:
            float: 峰度
        """
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3
        return kurtosis
    
    def validate_texture_features(self, original_data: np.ndarray,
                                texture_features: np.ndarray) -> Dict[str, Any]:
        """验证纹理特征提取结果
        
        Args:
            original_data: 原始数据
            texture_features: 纹理特征
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'input_shape': original_data.shape,
            'output_shape': texture_features.shape,
            'num_texture_features': texture_features.shape[2],
            'data_quality': {},
            'issues': []
        }
        
        # 检查数据质量
        nan_ratio = np.sum(np.isnan(texture_features)) / texture_features.size
        inf_ratio = np.sum(np.isinf(texture_features)) / texture_features.size
        
        validation_results['data_quality'] = {
            'nan_ratio': float(nan_ratio),
            'inf_ratio': float(inf_ratio),
            'value_range': (float(np.nanmin(texture_features)), float(np.nanmax(texture_features))),
            'mean_value': float(np.nanmean(texture_features)),
            'std_value': float(np.nanstd(texture_features))
        }
        
        # 检查潜在问题
        if nan_ratio > 0.01:
            validation_results['issues'].append(f"High NaN ratio: {nan_ratio:.2%}")
        
        if inf_ratio > 0:
            validation_results['issues'].append(f"Infinite values: {inf_ratio:.2%}")
        
        if texture_features.shape[2] == 0:
            validation_results['issues'].append("No texture features extracted")
        
        # 检查特征变化
        feature_variance = np.var(texture_features, axis=(0, 1))
        low_variance_count = np.sum(feature_variance < 1e-6)
        if low_variance_count > texture_features.shape[2] * 0.5:
            validation_results['issues'].append("Many features have very low variance")
        
        # 评估特征质量
        if len(validation_results['issues']) == 0:
            validation_results['quality'] = 'Good'
        elif len(validation_results['issues']) <= 2:
            validation_results['quality'] = 'Acceptable'
        else:
            validation_results['quality'] = 'Poor'
        
        return validation_results
    
    def get_feature_names(self) -> List[str]:
        """获取纹理特征名称列表
        
        Returns:
            List[str]: 特征名称列表
        """
        feature_names = []
        
        for method in self.methods:
            if method == 'GLCM':
                glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                for distance in self.distances:
                    for angle in self.angles:
                        for prop in glcm_props:
                            feature_names.append(f'GLCM_{prop}_d{distance}_a{angle}')
            
            elif method == 'LBP':
                feature_names.extend(['LBP_uniform', 'LBP_rotation_invariant'])
            
            elif method == 'Statistical':
                stat_features = ['variance', 'skewness', 'kurtosis', 'range', 'mean_abs_deviation']
                feature_names.extend([f'Stat_{feat}' for feat in stat_features])
        
        return feature_names