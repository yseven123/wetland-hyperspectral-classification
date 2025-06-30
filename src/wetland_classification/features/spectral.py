#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光谱特征提取器
Spectral Feature Extractor

提取高光谱数据的光谱特征，包括原始光谱、导数光谱、连续统去除等

作者: Wetland Research Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
from scipy import signal, interpolate
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..config import Config

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """光谱特征提取器
    
    支持的光谱特征：
    - 原始光谱反射率
    - 一阶导数光谱
    - 二阶导数光谱  
    - 连续统去除光谱
    - 光谱吸收深度
    - 光谱斜率和曲率
    - 光谱角度特征
    - 主要吸收带位置
    """
    
    def __init__(self, config: Config):
        """初始化光谱特征提取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.spectral_config = config.get('features.spectral', {})
        
        # 获取波长信息
        self.wavelengths = self._get_wavelengths()
        
        # 定义关键光谱区域
        self.spectral_regions = {
            'blue': (400, 500),
            'green': (500, 600), 
            'red': (600, 700),
            'red_edge': (700, 750),
            'nir': (750, 1300),
            'swir1': (1300, 1900),
            'swir2': (1900, 2500)
        }
        
        logger.info("SpectralFeatureExtractor initialized")
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """提取光谱特征
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            
        Returns:
            np.ndarray: 提取的光谱特征 (H, W, F)
        """
        logger.info("Extracting spectral features")
        
        height, width, bands = data.shape
        features_list = []
        
        # 1. 原始光谱
        if self.spectral_config.get('raw_bands', True):
            features_list.append(data)
            logger.debug("Added raw spectral bands")
        
        # 2. 一阶导数
        if self.spectral_config.get('first_derivative', True):
            first_derivative = self._compute_first_derivative(data)
            features_list.append(first_derivative)
            logger.debug("Added first derivative features")
        
        # 3. 二阶导数
        if self.spectral_config.get('second_derivative', False):
            second_derivative = self._compute_second_derivative(data)
            features_list.append(second_derivative)
            logger.debug("Added second derivative features")
        
        # 4. 连续统去除
        if self.spectral_config.get('continuum_removal', True):
            continuum_removed = self._compute_continuum_removal(data)
            features_list.append(continuum_removed)
            logger.debug("Added continuum removal features")
        
        # 5. 光谱吸收特征
        if self.spectral_config.get('absorption_features', False):
            absorption_features = self._compute_absorption_features(data)
            features_list.append(absorption_features)
            logger.debug("Added absorption features")
        
        # 6. 光谱统计特征
        if self.spectral_config.get('statistical_features', False):
            statistical_features = self._compute_statistical_features(data)
            features_list.append(statistical_features)
            logger.debug("Added statistical features")
        
        # 7. 光谱角度特征
        if self.spectral_config.get('angle_features', False):
            angle_features = self._compute_angle_features(data)
            features_list.append(angle_features)
            logger.debug("Added angle features")
        
        # 合并所有特征
        if features_list:
            combined_features = np.concatenate(features_list, axis=2)
            logger.info(f"Extracted spectral features: {data.shape} -> {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No spectral features extracted, returning original data")
            return data
    
    def _compute_first_derivative(self, data: np.ndarray) -> np.ndarray:
        """计算一阶导数光谱
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 一阶导数 (H, W, B-1)
        """
        height, width, bands = data.shape
        
        # 计算相邻波段间的差分
        first_derivative = np.diff(data, axis=2)
        
        # 如果有波长信息，按波长间隔归一化
        if self.wavelengths is not None and len(self.wavelengths) == bands:
            wavelength_diffs = np.diff(self.wavelengths)
            # 广播波长差分到所有像素
            wavelength_diffs = wavelength_diffs.reshape(1, 1, -1)
            first_derivative = first_derivative / wavelength_diffs
        
        return first_derivative
    
    def _compute_second_derivative(self, data: np.ndarray) -> np.ndarray:
        """计算二阶导数光谱
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 二阶导数 (H, W, B-2)
        """
        # 先计算一阶导数，再计算一阶导数的导数
        first_derivative = self._compute_first_derivative(data)
        second_derivative = np.diff(first_derivative, axis=2)
        
        return second_derivative
    
    def _compute_continuum_removal(self, data: np.ndarray) -> np.ndarray:
        """计算连续统去除光谱
        
        连续统去除可以突出吸收特征，常用于矿物识别
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 连续统去除后的数据 (H, W, B)
        """
        height, width, bands = data.shape
        continuum_removed = np.zeros_like(data)
        
        for i in range(height):
            for j in range(width):
                spectrum = data[i, j, :]
                cr_spectrum = self._continuum_removal_single_spectrum(spectrum)
                continuum_removed[i, j, :] = cr_spectrum
        
        return continuum_removed
    
    def _continuum_removal_single_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """对单个光谱进行连续统去除
        
        Args:
            spectrum: 输入光谱
            
        Returns:
            np.ndarray: 连续统去除后的光谱
        """
        if np.all(np.isnan(spectrum)) or len(spectrum) < 3:
            return spectrum
        
        # 找到光谱的凸包（连续统线）
        bands_indices = np.arange(len(spectrum))
        valid_mask = ~np.isnan(spectrum)
        
        if np.sum(valid_mask) < 3:
            return spectrum
        
        valid_bands = bands_indices[valid_mask]
        valid_spectrum = spectrum[valid_mask]
        
        # 计算凸包
        try:
            from scipy.spatial import ConvexHull
            
            # 创建点集
            points = np.column_stack([valid_bands, valid_spectrum])
            
            # 计算凸包
            hull = ConvexHull(points)
            hull_vertices = hull.vertices
            
            # 找到上凸包（连续统）
            # 按波段排序
            sorted_indices = np.argsort(points[hull_vertices, 0])
            sorted_hull_points = points[hull_vertices[sorted_indices]]
            
            # 线性插值得到连续统
            continuum_interp = interpolate.interp1d(
                sorted_hull_points[:, 0], 
                sorted_hull_points[:, 1],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            continuum_line = continuum_interp(valid_bands)
            
            # 连续统去除：spectrum / continuum
            continuum_removed = np.ones_like(spectrum)
            continuum_removed[valid_mask] = valid_spectrum / np.maximum(continuum_line, 1e-10)
            
        except ImportError:
            # 如果没有scipy.spatial，使用简化方法
            continuum_removed = self._simple_continuum_removal(spectrum)
        except Exception:
            # 如果凸包计算失败，返回原光谱
            continuum_removed = spectrum
        
        return continuum_removed
    
    def _simple_continuum_removal(self, spectrum: np.ndarray) -> np.ndarray:
        """简化的连续统去除方法
        
        使用线性连接光谱端点作为连续统
        
        Args:
            spectrum: 输入光谱
            
        Returns:
            np.ndarray: 连续统去除后的光谱
        """
        valid_mask = ~np.isnan(spectrum)
        if np.sum(valid_mask) < 2:
            return spectrum
        
        # 找到首尾有效点
        valid_indices = np.where(valid_mask)[0]
        start_idx, end_idx = valid_indices[0], valid_indices[-1]
        
        # 线性连续统
        continuum_line = np.linspace(
            spectrum[start_idx], 
            spectrum[end_idx], 
            len(spectrum)
        )
        
        # 连续统去除
        continuum_removed = spectrum / np.maximum(continuum_line, 1e-10)
        
        return continuum_removed
    
    def _compute_absorption_features(self, data: np.ndarray) -> np.ndarray:
        """计算光谱吸收特征
        
        包括吸收深度、吸收带宽度、吸收带位置等
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 吸收特征 (H, W, F)
        """
        height, width, bands = data.shape
        
        # 定义主要吸收带（以波长索引表示）
        absorption_bands = self._get_absorption_band_indices()
        
        num_features = len(absorption_bands) * 3  # 每个吸收带3个特征：深度、宽度、位置
        absorption_features = np.zeros((height, width, num_features))
        
        for i in range(height):
            for j in range(width):
                spectrum = data[i, j, :]
                features = self._extract_absorption_features_single_spectrum(
                    spectrum, absorption_bands
                )
                absorption_features[i, j, :] = features
        
        return absorption_features
    
    def _get_absorption_band_indices(self) -> Dict[str, Tuple[int, int]]:
        """获取主要吸收带的波段索引
        
        Returns:
            Dict[str, Tuple[int, int]]: 吸收带名称到波段索引范围的映射
        """
        if self.wavelengths is None:
            # 如果没有波长信息，使用默认的波段索引
            bands = self.config.get('data.hyperspectral.bands', 400)
            return {
                'water_940': (int(bands * 0.2), int(bands * 0.3)),
                'water_1450': (int(bands * 0.4), int(bands * 0.5)),
                'water_1940': (int(bands * 0.6), int(bands * 0.7)),
                'clay_2200': (int(bands * 0.8), int(bands * 0.9))
            }
        
        # 根据波长定义吸收带
        absorption_bands = {}
        
        # 水分吸收带
        for name, center_wl, width in [
            ('water_940', 940, 50),
            ('water_1450', 1450, 100), 
            ('water_1940', 1940, 100)
        ]:
            start_wl, end_wl = center_wl - width//2, center_wl + width//2
            start_idx = np.argmin(np.abs(self.wavelengths - start_wl))
            end_idx = np.argmin(np.abs(self.wavelengths - end_wl))
            absorption_bands[name] = (start_idx, end_idx)
        
        # 粘土矿物吸收带
        for name, center_wl, width in [
            ('clay_2200', 2200, 100),
            ('clay_2350', 2350, 100)
        ]:
            start_wl, end_wl = center_wl - width//2, center_wl + width//2
            start_idx = np.argmin(np.abs(self.wavelengths - start_wl))
            end_idx = np.argmin(np.abs(self.wavelengths - end_wl))
            absorption_bands[name] = (start_idx, end_idx)
        
        return absorption_bands
    
    def _extract_absorption_features_single_spectrum(self, spectrum: np.ndarray,
                                                   absorption_bands: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """提取单个光谱的吸收特征
        
        Args:
            spectrum: 输入光谱
            absorption_bands: 吸收带定义
            
        Returns:
            np.ndarray: 吸收特征向量
        """
        features = []
        
        for band_name, (start_idx, end_idx) in absorption_bands.items():
            if end_idx >= len(spectrum):
                # 超出范围，填充默认值
                features.extend([0.0, 0.0, 0.0])
                continue
            
            band_spectrum = spectrum[start_idx:end_idx+1]
            
            if len(band_spectrum) < 3 or np.all(np.isnan(band_spectrum)):
                features.extend([0.0, 0.0, 0.0])
                continue
            
            # 吸收深度：连续统与光谱的最大差值
            continuum_removed_band = self._continuum_removal_single_spectrum(band_spectrum)
            absorption_depth = np.max(1 - continuum_removed_band)
            
            # 吸收带宽度：吸收深度>0.1的波段数
            absorption_mask = (1 - continuum_removed_band) > 0.1
            absorption_width = np.sum(absorption_mask)
            
            # 吸收带位置：最大吸收处的相对位置
            max_absorption_idx = np.argmax(1 - continuum_removed_band)
            absorption_position = max_absorption_idx / len(band_spectrum)
            
            features.extend([absorption_depth, absorption_width, absorption_position])
        
        return np.array(features)
    
    def _compute_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """计算光谱统计特征
        
        包括均值、标准差、偏度、峰度等统计量
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 统计特征 (H, W, F)
        """
        height, width, bands = data.shape
        
        # 计算各种统计特征
        features = []
        
        # 均值
        mean_features = np.mean(data, axis=2, keepdims=True)
        features.append(mean_features)
        
        # 标准差
        std_features = np.std(data, axis=2, keepdims=True)
        features.append(std_features)
        
        # 最小值和最大值
        min_features = np.min(data, axis=2, keepdims=True)
        max_features = np.max(data, axis=2, keepdims=True)
        features.extend([min_features, max_features])
        
        # 范围
        range_features = max_features - min_features
        features.append(range_features)
        
        # 分位数
        q25_features = np.percentile(data, 25, axis=2, keepdims=True)
        q75_features = np.percentile(data, 75, axis=2, keepdims=True)
        features.extend([q25_features, q75_features])
        
        # 四分位距
        iqr_features = q75_features - q25_features
        features.append(iqr_features)
        
        # 变异系数
        cv_features = std_features / (mean_features + 1e-10)
        features.append(cv_features)
        
        # 光谱斜率（线性回归斜率）
        slope_features = self._compute_spectral_slope(data)
        features.append(slope_features)
        
        # 合并特征
        statistical_features = np.concatenate(features, axis=2)
        
        return statistical_features
    
    def _compute_spectral_slope(self, data: np.ndarray) -> np.ndarray:
        """计算光谱斜率
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 光谱斜率 (H, W, 1)
        """
        height, width, bands = data.shape
        slopes = np.zeros((height, width, 1))
        
        # 波段索引作为X坐标
        x = np.arange(bands)
        
        for i in range(height):
            for j in range(width):
                spectrum = data[i, j, :]
                
                if not np.all(np.isnan(spectrum)):
                    # 线性回归拟合
                    valid_mask = ~np.isnan(spectrum)
                    if np.sum(valid_mask) >= 2:
                        x_valid = x[valid_mask]
                        y_valid = spectrum[valid_mask]
                        
                        # 计算斜率
                        slope = np.polyfit(x_valid, y_valid, 1)[0]
                        slopes[i, j, 0] = slope
        
        return slopes
    
    def _compute_angle_features(self, data: np.ndarray) -> np.ndarray:
        """计算光谱角度特征
        
        包括与参考光谱的光谱角度映射(SAM)等
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            np.ndarray: 角度特征 (H, W, F)
        """
        height, width, bands = data.shape
        
        # 定义参考光谱（典型的湿地地物光谱）
        reference_spectra = self._get_reference_spectra(bands)
        
        num_references = len(reference_spectra)
        angle_features = np.zeros((height, width, num_references))
        
        for i in range(height):
            for j in range(width):
                spectrum = data[i, j, :]
                
                for k, ref_spectrum in enumerate(reference_spectra):
                    # 计算光谱角度映射
                    sam_angle = self._compute_sam(spectrum, ref_spectrum)
                    angle_features[i, j, k] = sam_angle
        
        return angle_features
    
    def _get_reference_spectra(self, bands: int) -> List[np.ndarray]:
        """获取参考光谱
        
        Args:
            bands: 波段数
            
        Returns:
            List[np.ndarray]: 参考光谱列表
        """
        # 简化的参考光谱（实际应用中应使用光谱库）
        reference_spectra = []
        
        # 水体光谱：蓝绿波段高，红外波段低
        water_spectrum = np.ones(bands)
        water_spectrum[:bands//4] = 0.8  # 蓝绿
        water_spectrum[bands//4:bands//2] = 0.6  # 红
        water_spectrum[bands//2:] = 0.1  # 近红外和短波红外
        reference_spectra.append(water_spectrum)
        
        # 植被光谱：红边特征明显
        vegetation_spectrum = np.ones(bands)
        vegetation_spectrum[:bands//4] = 0.4  # 蓝绿
        vegetation_spectrum[bands//4:bands//2] = 0.3  # 红
        vegetation_spectrum[bands//2:3*bands//4] = 0.8  # 近红外
        vegetation_spectrum[3*bands//4:] = 0.6  # 短波红外
        reference_spectra.append(vegetation_spectrum)
        
        # 土壤光谱：整体较平坦，略有上升趋势
        soil_spectrum = np.linspace(0.2, 0.4, bands)
        reference_spectra.append(soil_spectrum)
        
        return reference_spectra
    
    def _compute_sam(self, spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
        """计算光谱角度映射
        
        Args:
            spectrum1: 光谱1
            spectrum2: 光谱2
            
        Returns:
            float: 光谱角度（弧度）
        """
        # 去除NaN值
        valid_mask = ~(np.isnan(spectrum1) | np.isnan(spectrum2))
        
        if np.sum(valid_mask) < 2:
            return np.pi  # 最大角度
        
        s1_valid = spectrum1[valid_mask]
        s2_valid = spectrum2[valid_mask]
        
        # 计算余弦相似度，然后转换为角度
        try:
            cosine_similarity = 1 - cosine(s1_valid, s2_valid)
            # 限制在[-1, 1]范围内
            cosine_similarity = np.clip(cosine_similarity, -1, 1)
            angle = np.arccos(cosine_similarity)
        except:
            angle = np.pi
        
        return angle
    
    def _get_wavelengths(self) -> Optional[np.ndarray]:
        """获取波长信息
        
        Returns:
            Optional[np.ndarray]: 波长数组，如果不可用则返回None
        """
        # 从配置中获取波长范围和波段数
        wavelength_range = self.config.get('data.hyperspectral.wavelength_range', [400, 2500])
        bands = self.config.get('data.hyperspectral.bands', 400)
        
        if wavelength_range and bands:
            wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], bands)
            return wavelengths
        
        return None
    
    def extract_spectral_indices(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """提取常用的光谱指数
        
        Args:
            data: 输入数据 (H, W, B)
            
        Returns:
            Dict[str, np.ndarray]: 光谱指数字典
        """
        if self.wavelengths is None:
            logger.warning("No wavelength information available for spectral indices")
            return {}
        
        indices = {}
        
        # 定义波段位置
        band_positions = {}
        for name, wavelength in [
            ('blue', 470), ('green', 550), ('red', 670),
            ('red_edge', 720), ('nir', 800), ('swir1', 1600), ('swir2', 2200)
        ]:
            band_idx = np.argmin(np.abs(self.wavelengths - wavelength))
            band_positions[name] = band_idx
        
        height, width, bands = data.shape
        
        # 红边位置
        if 'red' in band_positions and 'nir' in band_positions:
            red_idx = band_positions['red']
            nir_idx = band_positions['nir']
            
            red_edge_region = data[:, :, red_idx:nir_idx]
            if red_edge_region.shape[2] > 0:
                # 红边位置：一阶导数最大值位置
                red_edge_derivative = np.diff(red_edge_region, axis=2)
                if red_edge_derivative.shape[2] > 0:
                    red_edge_position = np.argmax(red_edge_derivative, axis=2)
                    indices['red_edge_position'] = red_edge_position.astype(np.float32)
        
        # 光谱斜率指数
        if 'green' in band_positions and 'red' in band_positions:
            green_idx = band_positions['green']
            red_idx = band_positions['red']
            
            if red_idx > green_idx:
                slope_region = data[:, :, green_idx:red_idx+1]
                slope_values = self._compute_spectral_slope(slope_region)
                indices['green_red_slope'] = slope_values[:, :, 0]
        
        # 吸收深度指数
        if 'nir' in band_positions and 'swir1' in band_positions:
            nir_idx = band_positions['nir']
            swir1_idx = band_positions['swir1']
            
            if swir1_idx > nir_idx:
                # 1600nm附近的水分吸收
                absorption_region = data[:, :, nir_idx:swir1_idx+1]
                continuum_removed = self._compute_continuum_removal(absorption_region)
                water_absorption = 1 - np.min(continuum_removed, axis=2)
                indices['water_absorption_1600'] = water_absorption
        
        return indices
    
    def validate_spectral_features(self, original_data: np.ndarray,
                                 extracted_features: np.ndarray) -> Dict[str, Any]:
        """验证光谱特征提取结果
        
        Args:
            original_data: 原始数据
            extracted_features: 提取的特征
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'input_shape': original_data.shape,
            'output_shape': extracted_features.shape,
            'feature_expansion_ratio': extracted_features.shape[2] / original_data.shape[2],
            'data_quality': {},
            'issues': []
        }
        
        # 检查数据质量
        nan_ratio = np.sum(np.isnan(extracted_features)) / extracted_features.size
        inf_ratio = np.sum(np.isinf(extracted_features)) / extracted_features.size
        
        validation_results['data_quality'] = {
            'nan_ratio': float(nan_ratio),
            'inf_ratio': float(inf_ratio),
            'value_range': (float(np.nanmin(extracted_features)), float(np.nanmax(extracted_features))),
            'mean_value': float(np.nanmean(extracted_features)),
            'std_value': float(np.nanstd(extracted_features))
        }
        
        # 检查潜在问题
        if nan_ratio > 0.01:
            validation_results['issues'].append(f"High NaN ratio: {nan_ratio:.2%}")
        
        if inf_ratio > 0:
            validation_results['issues'].append(f"Infinite values detected: {inf_ratio:.2%}")
        
        if validation_results['feature_expansion_ratio'] < 1:
            validation_results['issues'].append("Feature dimension reduction occurred")
        
        # 评估特征质量
        if len(validation_results['issues']) == 0:
            validation_results['quality'] = 'Good'
        elif len(validation_results['issues']) <= 2:
            validation_results['quality'] = 'Acceptable'
        else:
            validation_results['quality'] = 'Poor'
        
        return validation_results