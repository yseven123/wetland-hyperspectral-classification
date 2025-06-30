#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证器
Data Validator

验证高光谱遥感数据和地面真值数据的质量和完整性

作者: Wetland Research Team
"""

import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False

from ..config import Config

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器
    
    验证功能包括：
    - 数据完整性检查
    - 数据质量评估
    - 格式兼容性验证
    - 空间范围一致性检查
    - 类别标签有效性验证
    """
    
    def __init__(self, config: Config):
        """初始化数据验证器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.data_config = config.get_data_config()
        
        # 验证阈值
        self.thresholds = {
            'max_missing_pixels_ratio': 0.05,  # 最大缺失像素比例
            'min_signal_to_noise_ratio': 10,   # 最小信噪比
            'max_cloud_cover_ratio': 0.1,      # 最大云覆盖比例
            'min_samples_per_class': 10,       # 每类最小样本数
            'max_spectral_outlier_ratio': 0.01, # 最大光谱异常值比例
        }
        
        logger.info("DataValidator initialized")
    
    def validate_hyperspectral(self, data: np.ndarray, 
                              metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证高光谱数据
        
        Args:
            data: 高光谱数据数组 (H, W, B)
            metadata: 元数据字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 问题列表)
        """
        logger.info("Validating hyperspectral data")
        
        issues = []
        
        # 1. 基本形状和类型检查
        shape_issues = self._validate_data_shape(data)
        issues.extend(shape_issues)
        
        # 2. 数值范围检查
        range_issues = self._validate_data_range(data)
        issues.extend(range_issues)
        
        # 3. 缺失值检查
        missing_issues = self._validate_missing_values(data)
        issues.extend(missing_issues)
        
        # 4. 光谱质量检查
        spectral_issues = self._validate_spectral_quality(data)
        issues.extend(spectral_issues)
        
        # 5. 元数据一致性检查
        metadata_issues = self._validate_metadata_consistency(data, metadata)
        issues.extend(metadata_issues)
        
        # 6. 空间完整性检查
        spatial_issues = self._validate_spatial_integrity(data)
        issues.extend(spatial_issues)
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Hyperspectral data validation passed")
        else:
            logger.warning(f"Hyperspectral data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def validate_samples(self, samples: Union[gpd.GeoDataFrame, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """验证样本数据
        
        Args:
            samples: 样本数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 问题列表)
        """
        logger.info("Validating sample data")
        
        issues = []
        
        # 1. 基本结构检查
        structure_issues = self._validate_sample_structure(samples)
        issues.extend(structure_issues)
        
        # 2. 类别标签检查
        label_issues = self._validate_class_labels(samples)
        issues.extend(label_issues)
        
        # 3. 空间数据检查
        if HAS_GEO_LIBS and isinstance(samples, gpd.GeoDataFrame):
            spatial_issues = self._validate_sample_geometry(samples)
            issues.extend(spatial_issues)
        
        # 4. 样本分布检查
        distribution_issues = self._validate_sample_distribution(samples)
        issues.extend(distribution_issues)
        
        # 5. 数据完整性检查
        completeness_issues = self._validate_sample_completeness(samples)
        issues.extend(completeness_issues)
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Sample data validation passed")
        else:
            logger.warning(f"Sample data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def validate_data_compatibility(self, hyperspectral_data: np.ndarray,
                                  samples: Union[gpd.GeoDataFrame, pd.DataFrame],
                                  metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证高光谱数据与样本数据的兼容性
        
        Args:
            hyperspectral_data: 高光谱数据
            samples: 样本数据
            metadata: 元数据
            
        Returns:
            Tuple[bool, List[str]]: (是否兼容, 问题列表)
        """
        logger.info("Validating data compatibility")
        
        issues = []
        
        # 1. 空间范围一致性检查
        if HAS_GEO_LIBS and isinstance(samples, gpd.GeoDataFrame):
            spatial_issues = self._validate_spatial_extent_compatibility(
                hyperspectral_data, samples, metadata
            )
            issues.extend(spatial_issues)
        
        # 2. 坐标系一致性检查
        if 'crs' in metadata and HAS_GEO_LIBS and isinstance(samples, gpd.GeoDataFrame):
            crs_issues = self._validate_crs_compatibility(samples, metadata)
            issues.extend(crs_issues)
        
        # 3. 样本位置有效性检查
        location_issues = self._validate_sample_locations(
            hyperspectral_data, samples, metadata
        )
        issues.extend(location_issues)
        
        # 4. 类别定义一致性检查
        class_issues = self._validate_class_definition_compatibility(samples)
        issues.extend(class_issues)
        
        is_compatible = len(issues) == 0
        
        if is_compatible:
            logger.info("Data compatibility validation passed")
        else:
            logger.warning(f"Data compatibility validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_compatible, issues
    
    def _validate_data_shape(self, data: np.ndarray) -> List[str]:
        """验证数据形状"""
        issues = []
        
        # 检查维度
        if data.ndim != 3:
            issues.append(f"Expected 3D data (H, W, B), got {data.ndim}D")
            return issues
        
        height, width, bands = data.shape
        
        # 检查最小尺寸
        if height < 10 or width < 10:
            issues.append(f"Image too small: {height}x{width}, minimum 10x10 required")
        
        # 检查波段数
        expected_bands = self.data_config.bands
        if expected_bands and abs(bands - expected_bands) > 50:  # 允许一定误差
            issues.append(f"Unexpected number of bands: {bands}, expected ~{expected_bands}")
        
        # 检查数据类型
        if not np.issubdtype(data.dtype, np.number):
            issues.append(f"Data must be numeric, got {data.dtype}")
        
        return issues
    
    def _validate_data_range(self, data: np.ndarray) -> List[str]:
        """验证数据数值范围"""
        issues = []
        
        # 检查数值范围的合理性
        min_val, max_val = np.nanmin(data), np.nanmax(data)
        
        # 反射率数据通常在0-1或0-10000范围内
        if min_val < -1.0:
            issues.append(f"Data contains unusually low values: {min_val}")
        
        if max_val > 20000:
            issues.append(f"Data contains unusually high values: {max_val}")
        
        # 检查数据是否全为常数
        if np.allclose(min_val, max_val, rtol=1e-10):
            issues.append("Data appears to be constant (no variation)")
        
        # 检查负值（对于反射率数据）
        negative_ratio = np.sum(data < 0) / data.size
        if negative_ratio > 0.01:  # 超过1%的负值
            issues.append(f"High proportion of negative values: {negative_ratio:.2%}")
        
        return issues
    
    def _validate_missing_values(self, data: np.ndarray) -> List[str]:
        """验证缺失值"""
        issues = []
        
        # 检查NaN值
        nan_ratio = np.isnan(data).sum() / data.size
        if nan_ratio > self.thresholds['max_missing_pixels_ratio']:
            issues.append(f"High NaN ratio: {nan_ratio:.2%}")
        
        # 检查无穷值
        inf_ratio = np.isinf(data).sum() / data.size
        if inf_ratio > 0:
            issues.append(f"Data contains infinite values: {inf_ratio:.2%}")
        
        # 检查NoData值
        nodata_value = self.data_config.nodata_value
        if nodata_value is not None:
            nodata_ratio = np.sum(data == nodata_value) / data.size
            if nodata_ratio > self.thresholds['max_missing_pixels_ratio']:
                issues.append(f"High NoData ratio: {nodata_ratio:.2%}")
        
        return issues
    
    def _validate_spectral_quality(self, data: np.ndarray) -> List[str]:
        """验证光谱质量"""
        issues = []
        
        # 计算每个波段的统计信息
        height, width, bands = data.shape
        
        # 检查异常波段
        band_means = np.nanmean(data.reshape(-1, bands), axis=0)
        band_stds = np.nanstd(data.reshape(-1, bands), axis=0)
        
        # 识别异常波段（均值或标准差异常）
        mean_outliers = np.abs(band_means - np.nanmedian(band_means)) > 3 * np.nanstd(band_means)
        std_outliers = band_stds < 0.01 * np.nanmean(band_stds)  # 方差过小的波段
        
        outlier_bands = np.where(mean_outliers | std_outliers)[0]
        if len(outlier_bands) > 0:
            issues.append(f"Potential problematic bands: {outlier_bands.tolist()}")
        
        # 检查光谱连续性
        spectral_continuity_issues = self._check_spectral_continuity(data)
        issues.extend(spectral_continuity_issues)
        
        # 检查信噪比
        snr_issues = self._estimate_signal_to_noise_ratio(data)
        issues.extend(snr_issues)
        
        return issues
    
    def _validate_metadata_consistency(self, data: np.ndarray, 
                                     metadata: Dict[str, Any]) -> List[str]:
        """验证元数据一致性"""
        issues = []
        
        # 检查形状一致性
        if 'shape' in metadata:
            expected_shape = metadata['shape']
            if isinstance(expected_shape, (list, tuple)) and len(expected_shape) >= 3:
                if tuple(data.shape) != tuple(expected_shape):
                    issues.append(f"Shape mismatch: data {data.shape} vs metadata {expected_shape}")
        
        # 检查波段数一致性
        if 'bands' in metadata:
            expected_bands = metadata['bands']
            if data.shape[2] != expected_bands:
                issues.append(f"Band count mismatch: data {data.shape[2]} vs metadata {expected_bands}")
        
        # 检查数据类型一致性
        if 'data_type' in metadata:
            expected_dtype = metadata['data_type']
            if str(data.dtype) != str(expected_dtype):
                issues.append(f"Data type mismatch: data {data.dtype} vs metadata {expected_dtype}")
        
        # 检查波长信息
        if 'wavelengths' in metadata:
            wavelengths = metadata['wavelengths']
            if isinstance(wavelengths, (list, np.ndarray)):
                if len(wavelengths) != data.shape[2]:
                    issues.append(f"Wavelength count mismatch: {len(wavelengths)} vs {data.shape[2]} bands")
        
        return issues
    
    def _validate_spatial_integrity(self, data: np.ndarray) -> List[str]:
        """验证空间完整性"""
        issues = []
        
        height, width, bands = data.shape
        
        # 检查边缘效应
        edge_thickness = min(5, height // 10, width // 10)
        if edge_thickness > 0:
            # 检查边缘像素的统计特性
            center_data = data[edge_thickness:-edge_thickness, 
                              edge_thickness:-edge_thickness, :]
            edge_data = np.concatenate([
                data[:edge_thickness, :, :].reshape(-1, bands),
                data[-edge_thickness:, :, :].reshape(-1, bands),
                data[:, :edge_thickness, :].reshape(-1, bands),
                data[:, -edge_thickness:, :].reshape(-1, bands)
            ])
            
            center_mean = np.nanmean(center_data)
            edge_mean = np.nanmean(edge_data)
            
            if abs(center_mean - edge_mean) / center_mean > 0.2:  # 20%差异
                issues.append("Significant edge effects detected")
        
        # 检查空间条带噪声
        stripe_issues = self._detect_stripes(data)
        issues.extend(stripe_issues)
        
        return issues
    
    def _validate_sample_structure(self, samples: Union[gpd.GeoDataFrame, pd.DataFrame]) -> List[str]:
        """验证样本数据结构"""
        issues = []
        
        # 检查是否为空
        if len(samples) == 0:
            issues.append("Sample data is empty")
            return issues
        
        # 检查必需列
        required_columns = ['class_id']
        alternative_columns = {
            'class_id': ['class', 'label', 'class_name']
        }
        
        for required_col in required_columns:
            if required_col not in samples.columns:
                # 检查是否有替代列名
                alternatives = alternative_columns.get(required_col, [])
                found_alternative = False
                for alt in alternatives:
                    if alt in samples.columns:
                        found_alternative = True
                        break
                
                if not found_alternative:
                    issues.append(f"Missing required column: {required_col}")
        
        # 检查地理空间数据的几何列
        if HAS_GEO_LIBS and isinstance(samples, gpd.GeoDataFrame):
            if 'geometry' not in samples.columns:
                issues.append("GeoDataFrame missing geometry column")
        
        return issues
    
    def _validate_class_labels(self, samples: Union[gpd.GeoDataFrame, pd.DataFrame]) -> List[str]:
        """验证类别标签"""
        issues = []
        
        # 获取类别列
        class_column = None
        for col in ['class_id', 'class', 'label']:
            if col in samples.columns:
                class_column = col
                break
        
        if class_column is None:
            issues.append("No class label column found")
            return issues
        
        class_labels = samples[class_column]
        
        # 检查缺失值
        missing_labels = class_labels.isna().sum()
        if missing_labels > 0:
            issues.append(f"{missing_labels} samples have missing class labels")
        
        # 检查类别值的有效性
        unique_classes = class_labels.dropna().unique()
        
        # 检查是否有足够的类别
        if len(unique_classes) < 2:
            issues.append(f"Too few classes: {len(unique_classes)} (minimum 2 required)")
        
        # 检查类别ID是否为正整数
        for class_id in unique_classes:
            if not isinstance(class_id, (int, np.integer)) or class_id <= 0:
                issues.append(f"Invalid class ID: {class_id} (must be positive integer)")
        
        # 检查类别是否在配置中定义
        config_classes = self.config.get_classes()
        if config_classes:
            undefined_classes = set(unique_classes) - set(config_classes.keys())
            if undefined_classes:
                issues.append(f"Classes not defined in config: {list(undefined_classes)}")
        
        return issues
    
    def _validate_sample_geometry(self, samples: gpd.GeoDataFrame) -> List[str]:
        """验证样本几何数据"""
        issues = []
        
        # 检查几何有效性
        invalid_geom = ~samples.geometry.is_valid
        if invalid_geom.any():
            issues.append(f"{invalid_geom.sum()} samples have invalid geometry")
        
        # 检查空几何
        empty_geom = samples.geometry.is_empty
        if empty_geom.any():
            issues.append(f"{empty_geom.sum()} samples have empty geometry")
        
        # 检查几何类型一致性
        geom_types = samples.geometry.geom_type.unique()
        if len(geom_types) > 1:
            issues.append(f"Mixed geometry types: {list(geom_types)}")
        
        # 检查坐标系
        if samples.crs is None:
            issues.append("No coordinate reference system (CRS) defined")
        
        return issues
    
    def _validate_sample_distribution(self, samples: Union[gpd.GeoDataFrame, pd.DataFrame]) -> List[str]:
        """验证样本分布"""
        issues = []
        
        # 获取类别列
        class_column = None
        for col in ['class_id', 'class', 'label']:
            if col in samples.columns:
                class_column = col
                break
        
        if class_column is None:
            return issues
        
        # 检查每个类别的样本数量
        class_counts = samples[class_column].value_counts()
        min_samples = self.thresholds['min_samples_per_class']
        
        insufficient_classes = class_counts[class_counts < min_samples]
        if len(insufficient_classes) > 0:
            issues.append(f"Classes with insufficient samples (<{min_samples}): {dict(insufficient_classes)}")
        
        # 检查类别不平衡
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 100:  # 100:1的不平衡比例
            issues.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        
        return issues
    
    def _validate_sample_completeness(self, samples: Union[gpd.GeoDataFrame, pd.DataFrame]) -> List[str]:
        """验证样本数据完整性"""
        issues = []
        
        # 检查整体缺失值
        total_missing = samples.isna().sum().sum()
        total_values = samples.size
        missing_ratio = total_missing / total_values
        
        if missing_ratio > 0.05:  # 5%的缺失值阈值
            issues.append(f"High missing value ratio: {missing_ratio:.2%}")
        
        # 检查关键列的缺失值
        critical_columns = ['class_id', 'class', 'label']
        for col in critical_columns:
            if col in samples.columns:
                col_missing = samples[col].isna().sum()
                if col_missing > 0:
                    issues.append(f"Missing values in critical column '{col}': {col_missing}")
        
        return issues
    
    def _validate_spatial_extent_compatibility(self, hyperspectral_data: np.ndarray,
                                             samples: gpd.GeoDataFrame,
                                             metadata: Dict[str, Any]) -> List[str]:
        """验证空间范围兼容性"""
        issues = []
        
        if 'bounds' not in metadata:
            return issues
        
        try:
            # 获取影像边界
            image_bounds = metadata['bounds']
            
            # 获取样本边界
            sample_bounds = samples.total_bounds
            
            # 检查样本是否在影像范围内
            samples_outside = (
                (sample_bounds[0] < image_bounds[0]) or  # 西边界
                (sample_bounds[1] < image_bounds[1]) or  # 南边界
                (sample_bounds[2] > image_bounds[2]) or  # 东边界
                (sample_bounds[3] > image_bounds[3])     # 北边界
            )
            
            if samples_outside:
                issues.append("Some samples are outside the hyperspectral image extent")
            
            # 检查重叠比例
            overlap_area = self._calculate_overlap_area(image_bounds, sample_bounds)
            sample_area = (sample_bounds[2] - sample_bounds[0]) * (sample_bounds[3] - sample_bounds[1])
            
            if sample_area > 0:
                overlap_ratio = overlap_area / sample_area
                if overlap_ratio < 0.9:  # 90%重叠阈值
                    issues.append(f"Low spatial overlap between image and samples: {overlap_ratio:.2%}")
        
        except Exception as e:
            issues.append(f"Failed to validate spatial extent compatibility: {e}")
        
        return issues
    
    def _validate_crs_compatibility(self, samples: gpd.GeoDataFrame,
                                   metadata: Dict[str, Any]) -> List[str]:
        """验证坐标系兼容性"""
        issues = []
        
        if 'crs' not in metadata or samples.crs is None:
            return issues
        
        try:
            image_crs = str(metadata['crs'])
            sample_crs = str(samples.crs)
            
            if image_crs != sample_crs:
                issues.append(f"CRS mismatch: image ({image_crs}) vs samples ({sample_crs})")
        
        except Exception as e:
            issues.append(f"Failed to validate CRS compatibility: {e}")
        
        return issues
    
    def _validate_sample_locations(self, hyperspectral_data: np.ndarray,
                                 samples: Union[gpd.GeoDataFrame, pd.DataFrame],
                                 metadata: Dict[str, Any]) -> List[str]:
        """验证样本位置有效性"""
        issues = []
        
        height, width, bands = hyperspectral_data.shape
        
        # 如果是表格数据且有x, y列
        if not HAS_GEO_LIBS or not isinstance(samples, gpd.GeoDataFrame):
            if 'x' in samples.columns and 'y' in samples.columns:
                # 假设x, y是像素坐标
                invalid_x = (samples['x'] < 0) | (samples['x'] >= width)
                invalid_y = (samples['y'] < 0) | (samples['y'] >= height)
                
                invalid_locations = invalid_x | invalid_y
                if invalid_locations.any():
                    issues.append(f"{invalid_locations.sum()} samples have invalid pixel coordinates")
        
        return issues
    
    def _validate_class_definition_compatibility(self, samples: Union[gpd.GeoDataFrame, pd.DataFrame]) -> List[str]:
        """验证类别定义兼容性"""
        issues = []
        
        # 获取样本中的类别
        class_column = None
        for col in ['class_id', 'class', 'label']:
            if col in samples.columns:
                class_column = col
                break
        
        if class_column is None:
            return issues
        
        sample_classes = set(samples[class_column].dropna().unique())
        config_classes = set(self.config.get_classes().keys())
        
        # 检查未定义的类别
        undefined_classes = sample_classes - config_classes
        if undefined_classes:
            issues.append(f"Sample classes not defined in config: {list(undefined_classes)}")
        
        # 检查未使用的类别
        unused_classes = config_classes - sample_classes
        if unused_classes:
            issues.append(f"Config classes not present in samples: {list(unused_classes)}")
        
        return issues
    
    def _check_spectral_continuity(self, data: np.ndarray) -> List[str]:
        """检查光谱连续性"""
        issues = []
        
        # 简单的光谱连续性检查
        # 计算相邻波段间的差异
        height, width, bands = data.shape
        
        if bands < 3:
            return issues
        
        # 重塑数据为 (pixels, bands)
        pixels = data.reshape(-1, bands)
        
        # 计算相邻波段差异
        band_diffs = np.diff(pixels, axis=1)
        
        # 检查异常大的跳跃
        diff_threshold = 3 * np.nanstd(band_diffs)
        large_jumps = np.abs(band_diffs) > diff_threshold
        
        jump_ratio = np.sum(large_jumps) / large_jumps.size
        if jump_ratio > 0.01:  # 1%的异常跳跃
            issues.append(f"Spectral discontinuities detected: {jump_ratio:.2%} of transitions")
        
        return issues
    
    def _estimate_signal_to_noise_ratio(self, data: np.ndarray) -> List[str]:
        """估算信噪比"""
        issues = []
        
        try:
            # 简单的SNR估算：使用信号强度与噪声标准差的比值
            height, width, bands = data.shape
            
            # 选择中心区域避免边缘效应
            center_h, center_w = height // 4, width // 4
            center_data = data[center_h:3*center_h, center_w:3*center_w, :]
            
            # 计算信号强度（均值）
            signal = np.nanmean(center_data, axis=(0, 1))
            
            # 估算噪声（使用局部标准差）
            noise = np.nanstd(center_data, axis=(0, 1))
            
            # 计算SNR
            snr = signal / (noise + 1e-10)  # 避免除零
            
            mean_snr = np.nanmean(snr)
            if mean_snr < self.thresholds['min_signal_to_noise_ratio']:
                issues.append(f"Low signal-to-noise ratio: {mean_snr:.1f}")
        
        except Exception as e:
            logger.warning(f"Failed to estimate SNR: {e}")
        
        return issues
    
    def _detect_stripes(self, data: np.ndarray) -> List[str]:
        """检测条带噪声"""
        issues = []
        
        try:
            height, width, bands = data.shape
            
            # 检查垂直条带（列方向的异常）
            col_means = np.nanmean(data, axis=(0, 2))  # 每列的平均值
            col_std = np.nanstd(col_means)
            col_outliers = np.abs(col_means - np.nanmean(col_means)) > 3 * col_std
            
            if np.sum(col_outliers) > width * 0.05:  # 超过5%的列异常
                issues.append("Vertical stripes detected")
            
            # 检查水平条带（行方向的异常）
            row_means = np.nanmean(data, axis=(1, 2))  # 每行的平均值
            row_std = np.nanstd(row_means)
            row_outliers = np.abs(row_means - np.nanmean(row_means)) > 3 * row_std
            
            if np.sum(row_outliers) > height * 0.05:  # 超过5%的行异常
                issues.append("Horizontal stripes detected")
        
        except Exception as e:
            logger.warning(f"Failed to detect stripes: {e}")
        
        return issues
    
    def _calculate_overlap_area(self, bounds1: Tuple[float, float, float, float],
                               bounds2: Tuple[float, float, float, float]) -> float:
        """计算两个边界框的重叠面积"""
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        # 计算重叠区域
        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)
        
        # 检查是否有重叠
        if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
            return 0.0
        
        # 计算重叠面积
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        return overlap_area
    
    def generate_data_quality_report(self, hyperspectral_data: np.ndarray,
                                   metadata: Dict[str, Any],
                                   samples: Optional[Union[gpd.GeoDataFrame, pd.DataFrame]] = None) -> Dict[str, Any]:
        """生成数据质量报告
        
        Args:
            hyperspectral_data: 高光谱数据
            metadata: 元数据
            samples: 样本数据 (可选)
            
        Returns:
            Dict[str, Any]: 质量报告
        """
        logger.info("Generating data quality report")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'hyperspectral_validation': {},
            'sample_validation': {},
            'compatibility_validation': {},
            'summary': {}
        }
        
        # 验证高光谱数据
        hs_valid, hs_issues = self.validate_hyperspectral(hyperspectral_data, metadata)
        report['hyperspectral_validation'] = {
            'is_valid': hs_valid,
            'issues': hs_issues,
            'data_info': {
                'shape': hyperspectral_data.shape,
                'dtype': str(hyperspectral_data.dtype),
                'size_mb': hyperspectral_data.nbytes / (1024 * 1024),
                'min_value': float(np.nanmin(hyperspectral_data)),
                'max_value': float(np.nanmax(hyperspectral_data)),
                'mean_value': float(np.nanmean(hyperspectral_data)),
                'std_value': float(np.nanstd(hyperspectral_data)),
                'nan_ratio': float(np.isnan(hyperspectral_data).sum() / hyperspectral_data.size),
            }
        }
        
        # 验证样本数据
        if samples is not None:
            sample_valid, sample_issues = self.validate_samples(samples)
            report['sample_validation'] = {
                'is_valid': sample_valid,
                'issues': sample_issues,
                'sample_info': {
                    'total_samples': len(samples),
                    'num_classes': len(samples.get('class_id', samples.get('class', [])).unique()) if 'class_id' in samples.columns or 'class' in samples.columns else 0,
                    'class_distribution': samples.get('class_id', samples.get('class', pd.Series())).value_counts().to_dict() if 'class_id' in samples.columns or 'class' in samples.columns else {}
                }
            }
            
            # 验证兼容性
            compat_valid, compat_issues = self.validate_data_compatibility(
                hyperspectral_data, samples, metadata
            )
            report['compatibility_validation'] = {
                'is_compatible': compat_valid,
                'issues': compat_issues
            }
        
        # 生成摘要
        total_issues = len(hs_issues)
        if samples is not None:
            total_issues += len(sample_issues) + len(compat_issues)
        
        report['summary'] = {
            'overall_status': 'PASS' if total_issues == 0 else 'FAIL',
            'total_issues': total_issues,
            'recommendations': self._generate_recommendations(report)
        }
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """基于验证结果生成建议"""
        recommendations = []
        
        # 基于高光谱数据问题的建议
        hs_issues = report['hyperspectral_validation']['issues']
        for issue in hs_issues:
            if 'missing' in issue.lower():
                recommendations.append("Consider data gap filling or interpolation")
            elif 'noise' in issue.lower():
                recommendations.append("Apply noise reduction preprocessing")
            elif 'stripe' in issue.lower():
                recommendations.append("Apply destriping algorithms")
            elif 'snr' in issue.lower():
                recommendations.append("Consider spectral smoothing or binning")
        
        # 基于样本数据问题的建议
        if 'sample_validation' in report:
            sample_issues = report['sample_validation']['issues']
            for issue in sample_issues:
                if 'insufficient' in issue.lower():
                    recommendations.append("Collect more training samples for underrepresented classes")
                elif 'imbalance' in issue.lower():
                    recommendations.append("Apply class balancing techniques")
                elif 'missing' in issue.lower():
                    recommendations.append("Clean sample data and fill missing labels")
        
        return recommendations