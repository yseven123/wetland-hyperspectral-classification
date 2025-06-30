#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
辐射定标处理器
Radiometric Correction

将数字信号转换为物理量（辐射亮度或反射率）

作者: Wetland Research Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
from scipy import interpolate

from ..config import Config

logger = logging.getLogger(__name__)


class RadiometricCorrector:
    """辐射定标校正器
    
    支持的定标类型：
    - DN to Radiance (数字信号转辐射亮度)
    - DN to Reflectance (数字信号转反射率)
    - TOA Reflectance (大气顶层反射率)
    - BOA Reflectance (地表反射率)
    """
    
    def __init__(self, config: Config):
        """初始化辐射定标校正器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.radiometric_config = config.get('preprocessing.radiometric', {})
        
        # 常用的太阳光谱辐照度常数 (W/m²/μm)
        self.solar_irradiance = self._get_solar_irradiance_constants()
        
        logger.info("RadiometricCorrector initialized")
    
    def correct(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """执行辐射定标校正
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            metadata: 元数据信息
            
        Returns:
            np.ndarray: 校正后的数据
        """
        if not self.radiometric_config.get('enabled', True):
            logger.info("Radiometric correction disabled")
            return data
        
        method = self.radiometric_config.get('method', 'TOA')
        logger.info(f"Applying radiometric correction: {method}")
        
        if method == 'RAD':
            return self._dn_to_radiance(data, metadata)
        elif method == 'TOA':
            return self._dn_to_toa_reflectance(data, metadata)
        elif method == 'BOA':
            return self._dn_to_boa_reflectance(data, metadata)
        else:
            logger.warning(f"Unknown radiometric method: {method}, applying TOA")
            return self._dn_to_toa_reflectance(data, metadata)
    
    def _dn_to_radiance(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """DN值转换为辐射亮度
        
        公式: L = DN * gain + offset
        
        Args:
            data: DN数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 辐射亮度数据 (W/m²/sr/μm)
        """
        logger.info("Converting DN to radiance")
        
        # 获取增益和偏移参数
        gains, offsets = self._get_calibration_parameters(data.shape[-1], metadata)
        
        # 应用定标公式
        radiance_data = data.astype(np.float32)
        
        for b in range(data.shape[-1]):
            radiance_data[:, :, b] = data[:, :, b] * gains[b] + offsets[b]
        
        # 处理负值和异常值
        radiance_data = np.clip(radiance_data, 0, None)
        
        logger.info("DN to radiance conversion completed")
        return radiance_data
    
    def _dn_to_toa_reflectance(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """DN值转换为大气顶层反射率
        
        公式: ρ = (π * L * d²) / (E_sun * cos(θ))
        
        Args:
            data: DN数据
            metadata: 元数据
            
        Returns:
            np.ndarray: TOA反射率数据
        """
        logger.info("Converting DN to TOA reflectance")
        
        # 首先转换为辐射亮度
        radiance_data = self._dn_to_radiance(data, metadata)
        
        # 获取太阳参数
        sun_elevation = self._get_sun_elevation(metadata)
        sun_zenith = 90.0 - sun_elevation
        sun_zenith_rad = np.radians(sun_zenith)
        cos_sun_zenith = np.cos(sun_zenith_rad)
        
        # 获取地球-太阳距离修正因子
        earth_sun_distance = self._get_earth_sun_distance(metadata)
        distance_correction = earth_sun_distance ** 2
        
        # 获取太阳光谱辐照度
        wavelengths = self._get_wavelengths(metadata)
        solar_irradiance = self._interpolate_solar_irradiance(wavelengths)
        
        # 计算TOA反射率
        toa_reflectance = np.zeros_like(radiance_data)
        
        for b in range(data.shape[-1]):
            if cos_sun_zenith > 0 and solar_irradiance[b] > 0:
                toa_reflectance[:, :, b] = (
                    np.pi * radiance_data[:, :, b] * distance_correction
                ) / (solar_irradiance[b] * cos_sun_zenith)
            else:
                logger.warning(f"Invalid solar parameters for band {b}")
                toa_reflectance[:, :, b] = 0
        
        # 限制反射率范围
        toa_reflectance = np.clip(toa_reflectance, 0, 1.2)
        
        logger.info("DN to TOA reflectance conversion completed")
        return toa_reflectance
    
    def _dn_to_boa_reflectance(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """DN值转换为地表反射率 (简化版本)
        
        注意：这是一个简化的BOA转换，完整的BOA需要大气校正
        
        Args:
            data: DN数据
            metadata: 元数据
            
        Returns:
            np.ndarray: BOA反射率数据
        """
        logger.info("Converting DN to BOA reflectance (simplified)")
        
        # 首先获取TOA反射率
        toa_reflectance = self._dn_to_toa_reflectance(data, metadata)
        
        # 简化的大气校正：减去路径辐射并应用透过率校正
        # 这是一个非常简化的方法，实际应用中需要更复杂的大气校正算法
        
        wavelengths = self._get_wavelengths(metadata)
        atmospheric_correction = self._get_simple_atmospheric_correction(wavelengths)
        
        boa_reflectance = np.zeros_like(toa_reflectance)
        
        for b in range(data.shape[-1]):
            path_radiance_correction = atmospheric_correction['path_radiance'][b]
            transmittance = atmospheric_correction['transmittance'][b]
            
            # 简化的BOA计算
            boa_reflectance[:, :, b] = (
                toa_reflectance[:, :, b] - path_radiance_correction
            ) / transmittance
        
        # 限制反射率范围
        boa_reflectance = np.clip(boa_reflectance, 0, 1.0)
        
        logger.info("DN to BOA reflectance conversion completed")
        return boa_reflectance
    
    def _get_calibration_parameters(self, num_bands: int, 
                                  metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """获取定标参数
        
        Args:
            num_bands: 波段数量
            metadata: 元数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (增益数组, 偏移数组)
        """
        # 尝试从元数据中获取定标参数
        if 'calibration' in metadata:
            cal_info = metadata['calibration']
            if 'gains' in cal_info and 'offsets' in cal_info:
                gains = np.array(cal_info['gains'])
                offsets = np.array(cal_info['offsets'])
                
                if len(gains) == num_bands and len(offsets) == num_bands:
                    return gains, offsets
        
        # 尝试从配置中获取
        scale_factor = self.radiometric_config.get('scale_factor', 0.0001)
        offset = self.radiometric_config.get('offset', 0.0)
        
        gains = np.full(num_bands, scale_factor)
        offsets = np.full(num_bands, offset)
        
        logger.info(f"Using default calibration parameters: gain={scale_factor}, offset={offset}")
        
        return gains, offsets
    
    def _get_sun_elevation(self, metadata: Dict[str, Any]) -> float:
        """获取太阳高度角
        
        Args:
            metadata: 元数据
            
        Returns:
            float: 太阳高度角 (度)
        """
        # 尝试从元数据获取
        if 'sun_elevation' in metadata:
            return float(metadata['sun_elevation'])
        elif 'solar_elevation' in metadata:
            return float(metadata['solar_elevation'])
        elif 'sun_zenith' in metadata:
            return 90.0 - float(metadata['sun_zenith'])
        
        # 默认值：45度
        default_elevation = 45.0
        logger.warning(f"Sun elevation not found in metadata, using default: {default_elevation}°")
        
        return default_elevation
    
    def _get_earth_sun_distance(self, metadata: Dict[str, Any]) -> float:
        """获取地球-太阳距离修正因子
        
        Args:
            metadata: 元数据
            
        Returns:
            float: 距离修正因子 (AU)
        """
        # 尝试从元数据获取
        if 'earth_sun_distance' in metadata:
            return float(metadata['earth_sun_distance'])
        
        # 尝试从采集日期计算
        if 'acquisition_date' in metadata:
            date_str = metadata['acquisition_date']
            try:
                from datetime import datetime
                date = datetime.strptime(date_str, '%Y-%m-%d')
                day_of_year = date.timetuple().tm_yday
                
                # 计算地球-太阳距离
                distance = self._calculate_earth_sun_distance(day_of_year)
                return distance
            
            except Exception as e:
                logger.warning(f"Failed to parse acquisition date: {e}")
        
        # 默认值：1.0 AU
        default_distance = 1.0
        logger.warning(f"Earth-sun distance not available, using default: {default_distance} AU")
        
        return default_distance
    
    def _calculate_earth_sun_distance(self, day_of_year: int) -> float:
        """计算地球-太阳距离
        
        Args:
            day_of_year: 一年中的第几天
            
        Returns:
            float: 地球-太阳距离 (AU)
        """
        # 简化的地球-太阳距离计算
        angle = 2 * np.pi * (day_of_year - 1) / 365.25
        distance = 1.0 - 0.01672 * np.cos(angle)
        
        return distance
    
    def _get_wavelengths(self, metadata: Dict[str, Any]) -> np.ndarray:
        """获取波长信息
        
        Args:
            metadata: 元数据
            
        Returns:
            np.ndarray: 波长数组 (nm)
        """
        # 尝试从元数据获取
        if 'wavelengths' in metadata:
            wavelengths = np.array(metadata['wavelengths'])
            return wavelengths
        elif 'wavelength' in metadata:
            wavelengths = np.array(metadata['wavelength'])
            return wavelengths
        
        # 尝试从配置获取波长范围
        wavelength_range = self.config.get('data.hyperspectral.wavelength_range', [400, 2500])
        num_bands = self.config.get('data.hyperspectral.bands', 400)
        
        # 生成等间隔波长
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_bands)
        
        logger.warning(f"Wavelengths not found in metadata, using generated wavelengths: {wavelength_range[0]}-{wavelength_range[1]} nm")
        
        return wavelengths
    
    def _get_solar_irradiance_constants(self) -> Dict[str, np.ndarray]:
        """获取太阳光谱辐照度常数
        
        Returns:
            Dict[str, np.ndarray]: 太阳辐照度数据
        """
        # 简化的太阳光谱辐照度数据 (W/m²/nm)
        # 实际应用中应使用标准的太阳光谱数据，如ASTM G173或Kurucz数据
        
        # 可见光-近红外范围的代表性值
        wavelengths = np.array([
            400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
            1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
            1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
            2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500
        ])
        
        irradiance = np.array([
            1.87, 1.94, 1.99, 1.97, 1.89, 1.86, 1.82, 1.78, 1.74, 1.69, 1.64,
            1.58, 1.52, 1.46, 1.40, 1.33, 1.27, 1.20, 1.13, 1.06, 0.99, 0.92,
            0.85, 0.78, 0.71, 0.64, 0.57, 0.50, 0.43, 0.36, 0.29, 0.22, 0.15,
            0.08, 0.05, 0.03, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001
        ])
        
        return {
            'wavelengths': wavelengths,
            'irradiance': irradiance
        }
    
    def _interpolate_solar_irradiance(self, target_wavelengths: np.ndarray) -> np.ndarray:
        """插值太阳辐照度到目标波长
        
        Args:
            target_wavelengths: 目标波长数组
            
        Returns:
            np.ndarray: 插值后的太阳辐照度
        """
        solar_wl = self.solar_irradiance['wavelengths']
        solar_irr = self.solar_irradiance['irradiance']
        
        # 线性插值
        f = interpolate.interp1d(
            solar_wl, solar_irr,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        interpolated_irradiance = f(target_wavelengths)
        
        # 确保非负值
        interpolated_irradiance = np.maximum(interpolated_irradiance, 0.001)
        
        return interpolated_irradiance
    
    def _get_simple_atmospheric_correction(self, wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """获取简化的大气校正参数
        
        Args:
            wavelengths: 波长数组
            
        Returns:
            Dict[str, np.ndarray]: 大气校正参数
        """
        num_bands = len(wavelengths)
        
        # 简化的路径辐射 (随波长变化)
        path_radiance = 0.05 * np.exp(-wavelengths / 1000)  # 短波长较高
        
        # 简化的大气透过率 (随波长变化)
        transmittance = 0.8 + 0.15 * np.exp(-(wavelengths - 800) ** 2 / (2 * 200 ** 2))  # 近红外较高
        transmittance = np.clip(transmittance, 0.3, 0.95)
        
        return {
            'path_radiance': path_radiance,
            'transmittance': transmittance
        }
    
    def validate_calibration(self, data: np.ndarray, 
                           calibrated_data: np.ndarray) -> Dict[str, Any]:
        """验证定标结果
        
        Args:
            data: 原始数据
            calibrated_data: 定标后数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'input_range': {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data))
            },
            'output_range': {
                'min': float(np.min(calibrated_data)),
                'max': float(np.max(calibrated_data)),
                'mean': float(np.mean(calibrated_data)),
                'std': float(np.std(calibrated_data))
            },
            'issues': []
        }
        
        # 检查异常值
        if np.any(calibrated_data < 0):
            negative_ratio = np.sum(calibrated_data < 0) / calibrated_data.size
            validation_results['issues'].append(f"Negative values: {negative_ratio:.2%}")
        
        if np.any(calibrated_data > 1.5):  # 反射率通常不超过1.5
            high_ratio = np.sum(calibrated_data > 1.5) / calibrated_data.size
            validation_results['issues'].append(f"Unusually high values: {high_ratio:.2%}")
        
        # 检查NaN和Inf
        if np.any(np.isnan(calibrated_data)):
            nan_ratio = np.sum(np.isnan(calibrated_data)) / calibrated_data.size
            validation_results['issues'].append(f"NaN values: {nan_ratio:.2%}")
        
        if np.any(np.isinf(calibrated_data)):
            inf_ratio = np.sum(np.isinf(calibrated_data)) / calibrated_data.size
            validation_results['issues'].append(f"Infinite values: {inf_ratio:.2%}")
        
        # 评估定标质量
        if len(validation_results['issues']) == 0:
            validation_results['quality'] = 'Good'
        elif len(validation_results['issues']) <= 2:
            validation_results['quality'] = 'Acceptable'
        else:
            validation_results['quality'] = 'Poor'
        
        return validation_results
    
    def apply_gain_offset_correction(self, data: np.ndarray,
                                   gains: np.ndarray,
                                   offsets: np.ndarray) -> np.ndarray:
        """应用增益偏移校正
        
        Args:
            data: 输入数据
            gains: 增益数组
            offsets: 偏移数组
            
        Returns:
            np.ndarray: 校正后的数据
        """
        corrected_data = data.astype(np.float32)
        
        for b in range(data.shape[-1]):
            corrected_data[:, :, b] = data[:, :, b] * gains[b] + offsets[b]
        
        return corrected_data
    
    def estimate_calibration_parameters(self, data: np.ndarray,
                                      reference_reflectance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """估算定标参数
        
        Args:
            data: 输入DN数据
            reference_reflectance: 参考反射率数据 (可选)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (估算的增益, 估算的偏移)
        """
        num_bands = data.shape[-1]
        
        if reference_reflectance is not None:
            # 基于参考数据估算
            gains = []
            offsets = []
            
            for b in range(num_bands):
                dn_values = data[:, :, b].flatten()
                ref_values = reference_reflectance[:, :, b].flatten()
                
                # 使用线性回归估算参数
                valid_mask = ~(np.isnan(dn_values) | np.isnan(ref_values))
                if np.sum(valid_mask) > 100:  # 足够的样本点
                    dn_clean = dn_values[valid_mask]
                    ref_clean = ref_values[valid_mask]
                    
                    # 线性回归: ref = gain * dn + offset
                    A = np.vstack([dn_clean, np.ones(len(dn_clean))]).T
                    gain, offset = np.linalg.lstsq(A, ref_clean, rcond=None)[0]
                    
                    gains.append(gain)
                    offsets.append(offset)
                else:
                    # 使用默认值
                    gains.append(0.0001)
                    offsets.append(0.0)
            
            return np.array(gains), np.array(offsets)
        
        else:
            # 基于经验估算
            # 假设DN范围对应常见的反射率范围
            dn_min = np.min(data, axis=(0, 1))
            dn_max = np.max(data, axis=(0, 1))
            
            # 假设反射率范围为0-0.8
            target_min = 0.0
            target_max = 0.8
            
            gains = (target_max - target_min) / (dn_max - dn_min + 1e-10)
            offsets = target_min - gains * dn_min
            
            return gains, offsets