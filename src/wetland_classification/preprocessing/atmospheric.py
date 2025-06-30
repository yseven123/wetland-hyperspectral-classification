#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大气校正处理器
Atmospheric Correction

去除大气影响，获得地表真实反射率

作者: Wetland Research Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
from scipy import optimize, interpolate

from ..config import Config

logger = logging.getLogger(__name__)


class AtmosphericCorrector:
    """大气校正处理器
    
    支持的大气校正方法：
    - FLAASH (Fast Line-of-sight Atmospheric Analysis of Spectral Hypercubes)
    - QUAC (Quick Atmospheric Correction)
    - DOS (Dark Object Subtraction)
    - 6SV (Second Simulation of Satellite Signal in Solar Spectrum)
    """
    
    def __init__(self, config: Config):
        """初始化大气校正处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.atmospheric_config = config.get('preprocessing.atmospheric', {})
        
        # 大气参数
        self.atmospheric_models = self._get_atmospheric_models()
        self.aerosol_models = self._get_aerosol_models()
        
        logger.info("AtmosphericCorrector initialized")
    
    def correct(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """执行大气校正
        
        Args:
            data: 输入高光谱数据 (H, W, B) - TOA反射率
            metadata: 元数据信息
            
        Returns:
            np.ndarray: 大气校正后的地表反射率
        """
        if not self.atmospheric_config.get('enabled', True):
            logger.info("Atmospheric correction disabled")
            return data
        
        method = self.atmospheric_config.get('method', 'DOS')
        logger.info(f"Applying atmospheric correction: {method}")
        
        if method == 'FLAASH':
            return self._flaash_correction(data, metadata)
        elif method == 'QUAC':
            return self._quac_correction(data, metadata)
        elif method == 'DOS':
            return self._dos_correction(data, metadata)
        elif method == '6SV':
            return self._6sv_correction(data, metadata)
        else:
            logger.warning(f"Unknown atmospheric method: {method}, applying DOS")
            return self._dos_correction(data, metadata)
    
    def _dos_correction(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """暗目标减法大气校正
        
        基本原理：暗目标（如深水、阴影）在可见光波段应该有很低的反射率，
        观测到的信号主要来自大气散射，可用于估算大气路径辐射
        
        Args:
            data: TOA反射率数据
            metadata: 元数据
            
        Returns:
            np.ndarray: DOS校正后的地表反射率
        """
        logger.info("Applying Dark Object Subtraction (DOS) correction")
        
        height, width, bands = data.shape
        corrected_data = data.copy().astype(np.float32)
        
        # 获取波长信息
        wavelengths = self._get_wavelengths(metadata)
        
        # 识别暗目标像素
        dark_pixels = self._identify_dark_pixels(data)
        
        if dark_pixels.size == 0:
            logger.warning("No dark pixels found, using minimum values")
            dark_values = np.min(data.reshape(-1, bands), axis=0)
        else:
            # 计算暗目标的平均反射率
            dark_values = np.mean(data[dark_pixels[:, 0], dark_pixels[:, 1], :], axis=0)
        
        # 波长相关的大气散射校正
        for b in range(bands):
            wavelength = wavelengths[b] if b < len(wavelengths) else 550  # 默认波长
            
            # 计算大气路径辐射 (Rayleigh散射模型)
            path_radiance = self._calculate_rayleigh_scattering(wavelength, dark_values[b])
            
            # 简化的大气透过率估算
            transmittance = self._estimate_atmospheric_transmittance(wavelength)
            
            # DOS校正公式
            corrected_data[:, :, b] = (data[:, :, b] - path_radiance) / transmittance
        
        # 限制反射率范围
        corrected_data = np.clip(corrected_data, 0, 1.0)
        
        logger.info("DOS correction completed")
        return corrected_data
    
    def _quac_correction(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """快速大气校正 (QUAC)
        
        基于场景的自适应大气校正方法
        
        Args:
            data: TOA反射率数据
            metadata: 元数据
            
        Returns:
            np.ndarray: QUAC校正后的地表反射率
        """
        logger.info("Applying Quick Atmospheric Correction (QUAC)")
        
        height, width, bands = data.shape
        corrected_data = data.copy().astype(np.float32)
        
        # 计算场景统计量
        scene_stats = self._calculate_scene_statistics(data)
        
        # 获取波长信息
        wavelengths = self._get_wavelengths(metadata)
        
        # 对每个波段进行校正
        for b in range(bands):
            # 获取当前波段的统计信息
            band_data = data[:, :, b]
            band_mean = scene_stats['mean'][b]
            band_std = scene_stats['std'][b]
            band_min = scene_stats['min'][b]
            band_max = scene_stats['max'][b]
            
            # QUAC算法：基于场景对比度调整
            # 假设场景包含从暗到亮的完整范围
            
            # 计算增益和偏移
            if band_max > band_min:
                # 线性拉伸到期望的反射率范围
                gain = 0.6 / (band_max - band_min)  # 期望最大反射率为0.6
                offset = -band_min * gain
                
                corrected_data[:, :, b] = band_data * gain + offset
            else:
                # 如果波段数据无变化，保持原值
                corrected_data[:, :, b] = band_data
        
        # 限制反射率范围
        corrected_data = np.clip(corrected_data, 0, 1.0)
        
        logger.info("QUAC correction completed")
        return corrected_data
    
    def _flaash_correction(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """FLAASH大气校正 (简化版本)
        
        基于MODTRAN大气辐射传输模型的大气校正
        
        Args:
            data: TOA反射率数据
            metadata: 元数据
            
        Returns:
            np.ndarray: FLAASH校正后的地表反射率
        """
        logger.info("Applying FLAASH atmospheric correction (simplified)")
        
        height, width, bands = data.shape
        corrected_data = data.copy().astype(np.float32)
        
        # 获取大气参数
        water_vapor = self.atmospheric_config.get('water_vapor', 2.5)  # g/cm²
        aerosol_model = self.atmospheric_config.get('aerosol_model', 'rural')
        visibility = self.atmospheric_config.get('visibility', 40.0)  # km
        
        # 获取波长信息
        wavelengths = self._get_wavelengths(metadata)
        
        # 获取太阳和观测几何
        sun_zenith = self._get_sun_zenith(metadata)
        view_zenith = self._get_view_zenith(metadata)
        
        # 对每个波段计算大气参数
        for b in range(bands):
            wavelength = wavelengths[b] if b < len(wavelengths) else 550
            
            # 计算大气透过率
            transmittance = self._calculate_transmittance(
                wavelength, water_vapor, aerosol_model, visibility,
                sun_zenith, view_zenith
            )
            
            # 计算大气路径辐射
            path_radiance = self._calculate_path_radiance(
                wavelength, aerosol_model, visibility, sun_zenith
            )
            
            # 计算球面反照率
            spherical_albedo = self._calculate_spherical_albedo(
                wavelength, aerosol_model, visibility
            )
            
            # FLAASH反演公式
            toa_refl = data[:, :, b]
            
            # 地表反射率计算
            numerator = toa_refl - path_radiance
            denominator = transmittance + spherical_albedo * (toa_refl - path_radiance)
            
            # 避免除零
            denominator = np.maximum(denominator, 0.01)
            
            corrected_data[:, :, b] = numerator / denominator
        
        # 限制反射率范围
        corrected_data = np.clip(corrected_data, 0, 1.0)
        
        logger.info("FLAASH correction completed")
        return corrected_data
    
    def _6sv_correction(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """6SV大气校正 (简化版本)
        
        基于6S/6SV辐射传输模型的大气校正
        
        Args:
            data: TOA反射率数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 6SV校正后的地表反射率
        """
        logger.info("Applying 6SV atmospheric correction (simplified)")
        
        # 注意：这是一个简化的6SV实现
        # 完整的6SV需要复杂的查找表和辐射传输计算
        
        height, width, bands = data.shape
        corrected_data = data.copy().astype(np.float32)
        
        # 获取大气和几何参数
        water_vapor = self.atmospheric_config.get('water_vapor', 2.5)
        aerosol_optical_depth = self._estimate_aerosol_optical_depth(metadata)
        sun_zenith = self._get_sun_zenith(metadata)
        view_zenith = self._get_view_zenith(metadata)
        relative_azimuth = self._get_relative_azimuth(metadata)
        
        # 获取波长信息
        wavelengths = self._get_wavelengths(metadata)
        
        # 6SV参数计算
        for b in range(bands):
            wavelength = wavelengths[b] if b < len(wavelengths) else 550
            
            # 计算6SV系数
            coefficients = self._calculate_6sv_coefficients(
                wavelength, water_vapor, aerosol_optical_depth,
                sun_zenith, view_zenith, relative_azimuth
            )
            
            # 6SV反演公式: ρ_surface = (ρ_TOA - xa) / (xb + xc * ρ_TOA)
            xa = coefficients['xa']  # 大气路径反射率
            xb = coefficients['xb']  # 直接透过率乘积
            xc = coefficients['xc']  # 环境效应系数
            
            toa_refl = data[:, :, b]
            
            numerator = toa_refl - xa
            denominator = xb + xc * toa_refl
            denominator = np.maximum(denominator, 0.01)  # 避免除零
            
            corrected_data[:, :, b] = numerator / denominator
        
        # 限制反射率范围
        corrected_data = np.clip(corrected_data, 0, 1.0)
        
        logger.info("6SV correction completed")
        return corrected_data
    
    def _identify_dark_pixels(self, data: np.ndarray, percentile: float = 1.0) -> np.ndarray:
        """识别暗目标像素
        
        Args:
            data: 输入数据
            percentile: 百分位阈值
            
        Returns:
            np.ndarray: 暗像素的坐标 (N, 2)
        """
        # 使用近红外波段识别暗目标
        if data.shape[-1] > 50:
            # 假设第50个波段附近是近红外
            nir_band = data[:, :, 50]
        else:
            # 使用最后一个波段
            nir_band = data[:, :, -1]
        
        # 计算阈值
        threshold = np.percentile(nir_band, percentile)
        
        # 找到暗像素
        dark_mask = nir_band <= threshold
        dark_pixels = np.where(dark_mask)
        
        # 返回坐标
        return np.column_stack((dark_pixels[0], dark_pixels[1]))
    
    def _calculate_rayleigh_scattering(self, wavelength: float, dark_value: float) -> float:
        """计算瑞利散射
        
        Args:
            wavelength: 波长 (nm)
            dark_value: 暗目标值
            
        Returns:
            float: 路径辐射
        """
        # 瑞利散射与波长的四次方成反比
        rayleigh_coeff = 0.1 * (550 / wavelength) ** 4
        
        # 结合暗目标信息估算路径辐射
        path_radiance = max(dark_value, rayleigh_coeff * 0.01)
        
        return path_radiance
    
    def _estimate_atmospheric_transmittance(self, wavelength: float) -> float:
        """估算大气透过率
        
        Args:
            wavelength: 波长 (nm)
            
        Returns:
            float: 大气透过率
        """
        # 简化的大气透过率模型
        # 考虑水汽吸收、臭氧吸收和气溶胶散射
        
        # 基础透过率
        base_transmittance = 0.85
        
        # 水汽吸收 (主要在940nm, 1140nm, 1380nm, 1900nm等)
        water_absorption = 0.0
        water_bands = [940, 1140, 1380, 1900, 2700]
        for wb in water_bands:
            if abs(wavelength - wb) < 50:
                water_absorption += 0.1 * np.exp(-(wavelength - wb)**2 / (2 * 20**2))
        
        # 瑞利散射
        rayleigh_scattering = 0.05 * (400 / wavelength) ** 4
        
        # 气溶胶散射
        aerosol_scattering = 0.02 * (550 / wavelength) ** 1.3
        
        transmittance = base_transmittance - water_absorption - rayleigh_scattering - aerosol_scattering
        
        return max(transmittance, 0.3)  # 最小透过率
    
    def _calculate_scene_statistics(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """计算场景统计量
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, np.ndarray]: 统计量字典
        """
        height, width, bands = data.shape
        pixels = data.reshape(-1, bands)
        
        stats = {
            'mean': np.mean(pixels, axis=0),
            'std': np.std(pixels, axis=0),
            'min': np.min(pixels, axis=0),
            'max': np.max(pixels, axis=0),
            'percentile_1': np.percentile(pixels, 1, axis=0),
            'percentile_99': np.percentile(pixels, 99, axis=0)
        }
        
        return stats
    
    def _calculate_transmittance(self, wavelength: float, water_vapor: float,
                               aerosol_model: str, visibility: float,
                               sun_zenith: float, view_zenith: float) -> float:
        """计算大气透过率
        
        Args:
            wavelength: 波长
            water_vapor: 水汽含量
            aerosol_model: 气溶胶模型
            visibility: 能见度
            sun_zenith: 太阳天顶角
            view_zenith: 观测天顶角
            
        Returns:
            float: 透过率
        """
        # 简化的透过率计算
        base_transmittance = 0.85
        
        # 路径长度修正
        air_mass = 1 / np.cos(np.radians(sun_zenith)) + 1 / np.cos(np.radians(view_zenith))
        path_length_factor = air_mass / 2
        
        # 水汽吸收
        water_absorption = self._calculate_water_absorption(wavelength, water_vapor) * path_length_factor
        
        # 气溶胶散射
        aerosol_extinction = self._calculate_aerosol_extinction(wavelength, aerosol_model, visibility)
        aerosol_effect = aerosol_extinction * path_length_factor
        
        # 瑞利散射
        rayleigh_effect = 0.05 * (400 / wavelength) ** 4 * path_length_factor
        
        transmittance = base_transmittance - water_absorption - aerosol_effect - rayleigh_effect
        
        return max(transmittance, 0.1)
    
    def _calculate_path_radiance(self, wavelength: float, aerosol_model: str,
                               visibility: float, sun_zenith: float) -> float:
        """计算路径辐射
        
        Args:
            wavelength: 波长
            aerosol_model: 气溶胶模型
            visibility: 能见度
            sun_zenith: 太阳天顶角
            
        Returns:
            float: 路径辐射
        """
        # 简化的路径辐射计算
        
        # 瑞利散射贡献
        rayleigh_path = 0.02 * (400 / wavelength) ** 4
        
        # 气溶胶散射贡献
        aerosol_path = 0.01 * (550 / wavelength) ** 1.3 * (23 / visibility)
        
        # 太阳角度影响
        sun_factor = 1 / np.cos(np.radians(sun_zenith))
        
        path_radiance = (rayleigh_path + aerosol_path) * sun_factor * 0.5
        
        return max(path_radiance, 0.001)
    
    def _calculate_spherical_albedo(self, wavelength: float, aerosol_model: str,
                                  visibility: float) -> float:
        """计算球面反照率
        
        Args:
            wavelength: 波长
            aerosol_model: 气溶胶模型
            visibility: 能见度
            
        Returns:
            float: 球面反照率
        """
        # 简化的球面反照率计算
        base_albedo = 0.05
        
        # 气溶胶贡献
        aerosol_albedo = 0.02 * (23 / visibility) * (550 / wavelength) ** 0.5
        
        spherical_albedo = base_albedo + aerosol_albedo
        
        return min(spherical_albedo, 0.2)
    
    def _calculate_6sv_coefficients(self, wavelength: float, water_vapor: float,
                                  aerosol_optical_depth: float, sun_zenith: float,
                                  view_zenith: float, relative_azimuth: float) -> Dict[str, float]:
        """计算6SV系数
        
        Args:
            wavelength: 波长
            water_vapor: 水汽
            aerosol_optical_depth: 气溶胶光学厚度
            sun_zenith: 太阳天顶角
            view_zenith: 观测天顶角
            relative_azimuth: 相对方位角
            
        Returns:
            Dict[str, float]: 6SV系数
        """
        # 简化的6SV系数计算
        # 实际的6SV需要复杂的查找表
        
        cos_sun = np.cos(np.radians(sun_zenith))
        cos_view = np.cos(np.radians(view_zenith))
        
        # xa: 大气路径反射率
        xa = 0.01 * aerosol_optical_depth * (550 / wavelength) ** 1.3
        
        # xb: 直接透过率乘积
        transmittance_sun = np.exp(-aerosol_optical_depth / cos_sun)
        transmittance_view = np.exp(-aerosol_optical_depth / cos_view)
        xb = transmittance_sun * transmittance_view
        
        # xc: 环境效应系数
        spherical_albedo = 0.05 + 0.02 * aerosol_optical_depth
        xc = spherical_albedo / (1 - spherical_albedo)
        
        return {
            'xa': xa,
            'xb': xb,
            'xc': xc
        }
    
    def _calculate_water_absorption(self, wavelength: float, water_vapor: float) -> float:
        """计算水汽吸收
        
        Args:
            wavelength: 波长 (nm)
            water_vapor: 水汽含量 (g/cm²)
            
        Returns:
            float: 水汽吸收系数
        """
        # 主要水汽吸收带
        water_bands = {
            940: 0.1,   # 强吸收带
            1140: 0.08,
            1380: 0.15,  # 很强吸收带
            1900: 0.12,
            2700: 0.2    # 极强吸收带
        }
        
        absorption = 0.0
        for center_wl, strength in water_bands.items():
            if abs(wavelength - center_wl) < 100:  # 100nm窗口
                absorption += strength * water_vapor * np.exp(-(wavelength - center_wl)**2 / (2 * 30**2))
        
        return absorption
    
    def _calculate_aerosol_extinction(self, wavelength: float, aerosol_model: str,
                                    visibility: float) -> float:
        """计算气溶胶消光
        
        Args:
            wavelength: 波长
            aerosol_model: 气溶胶模型
            visibility: 能见度
            
        Returns:
            float: 气溶胶消光系数
        """
        # Angstrom指数
        if aerosol_model == 'urban':
            angstrom_exp = 1.3
        elif aerosol_model == 'rural':
            angstrom_exp = 1.5
        elif aerosol_model == 'maritime':
            angstrom_exp = 0.5
        else:
            angstrom_exp = 1.3
        
        # 550nm处的气溶胶光学厚度
        aot_550 = 3.91 / visibility
        
        # 其他波长的气溶胶光学厚度
        aot = aot_550 * (550 / wavelength) ** angstrom_exp
        
        return aot
    
    def _estimate_aerosol_optical_depth(self, metadata: Dict[str, Any]) -> float:
        """估算气溶胶光学厚度
        
        Args:
            metadata: 元数据
            
        Returns:
            float: 气溶胶光学厚度
        """
        visibility = self.atmospheric_config.get('visibility', 40.0)
        
        # 使用能见度估算AOD
        aod = 3.91 / visibility
        
        return max(aod, 0.05)  # 最小AOD
    
    def _get_wavelengths(self, metadata: Dict[str, Any]) -> np.ndarray:
        """获取波长信息"""
        if 'wavelengths' in metadata:
            return np.array(metadata['wavelengths'])
        elif 'wavelength' in metadata:
            return np.array(metadata['wavelength'])
        else:
            # 使用默认波长
            num_bands = self.config.get('data.hyperspectral.bands', 400)
            wl_range = self.config.get('data.hyperspectral.wavelength_range', [400, 2500])
            return np.linspace(wl_range[0], wl_range[1], num_bands)
    
    def _get_sun_zenith(self, metadata: Dict[str, Any]) -> float:
        """获取太阳天顶角"""
        if 'sun_zenith' in metadata:
            return float(metadata['sun_zenith'])
        elif 'sun_elevation' in metadata:
            return 90.0 - float(metadata['sun_elevation'])
        else:
            return 45.0  # 默认值
    
    def _get_view_zenith(self, metadata: Dict[str, Any]) -> float:
        """获取观测天顶角"""
        if 'view_zenith' in metadata:
            return float(metadata['view_zenith'])
        else:
            return 0.0  # 假设星下点观测
    
    def _get_relative_azimuth(self, metadata: Dict[str, Any]) -> float:
        """获取相对方位角"""
        if 'relative_azimuth' in metadata:
            return float(metadata['relative_azimuth'])
        elif 'sun_azimuth' in metadata and 'view_azimuth' in metadata:
            return abs(float(metadata['sun_azimuth']) - float(metadata['view_azimuth']))
        else:
            return 0.0  # 默认值
    
    def _get_atmospheric_models(self) -> Dict[str, Dict[str, float]]:
        """获取大气模型参数"""
        return {
            'tropical': {'water_vapor': 4.1, 'ozone': 0.28},
            'mid_latitude_summer': {'water_vapor': 2.9, 'ozone': 0.30},
            'mid_latitude_winter': {'water_vapor': 0.85, 'ozone': 0.39},
            'subarctic_summer': {'water_vapor': 2.1, 'ozone': 0.34},
            'subarctic_winter': {'water_vapor': 0.42, 'ozone': 0.39},
            'us_standard': {'water_vapor': 1.4, 'ozone': 0.34}
        }
    
    def _get_aerosol_models(self) -> Dict[str, Dict[str, float]]:
        """获取气溶胶模型参数"""
        return {
            'rural': {'angstrom': 1.5, 'asymmetry': 0.7, 'ssa': 0.9},
            'urban': {'angstrom': 1.3, 'asymmetry': 0.65, 'ssa': 0.85},
            'maritime': {'angstrom': 0.5, 'asymmetry': 0.75, 'ssa': 0.95},
            'desert': {'angstrom': 0.1, 'asymmetry': 0.8, 'ssa': 0.9}
        }
    
    def validate_correction(self, toa_data: np.ndarray, 
                          corrected_data: np.ndarray) -> Dict[str, Any]:
        """验证大气校正结果
        
        Args:
            toa_data: TOA反射率数据
            corrected_data: 校正后数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'toa_stats': {
                'min': float(np.min(toa_data)),
                'max': float(np.max(toa_data)),
                'mean': float(np.mean(toa_data)),
                'std': float(np.std(toa_data))
            },
            'corrected_stats': {
                'min': float(np.min(corrected_data)),
                'max': float(np.max(corrected_data)),
                'mean': float(np.mean(corrected_data)),
                'std': float(np.std(corrected_data))
            },
            'issues': []
        }
        
        # 检查校正效果
        mean_change = np.mean(corrected_data) - np.mean(toa_data)
        if abs(mean_change) > 0.1:
            validation_results['issues'].append(f"Large mean change: {mean_change:.3f}")
        
        # 检查异常值
        if np.any(corrected_data < 0):
            negative_ratio = np.sum(corrected_data < 0) / corrected_data.size
            validation_results['issues'].append(f"Negative values: {negative_ratio:.2%}")
        
        if np.any(corrected_data > 1.0):
            high_ratio = np.sum(corrected_data > 1.0) / corrected_data.size
            validation_results['issues'].append(f"Values > 1.0: {high_ratio:.2%}")
        
        # 评估校正质量
        validation_results['quality'] = 'Good' if len(validation_results['issues']) == 0 else 'Needs Review'
        
        return validation_results