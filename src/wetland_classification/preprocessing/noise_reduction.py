#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
噪声去除处理器
Noise Reduction

去除高光谱数据中的各种噪声，提高数据质量

作者: Wetland Research Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
from scipy import ndimage, signal
from scipy.signal import savgol_filter, medfilt, wiener
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..config import Config

logger = logging.getLogger(__name__)


class NoiseReducer:
    """噪声去除处理器
    
    支持的噪声去除方法：
    - 光谱平滑：Savitzky-Golay滤波、高斯滤波、中值滤波
    - 空间去噪：双边滤波、非局部均值去噪、维纳滤波
    - 条带去除：傅里叶变换、统计去条带、小波去条带
    - 椒盐噪声去除：形态学滤波、中值滤波
    - 主成分去噪：PCA降噪、MNF变换
    """
    
    def __init__(self, config: Config):
        """初始化噪声去除处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.noise_config = config.get('preprocessing.noise_reduction', {})
        
        # 预定义的噪声波段（通常包含大气吸收和传感器噪声）
        self.bad_bands_default = [0, 1, 2, 3, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223]
        
        logger.info("NoiseReducer initialized")
    
    def reduce_noise(self, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """执行噪声去除
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            metadata: 元数据信息 (可选)
            
        Returns:
            np.ndarray: 去噪后的数据
        """
        if not self.noise_config.get('enabled', True):
            logger.info("Noise reduction disabled")
            return data
        
        logger.info("Applying noise reduction")
        
        denoised_data = data.copy().astype(np.float32)
        
        # 1. 移除坏波段
        denoised_data = self._remove_bad_bands(denoised_data, metadata)
        
        # 2. 光谱平滑
        if self.noise_config.get('spectral_smoothing', True):
            denoised_data = self._apply_spectral_smoothing(denoised_data)
        
        # 3. 条带去除
        if self.noise_config.get('stripe_removal', True):
            denoised_data = self._remove_stripes(denoised_data)
        
        # 4. 椒盐噪声去除
        if self.noise_config.get('spike_removal', True):
            denoised_data = self._remove_spikes(denoised_data)
        
        # 5. 空间去噪
        if self.noise_config.get('spatial_denoising', False):
            denoised_data = self._apply_spatial_denoising(denoised_data)
        
        # 6. PCA去噪 (可选)
        if self.noise_config.get('pca_denoising', False):
            denoised_data = self._apply_pca_denoising(denoised_data)
        
        logger.info("Noise reduction completed")
        return denoised_data
    
    def _remove_bad_bands(self, data: np.ndarray, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
        """移除坏波段
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 移除坏波段后的数据
        """
        height, width, bands = data.shape
        
        # 获取坏波段列表
        bad_bands = self.noise_config.get('bad_bands', [])
        if not bad_bands and metadata and 'bad_bands' in metadata:
            bad_bands = metadata['bad_bands']
        elif not bad_bands:
            # 使用默认的坏波段（针对AVIRIS等传感器）
            bad_bands = [b for b in self.bad_bands_default if b < bands]
        
        if not bad_bands:
            return data
        
        # 创建好波段的掩码
        good_bands = [b for b in range(bands) if b not in bad_bands]
        
        if len(good_bands) == 0:
            logger.warning("All bands marked as bad, keeping original data")
            return data
        
        logger.info(f"Removing {len(bad_bands)} bad bands, keeping {len(good_bands)} good bands")
        
        # 返回好波段的数据
        return data[:, :, good_bands]
    
    def _apply_spectral_smoothing(self, data: np.ndarray) -> np.ndarray:
        """应用光谱平滑
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 平滑后的数据
        """
        method = self.noise_config.get('method', 'savgol')
        logger.info(f"Applying spectral smoothing: {method}")
        
        height, width, bands = data.shape
        smoothed_data = data.copy()
        
        if method == 'savgol':
            # Savitzky-Golay滤波
            window_size = self.noise_config.get('window_size', 5)
            polynomial_order = self.noise_config.get('polynomial_order', 2)
            
            # 确保window_size是奇数且小于波段数
            window_size = min(window_size, bands - 1)
            if window_size % 2 == 0:
                window_size -= 1
            window_size = max(window_size, 3)
            
            polynomial_order = min(polynomial_order, window_size - 1)
            
            for i in range(height):
                for j in range(width):
                    spectrum = data[i, j, :]
                    if not np.all(np.isnan(spectrum)):
                        try:
                            smoothed_spectrum = savgol_filter(
                                spectrum, window_size, polynomial_order
                            )
                            smoothed_data[i, j, :] = smoothed_spectrum
                        except Exception:
                            # 如果平滑失败，保持原值
                            pass
        
        elif method == 'gaussian':
            # 高斯滤波
            sigma = self.noise_config.get('sigma', 1.0)
            
            for i in range(height):
                for j in range(width):
                    spectrum = data[i, j, :]
                    if not np.all(np.isnan(spectrum)):
                        smoothed_spectrum = ndimage.gaussian_filter1d(spectrum, sigma)
                        smoothed_data[i, j, :] = smoothed_spectrum
        
        elif method == 'median':
            # 中值滤波
            kernel_size = self.noise_config.get('kernel_size', 3)
            
            for i in range(height):
                for j in range(width):
                    spectrum = data[i, j, :]
                    if not np.all(np.isnan(spectrum)):
                        smoothed_spectrum = medfilt(spectrum, kernel_size)
                        smoothed_data[i, j, :] = smoothed_spectrum
        
        elif method == 'moving_average':
            # 移动平均
            window_size = self.noise_config.get('window_size', 5)
            
            for i in range(height):
                for j in range(width):
                    spectrum = data[i, j, :]
                    if not np.all(np.isnan(spectrum)):
                        smoothed_spectrum = self._moving_average(spectrum, window_size)
                        smoothed_data[i, j, :] = smoothed_spectrum
        
        return smoothed_data
    
    def _remove_stripes(self, data: np.ndarray) -> np.ndarray:
        """去除条带噪声
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 去条带后的数据
        """
        logger.info("Removing stripes")
        
        height, width, bands = data.shape
        destriped_data = data.copy()
        
        method = self.noise_config.get('stripe_method', 'statistical')
        
        if method == 'statistical':
            # 统计去条带方法
            destriped_data = self._statistical_destripe(data)
        
        elif method == 'fourier':
            # 傅里叶变换去条带
            destriped_data = self._fourier_destripe(data)
        
        elif method == 'wavelet':
            # 小波去条带
            destriped_data = self._wavelet_destripe(data)
        
        return destriped_data
    
    def _statistical_destripe(self, data: np.ndarray) -> np.ndarray:
        """统计去条带
        
        基于统计分析去除推扫式传感器的条带噪声
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 去条带后的数据
        """
        height, width, bands = data.shape
        destriped_data = data.copy()
        
        # 检测垂直条带（列方向异常）
        for b in range(bands):
            band_data = data[:, :, b]
            
            # 计算每列的均值
            col_means = np.nanmean(band_data, axis=0)
            overall_mean = np.nanmean(col_means)
            
            # 检测异常列
            col_std = np.nanstd(col_means)
            threshold = 2 * col_std
            
            for c in range(width):
                if abs(col_means[c] - overall_mean) > threshold:
                    # 使用相邻列的均值替代
                    neighbor_cols = []
                    for offset in [-2, -1, 1, 2]:
                        neighbor_col = c + offset
                        if 0 <= neighbor_col < width:
                            neighbor_cols.append(col_means[neighbor_col])
                    
                    if neighbor_cols:
                        correction_factor = np.mean(neighbor_cols) / col_means[c] if col_means[c] != 0 else 1
                        destriped_data[:, c, b] *= correction_factor
        
        # 检测水平条带（行方向异常）
        for b in range(bands):
            band_data = destriped_data[:, :, b]
            
            # 计算每行的均值
            row_means = np.nanmean(band_data, axis=1)
            overall_mean = np.nanmean(row_means)
            
            # 检测异常行
            row_std = np.nanstd(row_means)
            threshold = 2 * row_std
            
            for r in range(height):
                if abs(row_means[r] - overall_mean) > threshold:
                    # 使用相邻行的均值替代
                    neighbor_rows = []
                    for offset in [-2, -1, 1, 2]:
                        neighbor_row = r + offset
                        if 0 <= neighbor_row < height:
                            neighbor_rows.append(row_means[neighbor_row])
                    
                    if neighbor_rows:
                        correction_factor = np.mean(neighbor_rows) / row_means[r] if row_means[r] != 0 else 1
                        destriped_data[r, :, b] *= correction_factor
        
        return destriped_data
    
    def _fourier_destripe(self, data: np.ndarray) -> np.ndarray:
        """傅里叶变换去条带
        
        在频域中识别和去除周期性条带噪声
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 去条带后的数据
        """
        height, width, bands = data.shape
        destriped_data = data.copy()
        
        for b in range(bands):
            band_data = data[:, :, b]
            
            # 傅里叶变换
            fft_data = np.fft.fft2(band_data)
            fft_shift = np.fft.fftshift(fft_data)
            
            # 创建滤波器去除特定频率的条带
            center_y, center_x = height // 2, width // 2
            
            # 垂直条带滤波（去除水平频率成分）
            filter_mask = np.ones((height, width), dtype=complex)
            stripe_width = 3  # 滤波器宽度
            
            # 去除中心垂直线附近的频率（对应垂直条带）
            filter_mask[:, center_x-stripe_width:center_x+stripe_width+1] *= 0.1
            
            # 水平条带滤波（去除垂直频率成分）
            filter_mask[center_y-stripe_width:center_y+stripe_width+1, :] *= 0.1
            
            # 应用滤波器
            filtered_fft = fft_shift * filter_mask
            
            # 反傅里叶变换
            ifft_shift = np.fft.ifftshift(filtered_fft)
            destriped_band = np.real(np.fft.ifft2(ifft_shift))
            
            destriped_data[:, :, b] = destriped_band
        
        return destriped_data
    
    def _wavelet_destripe(self, data: np.ndarray) -> np.ndarray:
        """小波去条带
        
        使用小波变换在不同尺度上去除条带噪声
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 去条带后的数据
        """
        # 小波去条带的简化实现
        # 实际应用中需要pywt库
        logger.warning("Wavelet destriping not implemented, using statistical method")
        return self._statistical_destripe(data)
    
    def _remove_spikes(self, data: np.ndarray) -> np.ndarray:
        """去除椒盐噪声和尖峰噪声
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 去除尖峰后的数据
        """
        logger.info("Removing spikes and salt-and-pepper noise")
        
        height, width, bands = data.shape
        despike_data = data.copy()
        
        # 光谱尖峰检测和去除
        for i in range(height):
            for j in range(width):
                spectrum = data[i, j, :]
                cleaned_spectrum = self._detect_and_remove_spectral_spikes(spectrum)
                despike_data[i, j, :] = cleaned_spectrum
        
        # 空间椒盐噪声去除
        if HAS_CV2:
            for b in range(bands):
                band_data = despike_data[:, :, b].astype(np.float32)
                # 使用形态学开运算去除椒盐噪声
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned_band = cv2.morphologyEx(band_data, cv2.MORPH_OPEN, kernel)
                despike_data[:, :, b] = cleaned_band
        else:
            # 使用中值滤波代替
            for b in range(bands):
                band_data = despike_data[:, :, b]
                cleaned_band = ndimage.median_filter(band_data, size=3)
                despike_data[:, :, b] = cleaned_band
        
        return despike_data
    
    def _detect_and_remove_spectral_spikes(self, spectrum: np.ndarray) -> np.ndarray:
        """检测和去除光谱尖峰
        
        Args:
            spectrum: 输入光谱
            
        Returns:
            np.ndarray: 去除尖峰后的光谱
        """
        if len(spectrum) < 5:
            return spectrum
        
        cleaned_spectrum = spectrum.copy()
        
        # 计算一阶导数
        diff1 = np.diff(spectrum)
        
        # 检测异常大的跳跃
        diff_threshold = np.std(diff1) * 3
        
        spike_indices = []
        for i in range(len(diff1) - 1):
            # 检测正向尖峰后跟负向跳跃的模式
            if diff1[i] > diff_threshold and diff1[i+1] < -diff_threshold:
                spike_indices.append(i + 1)
            # 检测负向尖峰后跟正向跳跃的模式
            elif diff1[i] < -diff_threshold and diff1[i+1] > diff_threshold:
                spike_indices.append(i + 1)
        
        # 使用线性插值替换尖峰
        for spike_idx in spike_indices:
            if 0 < spike_idx < len(spectrum) - 1:
                # 使用相邻点的均值
                cleaned_spectrum[spike_idx] = (spectrum[spike_idx-1] + spectrum[spike_idx+1]) / 2
        
        return cleaned_spectrum
    
    def _apply_spatial_denoising(self, data: np.ndarray) -> np.ndarray:
        """应用空间去噪
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: 空间去噪后的数据
        """
        logger.info("Applying spatial denoising")
        
        height, width, bands = data.shape
        denoised_data = data.copy()
        
        method = self.noise_config.get('spatial_method', 'bilateral')
        
        if method == 'bilateral' and HAS_CV2:
            # 双边滤波
            d = 9
            sigma_color = 75
            sigma_space = 75
            
            for b in range(bands):
                band_data = data[:, :, b].astype(np.float32)
                # 归一化到0-255范围
                band_min, band_max = np.min(band_data), np.max(band_data)
                if band_max > band_min:
                    band_normalized = ((band_data - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                    denoised_band = cv2.bilateralFilter(band_normalized, d, sigma_color, sigma_space)
                    # 反归一化
                    denoised_data[:, :, b] = denoised_band.astype(np.float32) / 255 * (band_max - band_min) + band_min
                else:
                    denoised_data[:, :, b] = band_data
        
        elif method == 'non_local_means' and HAS_CV2:
            # 非局部均值去噪
            h = 10
            template_window_size = 7
            search_window_size = 21
            
            for b in range(bands):
                band_data = data[:, :, b].astype(np.float32)
                # 归一化
                band_min, band_max = np.min(band_data), np.max(band_data)
                if band_max > band_min:
                    band_normalized = ((band_data - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                    denoised_band = cv2.fastNlMeansDenoising(
                        band_normalized, None, h, template_window_size, search_window_size
                    )
                    denoised_data[:, :, b] = denoised_band.astype(np.float32) / 255 * (band_max - band_min) + band_min
                else:
                    denoised_data[:, :, b] = band_data
        
        elif method == 'gaussian':
            # 高斯滤波
            sigma = 1.0
            for b in range(bands):
                band_data = data[:, :, b]
                denoised_band = ndimage.gaussian_filter(band_data, sigma)
                denoised_data[:, :, b] = denoised_band
        
        elif method == 'wiener':
            # 维纳滤波
            for b in range(bands):
                band_data = data[:, :, b]
                denoised_band = wiener(band_data, (5, 5))
                denoised_data[:, :, b] = denoised_band
        
        return denoised_data
    
    def _apply_pca_denoising(self, data: np.ndarray) -> np.ndarray:
        """应用PCA去噪
        
        Args:
            data: 输入数据
            
        Returns:
            np.ndarray: PCA去噪后的数据
        """
        logger.info("Applying PCA denoising")
        
        height, width, bands = data.shape
        
        # 重塑数据为二维 (pixels, bands)
        pixels = data.reshape(-1, bands)
        
        # 移除无效像素
        valid_mask = ~np.any(np.isnan(pixels), axis=1)
        valid_pixels = pixels[valid_mask]
        
        if len(valid_pixels) == 0:
            return data
        
        # 标准化
        scaler = StandardScaler()
        scaled_pixels = scaler.fit_transform(valid_pixels)
        
        # PCA
        n_components = min(self.noise_config.get('pca_components', 50), bands, len(valid_pixels))
        pca = PCA(n_components=n_components)
        
        # 变换和反变换
        transformed = pca.fit_transform(scaled_pixels)
        reconstructed_scaled = pca.inverse_transform(transformed)
        
        # 反标准化
        reconstructed = scaler.inverse_transform(reconstructed_scaled)
        
        # 重建完整数据
        denoised_pixels = pixels.copy()
        denoised_pixels[valid_mask] = reconstructed
        
        denoised_data = denoised_pixels.reshape(height, width, bands)
        
        logger.info(f"PCA denoising: retained {n_components}/{bands} components")
        
        return denoised_data
    
    def _moving_average(self, spectrum: np.ndarray, window_size: int) -> np.ndarray:
        """移动平均平滑
        
        Args:
            spectrum: 输入光谱
            window_size: 窗口大小
            
        Returns:
            np.ndarray: 平滑后的光谱
        """
        if window_size >= len(spectrum):
            return spectrum
        
        # 使用卷积实现移动平均
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(spectrum, kernel, mode='same')
        
        # 处理边界效应
        half_window = window_size // 2
        for i in range(half_window):
            smoothed[i] = np.mean(spectrum[:i+half_window+1])
            smoothed[-(i+1)] = np.mean(spectrum[-(i+half_window+1):])
        
        return smoothed
    
    def estimate_noise_level(self, data: np.ndarray) -> Dict[str, float]:
        """估算噪声水平
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, float]: 噪声统计信息
        """
        height, width, bands = data.shape
        
        # 选择均匀区域估算噪声
        center_h, center_w = height // 4, width // 4
        center_region = data[center_h:3*center_h, center_w:3*center_w, :]
        
        # 计算局部标准差作为噪声估计
        noise_estimates = []
        
        for b in range(bands):
            band_data = center_region[:, :, b]
            
            # 使用Laplacian算子估算噪声
            laplacian = ndimage.laplace(band_data)
            noise_estimate = np.std(laplacian) / 6  # 经验公式
            
            noise_estimates.append(noise_estimate)
        
        return {
            'mean_noise_level': float(np.mean(noise_estimates)),
            'max_noise_level': float(np.max(noise_estimates)),
            'min_noise_level': float(np.min(noise_estimates)),
            'noise_variation': float(np.std(noise_estimates)),
            'snr_estimate': float(np.mean(data) / np.mean(noise_estimates)) if np.mean(noise_estimates) > 0 else float('inf')
        }
    
    def validate_denoising(self, original_data: np.ndarray,
                          denoised_data: np.ndarray) -> Dict[str, Any]:
        """验证去噪效果
        
        Args:
            original_data: 原始数据
            denoised_data: 去噪后数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'shape_preserved': original_data.shape == denoised_data.shape,
            'data_range_change': {},
            'noise_reduction': {},
            'signal_preservation': {},
            'issues': []
        }
        
        # 检查数据范围变化
        orig_min, orig_max = np.min(original_data), np.max(original_data)
        denoise_min, denoise_max = np.min(denoised_data), np.max(denoised_data)
        
        validation_results['data_range_change'] = {
            'original_range': (float(orig_min), float(orig_max)),
            'denoised_range': (float(denoise_min), float(denoise_max)),
            'range_change_ratio': float(abs((denoise_max - denoise_min) - (orig_max - orig_min)) / (orig_max - orig_min))
        }
        
        # 估算噪声减少
        orig_noise = self.estimate_noise_level(original_data)
        denoise_noise = self.estimate_noise_level(denoised_data)
        
        validation_results['noise_reduction'] = {
            'original_noise': orig_noise['mean_noise_level'],
            'denoised_noise': denoise_noise['mean_noise_level'],
            'noise_reduction_ratio': float(1 - denoise_noise['mean_noise_level'] / orig_noise['mean_noise_level']) if orig_noise['mean_noise_level'] > 0 else 0
        }
        
        # 信号保持评估
        signal_correlation = np.corrcoef(original_data.flatten(), denoised_data.flatten())[0, 1]
        validation_results['signal_preservation']['correlation'] = float(signal_correlation)
        
        # 检查潜在问题
        if validation_results['data_range_change']['range_change_ratio'] > 0.2:
            validation_results['issues'].append("Significant data range change")
        
        if signal_correlation < 0.9:
            validation_results['issues'].append("Low signal preservation")
        
        if validation_results['noise_reduction']['noise_reduction_ratio'] < 0.1:
            validation_results['issues'].append("Limited noise reduction achieved")
        
        # 评估去噪质量
        if len(validation_results['issues']) == 0:
            validation_results['quality'] = 'Excellent'
        elif len(validation_results['issues']) <= 1:
            validation_results['quality'] = 'Good'
        else:
            validation_results['quality'] = 'Needs Review'
        
        return validation_results
    
    def apply_custom_filter(self, data: np.ndarray, filter_func: callable) -> np.ndarray:
        """应用自定义滤波函数
        
        Args:
            data: 输入数据
            filter_func: 自定义滤波函数
            
        Returns:
            np.ndarray: 滤波后的数据
        """
        height, width, bands = data.shape
        filtered_data = np.zeros_like(data)
        
        for b in range(bands):
            filtered_data[:, :, b] = filter_func(data[:, :, b])
        
        return filtered_data
    
    def get_denoising_recommendations(self, data: np.ndarray) -> Dict[str, Any]:
        """获取去噪建议
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 去噪建议
        """
        noise_stats = self.estimate_noise_level(data)
        
        recommendations = {
            'noise_level': 'low' if noise_stats['mean_noise_level'] < 0.01 else 'medium' if noise_stats['mean_noise_level'] < 0.05 else 'high',
            'recommended_methods': [],
            'priority_order': [],
            'parameters': {}
        }
        
        if noise_stats['mean_noise_level'] > 0.05:  # 高噪声
            recommendations['recommended_methods'].extend(['spectral_smoothing', 'spatial_denoising', 'pca_denoising'])
            recommendations['priority_order'] = ['bad_band_removal', 'spectral_smoothing', 'stripe_removal', 'spatial_denoising']
            recommendations['parameters']['savgol_window'] = 7
            recommendations['parameters']['pca_components'] = 30
        
        elif noise_stats['mean_noise_level'] > 0.01:  # 中等噪声
            recommendations['recommended_methods'].extend(['spectral_smoothing', 'stripe_removal'])
            recommendations['priority_order'] = ['bad_band_removal', 'spectral_smoothing', 'spike_removal']
            recommendations['parameters']['savgol_window'] = 5
        
        else:  # 低噪声
            recommendations['recommended_methods'].extend(['spike_removal'])
            recommendations['priority_order'] = ['bad_band_removal', 'spike_removal']
            recommendations['parameters']['savgol_window'] = 3
        
        return recommendations