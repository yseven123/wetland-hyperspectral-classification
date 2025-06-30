#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强器
Data Augmentation

提供高光谱遥感数据的数据增强功能，提高模型泛化能力

作者: Wetland Research Team
"""

import random
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import logging

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from sklearn.utils import resample

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..config import Config

logger = logging.getLogger(__name__)


class DataAugmentation:
    """高光谱数据增强器
    
    支持多种增强技术：
    - 几何变换：旋转、翻转、缩放、平移
    - 光谱变换：噪声添加、亮度调整、对比度调整、光谱偏移
    - 混合增强：Mixup、CutMix
    - 空间增强：裁剪、填充
    """
    
    def __init__(self, config: Config):
        """初始化数据增强器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.augmentation_config = config.get('data_augmentation', {})
        
        # 设置随机种子
        self.random_state = config.get('runtime.random_seed', 42)
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        logger.info("DataAugmentation initialized")
    
    def augment_spectra(self, spectra: np.ndarray, labels: np.ndarray,
                       augmentation_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """增强光谱数据
        
        Args:
            spectra: 输入光谱数据 (N, B)
            labels: 标签数据 (N,)
            augmentation_factor: 增强倍数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (增强后的光谱, 增强后的标签)
        """
        logger.info(f"Augmenting spectra data with factor {augmentation_factor}")
        
        original_size = len(spectra)
        target_size = int(original_size * augmentation_factor)
        augmented_size = target_size - original_size
        
        if augmented_size <= 0:
            return spectra, labels
        
        augmented_spectra = []
        augmented_labels = []
        
        for _ in range(augmented_size):
            # 随机选择一个样本
            idx = np.random.randint(0, original_size)
            spectrum = spectra[idx].copy()
            label = labels[idx]
            
            # 应用光谱增强
            augmented_spectrum = self._apply_spectral_augmentations(spectrum)
            
            augmented_spectra.append(augmented_spectrum)
            augmented_labels.append(label)
        
        # 合并原始数据和增强数据
        combined_spectra = np.vstack([spectra, np.array(augmented_spectra)])
        combined_labels = np.concatenate([labels, np.array(augmented_labels)])
        
        logger.info(f"Augmented data from {original_size} to {len(combined_spectra)} samples")
        
        return combined_spectra, combined_labels
    
    def augment_hyperspectral_patches(self, patches: np.ndarray, labels: np.ndarray,
                                    augmentation_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """增强高光谱图像块
        
        Args:
            patches: 输入图像块 (N, H, W, B)
            labels: 标签数据 (N,)
            augmentation_factor: 增强倍数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (增强后的图像块, 增强后的标签)
        """
        logger.info(f"Augmenting hyperspectral patches with factor {augmentation_factor}")
        
        original_size = len(patches)
        target_size = int(original_size * augmentation_factor)
        augmented_size = target_size - original_size
        
        if augmented_size <= 0:
            return patches, labels
        
        augmented_patches = []
        augmented_labels = []
        
        for _ in range(augmented_size):
            # 随机选择一个样本
            idx = np.random.randint(0, original_size)
            patch = patches[idx].copy()
            label = labels[idx]
            
            # 应用几何和光谱增强
            augmented_patch = self._apply_spatial_augmentations(patch)
            augmented_patch = self._apply_spectral_augmentations_to_patch(augmented_patch)
            
            augmented_patches.append(augmented_patch)
            augmented_labels.append(label)
        
        # 合并原始数据和增强数据
        combined_patches = np.vstack([patches, np.array(augmented_patches)])
        combined_labels = np.concatenate([labels, np.array(augmented_labels)])
        
        logger.info(f"Augmented patches from {original_size} to {len(combined_patches)} samples")
        
        return combined_patches, combined_labels
    
    def _apply_spectral_augmentations(self, spectrum: np.ndarray) -> np.ndarray:
        """应用光谱增强技术
        
        Args:
            spectrum: 输入光谱 (B,)
            
        Returns:
            np.ndarray: 增强后的光谱
        """
        augmented_spectrum = spectrum.copy()
        
        # 1. 添加高斯噪声
        if self.augmentation_config.get('spectral_transforms', {}).get('noise_addition', {}).get('enabled', True):
            noise_level = self.augmentation_config.get('spectral_transforms', {}).get('noise_addition', {}).get('noise_level', 0.01)
            augmented_spectrum = self._add_gaussian_noise(augmented_spectrum, noise_level)
        
        # 2. 亮度调整
        if self.augmentation_config.get('spectral_transforms', {}).get('brightness_adjustment', {}).get('enabled', True):
            factor_range = self.augmentation_config.get('spectral_transforms', {}).get('brightness_adjustment', {}).get('factor_range', [0.8, 1.2])
            augmented_spectrum = self._adjust_brightness(augmented_spectrum, factor_range)
        
        # 3. 对比度调整
        if self.augmentation_config.get('spectral_transforms', {}).get('contrast_adjustment', {}).get('enabled', True):
            factor_range = self.augmentation_config.get('spectral_transforms', {}).get('contrast_adjustment', {}).get('factor_range', [0.9, 1.1])
            augmented_spectrum = self._adjust_contrast(augmented_spectrum, factor_range)
        
        # 4. 光谱偏移
        if self.augmentation_config.get('spectral_transforms', {}).get('spectral_shift', {}).get('enabled', True):
            shift_range = self.augmentation_config.get('spectral_transforms', {}).get('spectral_shift', {}).get('shift_range', [-2, 2])
            augmented_spectrum = self._apply_spectral_shift(augmented_spectrum, shift_range)
        
        # 5. 光谱平滑
        if np.random.random() < 0.3:  # 30%概率应用平滑
            augmented_spectrum = self._apply_spectral_smoothing(augmented_spectrum)
        
        return augmented_spectrum
    
    def _apply_spatial_augmentations(self, patch: np.ndarray) -> np.ndarray:
        """应用空间增强技术
        
        Args:
            patch: 输入图像块 (H, W, B)
            
        Returns:
            np.ndarray: 增强后的图像块
        """
        augmented_patch = patch.copy()
        
        # 1. 旋转
        if self.augmentation_config.get('geometric_transforms', {}).get('rotation', {}).get('enabled', True):
            angles = self.augmentation_config.get('geometric_transforms', {}).get('rotation', {}).get('angles', [90, 180, 270])
            if angles and np.random.random() < 0.5:
                angle = np.random.choice(angles)
                augmented_patch = self._rotate_patch(augmented_patch, angle)
        
        # 2. 翻转
        if self.augmentation_config.get('geometric_transforms', {}).get('flipping', {}).get('horizontal', True):
            if np.random.random() < 0.5:
                augmented_patch = np.fliplr(augmented_patch)
        
        if self.augmentation_config.get('geometric_transforms', {}).get('flipping', {}).get('vertical', True):
            if np.random.random() < 0.5:
                augmented_patch = np.flipud(augmented_patch)
        
        # 3. 缩放
        if self.augmentation_config.get('geometric_transforms', {}).get('scaling', {}).get('enabled', True):
            factors = self.augmentation_config.get('geometric_transforms', {}).get('scaling', {}).get('factors', [0.8, 1.2])
            if factors and np.random.random() < 0.3:
                factor = np.random.uniform(factors[0], factors[1])
                augmented_patch = self._scale_patch(augmented_patch, factor)
        
        # 4. 平移
        if self.augmentation_config.get('geometric_transforms', {}).get('translation', {}).get('enabled', True):
            pixel_range = self.augmentation_config.get('geometric_transforms', {}).get('translation', {}).get('pixels', [-2, 2])
            if pixel_range and np.random.random() < 0.3:
                dx = np.random.randint(pixel_range[0], pixel_range[1] + 1)
                dy = np.random.randint(pixel_range[0], pixel_range[1] + 1)
                augmented_patch = self._translate_patch(augmented_patch, dx, dy)
        
        return augmented_patch
    
    def _apply_spectral_augmentations_to_patch(self, patch: np.ndarray) -> np.ndarray:
        """对图像块应用光谱增强"""
        H, W, B = patch.shape
        augmented_patch = patch.copy()
        
        # 对每个像素应用光谱增强
        for i in range(H):
            for j in range(W):
                spectrum = augmented_patch[i, j, :]
                augmented_spectrum = self._apply_spectral_augmentations(spectrum)
                augmented_patch[i, j, :] = augmented_spectrum
        
        return augmented_patch
    
    def _add_gaussian_noise(self, spectrum: np.ndarray, noise_level: float) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level * np.std(spectrum), spectrum.shape)
        return spectrum + noise
    
    def _adjust_brightness(self, spectrum: np.ndarray, factor_range: List[float]) -> np.ndarray:
        """调整亮度"""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        return spectrum * factor
    
    def _adjust_contrast(self, spectrum: np.ndarray, factor_range: List[float]) -> np.ndarray:
        """调整对比度"""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        mean_val = np.mean(spectrum)
        return (spectrum - mean_val) * factor + mean_val
    
    def _apply_spectral_shift(self, spectrum: np.ndarray, shift_range: List[int]) -> np.ndarray:
        """应用光谱偏移"""
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        
        if shift == 0:
            return spectrum
        
        # 创建插值函数
        original_indices = np.arange(len(spectrum))
        shifted_indices = original_indices + shift
        
        # 使用线性插值
        f = interp1d(original_indices, spectrum, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        
        # 应用偏移
        shifted_spectrum = f(shifted_indices)
        
        return shifted_spectrum
    
    def _apply_spectral_smoothing(self, spectrum: np.ndarray) -> np.ndarray:
        """应用光谱平滑"""
        # 使用高斯滤波器进行平滑
        sigma = np.random.uniform(0.5, 1.5)
        return ndimage.gaussian_filter1d(spectrum, sigma)
    
    def _rotate_patch(self, patch: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像块"""
        if angle == 90:
            return np.rot90(patch, k=1)
        elif angle == 180:
            return np.rot90(patch, k=2)
        elif angle == 270:
            return np.rot90(patch, k=3)
        else:
            # 对于任意角度，使用scipy的旋转
            H, W, B = patch.shape
            rotated_patch = np.zeros_like(patch)
            
            for b in range(B):
                rotated_patch[:, :, b] = ndimage.rotate(
                    patch[:, :, b], angle, reshape=False, mode='reflect'
                )
            
            return rotated_patch
    
    def _scale_patch(self, patch: np.ndarray, factor: float) -> np.ndarray:
        """缩放图像块"""
        if not HAS_CV2:
            # 如果没有OpenCV，使用scipy进行缩放
            H, W, B = patch.shape
            new_H, new_W = int(H * factor), int(W * factor)
            
            scaled_patch = np.zeros((new_H, new_W, B), dtype=patch.dtype)
            for b in range(B):
                scaled_patch[:, :, b] = ndimage.zoom(
                    patch[:, :, b], factor, order=1
                )
            
            # 如果缩放后尺寸不同，需要裁剪或填充到原始尺寸
            if new_H != H or new_W != W:
                if new_H >= H and new_W >= W:
                    # 裁剪
                    start_h = (new_H - H) // 2
                    start_w = (new_W - W) // 2
                    scaled_patch = scaled_patch[start_h:start_h+H, start_w:start_w+W, :]
                else:
                    # 填充
                    padded_patch = np.zeros((H, W, B), dtype=patch.dtype)
                    start_h = (H - new_H) // 2
                    start_w = (W - new_W) // 2
                    padded_patch[start_h:start_h+new_H, start_w:start_w+new_W, :] = scaled_patch
                    scaled_patch = padded_patch
            
            return scaled_patch
        else:
            # 使用OpenCV进行缩放
            H, W, B = patch.shape
            
            # 对每个波段进行缩放
            scaled_bands = []
            for b in range(B):
                scaled_band = cv2.resize(patch[:, :, b], (W, H), interpolation=cv2.INTER_LINEAR)
                scaled_bands.append(scaled_band)
            
            return np.stack(scaled_bands, axis=2)
    
    def _translate_patch(self, patch: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """平移图像块"""
        H, W, B = patch.shape
        translated_patch = np.zeros_like(patch)
        
        # 计算有效区域
        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)
        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # 复制有效区域
        translated_patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
            patch[src_y_start:src_y_end, src_x_start:src_x_end, :]
        
        return translated_patch
    
    def mixup(self, spectra1: np.ndarray, labels1: np.ndarray,
             spectra2: np.ndarray, labels2: np.ndarray,
             alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Mixup数据增强
        
        Args:
            spectra1: 第一组光谱数据
            labels1: 第一组标签
            spectra2: 第二组光谱数据
            labels2: 第二组标签
            alpha: Beta分布参数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (混合后的光谱, 混合权重)
        """
        if len(spectra1) != len(spectra2):
            min_len = min(len(spectra1), len(spectra2))
            spectra1 = spectra1[:min_len]
            labels1 = labels1[:min_len]
            spectra2 = spectra2[:min_len]
            labels2 = labels2[:min_len]
        
        # 生成混合权重
        lam = np.random.beta(alpha, alpha, size=(len(spectra1), 1))
        
        # 混合光谱
        mixed_spectra = lam * spectra1 + (1 - lam) * spectra2
        
        # 混合标签权重
        mixed_labels = np.column_stack([lam.flatten(), 1 - lam.flatten()])
        
        return mixed_spectra, mixed_labels
    
    def cutmix(self, patches1: np.ndarray, labels1: np.ndarray,
              patches2: np.ndarray, labels2: np.ndarray,
              alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """CutMix数据增强
        
        Args:
            patches1: 第一组图像块
            labels1: 第一组标签
            patches2: 第二组图像块
            labels2: 第二组标签
            alpha: Beta分布参数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (混合后的图像块, 混合权重)
        """
        if len(patches1) != len(patches2):
            min_len = min(len(patches1), len(patches2))
            patches1 = patches1[:min_len]
            labels1 = labels1[:min_len]
            patches2 = patches2[:min_len]
            labels2 = labels2[:min_len]
        
        N, H, W, B = patches1.shape
        mixed_patches = patches1.copy()
        mixed_labels = []
        
        for i in range(N):
            # 生成随机区域
            lam = np.random.beta(alpha, alpha)
            cut_ratio = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_ratio)
            cut_h = int(H * cut_ratio)
            
            # 随机选择切割位置
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            # 混合图像
            mixed_patches[i, bby1:bby2, bbx1:bbx2, :] = patches2[i, bby1:bby2, bbx1:bbx2, :]
            
            # 计算实际混合比例
            cut_area = (bbx2 - bbx1) * (bby2 - bby1)
            total_area = H * W
            actual_lam = 1 - cut_area / total_area
            
            mixed_labels.append([actual_lam, 1 - actual_lam])
        
        return mixed_patches, np.array(mixed_labels)
    
    def balance_dataset(self, spectra: np.ndarray, labels: np.ndarray,
                       strategy: str = 'oversample') -> Tuple[np.ndarray, np.ndarray]:
        """平衡数据集
        
        Args:
            spectra: 光谱数据
            labels: 标签数据
            strategy: 平衡策略 ('oversample', 'undersample', 'augment')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (平衡后的光谱, 平衡后的标签)
        """
        logger.info(f"Balancing dataset using {strategy} strategy")
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if strategy == 'oversample':
            # 过采样到最大类别的数量
            max_count = np.max(counts)
            balanced_spectra = []
            balanced_labels = []
            
            for label in unique_labels:
                label_mask = labels == label
                label_spectra = spectra[label_mask]
                label_labels = labels[label_mask]
                
                # 重采样到目标数量
                resampled_spectra, resampled_labels = resample(
                    label_spectra, label_labels,
                    n_samples=max_count,
                    random_state=self.random_state,
                    replace=True
                )
                
                balanced_spectra.append(resampled_spectra)
                balanced_labels.append(resampled_labels)
            
            balanced_spectra = np.vstack(balanced_spectra)
            balanced_labels = np.concatenate(balanced_labels)
        
        elif strategy == 'undersample':
            # 欠采样到最小类别的数量
            min_count = np.min(counts)
            balanced_spectra = []
            balanced_labels = []
            
            for label in unique_labels:
                label_mask = labels == label
                label_spectra = spectra[label_mask]
                label_labels = labels[label_mask]
                
                # 重采样到目标数量
                resampled_spectra, resampled_labels = resample(
                    label_spectra, label_labels,
                    n_samples=min_count,
                    random_state=self.random_state,
                    replace=False
                )
                
                balanced_spectra.append(resampled_spectra)
                balanced_labels.append(resampled_labels)
            
            balanced_spectra = np.vstack(balanced_spectra)
            balanced_labels = np.concatenate(balanced_labels)
        
        elif strategy == 'augment':
            # 使用数据增强平衡数据集
            max_count = np.max(counts)
            balanced_spectra = [spectra]
            balanced_labels = [labels]
            
            for label in unique_labels:
                label_mask = labels == label
                label_spectra = spectra[label_mask]
                current_count = np.sum(label_mask)
                
                if current_count < max_count:
                    # 需要增强的样本数
                    needed_samples = max_count - current_count
                    augmentation_factor = 1 + needed_samples / current_count
                    
                    # 应用增强
                    augmented_spectra, augmented_labels = self.augment_spectra(
                        label_spectra, 
                        np.full(current_count, label),
                        augmentation_factor
                    )
                    
                    # 只取增强的部分
                    extra_spectra = augmented_spectra[current_count:]
                    extra_labels = augmented_labels[current_count:]
                    
                    balanced_spectra.append(extra_spectra)
                    balanced_labels.append(extra_labels)
            
            balanced_spectra = np.vstack(balanced_spectra)
            balanced_labels = np.concatenate(balanced_labels)
        
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy}")
        
        logger.info(f"Dataset balanced from {len(spectra)} to {len(balanced_spectra)} samples")
        
        return balanced_spectra, balanced_labels
    
    def create_augmentation_pipeline(self, transforms: List[str]) -> Callable:
        """创建增强流水线
        
        Args:
            transforms: 变换名称列表
            
        Returns:
            Callable: 增强函数
        """
        def augmentation_pipeline(data: np.ndarray, labels: np.ndarray = None):
            augmented_data = data.copy()
            augmented_labels = labels.copy() if labels is not None else None
            
            for transform in transforms:
                if transform == 'spectral_noise':
                    if augmented_data.ndim == 2:  # 光谱数据
                        for i in range(len(augmented_data)):
                            augmented_data[i] = self._add_gaussian_noise(augmented_data[i], 0.01)
                
                elif transform == 'rotation':
                    if augmented_data.ndim == 4:  # 图像块数据
                        for i in range(len(augmented_data)):
                            if np.random.random() < 0.5:
                                angle = np.random.choice([90, 180, 270])
                                augmented_data[i] = self._rotate_patch(augmented_data[i], angle)
                
                elif transform == 'flip':
                    if augmented_data.ndim == 4:  # 图像块数据
                        for i in range(len(augmented_data)):
                            if np.random.random() < 0.5:
                                augmented_data[i] = np.fliplr(augmented_data[i])
                            if np.random.random() < 0.5:
                                augmented_data[i] = np.flipud(augmented_data[i])
                
                # 可以添加更多变换...
            
            return augmented_data, augmented_labels
        
        return augmentation_pipeline
    
    def get_augmentation_statistics(self, original_data: np.ndarray,
                                  augmented_data: np.ndarray) -> Dict[str, float]:
        """获取增强统计信息
        
        Args:
            original_data: 原始数据
            augmented_data: 增强后数据
            
        Returns:
            Dict[str, float]: 统计信息
        """
        stats = {
            'original_size': len(original_data),
            'augmented_size': len(augmented_data),
            'augmentation_ratio': len(augmented_data) / len(original_data),
            'original_mean': float(np.mean(original_data)),
            'augmented_mean': float(np.mean(augmented_data)),
            'original_std': float(np.std(original_data)),
            'augmented_std': float(np.std(augmented_data)),
            'mean_difference': float(abs(np.mean(augmented_data) - np.mean(original_data))),
            'std_difference': float(abs(np.std(augmented_data) - np.std(original_data))),
        }
        
        return stats