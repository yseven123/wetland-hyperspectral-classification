#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
几何校正处理器
Geometric Correction

消除几何畸变，将影像投影到地理坐标系统

作者: Wetland Research Team
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_bounds, from_gcps
    from rasterio.control import GroundControlPoint as GCP
    import pyproj
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False
    warnings.warn("Geographic libraries not available. Geometric correction will be limited.")

from ..config import Config

logger = logging.getLogger(__name__)


class GeometricCorrector:
    """几何校正处理器
    
    支持的几何校正方法：
    - 控制点校正 (Ground Control Points)
    - 多项式变换
    - 有理函数模型 (RPC)
    - 地形校正 (DEM based)
    - 投影变换
    """
    
    def __init__(self, config: Config):
        """初始化几何校正处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.geometric_config = config.get('preprocessing.geometric', {})
        
        # 支持的重采样方法
        self.resampling_methods = {
            'nearest': Resampling.nearest if HAS_GEO_LIBS else 'nearest',
            'bilinear': Resampling.bilinear if HAS_GEO_LIBS else 'bilinear',
            'cubic': Resampling.cubic if HAS_GEO_LIBS else 'cubic',
            'cubic_spline': Resampling.cubic_spline if HAS_GEO_LIBS else 'cubic',
            'lanczos': Resampling.lanczos if HAS_GEO_LIBS else 'cubic',
        }
        
        logger.info("GeometricCorrector initialized")
    
    def correct(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """执行几何校正
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            metadata: 元数据信息
            
        Returns:
            np.ndarray: 几何校正后的数据
        """
        if not self.geometric_config.get('enabled', True):
            logger.info("Geometric correction disabled")
            return data
        
        logger.info("Applying geometric correction")
        
        # 检查是否有足够的地理信息
        if not self._has_geometric_info(metadata):
            logger.warning("Insufficient geometric information, applying basic correction")
            return self._basic_geometric_correction(data, metadata)
        
        # 根据可用信息选择校正方法
        if 'gcps' in metadata and len(metadata['gcps']) >= 3:
            return self._gcp_based_correction(data, metadata)
        elif 'transform' in metadata and 'crs' in metadata:
            return self._projection_based_correction(data, metadata)
        elif 'rpc' in metadata:
            return self._rpc_based_correction(data, metadata)
        else:
            return self._polynomial_correction(data, metadata)
    
    def _basic_geometric_correction(self, data: np.ndarray, 
                                  metadata: Dict[str, Any]) -> np.ndarray:
        """基础几何校正
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 校正后的数据
        """
        logger.info("Applying basic geometric correction")
        
        # 目标像素大小
        target_pixel_size = self.geometric_config.get('pixel_size', [30, 30])
        
        # 如果数据已经是目标分辨率，直接返回
        current_shape = data.shape[:2]
        
        # 简单的重采样到目标分辨率
        if 'pixel_size' in metadata:
            current_pixel_size = metadata['pixel_size']
            if isinstance(current_pixel_size, (list, tuple)) and len(current_pixel_size) >= 2:
                scale_x = current_pixel_size[0] / target_pixel_size[0]
                scale_y = current_pixel_size[1] / target_pixel_size[1]
                
                if abs(scale_x - 1.0) > 0.1 or abs(scale_y - 1.0) > 0.1:
                    # 需要重采样
                    return self._resample_data(data, scale_x, scale_y)
        
        return data
    
    def _gcp_based_correction(self, data: np.ndarray, 
                            metadata: Dict[str, Any]) -> np.ndarray:
        """基于控制点的几何校正
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 校正后的数据
        """
        if not HAS_GEO_LIBS:
            logger.warning("Geographic libraries not available for GCP correction")
            return self._polynomial_correction(data, metadata)
        
        logger.info("Applying GCP-based geometric correction")
        
        gcps = metadata['gcps']
        src_crs = metadata.get('src_crs', 'EPSG:4326')
        target_crs = self.geometric_config.get('target_crs', 'EPSG:4326')
        
        height, width, bands = data.shape
        
        try:
            # 创建变换矩阵
            transform = from_gcps(gcps)
            
            # 计算目标范围和分辨率
            pixel_size = self.geometric_config.get('pixel_size', [30, 30])
            
            # 计算地理范围
            corners = [
                (0, 0), (width, 0), (width, height), (0, height)
            ]
            
            geo_corners = []
            for col, row in corners:
                x, y = transform * (col, row)
                geo_corners.append((x, y))
            
            min_x = min(corner[0] for corner in geo_corners)
            max_x = max(corner[0] for corner in geo_corners)
            min_y = min(corner[1] for corner in geo_corners)
            max_y = max(corner[1] for corner in geo_corners)
            
            # 计算输出尺寸
            out_width = int((max_x - min_x) / pixel_size[0])
            out_height = int((max_y - min_y) / pixel_size[1])
            
            # 创建目标变换
            dst_transform = from_bounds(min_x, min_y, max_x, max_y, out_width, out_height)
            
            # 重投影每个波段
            corrected_data = np.zeros((out_height, out_width, bands), dtype=data.dtype)
            
            resampling_method = self.resampling_methods.get(
                self.geometric_config.get('resampling', 'bilinear'),
                Resampling.bilinear
            )
            
            for b in range(bands):
                reproject(
                    source=data[:, :, b],
                    destination=corrected_data[:, :, b],
                    src_transform=transform,
                    dst_transform=dst_transform,
                    src_crs=src_crs,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )
            
            logger.info(f"GCP correction: {data.shape} -> {corrected_data.shape}")
            return corrected_data
            
        except Exception as e:
            logger.error(f"GCP correction failed: {e}")
            return self._polynomial_correction(data, metadata)
    
    def _projection_based_correction(self, data: np.ndarray,
                                   metadata: Dict[str, Any]) -> np.ndarray:
        """基于投影的几何校正
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 校正后的数据
        """
        if not HAS_GEO_LIBS:
            logger.warning("Geographic libraries not available for projection correction")
            return data
        
        logger.info("Applying projection-based geometric correction")
        
        src_transform = metadata['transform']
        src_crs = metadata['crs']
        target_crs = self.geometric_config.get('target_crs', src_crs)
        
        # 如果目标投影与源投影相同，只需要重采样
        if str(src_crs) == str(target_crs):
            return self._resample_to_target_resolution(data, metadata)
        
        height, width, bands = data.shape
        
        try:
            # 计算目标变换和尺寸
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, target_crs, width, height, *metadata['bounds']
            )
            
            # 调整到目标像素大小
            target_pixel_size = self.geometric_config.get('pixel_size', [30, 30])
            if target_pixel_size:
                # 重新计算变换和尺寸
                bounds = metadata['bounds']
                
                # 变换边界到目标投影
                transformer = pyproj.Transformer.from_crs(src_crs, target_crs, always_xy=True)
                
                # 变换四个角点
                corners = [
                    (bounds[0], bounds[1]),  # 左下
                    (bounds[2], bounds[1]),  # 右下
                    (bounds[2], bounds[3]),  # 右上
                    (bounds[0], bounds[3])   # 左上
                ]
                
                transformed_corners = [transformer.transform(x, y) for x, y in corners]
                
                min_x = min(corner[0] for corner in transformed_corners)
                max_x = max(corner[0] for corner in transformed_corners)
                min_y = min(corner[1] for corner in transformed_corners)
                max_y = max(corner[1] for corner in transformed_corners)
                
                dst_width = int((max_x - min_x) / target_pixel_size[0])
                dst_height = int((max_y - min_y) / target_pixel_size[1])
                
                dst_transform = from_bounds(min_x, min_y, max_x, max_y, dst_width, dst_height)
            
            # 重投影数据
            corrected_data = np.zeros((dst_height, dst_width, bands), dtype=data.dtype)
            
            resampling_method = self.resampling_methods.get(
                self.geometric_config.get('resampling', 'bilinear'),
                Resampling.bilinear
            )
            
            for b in range(bands):
                reproject(
                    source=data[:, :, b],
                    destination=corrected_data[:, :, b],
                    src_transform=src_transform,
                    dst_transform=dst_transform,
                    src_crs=src_crs,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )
            
            logger.info(f"Projection correction: {data.shape} -> {corrected_data.shape}")
            return corrected_data
            
        except Exception as e:
            logger.error(f"Projection correction failed: {e}")
            return data
    
    def _rpc_based_correction(self, data: np.ndarray,
                            metadata: Dict[str, Any]) -> np.ndarray:
        """基于有理函数模型的几何校正
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 校正后的数据
        """
        logger.info("Applying RPC-based geometric correction")
        
        # RPC校正是复杂的过程，这里提供简化实现
        rpc_params = metadata['rpc']
        
        height, width, bands = data.shape
        
        # 创建地理坐标网格
        target_pixel_size = self.geometric_config.get('pixel_size', [30, 30])
        
        # 简化的RPC实现：使用多项式近似
        # 实际的RPC需要复杂的有理函数计算
        
        # 生成控制点网格
        control_points = self._generate_rpc_control_points(rpc_params, width, height)
        
        # 使用控制点进行多项式变换
        return self._apply_polynomial_transform(data, control_points)
    
    def _polynomial_correction(self, data: np.ndarray,
                             metadata: Dict[str, Any]) -> np.ndarray:
        """多项式几何校正
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 校正后的数据
        """
        logger.info("Applying polynomial geometric correction")
        
        # 检查是否有变换参数
        if 'polynomial_coeffs' in metadata:
            coeffs = metadata['polynomial_coeffs']
            return self._apply_polynomial_coefficients(data, coeffs)
        
        # 如果没有预定义的系数，使用默认的简单校正
        return self._apply_basic_polynomial_correction(data, metadata)
    
    def _resample_to_target_resolution(self, data: np.ndarray,
                                     metadata: Dict[str, Any]) -> np.ndarray:
        """重采样到目标分辨率
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 重采样后的数据
        """
        target_pixel_size = self.geometric_config.get('pixel_size', [30, 30])
        
        if 'pixel_size' not in metadata:
            logger.warning("No pixel size information available")
            return data
        
        current_pixel_size = metadata['pixel_size']
        if not isinstance(current_pixel_size, (list, tuple)):
            current_pixel_size = [current_pixel_size, current_pixel_size]
        
        # 计算缩放因子
        scale_x = current_pixel_size[0] / target_pixel_size[0]
        scale_y = current_pixel_size[1] / target_pixel_size[1]
        
        return self._resample_data(data, scale_x, scale_y)
    
    def _resample_data(self, data: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
        """重采样数据
        
        Args:
            data: 输入数据
            scale_x: X方向缩放因子
            scale_y: Y方向缩放因子
            
        Returns:
            np.ndarray: 重采样后的数据
        """
        height, width, bands = data.shape
        new_height = int(height / scale_y)
        new_width = int(width / scale_x)
        
        resampled_data = np.zeros((new_height, new_width, bands), dtype=data.dtype)
        
        # 选择重采样方法
        resampling_method = self.geometric_config.get('resampling', 'bilinear')
        
        if resampling_method == 'nearest':
            order = 0
        elif resampling_method == 'bilinear':
            order = 1
        elif resampling_method in ['cubic', 'cubic_spline']:
            order = 3
        else:
            order = 1  # 默认双线性
        
        # 对每个波段进行重采样
        for b in range(bands):
            resampled_data[:, :, b] = ndimage.zoom(
                data[:, :, b], 
                (1/scale_y, 1/scale_x), 
                order=order,
                mode='reflect'
            )
        
        logger.info(f"Resampled data: {data.shape} -> {resampled_data.shape}")
        return resampled_data
    
    def _generate_rpc_control_points(self, rpc_params: Dict[str, Any],
                                   width: int, height: int) -> List[Tuple]:
        """生成RPC控制点
        
        Args:
            rpc_params: RPC参数
            width: 影像宽度
            height: 影像高度
            
        Returns:
            List[Tuple]: 控制点列表 (image_x, image_y, geo_x, geo_y)
        """
        # 简化的RPC控制点生成
        # 实际的RPC需要复杂的有理函数计算
        
        control_points = []
        
        # 在影像上生成网格点
        grid_size = 10
        for i in range(grid_size):
            for j in range(grid_size):
                img_x = (i / (grid_size - 1)) * width
                img_y = (j / (grid_size - 1)) * height
                
                # 简化的地理坐标计算
                # 实际应使用RPC有理函数
                geo_x = rpc_params.get('long_off', 0) + (img_x - width/2) * rpc_params.get('long_scale', 1) / width
                geo_y = rpc_params.get('lat_off', 0) + (img_y - height/2) * rpc_params.get('lat_scale', 1) / height
                
                control_points.append((img_x, img_y, geo_x, geo_y))
        
        return control_points
    
    def _apply_polynomial_transform(self, data: np.ndarray,
                                  control_points: List[Tuple]) -> np.ndarray:
        """应用多项式变换
        
        Args:
            data: 输入数据
            control_points: 控制点列表
            
        Returns:
            np.ndarray: 变换后的数据
        """
        height, width, bands = data.shape
        
        # 提取控制点坐标
        src_points = np.array([(cp[0], cp[1]) for cp in control_points])
        dst_points = np.array([(cp[2], cp[3]) for cp in control_points])
        
        # 计算多项式系数（二次多项式）
        coeffs_x = self._fit_polynomial_2d(src_points, dst_points[:, 0])
        coeffs_y = self._fit_polynomial_2d(src_points, dst_points[:, 1])
        
        # 创建输出坐标网格
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        
        # 计算变换后的坐标
        geo_x = self._evaluate_polynomial_2d(x_indices, y_indices, coeffs_x)
        geo_y = self._evaluate_polynomial_2d(x_indices, y_indices, coeffs_y)
        
        # 确定输出范围
        min_x, max_x = np.min(geo_x), np.max(geo_x)
        min_y, max_y = np.min(geo_y), np.max(geo_y)
        
        # 计算输出尺寸
        pixel_size = self.geometric_config.get('pixel_size', [30, 30])
        out_width = int((max_x - min_x) / pixel_size[0])
        out_height = int((max_y - min_y) / pixel_size[1])
        
        # 创建输出网格
        out_x = np.linspace(min_x, max_x, out_width)
        out_y = np.linspace(min_y, max_y, out_height)
        out_xx, out_yy = np.meshgrid(out_x, out_y)
        
        # 反向变换：从地理坐标到影像坐标
        # 简化实现：使用最近邻插值
        corrected_data = np.zeros((out_height, out_width, bands), dtype=data.dtype)
        
        for i in range(out_height):
            for j in range(out_width):
                target_x, target_y = out_xx[i, j], out_yy[i, j]
                
                # 找到最接近的源像素
                distances = np.sqrt((geo_x - target_x)**2 + (geo_y - target_y)**2)
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                
                if min_idx[0] < height and min_idx[1] < width:
                    corrected_data[i, j, :] = data[min_idx[0], min_idx[1], :]
        
        return corrected_data
    
    def _fit_polynomial_2d(self, points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """拟合二维二次多项式
        
        Args:
            points: 输入点坐标 (N, 2)
            values: 对应值 (N,)
            
        Returns:
            np.ndarray: 多项式系数
        """
        x, y = points[:, 0], points[:, 1]
        
        # 构建设计矩阵 (二次多项式)
        # f(x,y) = a0 + a1*x + a2*y + a3*x² + a4*xy + a5*y²
        A = np.column_stack([
            np.ones(len(x)),
            x, y,
            x**2, x*y, y**2
        ])
        
        # 最小二乘拟合
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        
        return coeffs
    
    def _evaluate_polynomial_2d(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """计算二维二次多项式值
        
        Args:
            x: X坐标数组
            y: Y坐标数组
            coeffs: 多项式系数
            
        Returns:
            np.ndarray: 多项式值
        """
        return (coeffs[0] + 
                coeffs[1] * x + coeffs[2] * y +
                coeffs[3] * x**2 + coeffs[4] * x * y + coeffs[5] * y**2)
    
    def _apply_polynomial_coefficients(self, data: np.ndarray,
                                     coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """应用预定义的多项式系数
        
        Args:
            data: 输入数据
            coeffs: 多项式系数字典
            
        Returns:
            np.ndarray: 变换后的数据
        """
        height, width, bands = data.shape
        
        # 创建坐标网格
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        
        # 应用多项式变换
        if 'x_coeffs' in coeffs and 'y_coeffs' in coeffs:
            new_x = self._evaluate_polynomial_2d(x_indices, y_indices, coeffs['x_coeffs'])
            new_y = self._evaluate_polynomial_2d(x_indices, y_indices, coeffs['y_coeffs'])
            
            # 应用变换
            return self._remap_image(data, new_x, new_y)
        
        return data
    
    def _apply_basic_polynomial_correction(self, data: np.ndarray,
                                         metadata: Dict[str, Any]) -> np.ndarray:
        """应用基础多项式校正
        
        Args:
            data: 输入数据
            metadata: 元数据
            
        Returns:
            np.ndarray: 校正后的数据
        """
        # 简单的仿射变换校正
        # 这是一个占位符实现，实际应根据具体的几何畸变模型调整
        
        # 如果有仿射变换矩阵，应用它
        if 'affine_transform' in metadata:
            transform_matrix = metadata['affine_transform']
            return self._apply_affine_transform(data, transform_matrix)
        
        # 否则只是简单的重采样
        return self._resample_to_target_resolution(data, metadata)
    
    def _apply_affine_transform(self, data: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """应用仿射变换
        
        Args:
            data: 输入数据
            transform_matrix: 变换矩阵
            
        Returns:
            np.ndarray: 变换后的数据
        """
        height, width, bands = data.shape
        
        # 应用仿射变换到每个波段
        corrected_data = np.zeros_like(data)
        
        for b in range(bands):
            corrected_data[:, :, b] = ndimage.affine_transform(
                data[:, :, b], 
                transform_matrix,
                output_shape=(height, width),
                mode='reflect'
            )
        
        return corrected_data
    
    def _remap_image(self, data: np.ndarray, new_x: np.ndarray, new_y: np.ndarray) -> np.ndarray:
        """重新映射图像
        
        Args:
            data: 输入数据
            new_x: 新的X坐标
            new_y: 新的Y坐标
            
        Returns:
            np.ndarray: 重映射后的数据
        """
        height, width, bands = data.shape
        corrected_data = np.zeros_like(data)
        
        # 使用双线性插值重映射
        for b in range(bands):
            corrected_data[:, :, b] = ndimage.map_coordinates(
                data[:, :, b],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )
        
        return corrected_data
    
    def _has_geometric_info(self, metadata: Dict[str, Any]) -> bool:
        """检查是否有几何信息
        
        Args:
            metadata: 元数据
            
        Returns:
            bool: 是否有几何信息
        """
        required_info = ['transform', 'crs', 'gcps', 'rpc', 'bounds']
        return any(key in metadata for key in required_info)
    
    def create_ground_control_points(self, image_coords: List[Tuple[float, float]],
                                   geo_coords: List[Tuple[float, float]],
                                   crs: str = 'EPSG:4326') -> List:
        """创建地面控制点
        
        Args:
            image_coords: 影像坐标列表 [(col, row), ...]
            geo_coords: 地理坐标列表 [(x, y), ...]
            crs: 坐标参考系统
            
        Returns:
            List: GCP对象列表
        """
        if not HAS_GEO_LIBS:
            logger.warning("Geographic libraries not available for GCP creation")
            return []
        
        gcps = []
        for i, (img_coord, geo_coord) in enumerate(zip(image_coords, geo_coords)):
            gcp = GCP(
                row=img_coord[1],
                col=img_coord[0],
                x=geo_coord[0],
                y=geo_coord[1],
                id=str(i)
            )
            gcps.append(gcp)
        
        return gcps
    
    def estimate_geometric_accuracy(self, reference_points: List[Tuple],
                                  corrected_points: List[Tuple]) -> Dict[str, float]:
        """评估几何校正精度
        
        Args:
            reference_points: 参考点坐标
            corrected_points: 校正后点坐标
            
        Returns:
            Dict[str, float]: 精度统计
        """
        if len(reference_points) != len(corrected_points):
            raise ValueError("Reference and corrected points must have same length")
        
        ref_array = np.array(reference_points)
        corr_array = np.array(corrected_points)
        
        # 计算残差
        residuals = ref_array - corr_array
        
        # 计算统计量
        rmse_x = np.sqrt(np.mean(residuals[:, 0]**2))
        rmse_y = np.sqrt(np.mean(residuals[:, 1]**2))
        rmse_total = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        
        mean_error_x = np.mean(residuals[:, 0])
        mean_error_y = np.mean(residuals[:, 1])
        
        max_error = np.max(np.sqrt(np.sum(residuals**2, axis=1)))
        
        return {
            'rmse_x': float(rmse_x),
            'rmse_y': float(rmse_y),
            'rmse_total': float(rmse_total),
            'mean_error_x': float(mean_error_x),
            'mean_error_y': float(mean_error_y),
            'max_error': float(max_error),
            'num_points': len(reference_points)
        }
    
    def validate_correction(self, original_data: np.ndarray,
                          corrected_data: np.ndarray,
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """验证几何校正结果
        
        Args:
            original_data: 原始数据
            corrected_data: 校正后数据
            metadata: 元数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'original_shape': original_data.shape,
            'corrected_shape': corrected_data.shape,
            'size_change_ratio': corrected_data.size / original_data.size,
            'issues': []
        }
        
        # 检查尺寸变化
        size_change = abs(1 - validation_results['size_change_ratio'])
        if size_change > 0.5:  # 50%以上的尺寸变化
            validation_results['issues'].append(f"Large size change: {size_change:.1%}")
        
        # 检查数据完整性
        if np.any(np.isnan(corrected_data)):
            nan_ratio = np.sum(np.isnan(corrected_data)) / corrected_data.size
            validation_results['issues'].append(f"NaN values introduced: {nan_ratio:.2%}")
        
        # 检查数据范围变化
        orig_range = (np.min(original_data), np.max(original_data))
        corr_range = (np.min(corrected_data), np.max(corrected_data))
        
        if abs(orig_range[0] - corr_range[0]) / orig_range[1] > 0.1:
            validation_results['issues'].append("Significant range change detected")
        
        # 评估校正质量
        validation_results['quality'] = 'Good' if len(validation_results['issues']) == 0 else 'Needs Review'
        
        return validation_results