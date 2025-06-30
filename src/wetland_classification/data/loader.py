#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载器
Data Loader

支持多种格式的高光谱遥感数据和地面真值数据加载

作者: Wetland Research Team
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# 地理空间数据处理
try:
    import rasterio
    from rasterio.enums import Resampling
    import geopandas as gpd
    import fiona
    from shapely.geometry import Point
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False
    warnings.warn("Geographic libraries not available. Some functionality will be limited.")

# 高光谱数据处理
try:
    import spectral
    import h5py
    HAS_SPECTRAL_LIBS = True
except ImportError:
    HAS_SPECTRAL_LIBS = False
    warnings.warn("Spectral libraries not available. Some functionality will be limited.")

from ..config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """高光谱遥感数据加载器
    
    支持多种数据格式：
    - 高光谱数据: ENVI, GeoTIFF, HDF5, NetCDF
    - 地面真值: Shapefile, GeoJSON, CSV
    - 辅助数据: DEM, 土壤图, 气象数据
    """
    
    def __init__(self, config: Config):
        """初始化数据加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.data_config = config.get_data_config()
        
        # 支持的文件格式
        self.supported_hyperspectral_formats = ['.hdr', '.tif', '.tiff', '.h5', '.hdf5', '.nc', '.mat']
        self.supported_vector_formats = ['.shp', '.geojson', '.json', '.kml', '.gpkg']
        self.supported_table_formats = ['.csv', '.xlsx', '.xls', '.txt']
        
        logger.info("DataLoader initialized")
    
    def load_hyperspectral(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载高光谱数据
        
        Args:
            file_path: 高光谱数据文件路径
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (数据数组, 元数据字典)
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Hyperspectral file not found: {file_path}")
        
        logger.info(f"Loading hyperspectral data: {file_path}")
        
        # 根据文件扩展名选择加载方法
        suffix = file_path.suffix.lower()
        
        if suffix in ['.hdr']:
            return self._load_envi_data(file_path)
        elif suffix in ['.tif', '.tiff']:
            return self._load_geotiff_data(file_path)
        elif suffix in ['.h5', '.hdf5']:
            return self._load_hdf5_data(file_path)
        elif suffix in ['.nc']:
            return self._load_netcdf_data(file_path)
        elif suffix in ['.mat']:
            return self._load_matlab_data(file_path)
        else:
            raise ValueError(f"Unsupported hyperspectral format: {suffix}")
    
    def load_samples(self, file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """加载训练样本数据
        
        Args:
            file_path: 样本数据文件路径
            
        Returns:
            gpd.GeoDataFrame: 样本数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        if not HAS_GEO_LIBS:
            raise ImportError("Geographic libraries required for loading samples")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Sample file not found: {file_path}")
        
        logger.info(f"Loading sample data: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.shp']:
            return gpd.read_file(file_path)
        elif suffix in ['.geojson', '.json']:
            return gpd.read_file(file_path)
        elif suffix in ['.kml']:
            return gpd.read_file(file_path)
        elif suffix in ['.gpkg']:
            return gpd.read_file(file_path)
        elif suffix in ['.csv', '.txt']:
            return self._load_csv_samples(file_path)
        else:
            raise ValueError(f"Unsupported sample format: {suffix}")
    
    def extract_spectra_and_labels(self, hyperspectral_data: np.ndarray, 
                                 samples: gpd.GeoDataFrame,
                                 transform: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """从高光谱数据中提取光谱和标签
        
        Args:
            hyperspectral_data: 高光谱数据 (H, W, B)
            samples: 样本数据
            transform: 坐标变换 (可选)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (光谱数据, 标签)
        """
        if not HAS_GEO_LIBS:
            raise ImportError("Geographic libraries required for extracting spectra")
        
        logger.info("Extracting spectra and labels from samples")
        
        spectra_list = []
        labels_list = []
        
        height, width, bands = hyperspectral_data.shape
        
        for idx, row in samples.iterrows():
            try:
                # 获取几何信息和类别标签
                geometry = row.geometry
                class_id = row.get('class_id', row.get('class', row.get('label', 1)))
                
                if geometry.geom_type == 'Point':
                    # 点样本
                    x, y = geometry.x, geometry.y
                    
                    # 坐标转换 (如果提供了transform)
                    if transform:
                        col, row_idx = ~transform * (x, y)
                        col, row_idx = int(col), int(row_idx)
                    else:
                        # 假设坐标已经是像素坐标
                        col, row_idx = int(x), int(y)
                    
                    # 检查边界
                    if 0 <= row_idx < height and 0 <= col < width:
                        spectrum = hyperspectral_data[row_idx, col, :]
                        spectra_list.append(spectrum)
                        labels_list.append(class_id)
                
                elif geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    # 面样本 - 提取多个像素
                    # 这里简化处理，可以根据需要添加更复杂的采样策略
                    bounds = geometry.bounds
                    
                    if transform:
                        min_col, min_row = ~transform * (bounds[0], bounds[3])
                        max_col, max_row = ~transform * (bounds[2], bounds[1])
                    else:
                        min_col, min_row = bounds[0], bounds[1]
                        max_col, max_row = bounds[2], bounds[3]
                    
                    min_col, min_row = max(0, int(min_col)), max(0, int(min_row))
                    max_col, max_row = min(width, int(max_col)), min(height, int(max_row))
                    
                    # 在多边形内采样
                    for r in range(min_row, max_row):
                        for c in range(min_col, max_col):
                            if transform:
                                point_x, point_y = transform * (c, r)
                                point = Point(point_x, point_y)
                            else:
                                point = Point(c, r)
                            
                            if geometry.contains(point):
                                spectrum = hyperspectral_data[r, c, :]
                                spectra_list.append(spectrum)
                                labels_list.append(class_id)
                
            except Exception as e:
                logger.warning(f"Failed to extract spectrum from sample {idx}: {e}")
                continue
        
        if not spectra_list:
            raise ValueError("No valid spectra extracted from samples")
        
        spectra = np.array(spectra_list)
        labels = np.array(labels_list)
        
        logger.info(f"Extracted {len(spectra)} spectra with {len(np.unique(labels))} classes")
        
        return spectra, labels
    
    def split_data(self, spectra: np.ndarray, labels: np.ndarray,
                   train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                   stratify: bool = True, random_state: int = 42) -> Tuple[Tuple, Tuple, Tuple]:
        """分割数据为训练、验证和测试集
        
        Args:
            spectra: 光谱数据
            labels: 标签数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify: 是否分层采样
            random_state: 随机种子
            
        Returns:
            Tuple[Tuple, Tuple, Tuple]: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # 检查比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if stratify:
            # 分层采样
            # 首先分离出测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                spectra, labels, 
                test_size=test_ratio,
                stratify=labels,
                random_state=random_state
            )
            
            # 然后从剩余数据中分离训练集和验证集
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio_adjusted,
                stratify=y_temp,
                random_state=random_state
            )
        else:
            # 简单随机采样
            X_temp, X_test, y_temp, y_test = train_test_split(
                spectra, labels,
                test_size=test_ratio,
                random_state=random_state
            )
            
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio_adjusted,
                random_state=random_state
            )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _load_envi_data(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载ENVI格式数据"""
        if not HAS_SPECTRAL_LIBS:
            raise ImportError("Spectral library required for ENVI data")
        
        try:
            # 使用spectral库加载ENVI数据
            img = spectral.open_image(str(file_path))
            data = img.load()
            
            # 转换为numpy数组
            hyperspectral_data = np.array(data)
            
            # 提取元数据
            metadata = {
                'format': 'ENVI',
                'shape': hyperspectral_data.shape,
                'bands': img.nbands,
                'lines': img.nrows,
                'samples': img.ncols,
                'data_type': img.dtype,
                'wavelengths': getattr(img, 'bands', None),
                'band_names': getattr(img, 'band_names', None),
                'description': getattr(img, 'metadata', {}).get('description', ''),
                'acquisition_date': getattr(img, 'metadata', {}).get('acquisition date', ''),
            }
            
            logger.info(f"Loaded ENVI data: {hyperspectral_data.shape}")
            return hyperspectral_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load ENVI data: {e}")
            raise
    
    def _load_geotiff_data(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载GeoTIFF格式数据"""
        if not HAS_GEO_LIBS:
            raise ImportError("Rasterio required for GeoTIFF data")
        
        try:
            with rasterio.open(file_path) as src:
                # 读取数据
                data = src.read()  # (bands, height, width)
                
                # 转置为 (height, width, bands)
                hyperspectral_data = np.transpose(data, (1, 2, 0))
                
                # 提取元数据
                metadata = {
                    'format': 'GeoTIFF',
                    'shape': hyperspectral_data.shape,
                    'bands': src.count,
                    'height': src.height,
                    'width': src.width,
                    'data_type': str(src.dtype),
                    'crs': str(src.crs),
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': src.nodata,
                    'description': src.descriptions,
                }
                
                # 添加波长信息 (如果有的话)
                if 'wavelength' in src.tags():
                    metadata['wavelengths'] = src.tags()['wavelength']
                
            logger.info(f"Loaded GeoTIFF data: {hyperspectral_data.shape}")
            return hyperspectral_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load GeoTIFF data: {e}")
            raise
    
    def _load_hdf5_data(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载HDF5格式数据"""
        if not HAS_SPECTRAL_LIBS:
            raise ImportError("h5py required for HDF5 data")
        
        try:
            with h5py.File(file_path, 'r') as f:
                # 尝试找到主数据集
                data_keys = ['data', 'hyperspectral', 'reflectance', 'radiance']
                data_key = None
                
                for key in data_keys:
                    if key in f:
                        data_key = key
                        break
                
                if data_key is None:
                    # 使用第一个3D数据集
                    for key in f.keys():
                        if len(f[key].shape) == 3:
                            data_key = key
                            break
                
                if data_key is None:
                    raise ValueError("No suitable hyperspectral dataset found in HDF5 file")
                
                hyperspectral_data = f[data_key][:]
                
                # 提取元数据
                metadata = {
                    'format': 'HDF5',
                    'shape': hyperspectral_data.shape,
                    'data_key': data_key,
                    'data_type': str(hyperspectral_data.dtype),
                }
                
                # 尝试读取属性
                for attr_name in f.attrs:
                    metadata[attr_name] = f.attrs[attr_name]
                
                # 尝试读取波长信息
                if 'wavelengths' in f:
                    metadata['wavelengths'] = f['wavelengths'][:]
                elif 'wavelength' in f:
                    metadata['wavelengths'] = f['wavelength'][:]
                
            logger.info(f"Loaded HDF5 data: {hyperspectral_data.shape}")
            return hyperspectral_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load HDF5 data: {e}")
            raise
    
    def _load_netcdf_data(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载NetCDF格式数据"""
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 required for NetCDF data")
        
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                # 查找主要的数据变量
                data_vars = ['reflectance', 'radiance', 'data', 'hyperspectral']
                data_var = None
                
                for var in data_vars:
                    if var in dataset.variables:
                        data_var = var
                        break
                
                if data_var is None:
                    # 使用第一个3D变量
                    for var_name, var in dataset.variables.items():
                        if len(var.dimensions) == 3:
                            data_var = var_name
                            break
                
                if data_var is None:
                    raise ValueError("No suitable hyperspectral variable found in NetCDF file")
                
                hyperspectral_data = dataset.variables[data_var][:]
                
                # 提取元数据
                metadata = {
                    'format': 'NetCDF',
                    'shape': hyperspectral_data.shape,
                    'data_variable': data_var,
                    'data_type': str(hyperspectral_data.dtype),
                }
                
                # 添加全局属性
                for attr_name in dataset.ncattrs():
                    metadata[attr_name] = getattr(dataset, attr_name)
                
                # 尝试读取坐标信息
                if 'wavelength' in dataset.variables:
                    metadata['wavelengths'] = dataset.variables['wavelength'][:]
                
            logger.info(f"Loaded NetCDF data: {hyperspectral_data.shape}")
            return hyperspectral_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load NetCDF data: {e}")
            raise
    
    def _load_matlab_data(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载MATLAB格式数据"""
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy required for MATLAB data")
        
        try:
            mat_data = loadmat(file_path)
            
            # 查找主要的数据变量
            data_keys = ['data', 'hyperspectral', 'reflectance', 'radiance', 'image']
            data_key = None
            
            for key in data_keys:
                if key in mat_data and isinstance(mat_data[key], np.ndarray):
                    if len(mat_data[key].shape) == 3:
                        data_key = key
                        break
            
            if data_key is None:
                # 查找第一个3D数组
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 3:
                        data_key = key
                        break
            
            if data_key is None:
                raise ValueError("No suitable hyperspectral array found in MATLAB file")
            
            hyperspectral_data = mat_data[data_key]
            
            # 提取元数据
            metadata = {
                'format': 'MATLAB',
                'shape': hyperspectral_data.shape,
                'data_key': data_key,
                'data_type': str(hyperspectral_data.dtype),
            }
            
            # 添加其他变量作为元数据
            for key, value in mat_data.items():
                if not key.startswith('__') and key != data_key:
                    if isinstance(value, (int, float, str, list)):
                        metadata[key] = value
                    elif isinstance(value, np.ndarray) and value.size < 1000:
                        metadata[key] = value.tolist()
            
            logger.info(f"Loaded MATLAB data: {hyperspectral_data.shape}")
            return hyperspectral_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load MATLAB data: {e}")
            raise
    
    def _load_csv_samples(self, file_path: Path) -> gpd.GeoDataFrame:
        """加载CSV格式的样本数据"""
        if not HAS_GEO_LIBS:
            raise ImportError("GeoPandas required for CSV sample data")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必需的列
        required_cols = ['x', 'y', 'class_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"CSV file missing required columns: {missing_cols}")
        
        # 创建几何对象
        geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
        
        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # 设置坐标系 (如果指定了的话)
        if 'crs' in df.columns:
            gdf.crs = df['crs'].iloc[0]
        
        return gdf
    
    def get_data_info(self, data: np.ndarray) -> Dict[str, Any]:
        """获取数据基本信息
        
        Args:
            data: 高光谱数据
            
        Returns:
            Dict[str, Any]: 数据信息
        """
        return {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size_mb': data.nbytes / (1024 * 1024),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data)),
            'has_nan': bool(np.isnan(data).any()),
            'has_inf': bool(np.isinf(data).any()),
        }
    
    def resample_data(self, data: np.ndarray, target_shape: Tuple[int, int],
                     method: str = 'bilinear') -> np.ndarray:
        """重采样数据到目标空间分辨率
        
        Args:
            data: 输入数据 (H, W, B)
            target_shape: 目标形状 (height, width)
            method: 重采样方法
            
        Returns:
            np.ndarray: 重采样后的数据
        """
        if not HAS_GEO_LIBS:
            raise ImportError("Rasterio required for resampling")
        
        height, width, bands = data.shape
        target_height, target_width = target_shape
        
        # 如果已经是目标形状，直接返回
        if height == target_height and width == target_width:
            return data
        
        # 重采样每个波段
        resampled_data = np.zeros((target_height, target_width, bands), dtype=data.dtype)
        
        resampling_method = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
        }.get(method, Resampling.bilinear)
        
        for b in range(bands):
            from rasterio.warp import reproject, Resampling
            from rasterio.transform import from_bounds
            
            # 创建源和目标变换
            src_transform = from_bounds(0, 0, width, height, width, height)
            dst_transform = from_bounds(0, 0, width, height, target_width, target_height)
            
            reproject(
                source=data[:, :, b],
                destination=resampled_data[:, :, b],
                src_transform=src_transform,
                dst_transform=dst_transform,
                src_crs='EPSG:4326',
                dst_crs='EPSG:4326',
                resampling=resampling_method
            )
        
        logger.info(f"Resampled data from {data.shape} to {resampled_data.shape}")
        return resampled_data