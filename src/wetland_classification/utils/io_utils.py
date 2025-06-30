"""
输入输出工具模块
IO Utils Module

提供各种数据格式的读取、写入和转换功能，支持：
- 高光谱数据格式：GeoTIFF, ENVI, HDF5, NetCDF
- 矢量数据格式：Shapefile, GeoJSON, CSV
- 模型文件：Pickle, PyTorch, TensorFlow
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import yaml
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from spectral import envi
import xarray as xr
import torch

from .logger import get_logger

logger = get_logger(__name__)


def load_hyperspectral_data(
    file_path: Union[str, Path],
    bands: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    加载高光谱数据
    
    Args:
        file_path: 文件路径
        bands: 指定加载的波段列表，None表示加载所有波段
        bbox: 边界框 (min_x, min_y, max_x, max_y)
        chunks: 分块读取配置（用于大文件）
        
    Returns:
        Tuple[数据数组, 元数据字典]
    """
    file_path = Path(file_path)
    logger.info(f"正在加载高光谱数据: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.tif' or file_path.suffix.lower() == '.tiff':
            return _load_geotiff(file_path, bands, bbox)
        elif file_path.suffix.lower() == '.hdr':
            return _load_envi(file_path, bands, bbox)
        elif file_path.suffix.lower() == '.h5' or file_path.suffix.lower() == '.hdf5':
            return _load_hdf5(file_path, bands, bbox, chunks)
        elif file_path.suffix.lower() == '.nc':
            return _load_netcdf(file_path, bands, bbox)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
    except Exception as e:
        logger.error(f"加载高光谱数据失败: {e}")
        raise


def _load_geotiff(
    file_path: Path,
    bands: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """加载GeoTIFF格式的高光谱数据"""
    with rasterio.open(file_path) as src:
        # 获取元数据
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'bounds': src.bounds,
        }
        
        # 处理波段选择
        if bands is None:
            bands = list(range(1, src.count + 1))
        
        # 处理空间裁剪
        if bbox:
            window = rasterio.windows.from_bounds(*bbox, src.transform)
            data = src.read(bands, window=window)
            # 更新元数据
            metadata['transform'] = rasterio.windows.transform(window, src.transform)
            metadata['width'] = window.width
            metadata['height'] = window.height
        else:
            data = src.read(bands)
        
        # 转换为 (height, width, bands) 格式
        data = np.transpose(data, (1, 2, 0))
        metadata['bands'] = len(bands)
        metadata['wavelengths'] = _extract_wavelengths_from_metadata(src.tags())
        
    logger.info(f"成功加载GeoTIFF数据，形状: {data.shape}")
    return data, metadata


def _load_envi(
    file_path: Path,
    bands: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """加载ENVI格式的高光谱数据"""
    # 查找对应的.img文件
    img_path = file_path.with_suffix('.img')
    if not img_path.exists():
        img_path = file_path.with_suffix('')
    
    # 使用spectral库读取ENVI文件
    img = envi.open(str(file_path), str(img_path))
    
    metadata = {
        'shape': img.shape,
        'nbands': img.nbands,
        'nrows': img.nrows,
        'ncols': img.ncols,
        'dtype': img.dtype,
        'interleave': img.interleave,
        'wavelengths': img.bands.centers if hasattr(img.bands, 'centers') else None,
        'fwhm': img.bands.bandwidths if hasattr(img.bands, 'bandwidths') else None,
    }
    
    # 处理波段选择
    if bands is None:
        data = img.load()
    else:
        data = img.read_bands(bands)
    
    # 处理空间裁剪
    if bbox:
        # ENVI格式的空间裁剪需要根据具体情况实现
        logger.warning("ENVI格式暂不支持边界框裁剪")
    
    logger.info(f"成功加载ENVI数据，形状: {data.shape}")
    return data, metadata


def _load_hdf5(
    file_path: Path,
    bands: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """加载HDF5格式的高光谱数据"""
    with h5py.File(file_path, 'r') as f:
        # 查找数据集
        data_key = _find_main_dataset(f)
        dataset = f[data_key]
        
        metadata = {
            'shape': dataset.shape,
            'dtype': dataset.dtype,
            'chunks': dataset.chunks,
        }
        
        # 添加属性信息
        for key, value in dataset.attrs.items():
            metadata[key] = value
        
        # 处理波段选择和空间裁剪
        if bands is None and bbox is None:
            data = dataset[...]
        else:
            # 构建切片
            slices = [slice(None)] * len(dataset.shape)
            if bands is not None and len(dataset.shape) == 3:
                slices[2] = bands  # 假设最后一个维度是波段
            data = dataset[tuple(slices)]
    
    logger.info(f"成功加载HDF5数据，形状: {data.shape}")
    return data, metadata


def _load_netcdf(
    file_path: Path,
    bands: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """加载NetCDF格式的高光谱数据"""
    ds = xr.open_dataset(file_path)
    
    # 查找主要数据变量
    data_vars = list(ds.data_vars.keys())
    main_var = data_vars[0]  # 假设第一个是主要数据
    
    data_array = ds[main_var]
    
    metadata = {
        'dims': data_array.dims,
        'shape': data_array.shape,
        'attrs': dict(data_array.attrs),
        'coords': {k: v.values for k, v in data_array.coords.items()},
    }
    
    # 转换为numpy数组
    data = data_array.values
    
    # 处理波段选择
    if bands is not None:
        band_dim = _find_band_dimension(data_array)
        if band_dim is not None:
            data = data.take(bands, axis=band_dim)
    
    logger.info(f"成功加载NetCDF数据，形状: {data.shape}")
    return data, metadata


def load_ground_truth(
    file_path: Union[str, Path],
    label_column: str = 'class',
    encoding: str = 'utf-8'
) -> gpd.GeoDataFrame:
    """
    加载地面真实标签数据
    
    Args:
        file_path: 文件路径
        label_column: 标签列名
        encoding: 文件编码
        
    Returns:
        GeoDataFrame
    """
    file_path = Path(file_path)
    logger.info(f"正在加载地面真实数据: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.shp':
            gdf = gpd.read_file(file_path, encoding=encoding)
        elif file_path.suffix.lower() == '.geojson':
            gdf = gpd.read_file(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding=encoding)
            # 假设包含geometry列或经纬度列
            if 'geometry' in df.columns:
                gdf = gpd.GeoDataFrame(df)
            elif 'longitude' in df.columns and 'latitude' in df.columns:
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df.longitude, df.latitude)
                )
            else:
                raise ValueError("CSV文件必须包含geometry列或经纬度列")
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        # 验证标签列
        if label_column not in gdf.columns:
            raise ValueError(f"未找到标签列: {label_column}")
        
        logger.info(f"成功加载地面真实数据，共{len(gdf)}个样本")
        return gdf
        
    except Exception as e:
        logger.error(f"加载地面真实数据失败: {e}")
        raise


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    logger.info(f"正在加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info("成功加载配置文件")
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def save_classification_result(
    result: np.ndarray,
    output_path: Union[str, Path],
    metadata: Dict[str, Any],
    class_names: Optional[List[str]] = None
) -> None:
    """
    保存分类结果
    
    Args:
        result: 分类结果数组
        output_path: 输出路径
        metadata: 元数据信息
        class_names: 类别名称列表
    """
    output_path = Path(output_path)
    logger.info(f"正在保存分类结果: {output_path}")
    
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ['.tif', '.tiff']:
            _save_geotiff(result, output_path, metadata, class_names)
        elif output_path.suffix.lower() == '.png':
            _save_png(result, output_path, class_names)
        elif output_path.suffix.lower() == '.npz':
            _save_npz(result, output_path, metadata, class_names)
        else:
            raise ValueError(f"不支持的输出格式: {output_path.suffix}")
        
        logger.info("分类结果保存成功")
        
    except Exception as e:
        logger.error(f"保存分类结果失败: {e}")
        raise


def _save_geotiff(
    result: np.ndarray,
    output_path: Path,
    metadata: Dict[str, Any],
    class_names: Optional[List[str]] = None
) -> None:
    """保存为GeoTIFF格式"""
    # 确保结果是2D数组
    if len(result.shape) == 3:
        result = result[:, :, 0]
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=result.shape[0],
        width=result.shape[1],
        count=1,
        dtype=result.dtype,
        crs=metadata.get('crs'),
        transform=metadata.get('transform'),
        compress='lzw',
    ) as dst:
        dst.write(result, 1)
        
        # 添加类别名称到元数据
        if class_names:
            dst.update_tags(class_names=','.join(class_names))


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    保存评估指标
    
    Args:
        metrics: 指标字典
        output_path: 输出路径
        format: 输出格式 ('json', 'csv', 'yaml')
    """
    output_path = Path(output_path)
    logger.info(f"正在保存评估指标: {output_path}")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(metrics, f, default_flow_style=False, allow_unicode=True)
        elif format == 'csv':
            df = pd.DataFrame([metrics])
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        logger.info("评估指标保存成功")
        
    except Exception as e:
        logger.error(f"保存评估指标失败: {e}")
        raise


def save_model(
    model: Any,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    保存模型
    
    Args:
        model: 模型对象
        output_path: 输出路径
        metadata: 模型元数据
    """
    output_path = Path(output_path)
    logger.info(f"正在保存模型: {output_path}")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(model, 'state_dict'):  # PyTorch模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': metadata or {}
            }, output_path)
        else:  # Sklearn模型等
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'metadata': metadata or {}
                }, f)
        
        logger.info("模型保存成功")
        
    except Exception as e:
        logger.error(f"保存模型失败: {e}")
        raise


def load_model(model_path: Union[str, Path], model_class=None) -> Tuple[Any, Dict[str, Any]]:
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        model_class: PyTorch模型类（如果是PyTorch模型）
        
    Returns:
        Tuple[模型对象, 元数据]
    """
    model_path = Path(model_path)
    logger.info(f"正在加载模型: {model_path}")
    
    try:
        if model_path.suffix == '.pth' or model_path.suffix == '.pt':
            # PyTorch模型
            checkpoint = torch.load(model_path, map_location='cpu')
            if model_class:
                model = model_class()
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = checkpoint.get('model')
            metadata = checkpoint.get('metadata', {})
        else:
            # Pickle格式
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                model = data['model']
                metadata = data.get('metadata', {})
        
        logger.info("模型加载成功")
        return model, metadata
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def validate_hyperspectral_data(data: np.ndarray, metadata: Dict[str, Any]) -> bool:
    """
    验证高光谱数据的有效性
    
    Args:
        data: 数据数组
        metadata: 元数据
        
    Returns:
        是否有效
    """
    try:
        # 检查数据维度
        if len(data.shape) != 3:
            logger.error(f"数据维度错误，期望3维，实际{len(data.shape)}维")
            return False
        
        # 检查数据类型
        if not np.issubdtype(data.dtype, np.number):
            logger.error(f"数据类型错误，期望数值类型，实际{data.dtype}")
            return False
        
        # 检查是否包含无效值
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("数据包含NaN或无穷值")
        
        # 检查数据范围
        if np.min(data) < 0:
            logger.warning("数据包含负值")
        
        logger.info("高光谱数据验证通过")
        return True
        
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        return False


def check_spatial_alignment(
    raster_metadata: Dict[str, Any],
    vector_gdf: gpd.GeoDataFrame
) -> bool:
    """
    检查栅格和矢量数据的空间对齐
    
    Args:
        raster_metadata: 栅格数据元数据
        vector_gdf: 矢量数据
        
    Returns:
        是否对齐
    """
    try:
        # 检查坐标系
        raster_crs = raster_metadata.get('crs')
        vector_crs = vector_gdf.crs
        
        if raster_crs != vector_crs:
            logger.warning(f"坐标系不匹配: 栅格({raster_crs}) vs 矢量({vector_crs})")
            return False
        
        # 检查空间范围
        raster_bounds = raster_metadata.get('bounds')
        vector_bounds = vector_gdf.total_bounds
        
        if raster_bounds and not _bounds_intersect(raster_bounds, vector_bounds):
            logger.warning("栅格和矢量数据空间范围不重叠")
            return False
        
        logger.info("空间对齐检查通过")
        return True
        
    except Exception as e:
        logger.error(f"空间对齐检查失败: {e}")
        return False


# 辅助函数
def _extract_wavelengths_from_metadata(tags: Dict[str, str]) -> Optional[List[float]]:
    """从元数据中提取波长信息"""
    # 尝试从GDAL标签中提取波长
    wavelength_keys = ['wavelength', 'band_names', 'WAVELENGTH']
    for key in wavelength_keys:
        if key in tags:
            try:
                return [float(w) for w in tags[key].split(',')]
            except:
                continue
    return None


def _find_main_dataset(hdf5_group) -> str:
    """在HDF5文件中查找主要数据集"""
    def find_largest_dataset(group, path=""):
        largest_size = 0
        largest_path = ""
        
        for key in group.keys():
            current_path = f"{path}/{key}" if path else key
            item = group[key]
            
            if hasattr(item, 'shape'):  # 是数据集
                size = np.prod(item.shape)
                if size > largest_size:
                    largest_size = size
                    largest_path = current_path
            elif hasattr(item, 'keys'):  # 是组
                sub_path = find_largest_dataset(item, current_path)
                if sub_path:
                    sub_size = np.prod(group[sub_path].shape)
                    if sub_size > largest_size:
                        largest_size = sub_size
                        largest_path = sub_path
        
        return largest_path
    
    return find_largest_dataset(hdf5_group)


def _find_band_dimension(data_array) -> Optional[int]:
    """查找波段维度"""
    # 根据维度名称推断
    dim_names = data_array.dims
    band_indicators = ['band', 'bands', 'wavelength', 'spectral']
    
    for i, dim_name in enumerate(dim_names):
        if any(indicator in dim_name.lower() for indicator in band_indicators):
            return i
    
    # 如果无法确定，假设最后一个维度是波段
    return len(dim_names) - 1


def _bounds_intersect(bounds1: Tuple[float, float, float, float], 
                     bounds2: Tuple[float, float, float, float]) -> bool:
    """检查两个边界框是否相交"""
    min_x1, min_y1, max_x1, max_y1 = bounds1
    min_x2, min_y2, max_x2, max_y2 = bounds2
    
    return not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1)


def _save_png(result: np.ndarray, output_path: Path, class_names: Optional[List[str]] = None) -> None:
    """保存为PNG格式（用于可视化）"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(result, cmap='tab10')
    plt.colorbar(label='Class')
    plt.title('Classification Result')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _save_npz(result: np.ndarray, output_path: Path, metadata: Dict[str, Any], 
              class_names: Optional[List[str]] = None) -> None:
    """保存为NPZ格式"""
    np.savez_compressed(
        output_path,
        classification=result,
        metadata=metadata,
        class_names=class_names
    )