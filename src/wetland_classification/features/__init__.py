#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取模块
Feature Extraction Module

提供高光谱遥感数据的特征提取功能：
- 光谱特征：原始光谱、导数光谱、连续统去除
- 植被指数：NDVI、EVI、SAVI等各种植被指数
- 纹理特征：GLCM、LBP、GLRLM等纹理分析
- 空间特征：形态学特征、边缘特征、梯度特征

作者: Wetland Research Team
"""

from .spectral import SpectralFeatureExtractor
from .indices import VegetationIndexCalculator  
from .texture import TextureFeatureExtractor
from .spatial import SpatialFeatureExtractor

__all__ = [
    'SpectralFeatureExtractor',
    'VegetationIndexCalculator',
    'TextureFeatureExtractor', 
    'SpatialFeatureExtractor'
]