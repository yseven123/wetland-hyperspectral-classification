#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
植被指数计算器
Vegetation Index Calculator

计算各种植被指数和光谱指数，用于湿地生态系统分类

作者: Wetland Research Team
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import logging

import numpy as np

from ..config import Config

logger = logging.getLogger(__name__)


class VegetationIndexCalculator:
    """植被指数计算器
    
    支持的植被指数：
    - 基础植被指数：NDVI、EVI、SAVI、OSAVI、MSAVI
    - 水分指数：NDWI、MNDWI、NDMI、WBI、MSI
    - 叶绿素指数：MCARI、TCARI、CHL、CI
    - 红边指数：REP、NDRE、mNDRE、CIred-edge
    - 光化学指数：PRI、SIPI、NPCI
    - 土壤调节指数：SAVI、TSAVI、GESAVI
    - 抗大气影响指数：ARVI、GEMI、VARI
    """
    
    def __init__(self, config: Config):
        """初始化植被指数计算器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.indices_config = config.get('features.vegetation_indices', {})
        
        # 获取波长信息
        self.wavelengths = self._get_wavelengths()
        
        # 定义波段位置映射
        self.band_positions = self._initialize_band_positions()
        
        # 定义可用的植被指数
        self.available_indices = self._get_available_indices()
        
        logger.info("VegetationIndexCalculator initialized")
    
    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算植被指数
        
        Args:
            data: 输入高光谱数据 (H, W, B)
            
        Returns:
            np.ndarray: 植被指数特征 (H, W, N_indices)
        """
        logger.info("Calculating vegetation indices")
        
        if self.band_positions is None:
            logger.warning("No band position information available")
            return np.zeros((data.shape[0], data.shape[1], 1))
        
        # 获取要计算的指数列表
        indices_to_calculate = self.indices_config.get('indices', list(self.available_indices.keys()))
        
        height, width, bands = data.shape
        calculated_indices = []
        
        for index_name in indices_to_calculate:
            if index_name in self.available_indices:
                try:
                    index_func = self.available_indices[index_name]
                    index_values = index_func(data)
                    
                    if index_values is not None:
                        # 确保是三维数组
                        if index_values.ndim == 2:
                            index_values = index_values[:, :, np.newaxis]
                        
                        calculated_indices.append(index_values)
                        logger.debug(f"Calculated {index_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {index_name}: {e}")
                    continue
            else:
                logger.warning(f"Unknown vegetation index: {index_name}")
        
        if calculated_indices:
            indices_array = np.concatenate(calculated_indices, axis=2)
            logger.info(f"Calculated {len(calculated_indices)} vegetation indices: shape {indices_array.shape}")
            return indices_array
        else:
            logger.warning("No vegetation indices calculated")
            return np.zeros((height, width, 1))
    
    def _get_wavelengths(self) -> Optional[np.ndarray]:
        """获取波长信息"""
        wavelength_range = self.config.get('data.hyperspectral.wavelength_range', [400, 2500])
        bands = self.config.get('data.hyperspectral.bands', 400)
        
        if wavelength_range and bands:
            return np.linspace(wavelength_range[0], wavelength_range[1], bands)
        return None
    
    def _initialize_band_positions(self) -> Optional[Dict[str, int]]:
        """初始化波段位置映射"""
        if self.wavelengths is None:
            return None
        
        # 定义关键波长
        key_wavelengths = {
            'blue': 470,
            'green': 550,
            'red': 670,
            'red_edge': 720,
            'nir': 850,
            'nir2': 1240,
            'swir1': 1640,
            'swir2': 2130,
            
            # 精细波段定义
            'coastal': 443,
            'violet': 410,
            'cyan': 490,
            'yellow': 570,
            'red_680': 680,
            'red_edge_705': 705,
            'red_edge_740': 740,
            'red_edge_783': 783,
            'nir_865': 865,
            'water_vapor': 945,
            'cirrus': 1375,
            'swir_1610': 1610,
            'swir_2200': 2200,
        }
        
        band_positions = {}
        for name, wavelength in key_wavelengths.items():
            # 找到最接近的波段
            band_idx = np.argmin(np.abs(self.wavelengths - wavelength))
            band_positions[name] = band_idx
        
        return band_positions
    
    def _get_available_indices(self) -> Dict[str, Callable]:
        """获取可用的植被指数计算函数"""
        return {
            # 基础植被指数
            'NDVI': self._calculate_ndvi,
            'EVI': self._calculate_evi,
            'SAVI': self._calculate_savi,
            'OSAVI': self._calculate_osavi,
            'MSAVI': self._calculate_msavi,
            'GESAVI': self._calculate_gesavi,
            
            # 水分指数
            'NDWI': self._calculate_ndwi,
            'MNDWI': self._calculate_mndwi,
            'NDMI': self._calculate_ndmi,
            'WBI': self._calculate_wbi,
            'MSI': self._calculate_msi,
            'LSWI': self._calculate_lswi,
            
            # 叶绿素指数
            'MCARI': self._calculate_mcari,
            'TCARI': self._calculate_tcari,
            'CHL': self._calculate_chl,
            'CI': self._calculate_ci,
            'CIG': self._calculate_cig,
            
            # 红边指数
            'REP': self._calculate_rep,
            'NDRE': self._calculate_ndre,
            'mNDRE': self._calculate_mndre,
            'CIred_edge': self._calculate_ci_red_edge,
            'MTCI': self._calculate_mtci,
            
            # 光化学指数
            'PRI': self._calculate_pri,
            'SIPI': self._calculate_sipi,
            'NPCI': self._calculate_npci,
            'PRIxCI': self._calculate_pri_ci,
            
            # 抗大气影响指数
            'ARVI': self._calculate_arvi,
            'GEMI': self._calculate_gemi,
            'VARI': self._calculate_vari,
            'GNDVI': self._calculate_gndvi,
            
            # 土壤线性指数
            'TSAVI': self._calculate_tsavi,
            'ATSAVI': self._calculate_atsavi,
            
            # 综合指数
            'TNDVI': self._calculate_tndvi,
            'DVI': self._calculate_dvi,
            'IPVI': self._calculate_ipvi,
            'RVI': self._calculate_rvi,
            'TVI': self._calculate_tvi,
        }
    
    # 基础植被指数
    def _calculate_ndvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """归一化植被指数 NDVI = (NIR - Red) / (NIR + Red)"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        return np.clip(ndvi, -1, 1)
    
    def _calculate_evi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """增强植被指数 EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
        required_bands = ['nir', 'red', 'blue']
        if not all(band in self.band_positions for band in required_bands):
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        blue = data[:, :, self.band_positions['blue']].astype(np.float32)
        
        denominator = nir + 6 * red - 7.5 * blue + 1
        evi = np.where(denominator != 0, 2.5 * (nir - red) / denominator, 0)
        return np.clip(evi, -1, 1)
    
    def _calculate_savi(self, data: np.ndarray, L: float = 0.5) -> Optional[np.ndarray]:
        """土壤调节植被指数 SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        denominator = nir + red + L
        savi = np.where(denominator != 0, (nir - red) / denominator * (1 + L), 0)
        return savi
    
    def _calculate_osavi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """优化土壤调节植被指数 OSAVI = (NIR - Red) / (NIR + Red + 0.16)"""
        return self._calculate_savi(data, L=0.16)
    
    def _calculate_msavi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """修正土壤调节植被指数"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        # MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2
        term1 = 2 * nir + 1
        term2 = term1**2 - 8 * (nir - red)
        term2 = np.where(term2 >= 0, np.sqrt(term2), 0)
        
        msavi = (term1 - term2) / 2
        return msavi
    
    def _calculate_gesavi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """广义土壤调节植被指数"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        # 简化版本，实际需要土壤线参数
        denominator = nir + red + 0.5
        gesavi = np.where(denominator != 0, (nir - red) / denominator * 1.5, 0)
        return gesavi
    
    # 水分指数
    def _calculate_ndwi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """归一化水分指数 NDWI = (Green - NIR) / (Green + NIR)"""
        if 'green' not in self.band_positions or 'nir' not in self.band_positions:
            return None
        
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        
        denominator = green + nir
        ndwi = np.where(denominator != 0, (green - nir) / denominator, 0)
        return np.clip(ndwi, -1, 1)
    
    def _calculate_mndwi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """修正归一化水分指数 MNDWI = (Green - SWIR1) / (Green + SWIR1)"""
        if 'green' not in self.band_positions or 'swir1' not in self.band_positions:
            return None
        
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        swir1 = data[:, :, self.band_positions['swir1']].astype(np.float32)
        
        denominator = green + swir1
        mndwi = np.where(denominator != 0, (green - swir1) / denominator, 0)
        return np.clip(mndwi, -1, 1)
    
    def _calculate_ndmi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """归一化湿度指数 NDMI = (NIR - SWIR1) / (NIR + SWIR1)"""
        if 'nir' not in self.band_positions or 'swir1' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        swir1 = data[:, :, self.band_positions['swir1']].astype(np.float32)
        
        denominator = nir + swir1
        ndmi = np.where(denominator != 0, (nir - swir1) / denominator, 0)
        return np.clip(ndmi, -1, 1)
    
    def _calculate_wbi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """水分指数 WBI = Red / NIR"""
        if 'red' not in self.band_positions or 'nir' not in self.band_positions:
            return None
        
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        
        wbi = np.where(nir != 0, red / nir, 0)
        return wbi
    
    def _calculate_msi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """湿度应力指数 MSI = SWIR1 / NIR"""
        if 'swir1' not in self.band_positions or 'nir' not in self.band_positions:
            return None
        
        swir1 = data[:, :, self.band_positions['swir1']].astype(np.float32)
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        
        msi = np.where(nir != 0, swir1 / nir, 0)
        return msi
    
    def _calculate_lswi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """陆地表面水分指数 LSWI = (NIR - SWIR1) / (NIR + SWIR1)"""
        # LSWI与NDMI相同
        return self._calculate_ndmi(data)
    
    # 叶绿素指数
    def _calculate_mcari(self, data: np.ndarray) -> Optional[np.ndarray]:
        """修正叶绿素吸收指数"""
        required_bands = ['green', 'red', 'red_edge']
        if not all(band in self.band_positions for band in required_bands):
            return None
        
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        red_edge = data[:, :, self.band_positions['red_edge']].astype(np.float32)
        
        # MCARI = [(Red_edge - Red) - 0.2 * (Red_edge - Green)] * (Red_edge / Red)
        mcari = ((red_edge - red) - 0.2 * (red_edge - green)) * np.where(red != 0, red_edge / red, 0)
        return mcari
    
    def _calculate_tcari(self, data: np.ndarray) -> Optional[np.ndarray]:
        """变换叶绿素吸收指数"""
        required_bands = ['green', 'red', 'red_edge']
        if not all(band in self.band_positions for band in required_bands):
            return None
        
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        red_edge = data[:, :, self.band_positions['red_edge']].astype(np.float32)
        
        # TCARI = 3 * [(Red_edge - Red) - 0.2 * (Red_edge - Green) * (Red_edge / Red)]
        tcari = 3 * ((red_edge - red) - 0.2 * (red_edge - green) * np.where(red != 0, red_edge / red, 0))
        return tcari
    
    def _calculate_chl(self, data: np.ndarray) -> Optional[np.ndarray]:
        """叶绿素指数"""
        if 'red_edge' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        red_edge = data[:, :, self.band_positions['red_edge']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        chl = np.where(red != 0, red_edge / red - 1, 0)
        return chl
    
    def _calculate_ci(self, data: np.ndarray) -> Optional[np.ndarray]:
        """叶绿素指数 CI = (NIR / Red) - 1"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        ci = np.where(red != 0, nir / red - 1, 0)
        return ci
    
    def _calculate_cig(self, data: np.ndarray) -> Optional[np.ndarray]:
        """绿色叶绿素指数 CIG = (NIR / Green) - 1"""
        if 'nir' not in self.band_positions or 'green' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        
        cig = np.where(green != 0, nir / green - 1, 0)
        return cig
    
    # 红边指数
    def _calculate_rep(self, data: np.ndarray) -> Optional[np.ndarray]:
        """红边位置"""
        if 'red' not in self.band_positions or 'nir' not in self.band_positions:
            return None
        
        red_idx = self.band_positions['red']
        nir_idx = self.band_positions['nir']
        
        if nir_idx <= red_idx:
            return None
        
        # 计算红边区域的一阶导数
        red_edge_region = data[:, :, red_idx:nir_idx]
        if red_edge_region.shape[2] <= 1:
            return None
        
        derivative = np.diff(red_edge_region, axis=2)
        
        # 找到最大导数位置
        max_derivative_pos = np.argmax(derivative, axis=2).astype(np.float32)
        
        # 转换为波长
        if self.wavelengths is not None:
            red_edge_wavelengths = self.wavelengths[red_idx:nir_idx-1]
            rep = red_edge_wavelengths[0] + max_derivative_pos * (
                red_edge_wavelengths[-1] - red_edge_wavelengths[0]
            ) / len(red_edge_wavelengths)
            return rep
        
        return max_derivative_pos
    
    def _calculate_ndre(self, data: np.ndarray) -> Optional[np.ndarray]:
        """归一化红边指数 NDRE = (NIR - RedEdge) / (NIR + RedEdge)"""
        if 'nir' not in self.band_positions or 'red_edge' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red_edge = data[:, :, self.band_positions['red_edge']].astype(np.float32)
        
        denominator = nir + red_edge
        ndre = np.where(denominator != 0, (nir - red_edge) / denominator, 0)
        return np.clip(ndre, -1, 1)
    
    def _calculate_mndre(self, data: np.ndarray) -> Optional[np.ndarray]:
        """修正归一化红边指数"""
        # 使用不同的红边波段
        if 'red_edge_740' in self.band_positions and 'red_edge_705' in self.band_positions:
            red_edge_740 = data[:, :, self.band_positions['red_edge_740']].astype(np.float32)
            red_edge_705 = data[:, :, self.band_positions['red_edge_705']].astype(np.float32)
            
            denominator = red_edge_740 + red_edge_705
            mndre = np.where(denominator != 0, (red_edge_740 - red_edge_705) / denominator, 0)
            return np.clip(mndre, -1, 1)
        
        return self._calculate_ndre(data)
    
    def _calculate_ci_red_edge(self, data: np.ndarray) -> Optional[np.ndarray]:
        """红边叶绿素指数"""
        if 'nir' not in self.band_positions or 'red_edge' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red_edge = data[:, :, self.band_positions['red_edge']].astype(np.float32)
        
        ci_red_edge = np.where(red_edge != 0, nir / red_edge - 1, 0)
        return ci_red_edge
    
    def _calculate_mtci(self, data: np.ndarray) -> Optional[np.ndarray]:
        """MERIS陆地叶绿素指数"""
        # 需要特定的红边波段
        if all(band in self.band_positions for band in ['red_edge_705', 'red_edge_740', 'red']):
            red = data[:, :, self.band_positions['red']].astype(np.float32)
            red_edge_705 = data[:, :, self.band_positions['red_edge_705']].astype(np.float32)
            red_edge_740 = data[:, :, self.band_positions['red_edge_740']].astype(np.float32)
            
            denominator = red_edge_705 - red
            mtci = np.where(denominator != 0, (red_edge_740 - red_edge_705) / denominator, 0)
            return mtci
        
        return None
    
    # 光化学指数
    def _calculate_pri(self, data: np.ndarray) -> Optional[np.ndarray]:
        """光化学植被指数 PRI = (R531 - R570) / (R531 + R570)"""
        # 需要特定波长，使用最接近的波段
        if self.wavelengths is None:
            return None
        
        # 找到531nm和570nm附近的波段
        band_531 = np.argmin(np.abs(self.wavelengths - 531))
        band_570 = np.argmin(np.abs(self.wavelengths - 570))
        
        if band_531 == band_570:
            return None
        
        r531 = data[:, :, band_531].astype(np.float32)
        r570 = data[:, :, band_570].astype(np.float32)
        
        denominator = r531 + r570
        pri = np.where(denominator != 0, (r531 - r570) / denominator, 0)
        return np.clip(pri, -1, 1)
    
    def _calculate_sipi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """结构无关色素指数"""
        if self.wavelengths is None:
            return None
        
        # 使用445nm, 680nm, 800nm
        band_445 = np.argmin(np.abs(self.wavelengths - 445))
        band_680 = np.argmin(np.abs(self.wavelengths - 680))
        band_800 = np.argmin(np.abs(self.wavelengths - 800))
        
        r445 = data[:, :, band_445].astype(np.float32)
        r680 = data[:, :, band_680].astype(np.float32)
        r800 = data[:, :, band_800].astype(np.float32)
        
        denominator = r800 - r445
        sipi = np.where(denominator != 0, (r800 - r445) / (r800 + r680), 0)
        return sipi
    
    def _calculate_npci(self, data: np.ndarray) -> Optional[np.ndarray]:
        """归一化色素叶绿素指数"""
        if 'red' not in self.band_positions or 'blue' not in self.band_positions:
            return None
        
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        blue = data[:, :, self.band_positions['blue']].astype(np.float32)
        
        denominator = red + blue
        npci = np.where(denominator != 0, (red - blue) / denominator, 0)
        return np.clip(npci, -1, 1)
    
    def _calculate_pri_ci(self, data: np.ndarray) -> Optional[np.ndarray]:
        """PRI与CI的乘积"""
        pri = self._calculate_pri(data)
        ci = self._calculate_ci(data)
        
        if pri is not None and ci is not None:
            return pri * ci
        return None
    
    # 抗大气影响指数
    def _calculate_arvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """大气阻抗植被指数"""
        required_bands = ['nir', 'red', 'blue']
        if not all(band in self.band_positions for band in required_bands):
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        blue = data[:, :, self.band_positions['blue']].astype(np.float32)
        
        # ARVI = (NIR - (2*Red - Blue)) / (NIR + (2*Red - Blue))
        rb = 2 * red - blue
        denominator = nir + rb
        arvi = np.where(denominator != 0, (nir - rb) / denominator, 0)
        return np.clip(arvi, -1, 1)
    
    def _calculate_gemi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """全球环境监测指数"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        # GEMI = eta * (1 - 0.25 * eta) - (Red - 0.125) / (1 - Red)
        # 其中 eta = (2 * (NIR^2 - Red^2) + 1.5*NIR + 0.5*Red) / (NIR + Red + 0.5)
        denominator = nir + red + 0.5
        eta = np.where(denominator != 0, 
                      (2 * (nir**2 - red**2) + 1.5*nir + 0.5*red) / denominator, 0)
        
        red_term = np.where(red != 1, (red - 0.125) / (1 - red), 0)
        gemi = eta * (1 - 0.25 * eta) - red_term
        
        return gemi
    
    def _calculate_vari(self, data: np.ndarray) -> Optional[np.ndarray]:
        """可见光大气阻抗指数"""
        required_bands = ['green', 'red', 'blue']
        if not all(band in self.band_positions for band in required_bands):
            return None
        
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        blue = data[:, :, self.band_positions['blue']].astype(np.float32)
        
        # VARI = (Green - Red) / (Green + Red - Blue)
        denominator = green + red - blue
        vari = np.where(denominator != 0, (green - red) / denominator, 0)
        return vari
    
    def _calculate_gndvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """绿色归一化植被指数"""
        if 'nir' not in self.band_positions or 'green' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        
        denominator = nir + green
        gndvi = np.where(denominator != 0, (nir - green) / denominator, 0)
        return np.clip(gndvi, -1, 1)
    
    # 土壤线性指数
    def _calculate_tsavi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """变换土壤调节植被指数"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        # 简化版本，实际需要土壤线参数
        a, b = 1.0, 0.0  # 土壤线斜率和截距
        
        denominator = a * nir + red + 0.08 * (1 + a**2)
        tsavi = np.where(denominator != 0, 
                        a * (nir - a * red - b) / denominator, 0)
        return tsavi
    
    def _calculate_atsavi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """调整变换土壤调节植被指数"""
        # 简化实现
        return self._calculate_tsavi(data)
    
    # 综合指数
    def _calculate_tndvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """变换归一化植被指数"""
        ndvi = self._calculate_ndvi(data)
        if ndvi is not None:
            return np.sqrt(ndvi + 0.5)
        return None
    
    def _calculate_dvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """植被指数差值 DVI = NIR - Red"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        return nir - red
    
    def _calculate_ipvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """红外百分比植被指数"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        denominator = nir + red
        ipvi = np.where(denominator != 0, nir / denominator, 0)
        return ipvi
    
    def _calculate_rvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """比值植被指数 RVI = NIR / Red"""
        if 'nir' not in self.band_positions or 'red' not in self.band_positions:
            return None
        
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        
        rvi = np.where(red != 0, nir / red, 0)
        return rvi
    
    def _calculate_tvi(self, data: np.ndarray) -> Optional[np.ndarray]:
        """三角植被指数"""
        if 'green' not in self.band_positions or 'red' not in self.band_positions or 'nir' not in self.band_positions:
            return None
        
        green = data[:, :, self.band_positions['green']].astype(np.float32)
        red = data[:, :, self.band_positions['red']].astype(np.float32)
        nir = data[:, :, self.band_positions['nir']].astype(np.float32)
        
        # TVI = 0.5 * (120 * (NIR - Green) - 200 * (Red - Green))
        tvi = 0.5 * (120 * (nir - green) - 200 * (red - green))
        return tvi
    
    def get_index_description(self, index_name: str) -> str:
        """获取植被指数的描述信息"""
        descriptions = {
            'NDVI': '归一化植被指数，用于评估植被覆盖度和健康状况',
            'EVI': '增强植被指数，对高植被覆盖区域敏感度更高',
            'SAVI': '土壤调节植被指数，减少土壤背景影响',
            'NDWI': '归一化水分指数，用于检测水体和湿润表面',
            'MNDWI': '修正归一化水分指数，更适合水体提取',
            'NDMI': '归一化湿度指数，反映植被水分含量',
            'PRI': '光化学植被指数，反映植物光合作用效率',
            'REP': '红边位置，指示植被叶绿素含量',
            'MCARI': '修正叶绿素吸收指数，对叶绿素含量敏感',
            'ARVI': '大气阻抗植被指数，减少大气影响',
        }
        
        return descriptions.get(index_name, f"植被指数: {index_name}")
    
    def validate_indices(self, data: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
        """验证植被指数计算结果"""
        validation_results = {
            'input_shape': data.shape,
            'output_shape': indices.shape,
            'num_indices': indices.shape[2],
            'data_quality': {},
            'issues': []
        }
        
        # 检查数据质量
        nan_ratio = np.sum(np.isnan(indices)) / indices.size
        inf_ratio = np.sum(np.isinf(indices)) / indices.size
        
        validation_results['data_quality'] = {
            'nan_ratio': float(nan_ratio),
            'inf_ratio': float(inf_ratio),
            'value_range': (float(np.nanmin(indices)), float(np.nanmax(indices))),
            'mean_value': float(np.nanmean(indices)),
            'std_value': float(np.nanstd(indices))
        }
        
        # 检查合理性
        if nan_ratio > 0.1:
            validation_results['issues'].append(f"High NaN ratio: {nan_ratio:.2%}")
        
        if inf_ratio > 0:
            validation_results['issues'].append(f"Infinite values: {inf_ratio:.2%}")
        
        # 检查值域合理性
        min_val, max_val = validation_results['data_quality']['value_range']
        if min_val < -10 or max_val > 10:
            validation_results['issues'].append("Values outside reasonable range")
        
        validation_results['quality'] = 'Good' if len(validation_results['issues']) == 0 else 'Needs Review'
        
        return validation_results