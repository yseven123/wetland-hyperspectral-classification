"""
景观格局指数
===========

这个模块实现了全面的景观格局指数计算，用于分析湿地生态系统的空间格局特征。

主要功能：
- 面积指数：CA, PLAND, LPI等
- 形状指数：LSI, MSI, AWMSI等  
- 核心区指数：TCA, CPLAND, CAI等
- 聚集度指数：AI, IJI, COHESION等
- 多样性指数：PR, PD, SHDI, SIEI等
- 边缘指数：TE, ED, CWED等

参考标准：
- FRAGSTATS软件标准
- McGarigal景观生态学经典指标
- 湿地专用指数

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology
from skimage.morphology import disk, binary_erosion
from collections import defaultdict, Counter
import math
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class LandscapeMetricsCalculator:
    """
    景观格局指数计算器
    
    提供全面的景观格局指数计算功能，支持斑块级、类别级和景观级分析。
    """
    
    def __init__(self, 
                 pixel_size: float = 1.0,
                 unit: str = 'pixel',
                 eight_connectivity: bool = True,
                 **kwargs):
        """
        初始化景观指数计算器
        
        Parameters:
        -----------
        pixel_size : float, default=1.0
            像元大小
        unit : str, default='pixel'
            度量单位 ('pixel', 'meter', 'hectare')
        eight_connectivity : bool, default=True
            是否使用8连通
        """
        self.pixel_size = pixel_size
        self.unit = unit
        self.eight_connectivity = eight_connectivity
        self.config = kwargs
        
        # 连通性结构
        if eight_connectivity:
            self.connectivity = ndimage.generate_binary_structure(2, 2)
        else:
            self.connectivity = ndimage.generate_binary_structure(2, 1)
        
        # 缓存计算结果
        self.patch_cache = {}
        self.class_cache = {}
        self.landscape_cache = {}
    
    def calculate_all_metrics(self, 
                            classification_map: np.ndarray,
                            class_names: Optional[Dict[int, str]] = None,
                            levels: List[str] = ['patch', 'class', 'landscape'],
                            **kwargs) -> Dict[str, Any]:
        """
        计算所有景观指数
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        class_names : dict, optional
            类别名称映射
        levels : list, default=['patch', 'class', 'landscape']
            计算层级
            
        Returns:
        --------
        metrics : dict
            所有层级的指数结果
        """
        logger.info(f"开始计算景观指数 - 层级: {levels}")
        
        # 验证输入
        if classification_map.ndim != 2:
            raise ValueError("分类图像必须是2D数组")
        
        # 获取基本信息
        unique_classes = np.unique(classification_map)
        unique_classes = unique_classes[unique_classes != 0]  # 排除背景
        
        if class_names is None:
            class_names = {cls: f"Class_{cls}" for cls in unique_classes}
        
        # 计算基础数据
        patches_data = self._identify_patches(classification_map, unique_classes)
        
        results = {
            'metadata': {
                'image_shape': classification_map.shape,
                'pixel_size': self.pixel_size,
                'unit': self.unit,
                'total_classes': len(unique_classes),
                'classes': list(unique_classes),
                'class_names': class_names,
                'connectivity': '8-connected' if self.eight_connectivity else '4-connected'
            }
        }
        
        # 斑块级指数
        if 'patch' in levels:
            logger.info("计算斑块级指数")
            results['patch_metrics'] = self._calculate_patch_metrics(
                patches_data, classification_map
            )
        
        # 类别级指数
        if 'class' in levels:
            logger.info("计算类别级指数")
            results['class_metrics'] = self._calculate_class_metrics(
                patches_data, classification_map, unique_classes
            )
        
        # 景观级指数
        if 'landscape' in levels:
            logger.info("计算景观级指数")
            results['landscape_metrics'] = self._calculate_landscape_metrics(
                patches_data, classification_map, unique_classes
            )
        
        logger.info("景观指数计算完成")
        
        return results
    
    def _identify_patches(self, 
                         classification_map: np.ndarray, 
                         unique_classes: np.ndarray) -> Dict[int, Dict]:
        """识别和标记斑块"""
        patches_data = {}
        
        for class_id in unique_classes:
            # 创建类别掩码
            class_mask = classification_map == class_id
            
            # 连通组件标记
            labeled_patches, num_patches = ndimage.label(class_mask, self.connectivity)
            
            # 分析每个斑块
            patches_info = []
            for patch_id in range(1, num_patches + 1):
                patch_mask = labeled_patches == patch_id
                patch_info = self._analyze_patch(patch_mask, patch_id, class_id)
                patches_info.append(patch_info)
            
            patches_data[class_id] = {
                'num_patches': num_patches,
                'labeled_patches': labeled_patches,
                'patches_info': patches_info
            }
        
        return patches_data
    
    def _analyze_patch(self, 
                      patch_mask: np.ndarray, 
                      patch_id: int, 
                      class_id: int) -> Dict[str, Any]:
        """分析单个斑块的基本属性"""
        # 面积
        area_pixels = np.sum(patch_mask)
        area = area_pixels * (self.pixel_size ** 2)
        
        # 周长（基于像素边界）
        perimeter_pixels = self._calculate_perimeter(patch_mask)
        perimeter = perimeter_pixels * self.pixel_size
        
        # 质心
        coords = np.where(patch_mask)
        centroid = (np.mean(coords[0]), np.mean(coords[1]))
        
        # 外接矩形
        min_row, max_row = np.min(coords[0]), np.max(coords[0])
        min_col, max_col = np.min(coords[1]), np.max(coords[1])
        bounding_box = (min_row, min_col, max_row - min_row + 1, max_col - min_col + 1)
        
        # 回转半径
        gyration_radius = self._calculate_gyration_radius(coords)
        
        # 形状指数
        shape_index = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0
        
        # 分形维数
        fractal_dimension = 2 * np.log(perimeter_pixels / 4) / np.log(area_pixels) if area_pixels > 1 else 0
        
        return {
            'patch_id': patch_id,
            'class_id': class_id,
            'area_pixels': area_pixels,
            'area': area,
            'perimeter_pixels': perimeter_pixels,
            'perimeter': perimeter,
            'centroid': centroid,
            'bounding_box': bounding_box,
            'gyration_radius': gyration_radius,
            'shape_index': shape_index,
            'fractal_dimension': fractal_dimension,
            'coordinates': coords
        }
    
    def _calculate_perimeter(self, patch_mask: np.ndarray) -> float:
        """计算斑块周长"""
        # 使用边缘检测计算周长
        from skimage import measure
        
        # 找到轮廓
        contours = measure.find_contours(patch_mask.astype(float), 0.5)
        
        if not contours:
            return 0.0
        
        # 计算最大轮廓的长度
        max_perimeter = 0
        for contour in contours:
            # 计算轮廓长度
            if len(contour) > 1:
                perimeter = 0
                for i in range(len(contour)):
                    p1 = contour[i]
                    p2 = contour[(i + 1) % len(contour)]
                    perimeter += np.sqrt(np.sum((p2 - p1) ** 2))
                max_perimeter = max(max_perimeter, perimeter)
        
        return max_perimeter
    
    def _calculate_gyration_radius(self, coords: Tuple[np.ndarray, np.ndarray]) -> float:
        """计算回转半径"""
        if len(coords[0]) == 0:
            return 0.0
        
        # 质心
        centroid_y = np.mean(coords[0])
        centroid_x = np.mean(coords[1])
        
        # 到质心的距离的平方的平均值
        distances_sq = (coords[0] - centroid_y) ** 2 + (coords[1] - centroid_x) ** 2
        gyration_radius = np.sqrt(np.mean(distances_sq))
        
        return gyration_radius
    
    def _calculate_patch_metrics(self, 
                               patches_data: Dict[int, Dict], 
                               classification_map: np.ndarray) -> pd.DataFrame:
        """计算斑块级指数"""
        patch_metrics = []
        
        for class_id, class_data in patches_data.items():
            patches_info = class_data['patches_info']
            
            for patch_info in patches_info:
                metrics = {
                    'CLASS': class_id,
                    'PID': patch_info['patch_id'],
                    'AREA': patch_info['area'],
                    'PERIM': patch_info['perimeter'],
                    'SHAPE': patch_info['shape_index'],
                    'FRAC': patch_info['fractal_dimension'],
                    'GYRATE': patch_info['gyration_radius'],
                }
                
                # 核心区面积
                core_area = self._calculate_core_area(
                    patches_data[class_id]['labeled_patches'] == patch_info['patch_id']
                )
                metrics['CORE'] = core_area
                
                # 核心区指数
                if patch_info['area'] > 0:
                    metrics['CAI'] = core_area / patch_info['area'] * 100
                else:
                    metrics['CAI'] = 0
                
                # 邻近度指数
                proximity = self._calculate_proximity_index(
                    patch_info, patches_data, classification_map
                )
                metrics['PROX'] = proximity
                
                # 相似性指数
                similarity = self._calculate_similarity_index(
                    patch_info, patches_data, classification_map
                )
                metrics['SIMI'] = similarity
                
                patch_metrics.append(metrics)
        
        return pd.DataFrame(patch_metrics)
    
    def _calculate_core_area(self, patch_mask: np.ndarray, edge_depth: int = 1) -> float:
        """计算核心区面积"""
        if not np.any(patch_mask):
            return 0.0
        
        # 边缘侵蚀
        eroded = binary_erosion(patch_mask, disk(edge_depth))
        core_area_pixels = np.sum(eroded)
        core_area = core_area_pixels * (self.pixel_size ** 2)
        
        return core_area
    
    def _calculate_proximity_index(self, 
                                 patch_info: Dict[str, Any],
                                 patches_data: Dict[int, Dict],
                                 classification_map: np.ndarray,
                                 search_radius: float = 1000) -> float:
        """计算邻近度指数"""
        class_id = patch_info['class_id']
        patch_centroid = patch_info['centroid']
        patch_area = patch_info['area']
        
        if class_id not in patches_data:
            return 0.0
        
        proximity = 0.0
        
        # 搜索范围内的同类斑块
        for other_patch in patches_data[class_id]['patches_info']:
            if other_patch['patch_id'] == patch_info['patch_id']:
                continue
            
            other_centroid = other_patch['centroid']
            other_area = other_patch['area']
            
            # 计算距离
            distance = np.sqrt(
                (patch_centroid[0] - other_centroid[0]) ** 2 + 
                (patch_centroid[1] - other_centroid[1]) ** 2
            ) * self.pixel_size
            
            if distance <= search_radius and distance > 0:
                proximity += other_area / (distance ** 2)
        
        return proximity
    
    def _calculate_similarity_index(self, 
                                  patch_info: Dict[str, Any],
                                  patches_data: Dict[int, Dict],
                                  classification_map: np.ndarray) -> float:
        """计算相似性指数"""
        # 简化的相似性指数，基于形状和大小
        class_id = patch_info['class_id']
        patch_area = patch_info['area']
        patch_shape = patch_info['shape_index']
        
        if class_id not in patches_data:
            return 0.0
        
        similarities = []
        
        for other_patch in patches_data[class_id]['patches_info']:
            if other_patch['patch_id'] == patch_info['patch_id']:
                continue
            
            # 面积相似性
            area_ratio = min(patch_area, other_patch['area']) / max(patch_area, other_patch['area'])
            
            # 形状相似性
            shape_diff = abs(patch_shape - other_patch['shape_index'])
            shape_similarity = 1 / (1 + shape_diff)
            
            # 综合相似性
            similarity = (area_ratio + shape_similarity) / 2
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_class_metrics(self, 
                               patches_data: Dict[int, Dict],
                               classification_map: np.ndarray,
                               unique_classes: np.ndarray) -> pd.DataFrame:
        """计算类别级指数"""
        class_metrics = []
        total_area = classification_map.size * (self.pixel_size ** 2)
        
        for class_id in unique_classes:
            if class_id not in patches_data:
                continue
            
            class_data = patches_data[class_id]
            patches_info = class_data['patches_info']
            
            # 基本统计
            num_patches = class_data['num_patches']
            class_areas = [p['area'] for p in patches_info]
            class_perimeters = [p['perimeter'] for p in patches_info]
            class_shapes = [p['shape_index'] for p in patches_info]
            
            total_class_area = sum(class_areas)
            
            metrics = {
                'CLASS': class_id,
                'NP': num_patches,  # 斑块数量
                'CA': total_class_area,  # 类别面积
                'PLAND': (total_class_area / total_area) * 100,  # 景观百分比
                'LPI': (max(class_areas) / total_area) * 100 if class_areas else 0,  # 最大斑块指数
                'TE': sum(class_perimeters),  # 总边缘长度
                'ED': sum(class_perimeters) / total_area,  # 边缘密度
                'LSI': sum(class_perimeters) / (2 * np.sqrt(np.pi * total_class_area)) if total_class_area > 0 else 0,  # 景观形状指数
                'MSI': np.mean(class_shapes) if class_shapes else 0,  # 平均形状指数
                'AWMSI': self._area_weighted_mean_shape_index(class_areas, class_shapes),  # 面积加权平均形状指数
                'MPS': np.mean(class_areas) if class_areas else 0,  # 平均斑块大小
                'PSSD': np.std(class_areas) if len(class_areas) > 1 else 0,  # 斑块大小标准差
                'PSCOV': (np.std(class_areas) / np.mean(class_areas)) * 100 if class_areas and np.mean(class_areas) > 0 else 0,  # 斑块大小变异系数
            }
            
            # 核心区指数
            core_areas = [self._calculate_core_area(
                class_data['labeled_patches'] == p['patch_id']
            ) for p in patches_info]
            
            total_core_area = sum(core_areas)
            metrics['TCA'] = total_core_area  # 总核心区面积
            metrics['CPLAND'] = (total_core_area / total_area) * 100  # 核心区景观百分比
            metrics['DCAD'] = total_core_area / num_patches if num_patches > 0 else 0  # 核心区面积密度
            
            # 聚集度指数
            metrics['AI'] = self._calculate_aggregation_index(
                classification_map, class_id
            )
            
            # 内聚力指数  
            metrics['COHESION'] = self._calculate_cohesion_index(
                class_areas, class_perimeters
            )
            
            # 分离度指数
            metrics['DIVISION'] = self._calculate_division_index(
                class_areas, total_area
            )
            
            # 分割度指数
            metrics['SPLIT'] = self._calculate_split_index(
                class_areas, total_area
            )
            
            class_metrics.append(metrics)
        
        return pd.DataFrame(class_metrics)
    
    def _area_weighted_mean_shape_index(self, areas: List[float], shapes: List[float]) -> float:
        """计算面积加权平均形状指数"""
        if not areas or not shapes or len(areas) != len(shapes):
            return 0.0
        
        total_area = sum(areas)
        if total_area == 0:
            return 0.0
        
        weighted_sum = sum(area * shape for area, shape in zip(areas, shapes))
        return weighted_sum / total_area
    
    def _calculate_aggregation_index(self, classification_map: np.ndarray, class_id: int) -> float:
        """计算聚集度指数"""
        # 创建类别掩码
        class_mask = classification_map == class_id
        height, width = classification_map.shape
        
        # 计算相邻像素对
        same_adjacencies = 0
        total_adjacencies = 0
        
        # 检查水平相邻
        for i in range(height):
            for j in range(width - 1):
                if class_mask[i, j] or class_mask[i, j + 1]:
                    total_adjacencies += 1
                    if class_mask[i, j] and class_mask[i, j + 1]:
                        same_adjacencies += 1
        
        # 检查垂直相邻
        for i in range(height - 1):
            for j in range(width):
                if class_mask[i, j] or class_mask[i + 1, j]:
                    total_adjacencies += 1
                    if class_mask[i, j] and class_mask[i + 1, j]:
                        same_adjacencies += 1
        
        if total_adjacencies == 0:
            return 0.0
        
        return (same_adjacencies / total_adjacencies) * 100
    
    def _calculate_cohesion_index(self, areas: List[float], perimeters: List[float]) -> float:
        """计算内聚力指数"""
        if not areas or not perimeters:
            return 0.0
        
        # 计算相关参数
        total_area = sum(areas)
        total_perimeter = sum(perimeters)
        
        if total_area == 0:
            return 0.0
        
        # 内聚力指数公式
        z = total_perimeter
        a = total_area
        
        cohesion = (1 - z / (z + np.sqrt(a))) / (1 - 1 / np.sqrt(a)) if a > 1 else 0
        
        return cohesion * 100
    
    def _calculate_division_index(self, areas: List[float], total_area: float) -> float:
        """计算分离度指数"""
        if not areas or total_area == 0:
            return 0.0
        
        # 分离度指数 = 1 - sum((ai/A)^2)
        area_proportions = [area / total_area for area in areas]
        division = 1 - sum(prop ** 2 for prop in area_proportions)
        
        return division
    
    def _calculate_split_index(self, areas: List[float], total_area: float) -> float:
        """计算分割度指数"""
        if not areas or total_area == 0:
            return 0.0
        
        # 分割度指数 = A^2 / sum(ai^2)
        sum_area_squared = sum(area ** 2 for area in areas)
        
        if sum_area_squared == 0:
            return 0.0
        
        split = (total_area ** 2) / sum_area_squared
        
        return split
    
    def _calculate_landscape_metrics(self, 
                                   patches_data: Dict[int, Dict],
                                   classification_map: np.ndarray,
                                   unique_classes: np.ndarray) -> Dict[str, float]:
        """计算景观级指数"""
        total_area = classification_map.size * (self.pixel_size ** 2)
        
        # 基本统计
        total_patches = sum(data['num_patches'] for data in patches_data.values())
        all_areas = []
        all_perimeters = []
        
        for class_data in patches_data.values():
            for patch in class_data['patches_info']:
                all_areas.append(patch['area'])
                all_perimeters.append(patch['perimeter'])
        
        metrics = {
            'TA': total_area,  # 总面积
            'NP': total_patches,  # 总斑块数
            'PD': total_patches / total_area,  # 斑块密度
            'LPI': (max(all_areas) / total_area) * 100 if all_areas else 0,  # 最大斑块指数
            'TE': sum(all_perimeters),  # 总边缘长度
            'ED': sum(all_perimeters) / total_area,  # 边缘密度
            'MPS': np.mean(all_areas) if all_areas else 0,  # 平均斑块大小
            'PSSD': np.std(all_areas) if len(all_areas) > 1 else 0,  # 斑块大小标准差
            'PSCOV': (np.std(all_areas) / np.mean(all_areas)) * 100 if all_areas and np.mean(all_areas) > 0 else 0,  # 斑块大小变异系数
        }
        
        # 多样性指数
        class_areas = {}
        for class_id in unique_classes:
            if class_id in patches_data:
                class_areas[class_id] = sum(p['area'] for p in patches_data[class_id]['patches_info'])
            else:
                class_areas[class_id] = 0
        
        # 斑块丰富度
        metrics['PR'] = len([area for area in class_areas.values() if area > 0])
        
        # 斑块丰富度密度
        metrics['PRD'] = metrics['PR'] / total_area
        
        # Shannon多样性指数
        metrics['SHDI'] = self._calculate_shannon_diversity(class_areas, total_area)
        
        # Simpson多样性指数
        metrics['SIDI'] = self._calculate_simpson_diversity(class_areas, total_area)
        
        # Shannon均匀度指数
        if metrics['PR'] > 1:
            metrics['SHEI'] = metrics['SHDI'] / np.log(metrics['PR'])
        else:
            metrics['SHEI'] = 0
        
        # Simpson均匀度指数
        if metrics['PR'] > 1:
            metrics['SIEI'] = metrics['SIDI'] / (1 - 1/metrics['PR'])
        else:
            metrics['SIEI'] = 0
        
        # 景观形状指数
        total_perimeter = sum(all_perimeters)
        if total_area > 0:
            metrics['LSI'] = total_perimeter / (2 * np.sqrt(np.pi * total_area))
        else:
            metrics['LSI'] = 0
        
        # 平均形状指数
        all_shapes = []
        for class_data in patches_data.values():
            for patch in class_data['patches_info']:
                all_shapes.append(patch['shape_index'])
        
        metrics['MSI'] = np.mean(all_shapes) if all_shapes else 0
        
        # 面积加权平均形状指数
        metrics['AWMSI'] = self._area_weighted_mean_shape_index(all_areas, all_shapes)
        
        # 散布与并列指数
        metrics['IJI'] = self._calculate_interspersion_juxtaposition_index(
            classification_map, unique_classes
        )
        
        # 聚集度指数（景观级）
        all_ai = []
        for class_id in unique_classes:
            ai = self._calculate_aggregation_index(classification_map, class_id)
            if ai > 0:
                all_ai.append(ai)
        
        metrics['AI'] = np.mean(all_ai) if all_ai else 0
        
        return metrics
    
    def _calculate_shannon_diversity(self, class_areas: Dict[int, float], total_area: float) -> float:
        """计算Shannon多样性指数"""
        if total_area == 0:
            return 0.0
        
        shannon = 0.0
        for area in class_areas.values():
            if area > 0:
                proportion = area / total_area
                shannon -= proportion * np.log(proportion)
        
        return shannon
    
    def _calculate_simpson_diversity(self, class_areas: Dict[int, float], total_area: float) -> float:
        """计算Simpson多样性指数"""
        if total_area == 0:
            return 0.0
        
        simpson = 0.0
        for area in class_areas.values():
            if area > 0:
                proportion = area / total_area
                simpson += proportion ** 2
        
        return 1 - simpson
    
    def _calculate_interspersion_juxtaposition_index(self, 
                                                   classification_map: np.ndarray,
                                                   unique_classes: np.ndarray) -> float:
        """计算散布与并列指数"""
        height, width = classification_map.shape
        adjacency_matrix = np.zeros((len(unique_classes), len(unique_classes)))
        
        # 创建类别到索引的映射
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        
        # 计算相邻矩阵
        for i in range(height):
            for j in range(width):
                current_class = classification_map[i, j]
                if current_class not in class_to_idx:
                    continue
                
                current_idx = class_to_idx[current_class]
                
                # 检查4个邻居
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_class = classification_map[ni, nj]
                        if neighbor_class in class_to_idx:
                            neighbor_idx = class_to_idx[neighbor_class]
                            adjacency_matrix[current_idx, neighbor_idx] += 1
        
        # 计算IJI
        m = len(unique_classes)
        if m <= 1:
            return 0.0
        
        # 计算观察到的邻接类型数
        observed_adjacencies = 0
        total_adjacencies = np.sum(adjacency_matrix)
        
        for i in range(m):
            for j in range(i, m):  # 避免重复计算
                if adjacency_matrix[i, j] + adjacency_matrix[j, i] > 0:
                    observed_adjacencies += 1
        
        # 最大可能邻接类型数
        max_adjacencies = m * (m - 1) / 2 + m  # 包括自身邻接
        
        if max_adjacencies == 0:
            return 0.0
        
        return (observed_adjacencies / max_adjacencies) * 100


class WetlandMetricsCalculator(LandscapeMetricsCalculator):
    """
    湿地专用指数计算器
    
    在标准景观指数基础上，增加湿地生态系统专用指数。
    """
    
    def __init__(self, **kwargs):
        """初始化湿地指数计算器"""
        super().__init__(**kwargs)
        
        # 湿地类别映射（可根据实际情况调整）
        self.wetland_classes = {
            'water': [1, 'water', 'open_water'],
            'shallow_water': [2, 'shallow_water', 'shallow'],
            'wetland_vegetation': [3, 'wetland_vegetation', 'emergent', 'marsh'],
            'mudflat': [4, 'mudflat', 'mud', 'bare_soil'],
            'dry_land': [5, 'dry_land', 'upland'],
            'agriculture': [6, 'agriculture', 'cropland']
        }
    
    def calculate_wetland_specific_metrics(self, 
                                         classification_map: np.ndarray,
                                         water_level_data: Optional[np.ndarray] = None,
                                         **kwargs) -> Dict[str, Any]:
        """
        计算湿地专用指数
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        water_level_data : np.ndarray, optional
            水位数据
            
        Returns:
        --------
        wetland_metrics : dict
            湿地专用指数
        """
        logger.info("计算湿地专用指数")
        
        # 识别湿地相关类别
        wetland_map = self._identify_wetland_classes(classification_map)
        
        metrics = {}
        
        # 水体连通性指数
        metrics['WATER_CONNECTIVITY'] = self._calculate_water_connectivity(wetland_map)
        
        # 湿地边缘密度
        metrics['WETLAND_EDGE_DENSITY'] = self._calculate_wetland_edge_density(wetland_map)
        
        # 水陆交错带指数
        metrics['ECOTONE_INDEX'] = self._calculate_ecotone_index(wetland_map)
        
        # 湿地破碎化指数
        metrics['WETLAND_FRAGMENTATION'] = self._calculate_wetland_fragmentation(wetland_map)
        
        # 湿地完整性指数
        metrics['WETLAND_INTEGRITY'] = self._calculate_wetland_integrity(wetland_map)
        
        # 水位变化指数（如果有水位数据）
        if water_level_data is not None:
            metrics['WATER_LEVEL_VARIATION'] = self._calculate_water_level_variation(
                wetland_map, water_level_data
            )
        
        # 湿地类型多样性
        metrics['WETLAND_TYPE_DIVERSITY'] = self._calculate_wetland_type_diversity(wetland_map)
        
        # 核心湿地面积比例
        metrics['CORE_WETLAND_RATIO'] = self._calculate_core_wetland_ratio(wetland_map)
        
        logger.info("湿地专用指数计算完成")
        
        return metrics
    
    def _identify_wetland_classes(self, classification_map: np.ndarray) -> Dict[str, np.ndarray]:
        """识别湿地相关类别"""
        wetland_map = {}
        
        # 根据类别映射创建各类湿地掩码
        unique_classes = np.unique(classification_map)
        
        wetland_map['water'] = np.isin(classification_map, [1, 2])  # 水体和浅水
        wetland_map['vegetation'] = np.isin(classification_map, [3])  # 湿地植被
        wetland_map['mudflat'] = np.isin(classification_map, [4])  # 泥滩
        wetland_map['dry_land'] = np.isin(classification_map, [5, 6])  # 旱地
        
        # 综合湿地掩码
        wetland_map['all_wetland'] = (
            wetland_map['water'] | 
            wetland_map['vegetation'] | 
            wetland_map['mudflat']
        )
        
        return wetland_map
    
    def _calculate_water_connectivity(self, wetland_map: Dict[str, np.ndarray]) -> float:
        """计算水体连通性指数"""
        water_mask = wetland_map['water']
        
        if not np.any(water_mask):
            return 0.0
        
        # 连通组件分析
        labeled_water, num_components = ndimage.label(water_mask, self.connectivity)
        
        if num_components == 0:
            return 0.0
        
        # 计算各组件面积
        component_areas = []
        for i in range(1, num_components + 1):
            area = np.sum(labeled_water == i)
            component_areas.append(area)
        
        # 连通性 = 最大组件面积 / 总水体面积
        total_water_area = np.sum(water_mask)
        max_component_area = max(component_areas)
        
        connectivity = max_component_area / total_water_area
        
        return connectivity
    
    def _calculate_wetland_edge_density(self, wetland_map: Dict[str, np.ndarray]) -> float:
        """计算湿地边缘密度"""
        wetland_mask = wetland_map['all_wetland']
        
        # 计算边缘
        from scipy.ndimage import binary_dilation
        
        # 膨胀后减去原图得到边缘
        dilated = binary_dilation(wetland_mask)
        edge = dilated & (~wetland_mask)
        
        edge_length = np.sum(edge) * self.pixel_size
        total_area = wetland_mask.size * (self.pixel_size ** 2)
        
        edge_density = edge_length / total_area
        
        return edge_density
    
    def _calculate_ecotone_index(self, wetland_map: Dict[str, np.ndarray]) -> float:
        """计算水陆交错带指数"""
        water_mask = wetland_map['water']
        dry_mask = wetland_map['dry_land']
        
        # 计算水陆交界线长度
        from scipy.ndimage import binary_dilation
        
        # 水体边界
        water_boundary = binary_dilation(water_mask) & (~water_mask)
        
        # 旱地边界
        dry_boundary = binary_dilation(dry_mask) & (~dry_mask)
        
        # 交错带 = 水体边界与旱地边界的交集附近区域
        ecotone = water_boundary & dry_boundary
        
        ecotone_length = np.sum(ecotone) * self.pixel_size
        total_boundary = (np.sum(water_boundary) + np.sum(dry_boundary)) * self.pixel_size
        
        if total_boundary == 0:
            return 0.0
        
        ecotone_index = ecotone_length / total_boundary
        
        return ecotone_index
    
    def _calculate_wetland_fragmentation(self, wetland_map: Dict[str, np.ndarray]) -> float:
        """计算湿地破碎化指数"""
        wetland_mask = wetland_map['all_wetland']
        
        # 连通组件分析
        labeled_wetland, num_components = ndimage.label(wetland_mask, self.connectivity)
        
        if num_components == 0:
            return 1.0  # 完全破碎
        
        # 计算各组件面积
        component_areas = []
        for i in range(1, num_components + 1):
            area = np.sum(labeled_wetland == i)
            component_areas.append(area)
        
        # 破碎化指数 = 1 - (最大斑块面积比例)^2
        total_wetland_area = np.sum(wetland_mask)
        if total_wetland_area == 0:
            return 1.0
        
        max_patch_ratio = max(component_areas) / total_wetland_area
        fragmentation = 1 - (max_patch_ratio ** 2)
        
        return fragmentation
    
    def _calculate_wetland_integrity(self, wetland_map: Dict[str, np.ndarray]) -> float:
        """计算湿地完整性指数"""
        # 综合多个因子计算完整性
        
        # 1. 面积因子
        total_area = wetland_map['all_wetland'].size
        wetland_area = np.sum(wetland_map['all_wetland'])
        area_factor = wetland_area / total_area
        
        # 2. 连通性因子
        connectivity_factor = self._calculate_water_connectivity(wetland_map)
        
        # 3. 边缘效应因子（核心区比例）
        core_ratio = self._calculate_core_wetland_ratio(wetland_map)
        
        # 4. 多样性因子
        diversity_factor = self._calculate_wetland_type_diversity(wetland_map) / 4  # 归一化
        
        # 综合完整性指数
        integrity = (area_factor + connectivity_factor + core_ratio + diversity_factor) / 4
        
        return integrity
    
    def _calculate_water_level_variation(self, 
                                       wetland_map: Dict[str, np.ndarray],
                                       water_level_data: np.ndarray) -> float:
        """计算水位变化指数"""
        water_mask = wetland_map['water']
        
        if not np.any(water_mask):
            return 0.0
        
        # 水体区域的水位变化
        water_levels = water_level_data[water_mask]
        
        if len(water_levels) == 0:
            return 0.0
        
        # 变异系数
        mean_level = np.mean(water_levels)
        std_level = np.std(water_levels)
        
        if mean_level == 0:
            return 0.0
        
        variation_index = std_level / mean_level
        
        return variation_index
    
    def _calculate_wetland_type_diversity(self, wetland_map: Dict[str, np.ndarray]) -> int:
        """计算湿地类型多样性"""
        wetland_types = ['water', 'vegetation', 'mudflat']
        
        present_types = 0
        for wetland_type in wetland_types:
            if wetland_type in wetland_map and np.any(wetland_map[wetland_type]):
                present_types += 1
        
        return present_types
    
    def _calculate_core_wetland_ratio(self, wetland_map: Dict[str, np.ndarray]) -> float:
        """计算核心湿地面积比例"""
        wetland_mask = wetland_map['all_wetland']
        
        if not np.any(wetland_mask):
            return 0.0
        
        # 计算核心区（距边缘至少2个像素）
        from skimage.morphology import binary_erosion, disk
        
        core_wetland = binary_erosion(wetland_mask, disk(2))
        
        core_area = np.sum(core_wetland)
        total_wetland_area = np.sum(wetland_mask)
        
        if total_wetland_area == 0:
            return 0.0
        
        core_ratio = core_area / total_wetland_area
        
        return core_ratio


def calculate_landscape_metrics(classification_map: np.ndarray,
                              pixel_size: float = 1.0,
                              levels: List[str] = ['class', 'landscape'],
                              wetland_specific: bool = True,
                              **kwargs) -> Dict[str, Any]:
    """
    计算景观指数的便捷函数
    
    Parameters:
    -----------
    classification_map : np.ndarray
        分类图像
    pixel_size : float, default=1.0
        像元大小
    levels : list, default=['class', 'landscape']
        计算层级
    wetland_specific : bool, default=True
        是否计算湿地专用指数
        
    Returns:
    --------
    metrics : dict
        计算结果
    """
    if wetland_specific:
        calculator = WetlandMetricsCalculator(pixel_size=pixel_size, **kwargs)
    else:
        calculator = LandscapeMetricsCalculator(pixel_size=pixel_size, **kwargs)
    
    # 计算标准指数
    metrics = calculator.calculate_all_metrics(classification_map, levels=levels)
    
    # 计算湿地专用指数
    if wetland_specific and isinstance(calculator, WetlandMetricsCalculator):
        wetland_metrics = calculator.calculate_wetland_specific_metrics(classification_map)
        metrics['wetland_specific_metrics'] = wetland_metrics
    
    return metrics