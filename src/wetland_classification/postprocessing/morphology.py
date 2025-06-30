"""
形态学操作
=========

这个模块实现了高级形态学操作，用于优化分类结果的空间结构。
主要功能包括连通性分析、区域处理、边界优化等。

主要算法：
- 连通组件分析
- 区域属性分析
- 边界跟踪与优化
- 空洞填充
- 区域合并与分割
- 距离变换
- 骨架提取

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes, distance_transform_edt
from skimage import morphology, measure, segmentation, filters
from skimage.morphology import (
    binary_erosion, binary_dilation, binary_opening, binary_closing,
    disk, square, diamond, rectangle, area_opening, area_closing,
    remove_small_objects, remove_small_holes, skeletonize
)
from skimage.measure import regionprops, label as sk_label
from skimage.segmentation import watershed
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class MorphologyProcessor:
    """
    形态学处理器基类
    
    提供通用的形态学操作接口和工具函数。
    """
    
    def __init__(self, 
                 connectivity: int = 2,
                 preserve_topology: bool = True,
                 **kwargs):
        """
        初始化形态学处理器
        
        Parameters:
        -----------
        connectivity : int, default=2
            连通性 (1: 4连通, 2: 8连通)
        preserve_topology : bool, default=True
            是否保持拓扑结构
        """
        self.connectivity = connectivity
        self.preserve_topology = preserve_topology
        self.config = kwargs
    
    def process(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        处理分类图像
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        processed_map : np.ndarray
            处理后的分类图像
        """
        raise NotImplementedError("子类必须实现process方法")
    
    def _validate_input(self, classification_map: np.ndarray):
        """验证输入数据"""
        if classification_map.ndim != 2:
            raise ValueError("分类图像必须是2D数组")
        
        if classification_map.size == 0:
            raise ValueError("分类图像不能为空")


class ConnectedComponentAnalyzer(MorphologyProcessor):
    """
    连通组件分析器
    
    分析和处理分类图像中的连通组件，包括组件统计、过滤等。
    """
    
    def __init__(self, 
                 min_size: int = 10,
                 max_size: Optional[int] = None,
                 analyze_properties: bool = True,
                 **kwargs):
        """
        初始化连通组件分析器
        
        Parameters:
        -----------
        min_size : int, default=10
            最小组件大小（像素数）
        max_size : int, optional
            最大组件大小
        analyze_properties : bool, default=True
            是否分析组件属性
        """
        super().__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size
        self.analyze_properties = analyze_properties
        self.component_stats = {}
    
    def process(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        处理连通组件
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        processed_map : np.ndarray
            处理后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"开始连通组件分析 - 最小大小: {self.min_size}")
        
        # 获取唯一类别
        unique_classes = np.unique(classification_map)
        processed_map = classification_map.copy()
        
        self.component_stats = {}
        
        # 对每个类别进行连通组件分析
        for class_id in unique_classes:
            logger.debug(f"分析类别 {class_id} 的连通组件")
            
            # 创建二值掩码
            binary_mask = classification_map == class_id
            
            # 连通组件标记
            labeled_mask, num_components = ndimage.label(
                binary_mask, 
                structure=ndimage.generate_binary_structure(2, self.connectivity)
            )
            
            # 分析组件属性
            if self.analyze_properties:
                component_props = self._analyze_components(labeled_mask, num_components)
                self.component_stats[class_id] = component_props
            
            # 过滤组件
            filtered_mask = self._filter_components(labeled_mask, binary_mask)
            
            # 更新分类图像
            processed_map = np.where(filtered_mask, class_id, processed_map)
            processed_map = np.where((~filtered_mask) & binary_mask, 0, processed_map)
        
        # 后处理：分配未分类像素
        processed_map = self._reassign_unclassified_pixels(
            processed_map, classification_map, unique_classes
        )
        
        logger.info("连通组件分析完成")
        
        return processed_map
    
    def _analyze_components(self, labeled_mask: np.ndarray, 
                           num_components: int) -> Dict[str, Any]:
        """分析连通组件属性"""
        if num_components == 0:
            return {'num_components': 0, 'total_area': 0}
        
        # 使用regionprops分析组件
        props = regionprops(labeled_mask)
        
        areas = [prop.area for prop in props]
        centroids = [prop.centroid for prop in props]
        eccentricities = [prop.eccentricity for prop in props]
        solidity = [prop.solidity for prop in props]
        
        stats = {
            'num_components': num_components,
            'total_area': sum(areas),
            'areas': areas,
            'centroids': centroids,
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'mean_eccentricity': np.mean(eccentricities),
            'mean_solidity': np.mean(solidity)
        }
        
        return stats
    
    def _filter_components(self, labeled_mask: np.ndarray, 
                          binary_mask: np.ndarray) -> np.ndarray:
        """过滤连通组件"""
        filtered_mask = np.zeros_like(binary_mask, dtype=bool)
        
        # 获取组件信息
        props = regionprops(labeled_mask)
        
        for prop in props:
            area = prop.area
            
            # 大小过滤
            if area < self.min_size:
                continue
            
            if self.max_size is not None and area > self.max_size:
                continue
            
            # 将符合条件的组件添加到结果中
            component_mask = labeled_mask == prop.label
            filtered_mask |= component_mask
        
        return filtered_mask
    
    def _reassign_unclassified_pixels(self, processed_map: np.ndarray,
                                     original_map: np.ndarray,
                                     unique_classes: np.ndarray) -> np.ndarray:
        """重新分配未分类像素"""
        # 找到被移除的像素
        unclassified_mask = (processed_map == 0) & (original_map != 0)
        
        if not np.any(unclassified_mask):
            return processed_map
        
        logger.debug(f"重新分配 {np.sum(unclassified_mask)} 个未分类像素")
        
        # 使用最近邻分配
        unclassified_indices = np.where(unclassified_mask)
        classified_indices = np.where(processed_map != 0)
        
        if len(classified_indices[0]) == 0:
            return processed_map
        
        # 计算距离并分配最近的类别
        from scipy.spatial.distance import cdist
        
        unclassified_coords = np.column_stack(unclassified_indices)
        classified_coords = np.column_stack(classified_indices)
        classified_labels = processed_map[classified_indices]
        
        if len(unclassified_coords) > 0 and len(classified_coords) > 0:
            distances = cdist(unclassified_coords, classified_coords)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_labels = classified_labels[nearest_indices]
            
            processed_map[unclassified_indices] = nearest_labels
        
        return processed_map
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """获取组件统计信息"""
        return self.component_stats


class RegionProcessor(MorphologyProcessor):
    """
    区域处理器
    
    对分类区域进行高级处理，包括区域合并、分割、边界优化等。
    """
    
    def __init__(self, 
                 merge_threshold: float = 0.1,
                 split_threshold: float = 0.8,
                 boundary_smooth: bool = True,
                 **kwargs):
        """
        初始化区域处理器
        
        Parameters:
        -----------
        merge_threshold : float, default=0.1
            区域合并阈值
        split_threshold : float, default=0.8
            区域分割阈值
        boundary_smooth : bool, default=True
            是否进行边界平滑
        """
        super().__init__(**kwargs)
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.boundary_smooth = boundary_smooth
        self.region_stats = {}
    
    def process(self, classification_map: np.ndarray, 
               feature_map: Optional[np.ndarray] = None,
               **kwargs) -> np.ndarray:
        """
        处理分类区域
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
        feature_map : np.ndarray, optional
            特征图用于区域分析
            
        Returns:
        --------
        processed_map : np.ndarray
            处理后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info("开始区域处理")
        
        processed_map = classification_map.copy()
        
        # 1. 区域合并
        if self.merge_threshold > 0:
            processed_map = self._merge_similar_regions(processed_map, feature_map)
        
        # 2. 区域分割
        if self.split_threshold < 1.0:
            processed_map = self._split_heterogeneous_regions(processed_map, feature_map)
        
        # 3. 边界平滑
        if self.boundary_smooth:
            processed_map = self._smooth_boundaries(processed_map)
        
        logger.info("区域处理完成")
        
        return processed_map
    
    def _merge_similar_regions(self, classification_map: np.ndarray,
                              feature_map: Optional[np.ndarray] = None) -> np.ndarray:
        """合并相似区域"""
        logger.debug("开始区域合并")
        
        if feature_map is None:
            # 如果没有特征图，基于空间邻接性合并
            return self._merge_by_adjacency(classification_map)
        else:
            # 基于特征相似性合并
            return self._merge_by_similarity(classification_map, feature_map)
    
    def _merge_by_adjacency(self, classification_map: np.ndarray) -> np.ndarray:
        """基于邻接性的区域合并"""
        # 获取区域标签
        labeled_map, num_regions = ndimage.label(classification_map > 0)
        
        # 分析区域邻接关系
        adjacency_matrix = self._compute_adjacency_matrix(
            labeled_map, classification_map, num_regions
        )
        
        # 合并策略：合并面积较小且类别相同的邻接区域
        merged_map = classification_map.copy()
        
        # 这里可以实现具体的合并逻辑
        # 为简化，当前返回原图
        return merged_map
    
    def _merge_by_similarity(self, classification_map: np.ndarray,
                           feature_map: np.ndarray) -> np.ndarray:
        """基于特征相似性的区域合并"""
        # 计算每个区域的平均特征
        unique_classes = np.unique(classification_map)
        region_features = {}
        
        for class_id in unique_classes:
            if class_id == 0:  # 跳过背景
                continue
            
            class_mask = classification_map == class_id
            if feature_map.ndim == 2:
                region_features[class_id] = np.mean(feature_map[class_mask])
            else:
                region_features[class_id] = np.mean(feature_map[class_mask], axis=0)
        
        # 基于特征相似性合并区域
        merged_map = classification_map.copy()
        
        # 这里可以实现基于特征距离的合并逻辑
        # 为简化，当前返回原图
        return merged_map
    
    def _split_heterogeneous_regions(self, classification_map: np.ndarray,
                                   feature_map: Optional[np.ndarray] = None) -> np.ndarray:
        """分割异质区域"""
        logger.debug("开始区域分割")
        
        if feature_map is None:
            return classification_map
        
        # 对每个类别的大区域进行分割分析
        unique_classes = np.unique(classification_map)
        split_map = classification_map.copy()
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            
            class_mask = classification_map == class_id
            class_regions = self._analyze_region_homogeneity(
                class_mask, feature_map, class_id
            )
            
            # 如果区域异质性高，进行分割
            if class_regions['heterogeneity'] > self.split_threshold:
                split_result = self._watershed_split(class_mask, feature_map)
                # 更新分割结果到split_map
                # 这里需要实现具体的分割逻辑
        
        return split_map
    
    def _analyze_region_homogeneity(self, region_mask: np.ndarray,
                                  feature_map: np.ndarray,
                                  class_id: int) -> Dict[str, Any]:
        """分析区域同质性"""
        if feature_map.ndim == 2:
            region_features = feature_map[region_mask]
        else:
            region_features = feature_map[region_mask]
        
        # 计算特征统计
        mean_feature = np.mean(region_features, axis=0)
        std_feature = np.std(region_features, axis=0)
        
        # 计算异质性（变异系数）
        cv = np.mean(std_feature / (mean_feature + 1e-10))
        
        return {
            'class_id': class_id,
            'size': np.sum(region_mask),
            'mean_feature': mean_feature,
            'std_feature': std_feature,
            'heterogeneity': cv
        }
    
    def _watershed_split(self, region_mask: np.ndarray,
                        feature_map: np.ndarray) -> np.ndarray:
        """使用分水岭算法分割区域"""
        # 计算距离变换
        distance = distance_transform_edt(region_mask)
        
        # 找到局部最大值作为种子点
        local_maxima = morphology.local_maxima(distance, min_distance=10)
        markers, _ = ndimage.label(local_maxima)
        
        # 分水岭分割
        if feature_map.ndim == 2:
            gradient = filters.sobel(feature_map)
        else:
            gradient = np.sqrt(np.sum(np.gradient(feature_map, axis=(0, 1))**2, axis=0))
        
        labels = watershed(gradient, markers, mask=region_mask)
        
        return labels
    
    def _smooth_boundaries(self, classification_map: np.ndarray) -> np.ndarray:
        """平滑区域边界"""
        logger.debug("开始边界平滑")
        
        smoothed_map = classification_map.copy()
        unique_classes = np.unique(classification_map)
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            
            # 创建类别掩码
            class_mask = classification_map == class_id
            
            # 形态学平滑
            # 先开运算再闭运算
            smoothed_mask = binary_opening(class_mask, disk(2))
            smoothed_mask = binary_closing(smoothed_mask, disk(3))
            
            # 更新分类图
            smoothed_map = np.where(smoothed_mask, class_id, smoothed_map)
            smoothed_map = np.where((~smoothed_mask) & class_mask, 0, smoothed_map)
        
        return smoothed_map
    
    def _compute_adjacency_matrix(self, labeled_map: np.ndarray,
                                 classification_map: np.ndarray,
                                 num_regions: int) -> np.ndarray:
        """计算区域邻接矩阵"""
        adjacency = np.zeros((num_regions + 1, num_regions + 1), dtype=bool)
        
        # 检查4连通邻域
        for i in range(labeled_map.shape[0] - 1):
            for j in range(labeled_map.shape[1] - 1):
                current_label = labeled_map[i, j]
                
                # 检查右邻居
                right_label = labeled_map[i, j + 1]
                if current_label != right_label and current_label > 0 and right_label > 0:
                    adjacency[current_label, right_label] = True
                    adjacency[right_label, current_label] = True
                
                # 检查下邻居
                bottom_label = labeled_map[i + 1, j]
                if current_label != bottom_label and current_label > 0 and bottom_label > 0:
                    adjacency[current_label, bottom_label] = True
                    adjacency[bottom_label, current_label] = True
        
        return adjacency


class HoleFillingProcessor(MorphologyProcessor):
    """
    空洞填充处理器
    
    识别并填充分类区域中的空洞。
    """
    
    def __init__(self, 
                 min_hole_size: int = 5,
                 max_hole_size: Optional[int] = None,
                 fill_strategy: str = 'surrounding',
                 **kwargs):
        """
        初始化空洞填充处理器
        
        Parameters:
        -----------
        min_hole_size : int, default=5
            需要填充的最小空洞大小
        max_hole_size : int, optional
            需要填充的最大空洞大小
        fill_strategy : str, default='surrounding'
            填充策略 ('surrounding', 'majority', 'interpolation')
        """
        super().__init__(**kwargs)
        self.min_hole_size = min_hole_size
        self.max_hole_size = max_hole_size
        self.fill_strategy = fill_strategy
        self.filled_holes = []
    
    def process(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        填充空洞
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        processed_map : np.ndarray
            填充空洞后的分类图像
        """
        self._validate_input(classification_map)
        
        logger.info(f"开始空洞填充 - 策略: {self.fill_strategy}")
        
        processed_map = classification_map.copy()
        unique_classes = np.unique(classification_map)
        self.filled_holes = []
        
        # 对每个非背景类别进行空洞填充
        for class_id in unique_classes:
            if class_id == 0:  # 跳过背景
                continue
            
            logger.debug(f"处理类别 {class_id} 的空洞")
            processed_map = self._fill_class_holes(processed_map, class_id)
        
        logger.info(f"空洞填充完成 - 共填充 {len(self.filled_holes)} 个空洞")
        
        return processed_map
    
    def _fill_class_holes(self, classification_map: np.ndarray, 
                         class_id: int) -> np.ndarray:
        """填充特定类别的空洞"""
        # 创建类别掩码
        class_mask = classification_map == class_id
        
        # 识别空洞
        holes = self._identify_holes(class_mask)
        
        if len(holes) == 0:
            return classification_map
        
        processed_map = classification_map.copy()
        
        # 填充每个空洞
        for hole_info in holes:
            hole_mask = hole_info['mask']
            hole_size = hole_info['size']
            
            # 大小过滤
            if hole_size < self.min_hole_size:
                continue
            
            if self.max_hole_size is not None and hole_size > self.max_hole_size:
                continue
            
            # 根据策略填充空洞
            fill_value = self._determine_fill_value(
                processed_map, hole_mask, class_id
            )
            
            processed_map[hole_mask] = fill_value
            
            # 记录填充信息
            self.filled_holes.append({
                'class_id': class_id,
                'size': hole_size,
                'fill_value': fill_value,
                'centroid': hole_info['centroid']
            })
        
        return processed_map
    
    def _identify_holes(self, class_mask: np.ndarray) -> List[Dict[str, Any]]:
        """识别类别掩码中的空洞"""
        # 使用形态学方法识别空洞
        filled_mask = binary_fill_holes(class_mask)
        holes_mask = filled_mask & (~class_mask)
        
        if not np.any(holes_mask):
            return []
        
        # 标记连通的空洞
        labeled_holes, num_holes = ndimage.label(holes_mask)
        
        holes = []
        for i in range(1, num_holes + 1):
            hole_mask = labeled_holes == i
            hole_size = np.sum(hole_mask)
            
            # 计算空洞质心
            coords = np.where(hole_mask)
            centroid = (np.mean(coords[0]), np.mean(coords[1]))
            
            holes.append({
                'mask': hole_mask,
                'size': hole_size,
                'centroid': centroid,
                'label': i
            })
        
        return holes
    
    def _determine_fill_value(self, classification_map: np.ndarray,
                             hole_mask: np.ndarray,
                             surrounding_class: int) -> int:
        """确定空洞填充值"""
        if self.fill_strategy == 'surrounding':
            # 使用周围类别填充
            return surrounding_class
        
        elif self.fill_strategy == 'majority':
            # 使用空洞周围的多数类别
            # 扩展空洞边界
            dilated_hole = binary_dilation(hole_mask, disk(3))
            boundary_mask = dilated_hole & (~hole_mask)
            
            if np.any(boundary_mask):
                boundary_values = classification_map[boundary_mask]
                boundary_values = boundary_values[boundary_values != 0]  # 排除背景
                
                if len(boundary_values) > 0:
                    unique_values, counts = np.unique(boundary_values, return_counts=True)
                    majority_class = unique_values[np.argmax(counts)]
                    return majority_class
            
            return surrounding_class
        
        elif self.fill_strategy == 'interpolation':
            # 使用插值方法（简化实现）
            return surrounding_class
        
        else:
            return surrounding_class


class SkeletonProcessor(MorphologyProcessor):
    """
    骨架提取处理器
    
    提取分类区域的骨架结构，用于形状分析和特征提取。
    """
    
    def __init__(self, 
                 method: str = 'zhang',
                 preserve_endpoints: bool = True,
                 min_branch_length: int = 5,
                 **kwargs):
        """
        初始化骨架处理器
        
        Parameters:
        -----------
        method : str, default='zhang'
            骨架化方法 ('zhang', 'lee')
        preserve_endpoints : bool, default=True
            是否保持端点
        min_branch_length : int, default=5
            最小分支长度
        """
        super().__init__(**kwargs)
        self.method = method
        self.preserve_endpoints = preserve_endpoints
        self.min_branch_length = min_branch_length
        self.skeleton_stats = {}
    
    def process(self, classification_map: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        提取骨架
        
        Parameters:
        -----------
        classification_map : np.ndarray
            输入分类图像
            
        Returns:
        --------
        results : dict
            包含骨架和统计信息的字典
        """
        self._validate_input(classification_map)
        
        logger.info(f"开始骨架提取 - 方法: {self.method}")
        
        unique_classes = np.unique(classification_map)
        skeleton_map = np.zeros_like(classification_map)
        self.skeleton_stats = {}
        
        # 对每个类别提取骨架
        for class_id in unique_classes:
            if class_id == 0:
                continue
            
            logger.debug(f"提取类别 {class_id} 的骨架")
            
            # 创建类别掩码
            class_mask = classification_map == class_id
            
            # 提取骨架
            skeleton = self._extract_skeleton(class_mask)
            
            # 后处理骨架
            skeleton = self._postprocess_skeleton(skeleton)
            
            # 分析骨架属性
            skeleton_props = self._analyze_skeleton(skeleton, class_id)
            self.skeleton_stats[class_id] = skeleton_props
            
            # 添加到骨架图
            skeleton_map[skeleton] = class_id
        
        logger.info("骨架提取完成")
        
        return {
            'skeleton_map': skeleton_map,
            'skeleton_stats': self.skeleton_stats,
            'original_map': classification_map
        }
    
    def _extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """提取二值图像的骨架"""
        if self.method == 'zhang':
            skeleton = skeletonize(binary_mask, method='zhang')
        elif self.method == 'lee':
            skeleton = skeletonize(binary_mask, method='lee')
        else:
            skeleton = skeletonize(binary_mask)
        
        return skeleton
    
    def _postprocess_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """后处理骨架"""
        processed_skeleton = skeleton.copy()
        
        # 移除短分支
        if self.min_branch_length > 0:
            processed_skeleton = self._remove_short_branches(processed_skeleton)
        
        return processed_skeleton
    
    def _remove_short_branches(self, skeleton: np.ndarray) -> np.ndarray:
        """移除短分支"""
        # 简化实现：使用形态学操作
        # 实际应用中可以实现更复杂的分支分析算法
        pruned = skeleton.copy()
        
        # 多次开运算来移除短分支
        for _ in range(2):
            opened = binary_opening(pruned, disk(1))
            pruned = opened
        
        return pruned
    
    def _analyze_skeleton(self, skeleton: np.ndarray, 
                         class_id: int) -> Dict[str, Any]:
        """分析骨架属性"""
        skeleton_points = np.sum(skeleton)
        
        if skeleton_points == 0:
            return {
                'class_id': class_id,
                'skeleton_length': 0,
                'num_branches': 0,
                'num_endpoints': 0,
                'num_junctions': 0
            }
        
        # 分析连通性
        labeled_skeleton, num_components = ndimage.label(skeleton)
        
        # 分析端点和交叉点
        endpoints, junctions = self._find_skeleton_features(skeleton)
        
        return {
            'class_id': class_id,
            'skeleton_length': skeleton_points,
            'num_components': num_components,
            'num_branches': num_components,  # 简化
            'num_endpoints': np.sum(endpoints),
            'num_junctions': np.sum(junctions)
        }
    
    def _find_skeleton_features(self, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """寻找骨架特征点（端点和交叉点）"""
        # 端点：只有一个邻居的点
        # 交叉点：有超过两个邻居的点
        
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # 计算每个点的邻居数
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * skeleton  # 只考虑骨架点
        
        # 端点：有1个邻居
        endpoints = (neighbor_count == 1) & skeleton
        
        # 交叉点：有3个或更多邻居
        junctions = (neighbor_count >= 3) & skeleton
        
        return endpoints, junctions


def create_morphology_pipeline() -> List[MorphologyProcessor]:
    """
    创建形态学处理流水线
    
    Returns:
    --------
    processors : list
        形态学处理器列表
    """
    processors = [
        # 1. 连通组件分析和过滤
        ConnectedComponentAnalyzer(min_size=10, max_size=None),
        
        # 2. 空洞填充
        HoleFillingProcessor(min_hole_size=5, fill_strategy='majority'),
        
        # 3. 区域处理和边界平滑
        RegionProcessor(boundary_smooth=True),
    ]
    
    return processors


def apply_morphology_pipeline(classification_map: np.ndarray,
                             processors: Optional[List[MorphologyProcessor]] = None,
                             **kwargs) -> Dict[str, Any]:
    """
    应用形态学处理流水线
    
    Parameters:
    -----------
    classification_map : np.ndarray
        输入分类图像
    processors : list, optional
        处理器列表
        
    Returns:
    --------
    results : dict
        处理结果和统计信息
    """
    if processors is None:
        processors = create_morphology_pipeline()
    
    logger.info(f"开始形态学流水线处理 - 处理器数量: {len(processors)}")
    
    current_map = classification_map.copy()
    results = {
        'original': classification_map,
        'steps': [],
        'statistics': {}
    }
    
    for i, processor in enumerate(processors):
        processor_name = processor.__class__.__name__
        logger.info(f"应用处理器 {i+1}/{len(processors)}: {processor_name}")
        
        # 应用处理器
        if isinstance(processor, SkeletonProcessor):
            # 骨架处理器返回字典
            step_result = processor.process(current_map, **kwargs)
            processed_map = step_result['skeleton_map']
            results['statistics'][f'step_{i+1}_{processor_name}'] = step_result
        else:
            processed_map = processor.process(current_map, **kwargs)
        
        # 计算变化统计
        changes = np.sum(processed_map != current_map)
        total_pixels = current_map.size
        change_ratio = changes / total_pixels
        
        # 保存步骤结果
        step_info = {
            'processor': processor_name,
            'step': i + 1,
            'changes': changes,
            'change_ratio': change_ratio,
            'result': processed_map.copy()
        }
        
        results['steps'].append(step_info)
        
        logger.info(f"{processor_name} 完成 - 改变像素: {changes}/{total_pixels} ({change_ratio:.2%})")
        
        # 更新当前图像
        current_map = processed_map
    
    results['final'] = current_map
    
    logger.info("形态学流水线处理完成")
    
    return results