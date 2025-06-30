"""
连通性分析
=========

这个模块实现了景观连通性分析功能，用于评估湿地生态系统的结构和功能连通性。

主要功能：
- 结构连通性分析
- 功能连通性评估
- 生态廊道识别
- 最小费用路径分析
- 栖息地网络分析
- 基因流模拟
- 阻力面构建

理论基础：
- 图论和网络分析
- 景观生态学理论
- 保护生物学原理

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from sklearn.cluster import DBSCAN
from skimage import measure, morphology, graph
from skimage.graph import route_through_array
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class ConnectivityAnalyzer:
    """
    连通性分析器
    
    提供景观连通性的多种分析方法，包括结构连通性和功能连通性。
    """
    
    def __init__(self, 
                 max_distance: float = 1000.0,
                 pixel_size: float = 1.0,
                 connectivity_threshold: float = 0.5,
                 **kwargs):
        """
        初始化连通性分析器
        
        Parameters:
        -----------
        max_distance : float, default=1000.0
            最大连通距离
        pixel_size : float, default=1.0
            像元大小
        connectivity_threshold : float, default=0.5
            连通性阈值
        """
        self.max_distance = max_distance
        self.pixel_size = pixel_size
        self.connectivity_threshold = connectivity_threshold
        self.config = kwargs
        
        # 缓存
        self.patches_cache = {}
        self.distance_cache = {}
        self.connectivity_cache = {}
    
    def analyze_structural_connectivity(self, 
                                      classification_map: np.ndarray,
                                      habitat_classes: List[int],
                                      **kwargs) -> Dict[str, Any]:
        """
        分析结构连通性
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        habitat_classes : list
            栖息地类别列表
            
        Returns:
        --------
        results : dict
            结构连通性分析结果
        """
        logger.info("开始结构连通性分析")
        
        # 提取栖息地斑块
        habitat_patches = self._extract_habitat_patches(
            classification_map, habitat_classes
        )
        
        if len(habitat_patches) == 0:
            logger.warning("未找到栖息地斑块")
            return {'patches': [], 'connectivity_matrix': np.array([]), 'metrics': {}}
        
        # 计算斑块间距离
        distance_matrix = self._calculate_patch_distances(habitat_patches)
        
        # 构建连通性矩阵
        connectivity_matrix = self._build_structural_connectivity_matrix(
            distance_matrix, habitat_patches
        )
        
        # 计算连通性指标
        connectivity_metrics = self._calculate_structural_metrics(
            connectivity_matrix, habitat_patches, distance_matrix
        )
        
        # 识别连通组件
        connected_components = self._identify_connected_components(connectivity_matrix)
        
        results = {
            'patches': habitat_patches,
            'distance_matrix': distance_matrix,
            'connectivity_matrix': connectivity_matrix,
            'connectivity_metrics': connectivity_metrics,
            'connected_components': connected_components,
            'analysis_type': 'structural'
        }
        
        logger.info(f"结构连通性分析完成 - 发现 {len(habitat_patches)} 个栖息地斑块")
        
        return results
    
    def analyze_functional_connectivity(self, 
                                      classification_map: np.ndarray,
                                      resistance_map: np.ndarray,
                                      habitat_classes: List[int],
                                      species_mobility: float = 100.0,
                                      **kwargs) -> Dict[str, Any]:
        """
        分析功能连通性
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        resistance_map : np.ndarray
            阻力地图
        habitat_classes : list
            栖息地类别列表
        species_mobility : float, default=100.0
            物种迁移能力
            
        Returns:
        --------
        results : dict
            功能连通性分析结果
        """
        logger.info("开始功能连通性分析")
        
        # 提取栖息地斑块
        habitat_patches = self._extract_habitat_patches(
            classification_map, habitat_classes
        )
        
        if len(habitat_patches) == 0:
            logger.warning("未找到栖息地斑块")
            return {'patches': [], 'connectivity_matrix': np.array([]), 'metrics': {}}
        
        # 计算最小费用路径
        cost_distances = self._calculate_cost_distances(
            habitat_patches, resistance_map, species_mobility
        )
        
        # 构建功能连通性矩阵
        functional_connectivity = self._build_functional_connectivity_matrix(
            cost_distances, habitat_patches, species_mobility
        )
        
        # 计算功能连通性指标
        functional_metrics = self._calculate_functional_metrics(
            functional_connectivity, habitat_patches, cost_distances
        )
        
        # 识别关键廊道
        key_corridors = self._identify_key_corridors(
            habitat_patches, resistance_map, cost_distances
        )
        
        # 计算重要性指数
        importance_indices = self._calculate_importance_indices(
            functional_connectivity, habitat_patches
        )
        
        results = {
            'patches': habitat_patches,
            'cost_distances': cost_distances,
            'functional_connectivity': functional_connectivity,
            'functional_metrics': functional_metrics,
            'key_corridors': key_corridors,
            'importance_indices': importance_indices,
            'analysis_type': 'functional'
        }
        
        logger.info("功能连通性分析完成")
        
        return results
    
    def _extract_habitat_patches(self, 
                                classification_map: np.ndarray,
                                habitat_classes: List[int]) -> List[Dict[str, Any]]:
        """提取栖息地斑块"""
        # 创建栖息地掩码
        habitat_mask = np.isin(classification_map, habitat_classes)
        
        # 连通组件标记
        labeled_patches, num_patches = ndimage.label(
            habitat_mask, 
            structure=ndimage.generate_binary_structure(2, 2)
        )
        
        patches = []
        for patch_id in range(1, num_patches + 1):
            patch_mask = labeled_patches == patch_id
            
            # 计算斑块属性
            coords = np.where(patch_mask)
            area = len(coords[0]) * (self.pixel_size ** 2)
            centroid = (np.mean(coords[0]) * self.pixel_size, 
                       np.mean(coords[1]) * self.pixel_size)
            
            # 边界框
            min_row, max_row = np.min(coords[0]), np.max(coords[0])
            min_col, max_col = np.min(coords[1]), np.max(coords[1])
            
            # 计算形状复杂度
            perimeter = self._calculate_patch_perimeter(patch_mask)
            shape_index = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0
            
            patch_info = {
                'id': patch_id,
                'area': area,
                'centroid': centroid,
                'coordinates': coords,
                'bounding_box': (min_row, min_col, max_row, max_col),
                'perimeter': perimeter,
                'shape_index': shape_index,
                'mask': patch_mask
            }
            
            patches.append(patch_info)
        
        return patches
    
    def _calculate_patch_perimeter(self, patch_mask: np.ndarray) -> float:
        """计算斑块周长"""
        # 使用边缘检测
        from scipy.ndimage import binary_dilation
        
        # 膨胀后减去原图得到边缘
        dilated = binary_dilation(patch_mask)
        edge = dilated & (~patch_mask)
        
        perimeter = np.sum(edge) * self.pixel_size
        return perimeter
    
    def _calculate_patch_distances(self, patches: List[Dict[str, Any]]) -> np.ndarray:
        """计算斑块间距离矩阵"""
        n_patches = len(patches)
        distance_matrix = np.zeros((n_patches, n_patches))
        
        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                # 欧几里得距离
                centroid_i = patches[i]['centroid']
                centroid_j = patches[j]['centroid']
                
                distance = euclidean(centroid_i, centroid_j)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _build_structural_connectivity_matrix(self, 
                                            distance_matrix: np.ndarray,
                                            patches: List[Dict[str, Any]]) -> np.ndarray:
        """构建结构连通性矩阵"""
        n_patches = len(patches)
        connectivity_matrix = np.zeros((n_patches, n_patches))
        
        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                distance = distance_matrix[i, j]
                
                # 基于距离的连通性
                if distance <= self.max_distance:
                    # 距离越近，连通性越强
                    connectivity = 1.0 - (distance / self.max_distance)
                    
                    # 考虑斑块大小的影响
                    area_i = patches[i]['area']
                    area_j = patches[j]['area']
                    area_factor = np.sqrt(area_i * area_j) / max(area_i, area_j)
                    
                    connectivity *= area_factor
                    
                    connectivity_matrix[i, j] = connectivity
                    connectivity_matrix[j, i] = connectivity
        
        return connectivity_matrix
    
    def _calculate_structural_metrics(self, 
                                    connectivity_matrix: np.ndarray,
                                    patches: List[Dict[str, Any]],
                                    distance_matrix: np.ndarray) -> Dict[str, float]:
        """计算结构连通性指标"""
        n_patches = len(patches)
        
        if n_patches == 0:
            return {}
        
        # 连通度
        binary_connectivity = (connectivity_matrix > self.connectivity_threshold).astype(int)
        
        # 平均连通度
        total_connections = np.sum(binary_connectivity) / 2  # 除以2因为矩阵对称
        max_connections = n_patches * (n_patches - 1) / 2
        connectance = total_connections / max_connections if max_connections > 0 else 0
        
        # 平均路径长度
        if total_connections > 0:
            G = nx.from_numpy_array(binary_connectivity)
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                # 对于不连通的图，计算最大连通分量的平均路径长度
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            avg_path_length = float('inf')
        
        # 聚类系数
        G = nx.from_numpy_array(binary_connectivity)
        clustering_coefficient = nx.average_clustering(G)
        
        # 连通分量数
        num_components = nx.number_connected_components(G)
        
        # 最大连通分量大小
        if num_components > 0:
            largest_component_size = len(max(nx.connected_components(G), key=len))
        else:
            largest_component_size = 0
        
        # 平均最近邻距离
        min_distances = []
        for i in range(n_patches):
            distances = distance_matrix[i, :]
            distances = distances[distances > 0]  # 排除自身
            if len(distances) > 0:
                min_distances.append(np.min(distances))
        
        avg_nearest_neighbor_distance = np.mean(min_distances) if min_distances else 0
        
        metrics = {
            'connectance': connectance,
            'avg_path_length': avg_path_length,
            'clustering_coefficient': clustering_coefficient,
            'num_components': num_components,
            'largest_component_size': largest_component_size,
            'largest_component_ratio': largest_component_size / n_patches,
            'avg_nearest_neighbor_distance': avg_nearest_neighbor_distance,
            'total_patches': n_patches,
            'total_connections': int(total_connections)
        }
        
        return metrics
    
    def _identify_connected_components(self, connectivity_matrix: np.ndarray) -> List[List[int]]:
        """识别连通组件"""
        binary_connectivity = (connectivity_matrix > self.connectivity_threshold).astype(int)
        G = nx.from_numpy_array(binary_connectivity)
        
        components = []
        for component in nx.connected_components(G):
            components.append(list(component))
        
        # 按组件大小排序
        components.sort(key=len, reverse=True)
        
        return components
    
    def _calculate_cost_distances(self, 
                                patches: List[Dict[str, Any]],
                                resistance_map: np.ndarray,
                                species_mobility: float) -> np.ndarray:
        """计算最小费用距离"""
        n_patches = len(patches)
        cost_matrix = np.zeros((n_patches, n_patches))
        
        # 归一化阻力图
        resistance_normalized = resistance_map / np.max(resistance_map)
        
        for i in range(n_patches):
            source_coords = patches[i]['centroid']
            source_pixel = (int(source_coords[0] / self.pixel_size), 
                           int(source_coords[1] / self.pixel_size))
            
            for j in range(i + 1, n_patches):
                target_coords = patches[j]['centroid']
                target_pixel = (int(target_coords[0] / self.pixel_size), 
                               int(target_coords[1] / self.pixel_size))
                
                # 使用Dijkstra算法计算最小费用路径
                try:
                    path, cost = route_through_array(
                        resistance_normalized, 
                        source_pixel, 
                        target_pixel,
                        geometric=True,
                        fully_connected=True
                    )
                    
                    # 考虑物种迁移能力
                    effective_cost = cost * np.exp(-cost / species_mobility)
                    
                    cost_matrix[i, j] = effective_cost
                    cost_matrix[j, i] = effective_cost
                    
                except Exception as e:
                    # 如果路径计算失败，使用欧几里得距离 * 平均阻力
                    euclidean_dist = euclidean(source_coords, target_coords)
                    avg_resistance = np.mean(resistance_normalized)
                    cost_matrix[i, j] = euclidean_dist * avg_resistance
                    cost_matrix[j, i] = euclidean_dist * avg_resistance
        
        return cost_matrix
    
    def _build_functional_connectivity_matrix(self, 
                                            cost_distances: np.ndarray,
                                            patches: List[Dict[str, Any]],
                                            species_mobility: float) -> np.ndarray:
        """构建功能连通性矩阵"""
        n_patches = len(patches)
        functional_matrix = np.zeros((n_patches, n_patches))
        
        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                cost = cost_distances[i, j]
                
                # 基于费用距离的连通性（负指数衰减）
                connectivity = np.exp(-cost / species_mobility)
                
                # 考虑斑块质量
                area_i = patches[i]['area']
                area_j = patches[j]['area']
                quality_factor = np.sqrt(area_i * area_j)
                
                functional_connectivity = connectivity * quality_factor
                
                functional_matrix[i, j] = functional_connectivity
                functional_matrix[j, i] = functional_connectivity
        
        return functional_matrix
    
    def _calculate_functional_metrics(self, 
                                    functional_connectivity: np.ndarray,
                                    patches: List[Dict[str, Any]],
                                    cost_distances: np.ndarray) -> Dict[str, float]:
        """计算功能连通性指标"""
        n_patches = len(patches)
        
        if n_patches == 0:
            return {}
        
        # 总功能连通性
        total_functional_connectivity = np.sum(functional_connectivity) / 2
        
        # 平均功能连通性
        num_pairs = n_patches * (n_patches - 1) / 2
        avg_functional_connectivity = total_functional_connectivity / num_pairs if num_pairs > 0 else 0
        
        # 有效网格大小（基于概率连通性）
        total_area = sum(patch['area'] for patch in patches)
        prob_connectivity = functional_connectivity / np.max(functional_connectivity) if np.max(functional_connectivity) > 0 else functional_connectivity
        
        effective_mesh_size = 0
        for i in range(n_patches):
            for j in range(n_patches):
                if i != j:
                    effective_mesh_size += patches[i]['area'] * patches[j]['area'] * prob_connectivity[i, j]
        
        effective_mesh_size = effective_mesh_size / (total_area ** 2) if total_area > 0 else 0
        
        # 景观连通性指数
        landscape_connectivity_index = np.sum(functional_connectivity) / (total_area ** 2) if total_area > 0 else 0
        
        # 平均最小费用距离
        cost_distances_upper = cost_distances[np.triu_indices_from(cost_distances, k=1)]
        avg_cost_distance = np.mean(cost_distances_upper) if len(cost_distances_upper) > 0 else 0
        
        # 连通性的变异系数
        connectivity_values = functional_connectivity[np.triu_indices_from(functional_connectivity, k=1)]
        connectivity_cv = np.std(connectivity_values) / np.mean(connectivity_values) if np.mean(connectivity_values) > 0 else 0
        
        metrics = {
            'total_functional_connectivity': total_functional_connectivity,
            'avg_functional_connectivity': avg_functional_connectivity,
            'effective_mesh_size': effective_mesh_size,
            'landscape_connectivity_index': landscape_connectivity_index,
            'avg_cost_distance': avg_cost_distance,
            'connectivity_cv': connectivity_cv,
            'total_habitat_area': total_area
        }
        
        return metrics
    
    def _identify_key_corridors(self, 
                              patches: List[Dict[str, Any]],
                              resistance_map: np.ndarray,
                              cost_distances: np.ndarray) -> List[Dict[str, Any]]:
        """识别关键生态廊道"""
        n_patches = len(patches)
        corridors = []
        
        # 找到最重要的连接（基于最小生成树）
        # 将费用距离转换为权重
        weights = cost_distances.copy()
        weights[weights == 0] = np.inf  # 避免自连接
        
        # 计算最小生成树
        mst = minimum_spanning_tree(csr_matrix(weights))
        mst_array = mst.toarray()
        
        # 提取MST中的边
        mst_edges = np.where(mst_array > 0)
        
        for i, j in zip(mst_edges[0], mst_edges[1]):
            if i < j:  # 避免重复
                corridor_info = {
                    'source_patch': i,
                    'target_patch': j,
                    'cost_distance': cost_distances[i, j],
                    'importance': 'high',  # MST边都是高重要性
                    'type': 'minimum_spanning_tree'
                }
                corridors.append(corridor_info)
        
        # 找到其他重要连接（基于连通性阈值）
        connectivity_threshold = np.percentile(cost_distances[cost_distances > 0], 25)  # 前25%
        
        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                if cost_distances[i, j] <= connectivity_threshold:
                    # 检查是否已在MST中
                    is_mst_edge = any(
                        (corridor['source_patch'] == i and corridor['target_patch'] == j) or
                        (corridor['source_patch'] == j and corridor['target_patch'] == i)
                        for corridor in corridors
                    )
                    
                    if not is_mst_edge:
                        corridor_info = {
                            'source_patch': i,
                            'target_patch': j,
                            'cost_distance': cost_distances[i, j],
                            'importance': 'medium',
                            'type': 'threshold_based'
                        }
                        corridors.append(corridor_info)
        
        # 按重要性和费用距离排序
        corridors.sort(key=lambda x: (x['importance'] == 'medium', x['cost_distance']))
        
        return corridors
    
    def _calculate_importance_indices(self, 
                                    functional_connectivity: np.ndarray,
                                    patches: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """计算斑块重要性指数"""
        n_patches = len(patches)
        importance_indices = {}
        
        for i in range(n_patches):
            # 节点度（连接数）
            connections = np.sum(functional_connectivity[i, :] > self.connectivity_threshold)
            
            # 节点强度（连接强度总和）
            strength = np.sum(functional_connectivity[i, :])
            
            # 中心性指标
            G = nx.from_numpy_array(functional_connectivity)
            
            # 度中心性
            degree_centrality = nx.degree_centrality(G)[i] if n_patches > 1 else 0
            
            # 接近中心性
            try:
                closeness_centrality = nx.closeness_centrality(G, distance='weight')[i]
            except:
                closeness_centrality = 0
            
            # 介数中心性
            try:
                betweenness_centrality = nx.betweenness_centrality(G, weight='weight')[i]
            except:
                betweenness_centrality = 0
            
            # 移除该节点后的连通性变化
            G_removed = G.copy()
            G_removed.remove_node(i)
            
            original_components = nx.number_connected_components(G)
            new_components = nx.number_connected_components(G_removed)
            component_change = new_components - original_components
            
            # 综合重要性指数
            area_importance = patches[i]['area'] / sum(p['area'] for p in patches)
            connectivity_importance = strength / np.sum(functional_connectivity) if np.sum(functional_connectivity) > 0 else 0
            
            overall_importance = (area_importance + connectivity_importance + degree_centrality) / 3
            
            importance_indices[i] = {
                'area': patches[i]['area'],
                'connections': connections,
                'strength': strength,
                'degree_centrality': degree_centrality,
                'closeness_centrality': closeness_centrality,
                'betweenness_centrality': betweenness_centrality,
                'component_change': component_change,
                'area_importance': area_importance,
                'connectivity_importance': connectivity_importance,
                'overall_importance': overall_importance
            }
        
        return importance_indices


class ResistanceMapper:
    """
    阻力面构建器
    
    根据土地利用类型和生态特征构建物种迁移的阻力面。
    """
    
    def __init__(self, 
                 species_type: str = 'wetland_bird',
                 **kwargs):
        """
        初始化阻力面构建器
        
        Parameters:
        -----------
        species_type : str, default='wetland_bird'
            物种类型，影响阻力值设定
        """
        self.species_type = species_type
        self.config = kwargs
        
        # 预定义的阻力值
        self.resistance_values = self._get_default_resistance_values()
    
    def _get_default_resistance_values(self) -> Dict[str, float]:
        """获取默认阻力值"""
        if self.species_type == 'wetland_bird':
            return {
                'water': 1.0,
                'shallow_water': 1.2,
                'wetland_vegetation': 1.5,
                'mudflat': 2.0,
                'grassland': 3.0,
                'forest': 4.0,
                'agriculture': 5.0,
                'urban': 10.0,
                'road': 15.0,
                'industrial': 20.0
            }
        elif self.species_type == 'amphibian':
            return {
                'water': 1.0,
                'shallow_water': 1.1,
                'wetland_vegetation': 1.3,
                'mudflat': 1.8,
                'grassland': 4.0,
                'forest': 2.0,
                'agriculture': 8.0,
                'urban': 15.0,
                'road': 25.0,
                'industrial': 30.0
            }
        else:
            # 通用阻力值
            return {
                'water': 1.0,
                'shallow_water': 1.5,
                'wetland_vegetation': 2.0,
                'mudflat': 2.5,
                'grassland': 3.0,
                'forest': 3.5,
                'agriculture': 5.0,
                'urban': 10.0,
                'road': 15.0,
                'industrial': 20.0
            }
    
    def create_resistance_map(self, 
                            classification_map: np.ndarray,
                            class_mapping: Dict[int, str],
                            custom_resistance: Optional[Dict[str, float]] = None,
                            edge_effect: bool = True,
                            **kwargs) -> np.ndarray:
        """
        创建阻力地图
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        class_mapping : dict
            类别编码到类别名称的映射
        custom_resistance : dict, optional
            自定义阻力值
        edge_effect : bool, default=True
            是否考虑边缘效应
            
        Returns:
        --------
        resistance_map : np.ndarray
            阻力地图
        """
        logger.info(f"创建阻力地图 - 物种类型: {self.species_type}")
        
        # 使用自定义阻力值或默认值
        resistance_values = custom_resistance or self.resistance_values
        
        # 初始化阻力地图
        resistance_map = np.zeros_like(classification_map, dtype=float)
        
        # 分配阻力值
        for class_id, class_name in class_mapping.items():
            class_mask = classification_map == class_id
            
            # 查找匹配的阻力值
            resistance_value = self._find_resistance_value(class_name, resistance_values)
            resistance_map[class_mask] = resistance_value
        
        # 边缘效应处理
        if edge_effect:
            resistance_map = self._apply_edge_effects(
                resistance_map, classification_map, class_mapping
            )
        
        # 平滑处理
        resistance_map = self._smooth_resistance_map(resistance_map)
        
        logger.info("阻力地图创建完成")
        
        return resistance_map
    
    def _find_resistance_value(self, class_name: str, resistance_values: Dict[str, float]) -> float:
        """查找类别对应的阻力值"""
        # 直接匹配
        if class_name in resistance_values:
            return resistance_values[class_name]
        
        # 模糊匹配
        class_name_lower = class_name.lower()
        for key, value in resistance_values.items():
            if key.lower() in class_name_lower or class_name_lower in key.lower():
                return value
        
        # 默认值
        return 5.0
    
    def _apply_edge_effects(self, 
                          resistance_map: np.ndarray,
                          classification_map: np.ndarray,
                          class_mapping: Dict[int, str]) -> np.ndarray:
        """应用边缘效应"""
        from scipy.ndimage import distance_transform_edt, binary_dilation
        
        enhanced_resistance = resistance_map.copy()
        
        # 对于高质量栖息地，增加边缘的阻力
        high_quality_classes = ['water', 'shallow_water', 'wetland_vegetation']
        
        for class_id, class_name in class_mapping.items():
            if any(hq_class in class_name.lower() for hq_class in high_quality_classes):
                class_mask = classification_map == class_id
                
                # 计算到边缘的距离
                distance_to_edge = distance_transform_edt(class_mask)
                
                # 在边缘附近增加阻力
                edge_zone = (distance_to_edge <= 3) & class_mask
                enhanced_resistance[edge_zone] *= 1.5
        
        return enhanced_resistance
    
    def _smooth_resistance_map(self, resistance_map: np.ndarray) -> np.ndarray:
        """平滑阻力地图"""
        from scipy.ndimage import gaussian_filter
        
        # 轻微的高斯平滑
        smoothed = gaussian_filter(resistance_map, sigma=1.0)
        
        return smoothed


class EcologicalNetworkAnalyzer:
    """
    生态网络分析器
    
    基于图论方法分析生态网络的结构和功能特征。
    """
    
    def __init__(self, **kwargs):
        """初始化生态网络分析器"""
        self.config = kwargs
    
    def analyze_network_structure(self, 
                                connectivity_matrix: np.ndarray,
                                patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析网络结构特征
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            连通性矩阵
        patches : list
            斑块信息列表
            
        Returns:
        --------
        network_metrics : dict
            网络结构指标
        """
        logger.info("分析生态网络结构")
        
        # 构建网络图
        G = nx.from_numpy_array(connectivity_matrix)
        
        # 为节点添加属性
        for i, patch in enumerate(patches):
            G.nodes[i]['area'] = patch['area']
            G.nodes[i]['centroid'] = patch['centroid']
        
        # 网络基本特征
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        
        # 连通性指标
        is_connected = nx.is_connected(G)
        n_components = nx.number_connected_components(G)
        
        if is_connected:
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # 对于不连通的网络，计算最大连通分量的指标
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
        
        # 中心性指标
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # 聚类系数
        clustering_coefficient = nx.average_clustering(G)
        
        # 小世界特征
        # 计算随机网络的聚类系数和路径长度进行比较
        random_clustering = density  # 随机网络的聚类系数近似等于密度
        small_worldness = (clustering_coefficient / random_clustering) / (avg_path_length / np.log(n_nodes)) if n_nodes > 1 else 0
        
        # 网络效率
        global_efficiency = nx.global_efficiency(G)
        local_efficiency = nx.local_efficiency(G)
        
        # 度分布特征
        degrees = [G.degree(n) for n in G.nodes()]
        degree_assortativity = nx.degree_assortativity_coefficient(G)
        
        network_metrics = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'is_connected': is_connected,
            'n_components': n_components,
            'avg_path_length': avg_path_length,
            'diameter': diameter,
            'clustering_coefficient': clustering_coefficient,
            'small_worldness': small_worldness,
            'global_efficiency': global_efficiency,
            'local_efficiency': local_efficiency,
            'degree_assortativity': degree_assortativity,
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees) if degrees else 0,
            'degree_centrality': degree_centrality,
            'closeness_centrality': closeness_centrality,
            'betweenness_centrality': betweenness_centrality
        }
        
        return network_metrics
    
    def identify_keystone_patches(self, 
                                connectivity_matrix: np.ndarray,
                                patches: List[Dict[str, Any]],
                                removal_threshold: float = 0.1) -> List[int]:
        """
        识别关键石斑块
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            连通性矩阵
        patches : list
            斑块信息列表
        removal_threshold : float, default=0.1
            移除影响阈值
            
        Returns:
        --------
        keystone_patches : list
            关键石斑块索引列表
        """
        G = nx.from_numpy_array(connectivity_matrix)
        original_connectivity = nx.global_efficiency(G)
        
        keystone_patches = []
        
        for i in range(len(patches)):
            # 移除节点i
            G_removed = G.copy()
            G_removed.remove_node(i)
            
            # 计算移除后的连通性
            if G_removed.number_of_nodes() > 0:
                new_connectivity = nx.global_efficiency(G_removed)
                connectivity_loss = (original_connectivity - new_connectivity) / original_connectivity
                
                if connectivity_loss > removal_threshold:
                    keystone_patches.append(i)
        
        return keystone_patches
    
    def simulate_network_robustness(self, 
                                  connectivity_matrix: np.ndarray,
                                  patches: List[Dict[str, Any]],
                                  removal_strategy: str = 'random') -> Dict[str, Any]:
        """
        模拟网络鲁棒性
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            连通性矩阵
        patches : list
            斑块信息列表
        removal_strategy : str, default='random'
            移除策略 ('random', 'degree', 'betweenness', 'area')
            
        Returns:
        --------
        robustness_results : dict
            鲁棒性分析结果
        """
        G = nx.from_numpy_array(connectivity_matrix)
        n_nodes = G.number_of_nodes()
        
        # 确定移除顺序
        if removal_strategy == 'random':
            removal_order = list(range(n_nodes))
            np.random.shuffle(removal_order)
        elif removal_strategy == 'degree':
            degrees = G.degree()
            removal_order = sorted(degrees, key=lambda x: x[1], reverse=True)
            removal_order = [node for node, _ in removal_order]
        elif removal_strategy == 'betweenness':
            betweenness = nx.betweenness_centrality(G)
            removal_order = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            removal_order = [node for node, _ in removal_order]
        elif removal_strategy == 'area':
            areas = [(i, patches[i]['area']) for i in range(len(patches))]
            removal_order = sorted(areas, key=lambda x: x[1], reverse=True)
            removal_order = [node for node, _ in removal_order]
        
        # 模拟逐步移除
        connectivity_over_time = []
        components_over_time = []
        efficiency_over_time = []
        
        G_current = G.copy()
        
        for step in range(n_nodes):
            # 计算当前网络指标
            if G_current.number_of_nodes() > 0:
                connectivity = 1 - nx.number_connected_components(G_current) / G_current.number_of_nodes()
                efficiency = nx.global_efficiency(G_current)
                components = nx.number_connected_components(G_current)
            else:
                connectivity = 0
                efficiency = 0
                components = 0
            
            connectivity_over_time.append(connectivity)
            efficiency_over_time.append(efficiency)
            components_over_time.append(components)
            
            # 移除下一个节点
            if step < len(removal_order) and removal_order[step] in G_current:
                G_current.remove_node(removal_order[step])
        
        # 计算崩溃点（连通性下降50%的点）
        initial_connectivity = connectivity_over_time[0]
        collapse_threshold = initial_connectivity * 0.5
        
        collapse_point = None
        for i, conn in enumerate(connectivity_over_time):
            if conn <= collapse_threshold:
                collapse_point = i / n_nodes  # 作为比例
                break
        
        robustness_results = {
            'removal_strategy': removal_strategy,
            'removal_order': removal_order,
            'connectivity_over_time': connectivity_over_time,
            'efficiency_over_time': efficiency_over_time,
            'components_over_time': components_over_time,
            'collapse_point': collapse_point,
            'final_connectivity': connectivity_over_time[-1],
            'area_under_curve': np.trapz(connectivity_over_time)  # 鲁棒性的积分指标
        }
        
        return robustness_results


def analyze_landscape_connectivity(classification_map: np.ndarray,
                                 habitat_classes: List[int],
                                 analysis_type: str = 'both',
                                 resistance_map: Optional[np.ndarray] = None,
                                 species_mobility: float = 100.0,
                                 pixel_size: float = 1.0,
                                 **kwargs) -> Dict[str, Any]:
    """
    景观连通性分析的便捷函数
    
    Parameters:
    -----------
    classification_map : np.ndarray
        分类图像
    habitat_classes : list
        栖息地类别列表
    analysis_type : str, default='both'
        分析类型 ('structural', 'functional', 'both')
    resistance_map : np.ndarray, optional
        阻力地图
    species_mobility : float, default=100.0
        物种迁移能力
    pixel_size : float, default=1.0
        像元大小
        
    Returns:
    --------
    results : dict
        连通性分析结果
    """
    analyzer = ConnectivityAnalyzer(pixel_size=pixel_size, **kwargs)
    
    results = {}
    
    if analysis_type in ['structural', 'both']:
        structural_results = analyzer.analyze_structural_connectivity(
            classification_map, habitat_classes
        )
        results['structural'] = structural_results
    
    if analysis_type in ['functional', 'both']:
        if resistance_map is None:
            logger.warning("功能连通性分析需要阻力地图，将创建默认阻力地图")
            # 创建简单的阻力地图
            resistance_map = np.ones_like(classification_map, dtype=float)
            for class_id in habitat_classes:
                resistance_map[classification_map == class_id] = 1.0
            resistance_map[~np.isin(classification_map, habitat_classes)] = 5.0
        
        functional_results = analyzer.analyze_functional_connectivity(
            classification_map, resistance_map, habitat_classes, species_mobility
        )
        results['functional'] = functional_results
    
    return results