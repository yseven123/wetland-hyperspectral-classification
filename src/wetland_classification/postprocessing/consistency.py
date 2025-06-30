"""
一致性处理
=========

这个模块实现了时序和空间一致性检查与修正算法，确保分类结果的逻辑一致性。

主要功能：
- 时序一致性分析与修正
- 空间一致性检查
- 生态约束验证
- 多期影像一致性处理
- 先验知识融合
- 分类结果验证

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import ndimage
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class ConsistencyProcessor:
    """
    一致性处理器基类
    
    提供通用的一致性检查和修正接口。
    """
    
    def __init__(self, 
                 tolerance: float = 0.1,
                 confidence_threshold: float = 0.8,
                 **kwargs):
        """
        初始化一致性处理器
        
        Parameters:
        -----------
        tolerance : float, default=0.1
            一致性容忍度
        confidence_threshold : float, default=0.8
            置信度阈值
        """
        self.tolerance = tolerance
        self.confidence_threshold = confidence_threshold
        self.config = kwargs
        self.inconsistency_log = []
    
    def check_consistency(self, classification_map: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        检查一致性
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
            
        Returns:
        --------
        results : dict
            一致性检查结果
        """
        raise NotImplementedError("子类必须实现check_consistency方法")
    
    def correct_inconsistency(self, classification_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        修正不一致性
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
            
        Returns:
        --------
        corrected_map : np.ndarray
            修正后的分类图像
        """
        raise NotImplementedError("子类必须实现correct_inconsistency方法")


class TemporalConsistencyProcessor(ConsistencyProcessor):
    """
    时序一致性处理器
    
    处理多时相分类结果的时序一致性问题。
    """
    
    def __init__(self, 
                 max_change_rate: float = 0.2,
                 stable_class_period: int = 3,
                 transition_rules: Optional[Dict] = None,
                 **kwargs):
        """
        初始化时序一致性处理器
        
        Parameters:
        -----------
        max_change_rate : float, default=0.2
            最大变化率阈值
        stable_class_period : int, default=3
            稳定类别持续期数
        transition_rules : dict, optional
            类别转换规则
        """
        super().__init__(**kwargs)
        self.max_change_rate = max_change_rate
        self.stable_class_period = stable_class_period
        self.transition_rules = transition_rules or self._default_transition_rules()
        self.temporal_stats = {}
    
    def _default_transition_rules(self) -> Dict[str, List]:
        """
        默认的湿地类别转换规则
        
        Returns:
        --------
        rules : dict
            转换规则字典
        """
        # 湿地生态系统的合理转换规则
        rules = {
            'water_to': ['shallow_water', 'wetland_vegetation', 'mudflat'],
            'shallow_water_to': ['water', 'wetland_vegetation', 'mudflat'],
            'wetland_vegetation_to': ['shallow_water', 'dry_land', 'water'],
            'mudflat_to': ['shallow_water', 'wetland_vegetation', 'water'],
            'dry_land_to': ['wetland_vegetation', 'agriculture'],
            'agriculture_to': ['dry_land', 'wetland_vegetation'],
            
            # 不太可能的直接转换
            'forbidden_transitions': [
                ('water', 'agriculture'),
                ('agriculture', 'water'),
                ('urban', 'water'),
                ('water', 'urban')
            ]
        }
        
        return rules
    
    def check_consistency(self, 
                         temporal_maps: List[np.ndarray],
                         dates: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        检查时序一致性
        
        Parameters:
        -----------
        temporal_maps : list
            时序分类图像列表
        dates : list, optional
            对应的日期列表
            
        Returns:
        --------
        results : dict
            时序一致性检查结果
        """
        logger.info(f"开始时序一致性检查 - 时相数量: {len(temporal_maps)}")
        
        if len(temporal_maps) < 2:
            raise ValueError("至少需要两个时相的分类图像")
        
        # 检查图像尺寸一致性
        reference_shape = temporal_maps[0].shape
        for i, map_data in enumerate(temporal_maps):
            if map_data.shape != reference_shape:
                raise ValueError(f"第{i}个时相图像尺寸不一致")
        
        # 分析时序变化
        change_analysis = self._analyze_temporal_changes(temporal_maps, dates)
        
        # 检测异常变化
        anomalies = self._detect_temporal_anomalies(temporal_maps, change_analysis)
        
        # 验证转换合理性
        transition_validity = self._validate_transitions(temporal_maps)
        
        # 计算一致性指标
        consistency_metrics = self._calculate_consistency_metrics(
            temporal_maps, change_analysis, anomalies
        )
        
        results = {
            'change_analysis': change_analysis,
            'anomalies': anomalies,
            'transition_validity': transition_validity,
            'consistency_metrics': consistency_metrics,
            'inconsistent_pixels': len(anomalies['spatial_anomalies']) if anomalies else 0
        }
        
        self.temporal_stats = results
        
        logger.info(f"时序一致性检查完成 - 发现 {results['inconsistent_pixels']} 个不一致像素")
        
        return results
    
    def _analyze_temporal_changes(self, 
                                 temporal_maps: List[np.ndarray],
                                 dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """分析时序变化"""
        num_periods = len(temporal_maps)
        height, width = temporal_maps[0].shape
        
        # 计算变化矩阵
        change_maps = []
        change_rates = []
        
        for i in range(num_periods - 1):
            current_map = temporal_maps[i]
            next_map = temporal_maps[i + 1]
            
            # 变化检测
            change_map = (current_map != next_map).astype(int)
            change_maps.append(change_map)
            
            # 变化率
            change_rate = np.sum(change_map) / (height * width)
            change_rates.append(change_rate)
        
        # 频繁变化像素
        total_changes = np.sum(change_maps, axis=0)
        frequent_change_pixels = total_changes > (num_periods - 1) * self.max_change_rate
        
        # 类别变化统计
        class_transitions = self._analyze_class_transitions(temporal_maps)
        
        return {
            'change_maps': change_maps,
            'change_rates': change_rates,
            'total_changes': total_changes,
            'frequent_change_pixels': frequent_change_pixels,
            'class_transitions': class_transitions,
            'average_change_rate': np.mean(change_rates)
        }
    
    def _analyze_class_transitions(self, temporal_maps: List[np.ndarray]) -> Dict[str, Any]:
        """分析类别转换模式"""
        transitions = defaultdict(int)
        transition_matrix = {}
        
        # 获取所有类别
        all_classes = set()
        for map_data in temporal_maps:
            all_classes.update(np.unique(map_data))
        all_classes = sorted(list(all_classes))
        
        # 初始化转换矩阵
        for class1 in all_classes:
            transition_matrix[class1] = {class2: 0 for class2 in all_classes}
        
        # 统计转换
        for i in range(len(temporal_maps) - 1):
            current_map = temporal_maps[i]
            next_map = temporal_maps[i + 1]
            
            for y in range(current_map.shape[0]):
                for x in range(current_map.shape[1]):
                    from_class = current_map[y, x]
                    to_class = next_map[y, x]
                    
                    if from_class != to_class:
                        transition_key = f"{from_class}_{to_class}"
                        transitions[transition_key] += 1
                        transition_matrix[from_class][to_class] += 1
        
        return {
            'transitions': dict(transitions),
            'transition_matrix': transition_matrix,
            'most_common_transitions': dict(Counter(transitions).most_common(10))
        }
    
    def _detect_temporal_anomalies(self, 
                                  temporal_maps: List[np.ndarray],
                                  change_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """检测时序异常"""
        anomalies = {
            'spatial_anomalies': [],
            'rapid_changes': [],
            'oscillations': [],
            'implausible_transitions': []
        }
        
        height, width = temporal_maps[0].shape
        
        # 检测空间异常（变化过于频繁的像素）
        frequent_change_mask = change_analysis['frequent_change_pixels']
        frequent_change_coords = np.where(frequent_change_mask)
        
        for y, x in zip(frequent_change_coords[0], frequent_change_coords[1]):
            temporal_sequence = [map_data[y, x] for map_data in temporal_maps]
            anomalies['spatial_anomalies'].append({
                'coordinates': (y, x),
                'sequence': temporal_sequence,
                'change_count': change_analysis['total_changes'][y, x]
            })
        
        # 检测快速变化（变化率超过阈值）
        for i, change_rate in enumerate(change_analysis['change_rates']):
            if change_rate > self.max_change_rate:
                anomalies['rapid_changes'].append({
                    'period': i,
                    'change_rate': change_rate,
                    'threshold': self.max_change_rate
                })
        
        # 检测振荡模式（类别来回变化）
        oscillations = self._detect_oscillations(temporal_maps)
        anomalies['oscillations'] = oscillations
        
        # 检测不合理转换
        implausible = self._detect_implausible_transitions(temporal_maps)
        anomalies['implausible_transitions'] = implausible
        
        return anomalies
    
    def _detect_oscillations(self, temporal_maps: List[np.ndarray]) -> List[Dict[str, Any]]:
        """检测振荡模式"""
        oscillations = []
        height, width = temporal_maps[0].shape
        
        # 检查每个像素的时序序列
        for y in range(height):
            for x in range(width):
                sequence = [map_data[y, x] for map_data in temporal_maps]
                
                # 检测简单的二值振荡（A-B-A-B模式）
                if len(sequence) >= 4:
                    for i in range(len(sequence) - 3):
                        if (sequence[i] == sequence[i+2] and 
                            sequence[i+1] == sequence[i+3] and
                            sequence[i] != sequence[i+1]):
                            
                            oscillations.append({
                                'coordinates': (y, x),
                                'pattern': sequence[i:i+4],
                                'start_period': i
                            })
                            break
        
        return oscillations
    
    def _detect_implausible_transitions(self, temporal_maps: List[np.ndarray]) -> List[Dict[str, Any]]:
        """检测不合理的转换"""
        implausible = []
        forbidden = self.transition_rules.get('forbidden_transitions', [])
        
        if not forbidden:
            return implausible
        
        height, width = temporal_maps[0].shape
        
        for i in range(len(temporal_maps) - 1):
            current_map = temporal_maps[i]
            next_map = temporal_maps[i + 1]
            
            for from_class, to_class in forbidden:
                # 找到发生禁止转换的像素
                transition_mask = (current_map == from_class) & (next_map == to_class)
                transition_coords = np.where(transition_mask)
                
                for y, x in zip(transition_coords[0], transition_coords[1]):
                    implausible.append({
                        'coordinates': (y, x),
                        'from_class': from_class,
                        'to_class': to_class,
                        'period': i
                    })
        
        return implausible
    
    def _validate_transitions(self, temporal_maps: List[np.ndarray]) -> Dict[str, Any]:
        """验证转换的合理性"""
        valid_transitions = 0
        total_transitions = 0
        invalid_details = []
        
        height, width = temporal_maps[0].shape
        
        for i in range(len(temporal_maps) - 1):
            current_map = temporal_maps[i]
            next_map = temporal_maps[i + 1]
            
            for y in range(height):
                for x in range(width):
                    from_class = current_map[y, x]
                    to_class = next_map[y, x]
                    
                    if from_class != to_class:
                        total_transitions += 1
                        
                        if self._is_valid_transition(from_class, to_class):
                            valid_transitions += 1
                        else:
                            invalid_details.append({
                                'coordinates': (y, x),
                                'from_class': from_class,
                                'to_class': to_class,
                                'period': i
                            })
        
        validity_rate = valid_transitions / total_transitions if total_transitions > 0 else 1.0
        
        return {
            'valid_transitions': valid_transitions,
            'total_transitions': total_transitions,
            'validity_rate': validity_rate,
            'invalid_details': invalid_details
        }
    
    def _is_valid_transition(self, from_class: int, to_class: int) -> bool:
        """检查转换是否合理"""
        # 检查是否为禁止转换
        forbidden = self.transition_rules.get('forbidden_transitions', [])
        if (from_class, to_class) in forbidden:
            return False
        
        # 可以添加更多的转换规则检查
        # 这里简化为允许大多数转换
        return True
    
    def _calculate_consistency_metrics(self, 
                                     temporal_maps: List[np.ndarray],
                                     change_analysis: Dict[str, Any],
                                     anomalies: Dict[str, Any]) -> Dict[str, float]:
        """计算一致性指标"""
        total_pixels = temporal_maps[0].size
        
        # 稳定性指标（稳定像素比例）
        stable_pixels = np.sum(change_analysis['total_changes'] == 0)
        stability_ratio = stable_pixels / total_pixels
        
        # 变化合理性指标
        anomaly_pixels = len(anomalies['spatial_anomalies'])
        reasonableness_ratio = 1.0 - (anomaly_pixels / total_pixels)
        
        # 平均变化率
        avg_change_rate = change_analysis['average_change_rate']
        
        # 整体一致性分数
        consistency_score = (stability_ratio + reasonableness_ratio) / 2
        
        return {
            'stability_ratio': stability_ratio,
            'reasonableness_ratio': reasonableness_ratio,
            'average_change_rate': avg_change_rate,
            'consistency_score': consistency_score
        }
    
    def correct_inconsistency(self, 
                            temporal_maps: List[np.ndarray],
                            correction_strategy: str = 'majority_temporal',
                            **kwargs) -> List[np.ndarray]:
        """
        修正时序不一致性
        
        Parameters:
        -----------
        temporal_maps : list
            时序分类图像列表
        correction_strategy : str, default='majority_temporal'
            修正策略
            
        Returns:
        --------
        corrected_maps : list
            修正后的时序分类图像
        """
        logger.info(f"开始时序一致性修正 - 策略: {correction_strategy}")
        
        corrected_maps = [map_data.copy() for map_data in temporal_maps]
        
        if correction_strategy == 'majority_temporal':
            corrected_maps = self._majority_temporal_correction(corrected_maps)
        elif correction_strategy == 'smoothing':
            corrected_maps = self._temporal_smoothing(corrected_maps)
        elif correction_strategy == 'rule_based':
            corrected_maps = self._rule_based_correction(corrected_maps)
        else:
            logger.warning(f"未知的修正策略: {correction_strategy}")
        
        logger.info("时序一致性修正完成")
        
        return corrected_maps
    
    def _majority_temporal_correction(self, temporal_maps: List[np.ndarray]) -> List[np.ndarray]:
        """基于时序多数投票的修正"""
        corrected_maps = temporal_maps.copy()
        height, width = temporal_maps[0].shape
        
        # 检查需要修正的异常像素
        if not hasattr(self, 'temporal_stats') or not self.temporal_stats:
            logger.warning("请先运行一致性检查")
            return corrected_maps
        
        anomalies = self.temporal_stats.get('anomalies', {})
        spatial_anomalies = anomalies.get('spatial_anomalies', [])
        
        for anomaly in spatial_anomalies:
            y, x = anomaly['coordinates']
            sequence = anomaly['sequence']
            
            # 使用时序窗口内的多数类别
            window_size = min(self.stable_class_period, len(sequence))
            
            for i in range(len(sequence)):
                # 定义窗口范围
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(sequence), i + window_size // 2 + 1)
                
                window_values = sequence[start_idx:end_idx]
                
                # 计算多数类别
                unique_values, counts = np.unique(window_values, return_counts=True)
                majority_class = unique_values[np.argmax(counts)]
                
                # 如果当前值与多数类别差异较大，进行修正
                current_value = sequence[i]
                if current_value != majority_class:
                    corrected_maps[i][y, x] = majority_class
        
        return corrected_maps
    
    def _temporal_smoothing(self, temporal_maps: List[np.ndarray]) -> List[np.ndarray]:
        """时序平滑修正"""
        corrected_maps = temporal_maps.copy()
        
        # 对每个像素进行时序滑动窗口平滑
        height, width = temporal_maps[0].shape
        window_size = 3  # 滑动窗口大小
        
        for y in range(height):
            for x in range(width):
                sequence = [map_data[y, x] for map_data in temporal_maps]
                
                # 滑动窗口平滑
                smoothed_sequence = self._smooth_sequence(sequence, window_size)
                
                # 更新修正结果
                for i, value in enumerate(smoothed_sequence):
                    corrected_maps[i][y, x] = value
        
        return corrected_maps
    
    def _smooth_sequence(self, sequence: List[int], window_size: int) -> List[int]:
        """平滑时序序列"""
        smoothed = sequence.copy()
        half_window = window_size // 2
        
        for i in range(len(sequence)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(sequence), i + half_window + 1)
            
            window_values = sequence[start_idx:end_idx]
            
            # 使用众数平滑
            unique_values, counts = np.unique(window_values, return_counts=True)
            mode_value = unique_values[np.argmax(counts)]
            
            smoothed[i] = mode_value
        
        return smoothed
    
    def _rule_based_correction(self, temporal_maps: List[np.ndarray]) -> List[np.ndarray]:
        """基于规则的修正"""
        corrected_maps = temporal_maps.copy()
        
        # 修正不合理的转换
        if hasattr(self, 'temporal_stats') and self.temporal_stats:
            transition_validity = self.temporal_stats.get('transition_validity', {})
            invalid_details = transition_validity.get('invalid_details', [])
            
            for invalid in invalid_details:
                y, x = invalid['coordinates']
                period = invalid['period']
                from_class = invalid['from_class']
                to_class = invalid['to_class']
                
                # 简单策略：保持原类别
                corrected_maps[period + 1][y, x] = from_class
        
        return corrected_maps


class SpatialConsistencyProcessor(ConsistencyProcessor):
    """
    空间一致性处理器
    
    处理分类结果的空间逻辑一致性。
    """
    
    def __init__(self, 
                 neighborhood_size: int = 3,
                 homogeneity_threshold: float = 0.7,
                 ecological_rules: Optional[Dict] = None,
                 **kwargs):
        """
        初始化空间一致性处理器
        
        Parameters:
        -----------
        neighborhood_size : int, default=3
            邻域大小
        homogeneity_threshold : float, default=0.7
            同质性阈值
        ecological_rules : dict, optional
            生态规则
        """
        super().__init__(**kwargs)
        self.neighborhood_size = neighborhood_size
        self.homogeneity_threshold = homogeneity_threshold
        self.ecological_rules = ecological_rules or self._default_ecological_rules()
        self.spatial_stats = {}
    
    def _default_ecological_rules(self) -> Dict[str, Any]:
        """默认生态规则"""
        return {
            'adjacency_rules': {
                # 水体类别通常与湿地植被、浅水区相邻
                'water': ['shallow_water', 'wetland_vegetation', 'mudflat'],
                'shallow_water': ['water', 'wetland_vegetation', 'mudflat'],
                'wetland_vegetation': ['water', 'shallow_water', 'dry_land'],
                'mudflat': ['water', 'shallow_water', 'wetland_vegetation'],
                'dry_land': ['wetland_vegetation', 'agriculture'],
                'agriculture': ['dry_land']
            },
            'isolation_rules': {
                # 某些类别不应该孤立存在
                'min_cluster_size': {
                    'water': 5,
                    'wetland_vegetation': 3,
                    'agriculture': 10
                }
            },
            'elevation_rules': {
                # 基于地形的约束（如果有DEM数据）
                'water_max_slope': 5.0,  # 度
                'agriculture_min_elevation': 10.0  # 米
            }
        }
    
    def check_consistency(self, 
                         classification_map: np.ndarray,
                         auxiliary_data: Optional[Dict[str, np.ndarray]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        检查空间一致性
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        auxiliary_data : dict, optional
            辅助数据（DEM、坡度等）
            
        Returns:
        --------
        results : dict
            空间一致性检查结果
        """
        logger.info("开始空间一致性检查")
        
        # 空间同质性分析
        homogeneity_analysis = self._analyze_spatial_homogeneity(classification_map)
        
        # 邻接关系检查
        adjacency_check = self._check_adjacency_rules(classification_map)
        
        # 孤立区域检查
        isolation_check = self._check_isolation_rules(classification_map)
        
        # 地形一致性检查（如果有辅助数据）
        terrain_check = {}
        if auxiliary_data:
            terrain_check = self._check_terrain_consistency(
                classification_map, auxiliary_data
            )
        
        # 计算一致性指标
        consistency_metrics = self._calculate_spatial_metrics(
            homogeneity_analysis, adjacency_check, isolation_check
        )
        
        results = {
            'homogeneity_analysis': homogeneity_analysis,
            'adjacency_check': adjacency_check,
            'isolation_check': isolation_check,
            'terrain_check': terrain_check,
            'consistency_metrics': consistency_metrics
        }
        
        self.spatial_stats = results
        
        logger.info("空间一致性检查完成")
        
        return results
    
    def _analyze_spatial_homogeneity(self, classification_map: np.ndarray) -> Dict[str, Any]:
        """分析空间同质性"""
        height, width = classification_map.shape
        half_window = self.neighborhood_size // 2
        
        homogeneity_map = np.zeros_like(classification_map, dtype=float)
        heterogeneous_pixels = []
        
        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                # 提取邻域
                neighborhood = classification_map[
                    i - half_window:i + half_window + 1,
                    j - half_window:j + half_window + 1
                ]
                
                center_value = classification_map[i, j]
                
                # 计算同质性
                same_class_count = np.sum(neighborhood == center_value)
                total_pixels = neighborhood.size
                homogeneity = same_class_count / total_pixels
                
                homogeneity_map[i, j] = homogeneity
                
                # 记录异质性像素
                if homogeneity < self.homogeneity_threshold:
                    heterogeneous_pixels.append({
                        'coordinates': (i, j),
                        'center_class': center_value,
                        'homogeneity': homogeneity,
                        'neighborhood': neighborhood.copy()
                    })
        
        return {
            'homogeneity_map': homogeneity_map,
            'heterogeneous_pixels': heterogeneous_pixels,
            'average_homogeneity': np.mean(homogeneity_map),
            'heterogeneous_count': len(heterogeneous_pixels)
        }
    
    def _check_adjacency_rules(self, classification_map: np.ndarray) -> Dict[str, Any]:
        """检查邻接规则"""
        violations = []
        adjacency_rules = self.ecological_rules.get('adjacency_rules', {})
        
        if not adjacency_rules:
            return {'violations': [], 'violation_count': 0}
        
        height, width = classification_map.shape
        
        # 定义4连通邻域
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for i in range(height):
            for j in range(width):
                center_class = classification_map[i, j]
                
                # 获取该类别的允许邻居
                allowed_neighbors = adjacency_rules.get(str(center_class), [])
                if not allowed_neighbors:
                    continue
                
                # 检查每个邻居
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_class = classification_map[ni, nj]
                        
                        if str(neighbor_class) not in allowed_neighbors:
                            violations.append({
                                'center_coordinates': (i, j),
                                'center_class': center_class,
                                'neighbor_coordinates': (ni, nj),
                                'neighbor_class': neighbor_class,
                                'allowed_neighbors': allowed_neighbors
                            })
        
        return {
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def _check_isolation_rules(self, classification_map: np.ndarray) -> Dict[str, Any]:
        """检查孤立规则"""
        violations = []
        isolation_rules = self.ecological_rules.get('isolation_rules', {})
        min_cluster_sizes = isolation_rules.get('min_cluster_size', {})
        
        if not min_cluster_sizes:
            return {'violations': [], 'small_clusters': []}
        
        # 对每个类别检查聚类大小
        unique_classes = np.unique(classification_map)
        small_clusters = []
        
        for class_id in unique_classes:
            min_size = min_cluster_sizes.get(str(class_id), 1)
            
            if min_size <= 1:
                continue
            
            # 创建类别掩码
            class_mask = classification_map == class_id
            
            # 连通组件分析
            labeled_mask, num_components = ndimage.label(class_mask)
            
            # 检查每个组件的大小
            for component_id in range(1, num_components + 1):
                component_mask = labeled_mask == component_id
                component_size = np.sum(component_mask)
                
                if component_size < min_size:
                    # 获取组件坐标
                    coords = np.where(component_mask)
                    
                    small_clusters.append({
                        'class_id': class_id,
                        'component_id': component_id,
                        'size': component_size,
                        'min_required_size': min_size,
                        'coordinates': list(zip(coords[0], coords[1]))
                    })
                    
                    violations.extend([
                        {
                            'coordinates': (y, x),
                            'class_id': class_id,
                            'violation_type': 'small_cluster',
                            'cluster_size': component_size,
                            'min_size': min_size
                        }
                        for y, x in zip(coords[0], coords[1])
                    ])
        
        return {
            'violations': violations,
            'small_clusters': small_clusters,
            'violation_count': len(violations)
        }
    
    def _check_terrain_consistency(self, 
                                  classification_map: np.ndarray,
                                  auxiliary_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """检查地形一致性"""
        violations = []
        terrain_rules = self.ecological_rules.get('elevation_rules', {})
        
        dem = auxiliary_data.get('dem')
        slope = auxiliary_data.get('slope')
        
        if dem is None and slope is None:
            return {'violations': [], 'violation_count': 0}
        
        height, width = classification_map.shape
        
        for i in range(height):
            for j in range(width):
                class_id = classification_map[i, j]
                
                # 检查水体坡度约束
                if (str(class_id) == 'water' and slope is not None and
                    'water_max_slope' in terrain_rules):
                    
                    max_slope = terrain_rules['water_max_slope']
                    current_slope = slope[i, j]
                    
                    if current_slope > max_slope:
                        violations.append({
                            'coordinates': (i, j),
                            'class_id': class_id,
                            'violation_type': 'slope_constraint',
                            'current_slope': current_slope,
                            'max_allowed_slope': max_slope
                        })
                
                # 检查农田高程约束
                if (str(class_id) == 'agriculture' and dem is not None and
                    'agriculture_min_elevation' in terrain_rules):
                    
                    min_elevation = terrain_rules['agriculture_min_elevation']
                    current_elevation = dem[i, j]
                    
                    if current_elevation < min_elevation:
                        violations.append({
                            'coordinates': (i, j),
                            'class_id': class_id,
                            'violation_type': 'elevation_constraint',
                            'current_elevation': current_elevation,
                            'min_required_elevation': min_elevation
                        })
        
        return {
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def _calculate_spatial_metrics(self, 
                                  homogeneity_analysis: Dict[str, Any],
                                  adjacency_check: Dict[str, Any],
                                  isolation_check: Dict[str, Any]) -> Dict[str, float]:
        """计算空间一致性指标"""
        total_pixels = homogeneity_analysis['homogeneity_map'].size
        
        # 同质性指标
        avg_homogeneity = homogeneity_analysis['average_homogeneity']
        
        # 邻接规则违反率
        adjacency_violation_rate = adjacency_check['violation_count'] / total_pixels
        
        # 孤立规则违反率
        isolation_violation_rate = isolation_check['violation_count'] / total_pixels
        
        # 总体一致性分数
        consistency_score = (
            avg_homogeneity * 
            (1 - adjacency_violation_rate) * 
            (1 - isolation_violation_rate)
        )
        
        return {
            'average_homogeneity': avg_homogeneity,
            'adjacency_violation_rate': adjacency_violation_rate,
            'isolation_violation_rate': isolation_violation_rate,
            'spatial_consistency_score': consistency_score
        }
    
    def correct_inconsistency(self, 
                            classification_map: np.ndarray,
                            correction_strategy: str = 'neighborhood_majority',
                            **kwargs) -> np.ndarray:
        """
        修正空间不一致性
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        correction_strategy : str, default='neighborhood_majority'
            修正策略
            
        Returns:
        --------
        corrected_map : np.ndarray
            修正后的分类图像
        """
        logger.info(f"开始空间一致性修正 - 策略: {correction_strategy}")
        
        corrected_map = classification_map.copy()
        
        if correction_strategy == 'neighborhood_majority':
            corrected_map = self._neighborhood_majority_correction(corrected_map)
        elif correction_strategy == 'remove_small_clusters':
            corrected_map = self._remove_small_clusters(corrected_map)
        elif correction_strategy == 'smooth_boundaries':
            corrected_map = self._smooth_boundaries(corrected_map)
        else:
            logger.warning(f"未知的修正策略: {correction_strategy}")
        
        logger.info("空间一致性修正完成")
        
        return corrected_map
    
    def _neighborhood_majority_correction(self, classification_map: np.ndarray) -> np.ndarray:
        """基于邻域多数的修正"""
        if not hasattr(self, 'spatial_stats') or not self.spatial_stats:
            logger.warning("请先运行一致性检查")
            return classification_map
        
        corrected_map = classification_map.copy()
        homogeneity_analysis = self.spatial_stats.get('homogeneity_analysis', {})
        heterogeneous_pixels = homogeneity_analysis.get('heterogeneous_pixels', [])
        
        for pixel_info in heterogeneous_pixels:
            i, j = pixel_info['coordinates']
            neighborhood = pixel_info['neighborhood']
            
            # 计算邻域内的多数类别
            unique_values, counts = np.unique(neighborhood, return_counts=True)
            majority_class = unique_values[np.argmax(counts)]
            
            # 如果多数类别与中心像素不同，且多数类别占比足够高
            if majority_class != classification_map[i, j]:
                majority_ratio = np.max(counts) / neighborhood.size
                if majority_ratio >= self.homogeneity_threshold:
                    corrected_map[i, j] = majority_class
        
        return corrected_map
    
    def _remove_small_clusters(self, classification_map: np.ndarray) -> np.ndarray:
        """移除小聚类"""
        if not hasattr(self, 'spatial_stats') or not self.spatial_stats:
            logger.warning("请先运行一致性检查")
            return classification_map
        
        corrected_map = classification_map.copy()
        isolation_check = self.spatial_stats.get('isolation_check', {})
        small_clusters = isolation_check.get('small_clusters', [])
        
        for cluster_info in small_clusters:
            coordinates = cluster_info['coordinates']
            
            # 将小聚类的像素重新分配给周围的多数类别
            for y, x in coordinates:
                # 获取周围像素的类别
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = y + di, x + dj
                        if (0 <= ni < classification_map.shape[0] and 
                            0 <= nj < classification_map.shape[1]):
                            neighbors.append(corrected_map[ni, nj])
                
                if neighbors:
                    # 使用邻居中的多数类别
                    unique_values, counts = np.unique(neighbors, return_counts=True)
                    majority_class = unique_values[np.argmax(counts)]
                    corrected_map[y, x] = majority_class
        
        return corrected_map
    
    def _smooth_boundaries(self, classification_map: np.ndarray) -> np.ndarray:
        """平滑边界"""
        from scipy.ndimage import median_filter
        
        # 使用中值滤波进行边界平滑
        corrected_map = median_filter(classification_map, size=3)
        
        return corrected_map


def create_consistency_pipeline() -> List[ConsistencyProcessor]:
    """
    创建一致性处理流水线
    
    Returns:
    --------
    processors : list
        一致性处理器列表
    """
    processors = [
        # 空间一致性处理
        SpatialConsistencyProcessor(
            neighborhood_size=3,
            homogeneity_threshold=0.7
        )
    ]
    
    return processors


def apply_consistency_check(classification_map: Union[np.ndarray, List[np.ndarray]],
                          processor_type: str = 'spatial',
                          **kwargs) -> Dict[str, Any]:
    """
    应用一致性检查
    
    Parameters:
    -----------
    classification_map : np.ndarray or list
        分类图像或时序分类图像列表
    processor_type : str, default='spatial'
        处理器类型 ('spatial', 'temporal')
        
    Returns:
    --------
    results : dict
        一致性检查结果
    """
    if processor_type == 'spatial':
        if isinstance(classification_map, list):
            classification_map = classification_map[0]
        
        processor = SpatialConsistencyProcessor(**kwargs)
        return processor.check_consistency(classification_map, **kwargs)
    
    elif processor_type == 'temporal':
        if not isinstance(classification_map, list):
            raise ValueError("时序一致性检查需要时序分类图像列表")
        
        processor = TemporalConsistencyProcessor(**kwargs)
        return processor.check_consistency(classification_map, **kwargs)
    
    else:
        raise ValueError(f"不支持的处理器类型: {processor_type}")