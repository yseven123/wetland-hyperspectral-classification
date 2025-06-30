"""
交叉验证模块
===========

这个模块提供了多种交叉验证策略，用于评估分类模型的泛化能力和稳定性。

主要功能：
- 标准交叉验证：K折、分层K折、留一法
- 空间交叉验证：空间分块、缓冲区验证
- 时序交叉验证：时间序列分割、滑动窗口
- 自助法验证：Bootstrap、0.632估计
- 嵌套交叉验证：模型选择与性能评估
- 验证策略比较：多策略性能对比

特殊考虑：
- 遥感数据的空间自相关性
- 时序数据的时间依赖性
- 样本不平衡问题
- 计算效率优化

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Generator
import logging
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut,
    train_test_split, cross_val_score, cross_validate,
    StratifiedShuffleSplit, TimeSeriesSplit
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class CrossValidator:
    """
    交叉验证器基类
    
    提供统一的交叉验证接口和通用功能。
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 random_state: int = 42,
                 shuffle: bool = True,
                 **kwargs):
        """
        初始化交叉验证器
        
        Parameters:
        -----------
        n_splits : int, default=5
            折数
        random_state : int, default=42
            随机种子
        shuffle : bool, default=True
            是否打乱数据
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.config = kwargs
        
        # 设置随机种子
        np.random.seed(random_state)
    
    def validate(self, 
                classifier,
                X: np.ndarray,
                y: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        执行交叉验证
        
        Parameters:
        -----------
        classifier : object
            分类器实例
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
            
        Returns:
        --------
        validation_results : dict
            验证结果
        """
        raise NotImplementedError("子类必须实现validate方法")
    
    def _evaluate_fold(self, 
                      classifier,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, float]:
        """评估单个fold"""
        # 训练模型
        classifier_copy = clone(classifier)
        classifier_copy.fit(X_train, y_train)
        
        # 预测
        y_pred = classifier_copy.predict(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def _aggregate_results(self, fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """聚合fold结果"""
        if not fold_results:
            return {}
        
        # 提取所有指标
        all_metrics = list(fold_results[0].keys())
        
        aggregated = {}
        for metric in all_metrics:
            values = [result[metric] for result in fold_results]
            
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            aggregated[f'{metric}_values'] = values
        
        return aggregated


class StandardCrossValidator(CrossValidator):
    """
    标准交叉验证器
    
    实现K折、分层K折、留一法等标准验证策略。
    """
    
    def __init__(self, 
                 validation_type: str = 'stratified_kfold',
                 **kwargs):
        """
        初始化标准交叉验证器
        
        Parameters:
        -----------
        validation_type : str, default='stratified_kfold'
            验证类型 ('kfold', 'stratified_kfold', 'leave_one_out', 'leave_p_out')
        """
        super().__init__(**kwargs)
        self.validation_type = validation_type
        
        # 创建验证器
        self.cv_splitter = self._create_splitter()
    
    def _create_splitter(self):
        """创建数据分割器"""
        if self.validation_type == 'kfold':
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.validation_type == 'stratified_kfold':
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif self.validation_type == 'leave_one_out':
            return LeaveOneOut()
        elif self.validation_type == 'leave_p_out':
            p = self.config.get('p', 2)
            return LeavePOut(p=p)
        else:
            raise ValueError(f"不支持的验证类型: {self.validation_type}")
    
    def validate(self, 
                classifier,
                X: np.ndarray,
                y: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        执行标准交叉验证
        
        Parameters:
        -----------
        classifier : object
            分类器实例
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
            
        Returns:
        --------
        validation_results : dict
            验证结果
        """
        logger.info(f"执行{self.validation_type}交叉验证")
        start_time = time.time()
        
        fold_results = []
        fold_details = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, y)):
            logger.debug(f"执行第 {fold_idx + 1} 折验证")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 评估fold
            fold_metrics = self._evaluate_fold(
                classifier, X_train, y_train, X_test, y_test
            )
            
            fold_results.append(fold_metrics)
            
            # 记录fold详情
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_indices': train_idx.tolist(),
                'test_indices': test_idx.tolist(),
                'metrics': fold_metrics
            })
        
        # 聚合结果
        aggregated_results = self._aggregate_results(fold_results)
        
        validation_time = time.time() - start_time
        
        return {
            'validation_type': self.validation_type,
            'n_splits': len(fold_results),
            'aggregated_metrics': aggregated_results,
            'fold_details': fold_details,
            'validation_time': validation_time,
            'summary': self._create_summary(aggregated_results)
        }
    
    def _create_summary(self, aggregated_results: Dict[str, Any]) -> Dict[str, str]:
        """创建验证摘要"""
        summary = {}
        
        # 主要指标
        accuracy_mean = aggregated_results.get('accuracy_mean', 0)
        accuracy_std = aggregated_results.get('accuracy_std', 0)
        
        summary['accuracy'] = f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
        
        f1_mean = aggregated_results.get('f1_macro_mean', 0)
        f1_std = aggregated_results.get('f1_macro_std', 0)
        
        summary['f1_macro'] = f"{f1_mean:.4f} ± {f1_std:.4f}"
        
        # 性能评价
        if accuracy_mean >= 0.9:
            summary['performance_level'] = "优秀"
        elif accuracy_mean >= 0.8:
            summary['performance_level'] = "良好"
        elif accuracy_mean >= 0.7:
            summary['performance_level'] = "一般"
        else:
            summary['performance_level'] = "较差"
        
        # 稳定性评价
        if accuracy_std <= 0.02:
            summary['stability'] = "非常稳定"
        elif accuracy_std <= 0.05:
            summary['stability'] = "稳定"
        elif accuracy_std <= 0.1:
            summary['stability'] = "一般"
        else:
            summary['stability'] = "不稳定"
        
        return summary


class SpatialCrossValidator(CrossValidator):
    """
    空间交叉验证器
    
    考虑空间自相关性的交叉验证，适用于遥感数据。
    """
    
    def __init__(self, 
                 spatial_strategy: str = 'spatial_blocking',
                 block_size: Optional[int] = None,
                 buffer_distance: float = 0.0,
                 coordinates: Optional[np.ndarray] = None,
                 **kwargs):
        """
        初始化空间交叉验证器
        
        Parameters:
        -----------
        spatial_strategy : str, default='spatial_blocking'
            空间策略 ('spatial_blocking', 'buffered_leave_one_out', 'distance_based')
        block_size : int, optional
            空间块大小
        buffer_distance : float, default=0.0
            缓冲区距离
        coordinates : np.ndarray, optional
            样本坐标 (n_samples, 2)
        """
        super().__init__(**kwargs)
        self.spatial_strategy = spatial_strategy
        self.block_size = block_size
        self.buffer_distance = buffer_distance
        self.coordinates = coordinates
    
    def validate(self, 
                classifier,
                X: np.ndarray,
                y: np.ndarray,
                coordinates: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行空间交叉验证
        
        Parameters:
        -----------
        classifier : object
            分类器实例
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
        coordinates : np.ndarray, optional
            样本坐标
            
        Returns:
        --------
        validation_results : dict
            验证结果
        """
        if coordinates is not None:
            self.coordinates = coordinates
        
        if self.coordinates is None:
            raise ValueError("空间交叉验证需要提供样本坐标")
        
        logger.info(f"执行空间交叉验证 - 策略: {self.spatial_strategy}")
        start_time = time.time()
        
        # 根据策略生成训练/测试分割
        if self.spatial_strategy == 'spatial_blocking':
            splits = self._spatial_blocking_splits(X, y)
        elif self.spatial_strategy == 'buffered_leave_one_out':
            splits = self._buffered_loo_splits(X, y)
        elif self.spatial_strategy == 'distance_based':
            splits = self._distance_based_splits(X, y)
        else:
            raise ValueError(f"不支持的空间策略: {self.spatial_strategy}")
        
        fold_results = []
        fold_details = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.debug(f"执行第 {fold_idx + 1} 折空间验证")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 评估fold
            fold_metrics = self._evaluate_fold(
                classifier, X_train, y_train, X_test, y_test
            )
            
            fold_results.append(fold_metrics)
            
            # 记录fold详情
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'spatial_separation': self._calculate_spatial_separation(train_idx, test_idx),
                'metrics': fold_metrics
            })
        
        # 聚合结果
        aggregated_results = self._aggregate_results(fold_results)
        
        validation_time = time.time() - start_time
        
        return {
            'validation_type': f'spatial_{self.spatial_strategy}',
            'n_splits': len(fold_results),
            'aggregated_metrics': aggregated_results,
            'fold_details': fold_details,
            'validation_time': validation_time,
            'spatial_parameters': {
                'strategy': self.spatial_strategy,
                'block_size': self.block_size,
                'buffer_distance': self.buffer_distance
            },
            'summary': self._create_summary(aggregated_results)
        }
    
    def _spatial_blocking_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """空间分块分割"""
        if self.block_size is None:
            # 自动确定块大小
            n_samples = len(X)
            self.block_size = int(np.sqrt(n_samples / self.n_splits))
        
        # 计算空间网格
        min_x, min_y = np.min(self.coordinates, axis=0)
        max_x, max_y = np.max(self.coordinates, axis=0)
        
        x_blocks = int(np.ceil((max_x - min_x) / self.block_size))
        y_blocks = int(np.ceil((max_y - min_y) / self.block_size))
        
        # 为每个样本分配块ID
        block_ids = []
        for coord in self.coordinates:
            x_block = int((coord[0] - min_x) / self.block_size)
            y_block = int((coord[1] - min_y) / self.block_size)
            
            # 确保块ID在有效范围内
            x_block = min(x_block, x_blocks - 1)
            y_block = min(y_block, y_blocks - 1)
            
            block_id = y_block * x_blocks + x_block
            block_ids.append(block_id)
        
        block_ids = np.array(block_ids)
        unique_blocks = np.unique(block_ids)
        
        # 随机分配块到折
        np.random.shuffle(unique_blocks)
        blocks_per_fold = len(unique_blocks) // self.n_splits
        
        splits = []
        for fold in range(self.n_splits):
            # 确定测试块
            start_idx = fold * blocks_per_fold
            if fold == self.n_splits - 1:
                # 最后一折包含剩余所有块
                test_blocks = unique_blocks[start_idx:]
            else:
                end_idx = start_idx + blocks_per_fold
                test_blocks = unique_blocks[start_idx:end_idx]
            
            # 生成训练/测试索引
            test_mask = np.isin(block_ids, test_blocks)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _buffered_loo_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """带缓冲区的留一法"""
        n_samples = len(X)
        splits = []
        
        # 限制样本数量以避免过多的分割
        max_samples = min(n_samples, 100)
        sample_indices = np.random.choice(n_samples, max_samples, replace=False)
        
        for test_idx in sample_indices:
            test_coord = self.coordinates[test_idx:test_idx+1]
            
            # 计算所有样本到测试样本的距离
            distances = cdist(self.coordinates, test_coord).flatten()
            
            # 创建缓冲区
            buffer_mask = distances <= self.buffer_distance
            
            # 测试集：目标样本
            test_indices = np.array([test_idx])
            
            # 训练集：缓冲区外的样本
            train_mask = ~buffer_mask
            train_indices = np.where(train_mask)[0]
            
            if len(train_indices) > 0:  # 确保训练集不为空
                splits.append((train_indices, test_indices))
        
        return splits
    
    def _distance_based_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """基于距离的分割"""
        n_samples = len(X)
        
        # 使用K-means聚类创建空间分组
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_splits, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(self.coordinates)
        
        splits = []
        for fold in range(self.n_splits):
            test_mask = (cluster_labels == fold)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _calculate_spatial_separation(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
        """计算空间分离度"""
        train_coords = self.coordinates[train_idx]
        test_coords = self.coordinates[test_idx]
        
        # 计算最近距离
        distances = cdist(test_coords, train_coords)
        min_distances = np.min(distances, axis=1)
        
        separation = {
            'min_distance': np.min(min_distances),
            'mean_distance': np.mean(min_distances),
            'max_distance': np.max(min_distances),
            'median_distance': np.median(min_distances)
        }
        
        return separation
    
    def _create_summary(self, aggregated_results: Dict[str, Any]) -> Dict[str, str]:
        """创建空间验证摘要"""
        summary = {}
        
        # 基础摘要
        accuracy_mean = aggregated_results.get('accuracy_mean', 0)
        accuracy_std = aggregated_results.get('accuracy_std', 0)
        
        summary['accuracy'] = f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
        
        # 空间效应评估
        # 比较空间验证与随机验证的差异可以揭示空间自相关的影响
        if accuracy_std > 0.1:
            summary['spatial_effect'] = "显著空间效应，存在空间自相关"
        elif accuracy_std > 0.05:
            summary['spatial_effect'] = "中等空间效应"
        else:
            summary['spatial_effect'] = "空间效应较小"
        
        return summary


class TemporalCrossValidator(CrossValidator):
    """
    时序交叉验证器
    
    考虑时间依赖性的交叉验证，适用于时序遥感数据。
    """
    
    def __init__(self, 
                 temporal_strategy: str = 'time_series_split',
                 n_test_splits: int = 1,
                 gap: int = 0,
                 **kwargs):
        """
        初始化时序交叉验证器
        
        Parameters:
        -----------
        temporal_strategy : str, default='time_series_split'
            时序策略 ('time_series_split', 'sliding_window', 'expanding_window')
        n_test_splits : int, default=1
            测试分割数
        gap : int, default=0
            训练和测试之间的间隔
        """
        super().__init__(**kwargs)
        self.temporal_strategy = temporal_strategy
        self.n_test_splits = n_test_splits
        self.gap = gap
    
    def validate(self, 
                classifier,
                X: np.ndarray,
                y: np.ndarray,
                time_indices: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行时序交叉验证
        
        Parameters:
        -----------
        classifier : object
            分类器实例
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
        time_indices : np.ndarray, optional
            时间索引
            
        Returns:
        --------
        validation_results : dict
            验证结果
        """
        logger.info(f"执行时序交叉验证 - 策略: {self.temporal_strategy}")
        start_time = time.time()
        
        # 生成时序分割
        if self.temporal_strategy == 'time_series_split':
            splits = self._time_series_splits(X, y, time_indices)
        elif self.temporal_strategy == 'sliding_window':
            splits = self._sliding_window_splits(X, y, time_indices)
        elif self.temporal_strategy == 'expanding_window':
            splits = self._expanding_window_splits(X, y, time_indices)
        else:
            raise ValueError(f"不支持的时序策略: {self.temporal_strategy}")
        
        fold_results = []
        fold_details = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.debug(f"执行第 {fold_idx + 1} 折时序验证")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 评估fold
            fold_metrics = self._evaluate_fold(
                classifier, X_train, y_train, X_test, y_test
            )
            
            fold_results.append(fold_metrics)
            
            # 记录fold详情
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'temporal_gap': self._calculate_temporal_gap(train_idx, test_idx, time_indices),
                'metrics': fold_metrics
            })
        
        # 聚合结果
        aggregated_results = self._aggregate_results(fold_results)
        
        validation_time = time.time() - start_time
        
        return {
            'validation_type': f'temporal_{self.temporal_strategy}',
            'n_splits': len(fold_results),
            'aggregated_metrics': aggregated_results,
            'fold_details': fold_details,
            'validation_time': validation_time,
            'temporal_parameters': {
                'strategy': self.temporal_strategy,
                'n_test_splits': self.n_test_splits,
                'gap': self.gap
            },
            'summary': self._create_summary(aggregated_results)
        }
    
    def _time_series_splits(self, X: np.ndarray, y: np.ndarray, time_indices: Optional[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """时间序列分割"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        
        if time_indices is not None:
            # 按时间索引排序
            sorted_indices = np.argsort(time_indices)
            splits = []
            
            for train_idx, test_idx in tscv.split(X):
                # 将索引映射回原始数据
                actual_train_idx = sorted_indices[train_idx]
                actual_test_idx = sorted_indices[test_idx]
                splits.append((actual_train_idx, actual_test_idx))
            
            return splits
        else:
            # 假设数据已按时间排序
            return list(tscv.split(X))
    
    def _sliding_window_splits(self, X: np.ndarray, y: np.ndarray, time_indices: Optional[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """滑动窗口分割"""
        n_samples = len(X)
        window_size = n_samples // (self.n_splits + 1)
        
        splits = []
        for i in range(self.n_splits):
            train_start = i * window_size
            train_end = train_start + window_size
            test_start = train_end + self.gap
            test_end = min(test_start + window_size, n_samples)
            
            if test_end > test_start:
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))
        
        return splits
    
    def _expanding_window_splits(self, X: np.ndarray, y: np.ndarray, time_indices: Optional[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """扩展窗口分割"""
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        splits = []
        for i in range(self.n_splits):
            train_end = n_samples - (self.n_splits - i) * test_size - self.gap
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if train_end > 0 and test_end <= n_samples:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)
                splits.append((train_idx, test_idx))
        
        return splits
    
    def _calculate_temporal_gap(self, train_idx: np.ndarray, test_idx: np.ndarray, time_indices: Optional[np.ndarray]) -> Dict[str, Any]:
        """计算时间间隔"""
        if time_indices is None:
            # 使用索引作为时间代理
            gap_info = {
                'index_gap': np.min(test_idx) - np.max(train_idx),
                'temporal_gap': None
            }
        else:
            train_times = time_indices[train_idx]
            test_times = time_indices[test_idx]
            
            gap_info = {
                'index_gap': np.min(test_idx) - np.max(train_idx),
                'temporal_gap': np.min(test_times) - np.max(train_times),
                'train_time_range': (np.min(train_times), np.max(train_times)),
                'test_time_range': (np.min(test_times), np.max(test_times))
            }
        
        return gap_info
    
    def _create_summary(self, aggregated_results: Dict[str, Any]) -> Dict[str, str]:
        """创建时序验证摘要"""
        summary = {}
        
        # 基础摘要
        accuracy_mean = aggregated_results.get('accuracy_mean', 0)
        accuracy_std = aggregated_results.get('accuracy_std', 0)
        
        summary['accuracy'] = f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
        
        # 时序稳定性评估
        if accuracy_std > 0.15:
            summary['temporal_stability'] = "时序性能不稳定，可能存在概念漂移"
        elif accuracy_std > 0.08:
            summary['temporal_stability'] = "时序性能中等稳定"
        else:
            summary['temporal_stability'] = "时序性能稳定"
        
        return summary


class BootstrapValidator(CrossValidator):
    """
    自助法验证器
    
    使用Bootstrap方法进行模型验证。
    """
    
    def __init__(self, 
                 n_bootstrap: int = 100,
                 bootstrap_method: str = 'bootstrap_632',
                 **kwargs):
        """
        初始化自助法验证器
        
        Parameters:
        -----------
        n_bootstrap : int, default=100
            Bootstrap采样次数
        bootstrap_method : str, default='bootstrap_632'
            Bootstrap方法 ('bootstrap', 'bootstrap_632', 'bootstrap_632_plus')
        """
        super().__init__(**kwargs)
        self.n_bootstrap = n_bootstrap
        self.bootstrap_method = bootstrap_method
    
    def validate(self, 
                classifier,
                X: np.ndarray,
                y: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        执行Bootstrap验证
        
        Parameters:
        -----------
        classifier : object
            分类器实例
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
            
        Returns:
        --------
        validation_results : dict
            验证结果
        """
        logger.info(f"执行Bootstrap验证 - 方法: {self.bootstrap_method}")
        start_time = time.time()
        
        n_samples = len(X)
        bootstrap_results = []
        
        for bootstrap_idx in range(self.n_bootstrap):
            # Bootstrap采样
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), boot_indices)
            
            if len(oob_indices) == 0:
                continue  # 跳过没有OOB样本的情况
            
            X_boot, y_boot = X[boot_indices], y[boot_indices]
            X_oob, y_oob = X[oob_indices], y[oob_indices]
            
            # 训练和评估
            classifier_copy = clone(classifier)
            classifier_copy.fit(X_boot, y_boot)
            
            # OOB评估
            y_oob_pred = classifier_copy.predict(X_oob)
            oob_accuracy = accuracy_score(y_oob, y_oob_pred)
            
            # 重采样评估（用于.632方法）
            y_boot_pred = classifier_copy.predict(X_boot)
            boot_accuracy = accuracy_score(y_boot, y_boot_pred)
            
            bootstrap_results.append({
                'bootstrap_idx': bootstrap_idx,
                'oob_accuracy': oob_accuracy,
                'boot_accuracy': boot_accuracy,
                'oob_size': len(oob_indices),
                'boot_size': len(boot_indices)
            })
        
        # 计算最终估计
        final_estimate = self._calculate_bootstrap_estimate(bootstrap_results)
        
        validation_time = time.time() - start_time
        
        return {
            'validation_type': f'bootstrap_{self.bootstrap_method}',
            'n_bootstrap': len(bootstrap_results),
            'final_estimate': final_estimate,
            'bootstrap_results': bootstrap_results,
            'validation_time': validation_time,
            'summary': self._create_bootstrap_summary(final_estimate, bootstrap_results)
        }
    
    def _calculate_bootstrap_estimate(self, bootstrap_results: List[Dict]) -> Dict[str, float]:
        """计算Bootstrap估计"""
        oob_accuracies = [result['oob_accuracy'] for result in bootstrap_results]
        boot_accuracies = [result['boot_accuracy'] for result in bootstrap_results]
        
        oob_mean = np.mean(oob_accuracies)
        boot_mean = np.mean(boot_accuracies)
        
        if self.bootstrap_method == 'bootstrap':
            # 标准Bootstrap估计
            estimate = oob_mean
        elif self.bootstrap_method == 'bootstrap_632':
            # .632估计
            estimate = 0.632 * oob_mean + 0.368 * boot_mean
        elif self.bootstrap_method == 'bootstrap_632_plus':
            # .632+估计（简化版）
            gamma = oob_mean - boot_mean
            weight = 0.632 / (1 - 0.368 * gamma) if gamma != 0.368 else 0.632
            estimate = weight * oob_mean + (1 - weight) * boot_mean
        else:
            estimate = oob_mean
        
        return {
            'final_accuracy': estimate,
            'oob_accuracy_mean': oob_mean,
            'oob_accuracy_std': np.std(oob_accuracies),
            'boot_accuracy_mean': boot_mean,
            'boot_accuracy_std': np.std(boot_accuracies)
        }
    
    def _create_bootstrap_summary(self, final_estimate: Dict[str, float], bootstrap_results: List[Dict]) -> Dict[str, str]:
        """创建Bootstrap摘要"""
        summary = {}
        
        final_acc = final_estimate['final_accuracy']
        oob_std = final_estimate['oob_accuracy_std']
        
        summary['final_accuracy'] = f"{final_acc:.4f}"
        summary['confidence_interval'] = f"[{final_acc - 1.96*oob_std:.4f}, {final_acc + 1.96*oob_std:.4f}]"
        
        # 评估可靠性
        if oob_std < 0.02:
            summary['reliability'] = "高可靠性"
        elif oob_std < 0.05:
            summary['reliability'] = "中等可靠性"
        else:
            summary['reliability'] = "低可靠性"
        
        return summary


class ValidationComparator:
    """
    验证策略比较器
    
    比较不同验证策略的结果，帮助选择最适合的验证方法。
    """
    
    def __init__(self):
        """初始化验证比较器"""
        self.comparison_results = {}
    
    def compare_validation_strategies(self, 
                                    classifier,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    strategies: List[str] = None,
                                    coordinates: Optional[np.ndarray] = None,
                                    time_indices: Optional[np.ndarray] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        比较多种验证策略
        
        Parameters:
        -----------
        classifier : object
            分类器实例
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
        strategies : list, optional
            验证策略列表
        coordinates : np.ndarray, optional
            空间坐标
        time_indices : np.ndarray, optional
            时间索引
            
        Returns:
        --------
        comparison_results : dict
            比较结果
        """
        if strategies is None:
            strategies = ['kfold', 'stratified_kfold', 'spatial_blocking', 'bootstrap']
        
        logger.info(f"比较 {len(strategies)} 种验证策略")
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"执行验证策略: {strategy}")
            
            try:
                if strategy in ['kfold', 'stratified_kfold', 'leave_one_out']:
                    validator = StandardCrossValidator(validation_type=strategy)
                    result = validator.validate(classifier, X, y)
                
                elif strategy in ['spatial_blocking', 'buffered_leave_one_out', 'distance_based']:
                    if coordinates is None:
                        logger.warning(f"跳过空间验证策略 {strategy}：缺少坐标信息")
                        continue
                    validator = SpatialCrossValidator(spatial_strategy=strategy, coordinates=coordinates)
                    result = validator.validate(classifier, X, y)
                
                elif strategy in ['time_series_split', 'sliding_window', 'expanding_window']:
                    validator = TemporalCrossValidator(temporal_strategy=strategy)
                    result = validator.validate(classifier, X, y, time_indices=time_indices)
                
                elif strategy in ['bootstrap', 'bootstrap_632']:
                    validator = BootstrapValidator(bootstrap_method=strategy)
                    result = validator.validate(classifier, X, y)
                
                else:
                    logger.warning(f"未知的验证策略: {strategy}")
                    continue
                
                results[strategy] = result
                
            except Exception as e:
                logger.error(f"验证策略 {strategy} 执行失败: {e}")
                continue
        
        # 创建比较分析
        comparison_analysis = self._analyze_strategy_comparison(results)
        
        self.comparison_results = {
            'individual_results': results,
            'comparison_analysis': comparison_analysis,
            'recommendations': self._generate_strategy_recommendations(comparison_analysis)
        }
        
        return self.comparison_results
    
    def _analyze_strategy_comparison(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """分析策略比较结果"""
        if not results:
            return {}
        
        # 提取关键指标
        strategy_metrics = {}
        for strategy, result in results.items():
            if 'aggregated_metrics' in result:
                metrics = result['aggregated_metrics']
                strategy_metrics[strategy] = {
                    'accuracy_mean': metrics.get('accuracy_mean', 0),
                    'accuracy_std': metrics.get('accuracy_std', 0),
                    'f1_macro_mean': metrics.get('f1_macro_mean', 0),
                    'f1_macro_std': metrics.get('f1_macro_std', 0)
                }
            elif 'final_estimate' in result:  # Bootstrap结果
                estimate = result['final_estimate']
                strategy_metrics[strategy] = {
                    'accuracy_mean': estimate.get('final_accuracy', 0),
                    'accuracy_std': estimate.get('oob_accuracy_std', 0),
                    'f1_macro_mean': 0,  # Bootstrap通常只计算精度
                    'f1_macro_std': 0
                }
        
        # 统计分析
        accuracies = [metrics['accuracy_mean'] for metrics in strategy_metrics.values()]
        accuracy_stds = [metrics['accuracy_std'] for metrics in strategy_metrics.values()]
        
        analysis = {
            'strategy_metrics': strategy_metrics,
            'accuracy_range': (min(accuracies), max(accuracies)) if accuracies else (0, 0),
            'accuracy_variance': np.var(accuracies) if len(accuracies) > 1 else 0,
            'stability_range': (min(accuracy_stds), max(accuracy_stds)) if accuracy_stds else (0, 0),
            'most_optimistic_strategy': max(strategy_metrics.keys(), key=lambda k: strategy_metrics[k]['accuracy_mean']) if strategy_metrics else None,
            'most_conservative_strategy': min(strategy_metrics.keys(), key=lambda k: strategy_metrics[k]['accuracy_mean']) if strategy_metrics else None,
            'most_stable_strategy': min(strategy_metrics.keys(), key=lambda k: strategy_metrics[k]['accuracy_std']) if strategy_metrics else None
        }
        
        return analysis
    
    def _generate_strategy_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成策略推荐"""
        recommendations = []
        
        if not analysis:
            return ["无法生成推荐：分析结果为空"]
        
        accuracy_variance = analysis.get('accuracy_variance', 0)
        
        if accuracy_variance > 0.01:
            recommendations.append(
                "不同验证策略结果差异较大，建议仔细考虑数据的空间和时间特性"
            )
        
        most_conservative = analysis.get('most_conservative_strategy')
        if most_conservative:
            recommendations.append(
                f"最保守的估计来自 {most_conservative}，适合作为性能下界"
            )
        
        most_stable = analysis.get('most_stable_strategy')
        if most_stable:
            recommendations.append(
                f"最稳定的策略是 {most_stable}，推荐用于模型评估"
            )
        
        # 根据数据特性给出具体建议
        strategy_metrics = analysis.get('strategy_metrics', {})
        
        if 'spatial_blocking' in strategy_metrics and 'stratified_kfold' in strategy_metrics:
            spatial_acc = strategy_metrics['spatial_blocking']['accuracy_mean']
            standard_acc = strategy_metrics['stratified_kfold']['accuracy_mean']
            
            if abs(spatial_acc - standard_acc) > 0.05:
                recommendations.append(
                    "空间验证与标准验证结果差异显著，数据存在空间自相关，推荐使用空间验证"
                )
        
        if not recommendations:
            recommendations.append("各种验证策略结果一致，可选择计算效率较高的策略")
        
        return recommendations
    
    def plot_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制策略比较图"""
        if not self.comparison_results:
            raise ValueError("尚未执行策略比较")
        
        strategy_metrics = self.comparison_results['comparison_analysis'].get('strategy_metrics', {})
        
        if not strategy_metrics:
            raise ValueError("无可绘制的策略指标")
        
        strategies = list(strategy_metrics.keys())
        accuracies = [strategy_metrics[s]['accuracy_mean'] for s in strategies]
        accuracy_stds = [strategy_metrics[s]['accuracy_std'] for s in strategies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 精度比较
        ax1.bar(strategies, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.7)
        ax1.set_title('验证策略精度比较')
        ax1.set_ylabel('精度')
        ax1.set_xlabel('验证策略')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 稳定性比较
        ax2.bar(strategies, accuracy_stds, alpha=0.7, color='orange')
        ax2.set_title('验证策略稳定性比较')
        ax2.set_ylabel('精度标准差')
        ax2.set_xlabel('验证策略')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def cross_validate_classifier(classifier,
                            X: np.ndarray,
                            y: np.ndarray,
                            validation_type: str = 'stratified_kfold',
                            n_splits: int = 5,
                            **kwargs) -> Dict[str, Any]:
    """
    交叉验证的便捷函数
    
    Parameters:
    -----------
    classifier : object
        分类器实例
    X : np.ndarray
        特征数据
    y : np.ndarray
        标签数据
    validation_type : str, default='stratified_kfold'
        验证类型
    n_splits : int, default=5
        折数
        
    Returns:
    --------
    validation_results : dict
        验证结果
    """
    if validation_type in ['kfold', 'stratified_kfold', 'leave_one_out']:
        validator = StandardCrossValidator(
            validation_type=validation_type,
            n_splits=n_splits,
            **kwargs
        )
    elif validation_type in ['spatial_blocking', 'buffered_leave_one_out']:
        validator = SpatialCrossValidator(
            spatial_strategy=validation_type,
            n_splits=n_splits,
            **kwargs
        )
    elif validation_type in ['time_series_split', 'sliding_window']:
        validator = TemporalCrossValidator(
            temporal_strategy=validation_type,
            n_splits=n_splits,
            **kwargs
        )
    elif validation_type in ['bootstrap', 'bootstrap_632']:
        validator = BootstrapValidator(
            bootstrap_method=validation_type,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的验证类型: {validation_type}")
    
    return validator.validate(classifier, X, y, **kwargs)