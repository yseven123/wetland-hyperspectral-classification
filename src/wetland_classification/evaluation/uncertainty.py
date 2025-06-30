"""
不确定性分析模块
===============

提供多种不确定性量化方法，用于评估分类结果的可靠性和置信度。
支持模型不确定性、预测不确定性和空间不确定性分析。

主要功能：
- 贝叶斯不确定性量化
- 集成模型不确定性
- 空间不确定性分析
- 置信度映射生成
- 不确定性可视化

作者: 湿地遥感团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UncertaintyAnalyzer:
    """
    不确定性分析器
    
    提供多种方法量化分类结果的不确定性，包括：
    - 预测不确定性
    - 模型不确定性
    - 空间不确定性
    - 认知不确定性和随机不确定性
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化不确定性分析器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.uncertainty_methods = {
            'entropy': self._entropy_uncertainty,
            'variance': self._variance_uncertainty,
            'confidence': self._confidence_uncertainty,
            'margin': self._margin_uncertainty,
            'bayesian': self._bayesian_uncertainty,
            'ensemble': self._ensemble_uncertainty
        }
        
        # 默认参数
        self.default_params = {
            'n_bootstrap': 100,
            'confidence_level': 0.95,
            'spatial_window': 5,
            'uncertainty_threshold': 0.5
        }
        
        # 更新参数
        self.params = {**self.default_params, **self.config.get('uncertainty', {})}
        
        logger.info("不确定性分析器初始化完成")
    
    def analyze_uncertainty(self, 
                          predictions: np.ndarray,
                          probabilities: Optional[np.ndarray] = None,
                          models: Optional[List] = None,
                          spatial_coords: Optional[np.ndarray] = None,
                          method: str = 'entropy') -> Dict[str, np.ndarray]:
        """
        综合不确定性分析
        
        Args:
            predictions: 预测结果 [n_samples,] 或 [n_samples, n_models]
            probabilities: 预测概率 [n_samples, n_classes] 或 [n_samples, n_models, n_classes]
            models: 模型列表（用于集成不确定性）
            spatial_coords: 空间坐标 [n_samples, 2]
            method: 不确定性计算方法
            
        Returns:
            包含各种不确定性指标的字典
        """
        logger.info(f"开始不确定性分析，方法: {method}")
        
        uncertainty_results = {}
        
        try:
            # 1. 预测不确定性
            if probabilities is not None:
                uncertainty_results['predictive'] = self._calculate_predictive_uncertainty(
                    probabilities, method
                )
            
            # 2. 模型不确定性（集成方法）
            if models is not None and len(models) > 1:
                uncertainty_results['model'] = self._calculate_model_uncertainty(
                    predictions, probabilities
                )
            
            # 3. 空间不确定性
            if spatial_coords is not None:
                uncertainty_results['spatial'] = self._calculate_spatial_uncertainty(
                    predictions, spatial_coords
                )
            
            # 4. 总不确定性
            uncertainty_results['total'] = self._calculate_total_uncertainty(
                uncertainty_results
            )
            
            # 5. 置信度映射
            uncertainty_results['confidence'] = self._calculate_confidence_map(
                uncertainty_results['total']
            )
            
            logger.info("不确定性分析完成")
            return uncertainty_results
            
        except Exception as e:
            logger.error(f"不确定性分析失败: {str(e)}")
            raise
    
    def _calculate_predictive_uncertainty(self, 
                                        probabilities: np.ndarray,
                                        method: str) -> np.ndarray:
        """计算预测不确定性"""
        if method not in self.uncertainty_methods:
            raise ValueError(f"不支持的不确定性方法: {method}")
        
        return self.uncertainty_methods[method](probabilities)
    
    def _entropy_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """基于熵的不确定性"""
        # 确保概率值不为0（避免log(0)）
        probabilities = np.clip(probabilities, 1e-8, 1.0)
        
        if probabilities.ndim == 2:
            # 单模型情况 [n_samples, n_classes]
            entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        elif probabilities.ndim == 3:
            # 多模型情况 [n_samples, n_models, n_classes]
            # 计算平均概率
            mean_probs = np.mean(probabilities, axis=1)
            mean_probs = np.clip(mean_probs, 1e-8, 1.0)
            entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)
        else:
            raise ValueError("概率数组维度错误")
        
        # 归一化到[0,1]
        max_entropy = np.log(probabilities.shape[-1])
        return entropy / max_entropy
    
    def _variance_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """基于方差的不确定性"""
        if probabilities.ndim == 2:
            # 单模型情况，使用概率的方差
            variance = np.var(probabilities, axis=1)
        elif probabilities.ndim == 3:
            # 多模型情况，计算模型间的方差
            variance = np.var(probabilities, axis=1)
            variance = np.mean(variance, axis=1)
        else:
            raise ValueError("概率数组维度错误")
        
        return variance
    
    def _confidence_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """基于置信度的不确定性"""
        if probabilities.ndim == 2:
            max_prob = np.max(probabilities, axis=1)
        elif probabilities.ndim == 3:
            mean_probs = np.mean(probabilities, axis=1)
            max_prob = np.max(mean_probs, axis=1)
        else:
            raise ValueError("概率数组维度错误")
        
        # 不确定性 = 1 - 最大概率
        return 1.0 - max_prob
    
    def _margin_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """基于边界的不确定性"""
        if probabilities.ndim == 2:
            sorted_probs = np.sort(probabilities, axis=1)
        elif probabilities.ndim == 3:
            mean_probs = np.mean(probabilities, axis=1)
            sorted_probs = np.sort(mean_probs, axis=1)
        else:
            raise ValueError("概率数组维度错误")
        
        # 边界 = 最大概率 - 第二大概率
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # 不确定性 = 1 - 边界
        return 1.0 - margin
    
    def _bayesian_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """贝叶斯不确定性（需要多次采样）"""
        if probabilities.ndim != 3:
            logger.warning("贝叶斯不确定性需要多次采样的概率，使用熵不确定性代替")
            return self._entropy_uncertainty(probabilities)
        
        # 认知不确定性（模型间的不一致性）
        mean_probs = np.mean(probabilities, axis=1)
        epistemic = self._entropy_uncertainty(mean_probs)
        
        # 随机不确定性（每个模型内部的不确定性）
        aleatoric = np.mean([self._entropy_uncertainty(probabilities[:, i, :]) 
                           for i in range(probabilities.shape[1])], axis=0)
        
        # 总不确定性
        return epistemic + aleatoric
    
    def _ensemble_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """集成模型不确定性"""
        if probabilities.ndim != 3:
            raise ValueError("集成不确定性需要多个模型的概率")
        
        # 计算模型间的分歧
        mean_probs = np.mean(probabilities, axis=1)
        disagreement = np.var(probabilities, axis=1)
        disagreement = np.mean(disagreement, axis=1)
        
        # 结合熵不确定性
        entropy_uncertainty = self._entropy_uncertainty(mean_probs)
        
        return entropy_uncertainty + disagreement
    
    def _calculate_model_uncertainty(self, 
                                   predictions: np.ndarray,
                                   probabilities: Optional[np.ndarray] = None) -> np.ndarray:
        """计算模型不确定性"""
        if predictions.ndim == 1:
            logger.warning("单模型预测，无法计算模型不确定性")
            return np.zeros(len(predictions))
        
        # 计算预测的一致性
        if predictions.ndim == 2:  # [n_samples, n_models]
            # 计算每个样本的预测分歧
            unique_preds = np.array([len(np.unique(pred)) for pred in predictions])
            max_possible = predictions.shape[1]
            disagreement = unique_preds / max_possible
        else:
            raise ValueError("预测数组维度错误")
        
        return disagreement
    
    def _calculate_spatial_uncertainty(self, 
                                     predictions: np.ndarray,
                                     spatial_coords: np.ndarray) -> np.ndarray:
        """计算空间不确定性"""
        logger.info("计算空间不确定性")
        
        n_samples = len(predictions)
        spatial_uncertainty = np.zeros(n_samples)
        window_size = self.params['spatial_window']
        
        for i in range(n_samples):
            # 计算距离
            distances = cdist([spatial_coords[i]], spatial_coords)[0]
            
            # 找到邻近点
            neighbors = np.argsort(distances)[1:window_size+1]  # 排除自己
            
            if len(neighbors) > 0:
                # 计算邻域内的类别一致性
                neighbor_preds = predictions[neighbors]
                if predictions.ndim == 1:
                    center_pred = predictions[i]
                    consistency = np.mean(neighbor_preds == center_pred)
                else:
                    # 多模型情况
                    center_pred = predictions[i]
                    consistency = np.mean([
                        np.mean(neighbor_preds[:, j] == center_pred[j])
                        for j in range(predictions.shape[1])
                    ])
                
                spatial_uncertainty[i] = 1.0 - consistency
        
        return spatial_uncertainty
    
    def _calculate_total_uncertainty(self, uncertainty_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """计算总不确定性"""
        uncertainties = []
        weights = []
        
        # 添加预测不确定性
        if 'predictive' in uncertainty_dict:
            uncertainties.append(uncertainty_dict['predictive'])
            weights.append(0.5)
        
        # 添加模型不确定性
        if 'model' in uncertainty_dict:
            uncertainties.append(uncertainty_dict['model'])
            weights.append(0.3)
        
        # 添加空间不确定性
        if 'spatial' in uncertainty_dict:
            uncertainties.append(uncertainty_dict['spatial'])
            weights.append(0.2)
        
        if not uncertainties:
            raise ValueError("没有可用的不确定性指标")
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 加权平均
        total_uncertainty = np.zeros_like(uncertainties[0])
        for unc, weight in zip(uncertainties, weights):
            total_uncertainty += weight * unc
        
        return total_uncertainty
    
    def _calculate_confidence_map(self, uncertainty: np.ndarray) -> np.ndarray:
        """计算置信度映射"""
        return 1.0 - uncertainty
    
    def bootstrap_uncertainty(self, 
                            model,
                            X: np.ndarray,
                            y: np.ndarray,
                            n_bootstrap: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        自举法不确定性估计
        
        Args:
            model: 分类模型
            X: 特征数据
            y: 标签数据
            n_bootstrap: 自举次数
            
        Returns:
            包含不确定性统计的字典
        """
        n_bootstrap = n_bootstrap or self.params['n_bootstrap']
        logger.info(f"开始自举不确定性估计，次数: {n_bootstrap}")
        
        n_samples = len(X)
        predictions = np.zeros((n_samples, n_bootstrap))
        
        try:
            for i in range(n_bootstrap):
                # 自举采样
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                # 训练模型
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_copy.fit(X_boot, y_boot)
                
                # 预测
                predictions[:, i] = model_copy.predict(X)
            
            # 计算统计量
            results = {
                'mean_prediction': np.mean(predictions, axis=1),
                'std_prediction': np.std(predictions, axis=1),
                'prediction_variance': np.var(predictions, axis=1),
                'prediction_range': np.max(predictions, axis=1) - np.min(predictions, axis=1)
            }
            
            # 计算置信区间
            alpha = 1 - self.params['confidence_level']
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            results['confidence_lower'] = np.percentile(predictions, lower_percentile, axis=1)
            results['confidence_upper'] = np.percentile(predictions, upper_percentile, axis=1)
            results['confidence_width'] = results['confidence_upper'] - results['confidence_lower']
            
            logger.info("自举不确定性估计完成")
            return results
            
        except Exception as e:
            logger.error(f"自举不确定性估计失败: {str(e)}")
            raise
    
    def cross_validation_uncertainty(self,
                                   model,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   cv: int = 5) -> Dict[str, float]:
        """
        交叉验证不确定性估计
        
        Args:
            model: 分类模型
            X: 特征数据
            y: 标签数据
            cv: 交叉验证折数
            
        Returns:
            不确定性统计
        """
        logger.info(f"开始交叉验证不确定性估计，折数: {cv}")
        
        try:
            # 交叉验证预测
            cv_predictions = cross_val_predict(model, X, y, cv=cv)
            
            # 计算不确定性
            accuracy_scores = []
            for train_idx, test_idx in KFold(n_splits=cv, shuffle=True, random_state=42).split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                accuracy_scores.append(accuracy_score(y_test, pred))
            
            results = {
                'mean_accuracy': np.mean(accuracy_scores),
                'std_accuracy': np.std(accuracy_scores),
                'cv_uncertainty': np.std(accuracy_scores) / np.mean(accuracy_scores)  # 变异系数
            }
            
            logger.info("交叉验证不确定性估计完成")
            return results
            
        except Exception as e:
            logger.error(f"交叉验证不确定性估计失败: {str(e)}")
            raise
    
    def uncertainty_threshold_analysis(self,
                                     uncertainty: np.ndarray,
                                     predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     thresholds: Optional[List[float]] = None) -> pd.DataFrame:
        """
        不确定性阈值分析
        
        Args:
            uncertainty: 不确定性值
            predictions: 预测结果
            ground_truth: 真实标签
            thresholds: 阈值列表
            
        Returns:
            阈值分析结果DataFrame
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        logger.info("开始不确定性阈值分析")
        
        results = []
        
        for threshold in thresholds:
            # 选择低不确定性的样本
            reliable_mask = uncertainty <= threshold
            
            if np.sum(reliable_mask) == 0:
                continue
            
            # 计算可靠样本的精度
            reliable_predictions = predictions[reliable_mask]
            reliable_truth = ground_truth[reliable_mask]
            reliable_accuracy = accuracy_score(reliable_truth, reliable_predictions)
            
            results.append({
                'threshold': threshold,
                'reliable_ratio': np.mean(reliable_mask),
                'reliable_accuracy': reliable_accuracy,
                'rejected_ratio': 1 - np.mean(reliable_mask),
                'n_reliable': np.sum(reliable_mask),
                'n_rejected': np.sum(~reliable_mask)
            })
        
        results_df = pd.DataFrame(results)
        logger.info("不确定性阈值分析完成")
        
        return results_df
    
    def visualize_uncertainty(self,
                            uncertainty_results: Dict[str, np.ndarray],
                            spatial_coords: Optional[np.ndarray] = None,
                            predictions: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> None:
        """
        可视化不确定性分析结果
        
        Args:
            uncertainty_results: 不确定性分析结果
            spatial_coords: 空间坐标
            predictions: 预测结果
            save_path: 保存路径
        """
        logger.info("开始生成不确定性可视化")
        
        # 设置图形参数
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 不确定性分布直方图
        n_uncertainties = len(uncertainty_results)
        n_cols = 3
        n_rows = (n_uncertainties + n_cols - 1) // n_cols
        
        plot_idx = 1
        for name, uncertainty in uncertainty_results.items():
            plt.subplot(n_rows + 1, n_cols, plot_idx)
            plt.hist(uncertainty, bins=50, alpha=0.7, density=True)
            plt.title(f'{name.capitalize()} Uncertainty Distribution')
            plt.xlabel('Uncertainty')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 2. 不确定性相关性矩阵
        if len(uncertainty_results) > 1:
            plt.subplot(n_rows + 1, n_cols, plot_idx)
            uncertainty_df = pd.DataFrame(uncertainty_results)
            correlation_matrix = uncertainty_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Uncertainty Correlation Matrix')
            plot_idx += 1
        
        # 3. 空间不确定性分布
        if spatial_coords is not None and 'total' in uncertainty_results:
            plt.subplot(n_rows + 1, n_cols, plot_idx)
            scatter = plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                                c=uncertainty_results['total'], 
                                cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Total Uncertainty')
            plt.title('Spatial Uncertainty Distribution')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plot_idx += 1
        
        # 4. 不确定性与预测类别的关系
        if predictions is not None and 'total' in uncertainty_results:
            plt.subplot(n_rows + 1, n_cols, plot_idx)
            unique_classes = np.unique(predictions)
            uncertainty_by_class = [
                uncertainty_results['total'][predictions == cls] 
                for cls in unique_classes
            ]
            plt.boxplot(uncertainty_by_class, labels=unique_classes)
            plt.title('Uncertainty by Predicted Class')
            plt.xlabel('Predicted Class')
            plt.ylabel('Total Uncertainty')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"不确定性可视化保存至: {save_path}")
        
        plt.show()
    
    def generate_uncertainty_report(self,
                                  uncertainty_results: Dict[str, np.ndarray],
                                  threshold_analysis: Optional[pd.DataFrame] = None,
                                  bootstrap_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        生成不确定性分析报告
        
        Args:
            uncertainty_results: 不确定性分析结果
            threshold_analysis: 阈值分析结果
            bootstrap_results: 自举分析结果
            
        Returns:
            完整的不确定性报告
        """
        logger.info("生成不确定性分析报告")
        
        report = {
            'summary': {},
            'statistics': {},
            'recommendations': []
        }
        
        # 基本统计
        for name, uncertainty in uncertainty_results.items():
            report['statistics'][name] = {
                'mean': float(np.mean(uncertainty)),
                'std': float(np.std(uncertainty)),
                'min': float(np.min(uncertainty)),
                'max': float(np.max(uncertainty)),
                'median': float(np.median(uncertainty)),
                'q25': float(np.percentile(uncertainty, 25)),
                'q75': float(np.percentile(uncertainty, 75))
            }
        
        # 总体评估
        if 'total' in uncertainty_results:
            total_uncertainty = uncertainty_results['total']
            high_uncertainty_ratio = np.mean(total_uncertainty > self.params['uncertainty_threshold'])
            
            report['summary'] = {
                'total_samples': len(total_uncertainty),
                'high_uncertainty_ratio': float(high_uncertainty_ratio),
                'mean_uncertainty': float(np.mean(total_uncertainty)),
                'uncertainty_threshold': self.params['uncertainty_threshold']
            }
            
            # 生成建议
            if high_uncertainty_ratio > 0.3:
                report['recommendations'].append(
                    "高不确定性样本比例较高(>30%)，建议增加训练数据或改进模型"
                )
            if high_uncertainty_ratio > 0.5:
                report['recommendations'].append(
                    "超过50%的样本具有高不确定性，分类结果可靠性较低"
                )
            
            if np.mean(total_uncertainty) < 0.2:
                report['recommendations'].append(
                    "总体不确定性较低，分类结果相对可靠"
                )
        
        # 添加阈值分析
        if threshold_analysis is not None:
            report['threshold_analysis'] = threshold_analysis.to_dict('records')
        
        # 添加自举分析
        if bootstrap_results is not None:
            report['bootstrap_analysis'] = {
                k: float(v.mean()) if isinstance(v, np.ndarray) else float(v)
                for k, v in bootstrap_results.items()
                if k != 'predictions'
            }
        
        logger.info("不确定性分析报告生成完成")
        return report


class SpatialUncertaintyAnalyzer:
    """
    空间不确定性分析器
    
    专门处理空间相关的不确定性分析，包括：
    - 空间自相关性分析
    - 邻域一致性分析
    - 边界不确定性分析
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化空间不确定性分析器"""
        self.config = config or {}
        self.params = {
            'neighborhood_size': 8,
            'distance_threshold': 100,
            'spatial_lag': 1
        }
        self.params.update(self.config.get('spatial_uncertainty', {}))
    
    def spatial_autocorrelation_uncertainty(self,
                                          uncertainty: np.ndarray,
                                          spatial_coords: np.ndarray) -> Dict[str, float]:
        """
        计算不确定性的空间自相关性
        
        Args:
            uncertainty: 不确定性值
            spatial_coords: 空间坐标
            
        Returns:
            空间自相关统计
        """
        from scipy.spatial.distance import pdist, squareform
        
        # 计算距离矩阵
        distances = squareform(pdist(spatial_coords))
        
        # 创建空间权重矩阵（基于距离阈值）
        weights = (distances <= self.params['distance_threshold']).astype(float)
        np.fill_diagonal(weights, 0)  # 对角线为0
        
        # 行标准化
        row_sums = weights.sum(axis=1)
        weights = np.divide(weights, row_sums[:, np.newaxis], 
                          out=np.zeros_like(weights), where=row_sums[:, np.newaxis]!=0)
        
        # 计算Moran's I
        n = len(uncertainty)
        mean_uncertainty = np.mean(uncertainty)
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if weights[i, j] > 0:
                    numerator += weights[i, j] * (uncertainty[i] - mean_uncertainty) * (uncertainty[j] - mean_uncertainty)
        
        denominator = np.sum((uncertainty - mean_uncertainty) ** 2)
        
        if denominator > 0:
            morans_i = (n / np.sum(weights)) * (numerator / denominator)
        else:
            morans_i = 0
        
        return {
            'morans_i': morans_i,
            'spatial_clustering': 'positive' if morans_i > 0 else 'negative' if morans_i < 0 else 'random'
        }
    
    def boundary_uncertainty_analysis(self,
                                    predictions: np.ndarray,
                                    spatial_coords: np.ndarray,
                                    uncertainty: np.ndarray) -> np.ndarray:
        """
        分析类别边界附近的不确定性
        
        Args:
            predictions: 预测结果
            spatial_coords: 空间坐标
            uncertainty: 不确定性值
            
        Returns:
            边界不确定性指标
        """
        from scipy.spatial import cKDTree
        
        # 构建空间索引
        tree = cKDTree(spatial_coords)
        
        boundary_uncertainty = np.zeros(len(predictions))
        
        for i, (coord, pred) in enumerate(zip(spatial_coords, predictions)):
            # 查找邻近点
            distances, indices = tree.query(coord, k=self.params['neighborhood_size']+1)
            neighbor_indices = indices[1:]  # 排除自己
            
            # 检查邻域中的类别多样性
            neighbor_preds = predictions[neighbor_indices]
            unique_classes = len(np.unique(neighbor_preds))
            
            # 如果邻域中存在多个类别，则认为是边界区域
            if unique_classes > 1:
                boundary_uncertainty[i] = uncertainty[i] * (unique_classes / len(neighbor_preds))
            else:
                boundary_uncertainty[i] = uncertainty[i] * 0.5  # 非边界区域的不确定性降权
        
        return boundary_uncertainty


def calculate_ensemble_uncertainty(predictions_list: List[np.ndarray],
                                 probabilities_list: Optional[List[np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """
    计算集成模型的不确定性
    
    Args:
        predictions_list: 多个模型的预测结果列表
        probabilities_list: 多个模型的预测概率列表
        
    Returns:
        集成不确定性指标
    """
    predictions_array = np.array(predictions_list).T  # [n_samples, n_models]
    
    results = {}
    
    # 1. 预测分歧
    disagreement = np.array([
        len(np.unique(pred_row)) / len(pred_row)
        for pred_row in predictions_array
    ])
    results['disagreement'] = disagreement
    
    # 2. 投票熵
    voting_entropy = np.zeros(len(predictions_array))
    for i, pred_row in enumerate(predictions_array):
        unique, counts = np.unique(pred_row, return_counts=True)
        probs = counts / len(pred_row)
        voting_entropy[i] = -np.sum(probs * np.log(probs + 1e-8))
    
    # 归一化
    max_entropy = np.log(len(predictions_list))
    results['voting_entropy'] = voting_entropy / max_entropy
    
    # 3. 概率分歧（如果提供了概率）
    if probabilities_list is not None:
        prob_array = np.array(probabilities_list)  # [n_models, n_samples, n_classes]
        prob_array = prob_array.transpose(1, 0, 2)  # [n_samples, n_models, n_classes]
        
        # 计算模型间的KL散度
        mean_probs = np.mean(prob_array, axis=1)
        kl_divergences = np.zeros(len(prob_array))
        
        for i in range(len(prob_array)):
            kl_div = 0
            for j in range(prob_array.shape[1]):
                kl_div += stats.entropy(prob_array[i, j], mean_probs[i])
            kl_divergences[i] = kl_div / prob_array.shape[1]
        
        results['probability_disagreement'] = kl_divergences
    
    # 4. 总体集成不确定性
    ensemble_uncertainty = results['disagreement']
    if 'voting_entropy' in results:
        ensemble_uncertainty = 0.5 * (ensemble_uncertainty + results['voting_entropy'])
    
    results['total_ensemble_uncertainty'] = ensemble_uncertainty
    
    return results


# 工具函数
def load_uncertainty_config(config_path: str) -> Dict:
    """加载不确定性分析配置"""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('uncertainty', {})


def save_uncertainty_results(results: Dict[str, np.ndarray], 
                           output_path: str) -> None:
    """保存不确定性分析结果"""
    import pickle
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"不确定性结果已保存至: {output_path}")


if __name__ == "__main__":
    # 示例用法
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, random_state=42)
    
    # 训练模型并获取概率
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    
    # 创建不确定性分析器
    analyzer = UncertaintyAnalyzer()
    
    # 分析不确定性
    uncertainty_results = analyzer.analyze_uncertainty(
        predictions=predictions,
        probabilities=probabilities,
        method='entropy'
    )
    
    print("不确定性分析完成！")
    for name, values in uncertainty_results.items():
        print(f"{name}: 均值={np.mean(values):.3f}, 标准差={np.std(values):.3f}")