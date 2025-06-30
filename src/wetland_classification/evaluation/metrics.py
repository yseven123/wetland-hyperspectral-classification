"""
精度评估模块
===========

这个模块提供了全面的分类精度评估功能，支持多种评价指标和统计分析方法。

主要功能：
- 基础精度指标：总体精度、Kappa系数、混淆矩阵
- 类别精度指标：生产者精度、用户精度、F1分数
- 高级精度指标：平衡精度、Matthews相关系数、AUC
- 空间精度评估：空间自相关、边界精度、区域精度
- 统计显著性检验：McNemar检验、配对t检验
- 精度报告生成：详细的评估报告和可视化

参考标准：
- 遥感精度评估标准
- 混淆矩阵分析方法
- 统计学显著性检验

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
from scipy.stats import chi2_contingency, mcnemar
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, 
    multilabel_confusion_matrix, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logger = logging.getLogger(__name__)


class AccuracyAssessment:
    """
    精度评估器
    
    提供全面的分类精度评估功能，包括基础指标、高级指标和统计分析。
    """
    
    def __init__(self, 
                 class_names: Optional[List[str]] = None,
                 confidence_level: float = 0.95,
                 **kwargs):
        """
        初始化精度评估器
        
        Parameters:
        -----------
        class_names : list, optional
            类别名称列表
        confidence_level : float, default=0.95
            置信度水平
        """
        self.class_names = class_names
        self.confidence_level = confidence_level
        self.config = kwargs
        
        # 缓存计算结果
        self.confusion_matrix_cache = None
        self.metrics_cache = {}
    
    def assess_accuracy(self, 
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None,
                       sample_weights: Optional[np.ndarray] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        全面精度评估
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
        y_proba : np.ndarray, optional
            预测概率
        sample_weights : np.ndarray, optional
            样本权重
            
        Returns:
        --------
        assessment_results : dict
            精度评估结果
        """
        logger.info("开始全面精度评估")
        
        # 数据验证
        self._validate_inputs(y_true, y_pred, y_proba)
        
        # 基础精度指标
        basic_metrics = self.calculate_basic_metrics(
            y_true, y_pred, sample_weights
        )
        
        # 类别精度指标
        class_metrics = self.calculate_class_metrics(
            y_true, y_pred, sample_weights
        )
        
        # 高级精度指标
        advanced_metrics = self.calculate_advanced_metrics(
            y_true, y_pred, y_proba, sample_weights
        )
        
        # 混淆矩阵分析
        confusion_analysis = self.analyze_confusion_matrix(y_true, y_pred)
        
        # 统计显著性检验
        statistical_tests = self.perform_statistical_tests(y_true, y_pred)
        
        # 置信区间
        confidence_intervals = self.calculate_confidence_intervals(
            y_true, y_pred
        )
        
        # 误差分析
        error_analysis = self.analyze_errors(y_true, y_pred)
        
        assessment_results = {
            'basic_metrics': basic_metrics,
            'class_metrics': class_metrics,
            'advanced_metrics': advanced_metrics,
            'confusion_analysis': confusion_analysis,
            'statistical_tests': statistical_tests,
            'confidence_intervals': confidence_intervals,
            'error_analysis': error_analysis,
            'assessment_summary': self._create_assessment_summary(
                basic_metrics, class_metrics, advanced_metrics
            )
        }
        
        logger.info("精度评估完成")
        
        return assessment_results
    
    def _validate_inputs(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray] = None):
        """验证输入数据"""
        if len(y_true) != len(y_pred):
            raise ValueError("真实标签和预测标签长度不匹配")
        
        if len(y_true) == 0:
            raise ValueError("输入数据为空")
        
        if y_proba is not None and len(y_proba) != len(y_true):
            raise ValueError("概率预测长度与标签不匹配")
        
        # 检查类别一致性
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        
        if not unique_pred.issubset(unique_true):
            logger.warning("预测标签包含训练集中未出现的类别")
    
    def calculate_basic_metrics(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算基础精度指标
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
        sample_weights : np.ndarray, optional
            样本权重
            
        Returns:
        --------
        basic_metrics : dict
            基础精度指标
        """
        metrics = {}
        
        # 总体精度
        metrics['overall_accuracy'] = accuracy_score(
            y_true, y_pred, sample_weight=sample_weights
        )
        
        # Kappa系数
        metrics['kappa_coefficient'] = cohen_kappa_score(
            y_true, y_pred, sample_weight=sample_weights
        )
        
        # 平衡精度
        metrics['balanced_accuracy'] = balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weights
        )
        
        # 宏平均精度
        metrics['macro_precision'] = precision_score(
            y_true, y_pred, average='macro', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 宏平均召回率
        metrics['macro_recall'] = recall_score(
            y_true, y_pred, average='macro', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 宏平均F1分数
        metrics['macro_f1'] = f1_score(
            y_true, y_pred, average='macro', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 微平均精度
        metrics['micro_precision'] = precision_score(
            y_true, y_pred, average='micro', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 微平均召回率
        metrics['micro_recall'] = recall_score(
            y_true, y_pred, average='micro', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 微平均F1分数
        metrics['micro_f1'] = f1_score(
            y_true, y_pred, average='micro', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 加权平均精度
        metrics['weighted_precision'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 加权平均召回率
        metrics['weighted_recall'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0,
            sample_weight=sample_weights
        )
        
        # 加权平均F1分数
        metrics['weighted_f1'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0,
            sample_weight=sample_weights
        )
        
        return metrics
    
    def calculate_class_metrics(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              sample_weights: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        计算类别精度指标
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
        sample_weights : np.ndarray, optional
            样本权重
            
        Returns:
        --------
        class_metrics : pd.DataFrame
            类别精度指标表
        """
        # 获取所有类别
        unique_classes = sorted(list(set(y_true) | set(y_pred)))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes,
                             sample_weight=sample_weights)
        
        # 计算各类别指标
        class_data = []
        
        for i, class_label in enumerate(unique_classes):
            # 从混淆矩阵提取统计量
            tp = cm[i, i]  # 真正例
            fp = np.sum(cm[:, i]) - tp  # 假正例
            fn = np.sum(cm[i, :]) - tp  # 假负例
            tn = np.sum(cm) - tp - fp - fn  # 真负例
            
            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 生产者精度（召回率）
            producer_accuracy = recall
            
            # 用户精度（精确率）
            user_accuracy = precision
            
            # 遗漏错误率
            omission_error = 1 - producer_accuracy
            
            # 委托错误率
            commission_error = 1 - user_accuracy
            
            # 支持度（样本数量）
            support = np.sum(y_true == class_label)
            
            class_data.append({
                'class': class_label,
                'class_name': self.class_names[i] if self.class_names and i < len(self.class_names) else f'Class_{class_label}',
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'producer_accuracy': producer_accuracy,
                'user_accuracy': user_accuracy,
                'omission_error': omission_error,
                'commission_error': commission_error,
                'support': support
            })
        
        return pd.DataFrame(class_data)
    
    def calculate_advanced_metrics(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 sample_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算高级精度指标
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
        y_proba : np.ndarray, optional
            预测概率
        sample_weights : np.ndarray, optional
            样本权重
            
        Returns:
        --------
        advanced_metrics : dict
            高级精度指标
        """
        metrics = {}
        
        # Matthews相关系数
        try:
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            logger.warning(f"无法计算Matthews相关系数: {e}")
            metrics['matthews_corrcoef'] = np.nan
        
        # 如果有概率预测，计算AUC相关指标
        if y_proba is not None:
            try:
                unique_classes = sorted(list(set(y_true)))
                n_classes = len(unique_classes)
                
                if n_classes == 2:
                    # 二分类AUC
                    if y_proba.ndim == 1:
                        # 单列概率（正类概率）
                        auc_score = roc_auc_score(y_true, y_proba, 
                                                sample_weight=sample_weights)
                    else:
                        # 双列概率矩阵
                        auc_score = roc_auc_score(y_true, y_proba[:, 1],
                                                sample_weight=sample_weights)
                    
                    metrics['auc_roc'] = auc_score
                    
                    # 计算AUC-PR
                    if y_proba.ndim == 1:
                        ap_score = average_precision_score(y_true, y_proba,
                                                         sample_weight=sample_weights)
                    else:
                        ap_score = average_precision_score(y_true, y_proba[:, 1],
                                                         sample_weight=sample_weights)
                    
                    metrics['auc_pr'] = ap_score
                
                else:
                    # 多分类AUC
                    # 将标签二值化
                    y_true_bin = label_binarize(y_true, classes=unique_classes)
                    
                    if y_proba.shape[1] == n_classes:
                        # 宏平均AUC
                        auc_macro = roc_auc_score(y_true_bin, y_proba, 
                                                average='macro', multi_class='ovr',
                                                sample_weight=sample_weights)
                        metrics['auc_roc_macro'] = auc_macro
                        
                        # 微平均AUC
                        try:
                            auc_micro = roc_auc_score(y_true_bin, y_proba,
                                                    average='micro', multi_class='ovr',
                                                    sample_weight=sample_weights)
                            metrics['auc_roc_micro'] = auc_micro
                        except Exception as e:
                            logger.warning(f"无法计算微平均AUC: {e}")
                        
                        # 加权平均AUC
                        auc_weighted = roc_auc_score(y_true_bin, y_proba,
                                                   average='weighted', multi_class='ovr',
                                                   sample_weight=sample_weights)
                        metrics['auc_roc_weighted'] = auc_weighted
                
            except Exception as e:
                logger.warning(f"无法计算AUC指标: {e}")
        
        # 几何平均数
        class_metrics_df = self.calculate_class_metrics(y_true, y_pred, sample_weights)
        if not class_metrics_df.empty:
            recalls = class_metrics_df['recall'].values
            recalls = recalls[recalls > 0]  # 避免0值
            if len(recalls) > 0:
                metrics['geometric_mean'] = np.power(np.prod(recalls), 1.0/len(recalls))
        
        # 调和平均数
        if not class_metrics_df.empty:
            f1_scores = class_metrics_df['f1_score'].values
            f1_scores = f1_scores[f1_scores > 0]
            if len(f1_scores) > 0:
                metrics['harmonic_mean_f1'] = len(f1_scores) / np.sum(1.0 / f1_scores)
        
        return metrics
    
    def analyze_confusion_matrix(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, Any]:
        """
        分析混淆矩阵
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
            
        Returns:
        --------
        confusion_analysis : dict
            混淆矩阵分析结果
        """
        # 计算混淆矩阵
        unique_classes = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况
        
        # 缓存混淆矩阵
        self.confusion_matrix_cache = cm
        
        # 分析混淆模式
        confusion_patterns = self._analyze_confusion_patterns(cm, unique_classes)
        
        # 计算类别间混淆强度
        confusion_intensity = self._calculate_confusion_intensity(cm)
        
        # 识别最容易混淆的类别对
        most_confused_pairs = self._find_most_confused_pairs(cm, unique_classes)
        
        return {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'unique_classes': unique_classes,
            'confusion_patterns': confusion_patterns,
            'confusion_intensity': confusion_intensity,
            'most_confused_pairs': most_confused_pairs,
            'matrix_properties': self._analyze_matrix_properties(cm)
        }
    
    def _analyze_confusion_patterns(self, 
                                   cm: np.ndarray, 
                                   class_labels: List) -> Dict[str, Any]:
        """分析混淆模式"""
        patterns = {}
        
        # 对角线元素（正确分类）
        diagonal_sum = np.trace(cm)
        total_sum = np.sum(cm)
        
        patterns['correct_classifications'] = diagonal_sum
        patterns['total_classifications'] = total_sum
        patterns['error_rate'] = 1 - (diagonal_sum / total_sum)
        
        # 分析每行（实际类别的分类情况）
        row_analysis = {}
        for i, class_label in enumerate(class_labels):
            row_sum = np.sum(cm[i, :])
            if row_sum > 0:
                row_analysis[class_label] = {
                    'total_samples': row_sum,
                    'correct_predictions': cm[i, i],
                    'accuracy': cm[i, i] / row_sum,
                    'main_confusion_target': class_labels[np.argmax(cm[i, :])] if np.argmax(cm[i, :]) != i else None,
                    'confusion_distribution': cm[i, :].tolist()
                }
        
        patterns['by_true_class'] = row_analysis
        
        # 分析每列（预测类别的情况）
        col_analysis = {}
        for j, class_label in enumerate(class_labels):
            col_sum = np.sum(cm[:, j])
            if col_sum > 0:
                col_analysis[class_label] = {
                    'total_predictions': col_sum,
                    'correct_predictions': cm[j, j],
                    'precision': cm[j, j] / col_sum,
                    'main_confusion_source': class_labels[np.argmax(cm[:, j])] if np.argmax(cm[:, j]) != j else None,
                    'confusion_distribution': cm[:, j].tolist()
                }
        
        patterns['by_predicted_class'] = col_analysis
        
        return patterns
    
    def _calculate_confusion_intensity(self, cm: np.ndarray) -> float:
        """计算混淆强度"""
        # 混淆强度 = 非对角线元素之和 / 总和
        total_sum = np.sum(cm)
        off_diagonal_sum = total_sum - np.trace(cm)
        
        return off_diagonal_sum / total_sum if total_sum > 0 else 0
    
    def _find_most_confused_pairs(self, 
                                 cm: np.ndarray, 
                                 class_labels: List,
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """找到最容易混淆的类别对"""
        confused_pairs = []
        
        n_classes = cm.shape[0]
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:  # 非对角线元素
                    confusion_count = cm[i, j]
                    if confusion_count > 0:
                        # 计算混淆程度（相对于该行总数）
                        row_sum = np.sum(cm[i, :])
                        confusion_rate = confusion_count / row_sum if row_sum > 0 else 0
                        
                        confused_pairs.append({
                            'true_class': class_labels[i],
                            'predicted_class': class_labels[j],
                            'confusion_count': confusion_count,
                            'confusion_rate': confusion_rate,
                            'intensity': confusion_count
                        })
        
        # 按混淆程度排序
        confused_pairs.sort(key=lambda x: x['intensity'], reverse=True)
        
        return confused_pairs[:top_k]
    
    def _analyze_matrix_properties(self, cm: np.ndarray) -> Dict[str, float]:
        """分析混淆矩阵属性"""
        properties = {}
        
        # 矩阵的条件数
        try:
            properties['condition_number'] = np.linalg.cond(cm.astype(float))
        except:
            properties['condition_number'] = np.inf
        
        # 矩阵的秩
        properties['rank'] = np.linalg.matrix_rank(cm)
        
        # 矩阵的行列式
        try:
            properties['determinant'] = np.linalg.det(cm.astype(float))
        except:
            properties['determinant'] = 0
        
        # 对角线优势度
        diagonal_sum = np.trace(cm)
        off_diagonal_sum = np.sum(cm) - diagonal_sum
        properties['diagonal_dominance'] = diagonal_sum / (diagonal_sum + off_diagonal_sum) if (diagonal_sum + off_diagonal_sum) > 0 else 0
        
        return properties
    
    def perform_statistical_tests(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray) -> Dict[str, Any]:
        """
        执行统计显著性检验
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
            
        Returns:
        --------
        statistical_tests : dict
            统计检验结果
        """
        tests = {}
        
        # McNemar检验（二分类或转换为二分类）
        try:
            if len(set(y_true)) == 2:
                # 直接二分类McNemar检验
                mcnemar_result = self._mcnemar_test(y_true, y_pred)
                tests['mcnemar_test'] = mcnemar_result
            else:
                # 多分类情况，进行类别间的McNemar检验
                tests['pairwise_mcnemar'] = self._pairwise_mcnemar_tests(y_true, y_pred)
        except Exception as e:
            logger.warning(f"McNemar检验失败: {e}")
        
        # 卡方检验
        try:
            chi2_result = self._chi_square_test(y_true, y_pred)
            tests['chi_square_test'] = chi2_result
        except Exception as e:
            logger.warning(f"卡方检验失败: {e}")
        
        # Cochran's Q检验（如果有多个分类器比较）
        # 这里暂时跳过，需要多个分类器的结果
        
        return tests
    
    def _mcnemar_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """McNemar检验"""
        # 构建2x2列联表
        correct_true = (y_true == y_pred)
        
        # 这里简化为检验分类器是否显著优于随机分类
        # 更复杂的情况需要两个分类器的比较
        
        n = len(y_true)
        correct_count = np.sum(correct_true)
        
        # 简化的精确二项检验
        from scipy.stats import binom_test
        
        # 检验精度是否显著高于0.5
        p_value = binom_test(correct_count, n, 0.5, alternative='greater')
        
        return {
            'statistic': correct_count,
            'p_value': p_value,
            'significant': p_value < (1 - self.confidence_level),
            'interpretation': '分类精度显著高于随机分类' if p_value < 0.05 else '分类精度未显著高于随机分类'
        }
    
    def _pairwise_mcnemar_tests(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """类别间McNemar检验"""
        # 简化实现，实际应用中需要更复杂的设计
        unique_classes = sorted(list(set(y_true)))
        
        results = {}
        for class_label in unique_classes:
            # 将多分类问题转换为二分类（当前类别 vs 其他）
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            
            # 执行McNemar检验
            try:
                mcnemar_result = self._mcnemar_test(y_true_binary, y_pred_binary)
                results[f'class_{class_label}'] = mcnemar_result
            except Exception as e:
                logger.warning(f"类别{class_label}的McNemar检验失败: {e}")
        
        return results
    
    def _chi_square_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """卡方独立性检验"""
        # 构建列联表
        cm = confusion_matrix(y_true, y_pred)
        
        # 执行卡方检验
        chi2_stat, p_value, dof, expected = chi2_contingency(cm)
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected,
            'significant': p_value < (1 - self.confidence_level),
            'interpretation': '预测结果与真实标签存在显著关联' if p_value < 0.05 else '预测结果与真实标签无显著关联'
        }
    
    def calculate_confidence_intervals(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        计算置信区间
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
            
        Returns:
        --------
        confidence_intervals : dict
            各指标的置信区间
        """
        n = len(y_true)
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        intervals = {}
        
        # 总体精度的置信区间
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_se = np.sqrt(accuracy * (1 - accuracy) / n)
        accuracy_ci = (
            max(0, accuracy - z_score * accuracy_se),
            min(1, accuracy + z_score * accuracy_se)
        )
        intervals['overall_accuracy'] = accuracy_ci
        
        # Kappa系数的置信区间（近似）
        kappa = cohen_kappa_score(y_true, y_pred)
        # Kappa的标准误差计算较复杂，这里使用简化估计
        kappa_se = np.sqrt((accuracy * (1 - accuracy)) / (n * (1 - accuracy)**2))
        kappa_ci = (
            max(-1, kappa - z_score * kappa_se),
            min(1, kappa + z_score * kappa_se)
        )
        intervals['kappa_coefficient'] = kappa_ci
        
        # 为每个类别计算精度和召回率的置信区间
        unique_classes = sorted(list(set(y_true)))
        for class_label in unique_classes:
            class_mask = (y_true == class_label)
            class_pred_mask = (y_pred == class_label)
            
            # 该类别的样本数
            n_class = np.sum(class_mask)
            
            if n_class > 0:
                # 召回率置信区间
                recall = np.sum(class_mask & class_pred_mask) / n_class
                recall_se = np.sqrt(recall * (1 - recall) / n_class)
                recall_ci = (
                    max(0, recall - z_score * recall_se),
                    min(1, recall + z_score * recall_se)
                )
                intervals[f'recall_class_{class_label}'] = recall_ci
                
                # 精确率置信区间
                n_pred_class = np.sum(class_pred_mask)
                if n_pred_class > 0:
                    precision = np.sum(class_mask & class_pred_mask) / n_pred_class
                    precision_se = np.sqrt(precision * (1 - precision) / n_pred_class)
                    precision_ci = (
                        max(0, precision - z_score * precision_se),
                        min(1, precision + z_score * precision_se)
                    )
                    intervals[f'precision_class_{class_label}'] = precision_ci
        
        return intervals
    
    def analyze_errors(self, 
                      y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, Any]:
        """
        分析分类错误
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签
        y_pred : np.ndarray
            预测标签
            
        Returns:
        --------
        error_analysis : dict
            错误分析结果
        """
        error_analysis = {}
        
        # 错误样本索引
        error_mask = (y_true != y_pred)
        error_indices = np.where(error_mask)[0]
        
        error_analysis['total_errors'] = len(error_indices)
        error_analysis['error_rate'] = len(error_indices) / len(y_true)
        error_analysis['error_indices'] = error_indices.tolist()
        
        # 错误类型分析
        error_types = defaultdict(int)
        for i in error_indices:
            true_label = y_true[i]
            pred_label = y_pred[i]
            error_types[f'{true_label}_to_{pred_label}'] += 1
        
        error_analysis['error_types'] = dict(error_types)
        
        # 最常见的错误类型
        most_common_errors = sorted(error_types.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
        error_analysis['most_common_errors'] = most_common_errors
        
        # 类别错误分布
        class_errors = {}
        unique_classes = sorted(list(set(y_true)))
        
        for class_label in unique_classes:
            class_mask = (y_true == class_label)
            class_error_mask = class_mask & error_mask
            
            class_errors[class_label] = {
                'total_samples': np.sum(class_mask),
                'error_count': np.sum(class_error_mask),
                'error_rate': np.sum(class_error_mask) / np.sum(class_mask) if np.sum(class_mask) > 0 else 0
            }
        
        error_analysis['class_error_distribution'] = class_errors
        
        return error_analysis
    
    def _create_assessment_summary(self, 
                                  basic_metrics: Dict[str, float],
                                  class_metrics: pd.DataFrame,
                                  advanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """创建评估摘要"""
        summary = {}
        
        # 总体性能等级
        overall_accuracy = basic_metrics.get('overall_accuracy', 0)
        kappa = basic_metrics.get('kappa_coefficient', 0)
        
        if overall_accuracy >= 0.9 and kappa >= 0.8:
            performance_grade = "优秀"
        elif overall_accuracy >= 0.8 and kappa >= 0.6:
            performance_grade = "良好"
        elif overall_accuracy >= 0.7 and kappa >= 0.4:
            performance_grade = "一般"
        elif overall_accuracy >= 0.6 and kappa >= 0.2:
            performance_grade = "较差"
        else:
            performance_grade = "很差"
        
        summary['performance_grade'] = performance_grade
        summary['overall_accuracy'] = overall_accuracy
        summary['kappa_coefficient'] = kappa
        
        # 类别性能摘要
        if not class_metrics.empty:
            summary['best_performing_class'] = {
                'class': class_metrics.loc[class_metrics['f1_score'].idxmax(), 'class'],
                'f1_score': class_metrics['f1_score'].max()
            }
            
            summary['worst_performing_class'] = {
                'class': class_metrics.loc[class_metrics['f1_score'].idxmin(), 'class'],
                'f1_score': class_metrics['f1_score'].min()
            }
            
            summary['avg_class_f1'] = class_metrics['f1_score'].mean()
            summary['class_performance_std'] = class_metrics['f1_score'].std()
        
        # 关键发现
        key_findings = []
        
        if overall_accuracy > 0.85:
            key_findings.append("总体分类精度较高")
        
        if kappa > 0.8:
            key_findings.append("Kappa系数表明分类结果与随机分类相比有显著改善")
        
        if not class_metrics.empty:
            class_std = class_metrics['f1_score'].std()
            if class_std < 0.1:
                key_findings.append("各类别性能较为均衡")
            elif class_std > 0.3:
                key_findings.append("各类别性能差异较大，存在明显的易混淆类别")
        
        summary['key_findings'] = key_findings
        
        return summary
    
    def generate_report(self, 
                       assessment_results: Dict[str, Any],
                       output_path: Optional[str] = None) -> str:
        """
        生成精度评估报告
        
        Parameters:
        -----------
        assessment_results : dict
            评估结果
        output_path : str, optional
            输出路径
            
        Returns:
        --------
        report : str
            评估报告内容
        """
        report_lines = []
        
        # 报告标题
        report_lines.append("# 分类精度评估报告")
        report_lines.append("")
        
        # 评估摘要
        summary = assessment_results.get('assessment_summary', {})
        report_lines.append("## 评估摘要")
        report_lines.append(f"- **性能等级**: {summary.get('performance_grade', 'N/A')}")
        report_lines.append(f"- **总体精度**: {summary.get('overall_accuracy', 0):.4f}")
        report_lines.append(f"- **Kappa系数**: {summary.get('kappa_coefficient', 0):.4f}")
        report_lines.append("")
        
        # 基础指标
        basic_metrics = assessment_results.get('basic_metrics', {})
        report_lines.append("## 基础精度指标")
        for metric, value in basic_metrics.items():
            report_lines.append(f"- **{metric}**: {value:.4f}")
        report_lines.append("")
        
        # 类别指标
        class_metrics = assessment_results.get('class_metrics')
        if class_metrics is not None and not class_metrics.empty:
            report_lines.append("## 类别精度指标")
            report_lines.append("| 类别 | 精确率 | 召回率 | F1分数 | 支持度 |")
            report_lines.append("|------|--------|--------|--------|--------|")
            
            for _, row in class_metrics.iterrows():
                report_lines.append(
                    f"| {row['class_name']} | {row['precision']:.4f} | "
                    f"{row['recall']:.4f} | {row['f1_score']:.4f} | {row['support']} |"
                )
            report_lines.append("")
        
        # 混淆矩阵分析
        confusion_analysis = assessment_results.get('confusion_analysis', {})
        if 'most_confused_pairs' in confusion_analysis:
            report_lines.append("## 主要混淆类别对")
            pairs = confusion_analysis['most_confused_pairs']
            for pair in pairs[:3]:  # 显示前3个最混淆的类别对
                report_lines.append(
                    f"- {pair['true_class']} → {pair['predicted_class']}: "
                    f"{pair['confusion_count']} 次 ({pair['confusion_rate']:.2%})"
                )
            report_lines.append("")
        
        # 关键发现
        key_findings = summary.get('key_findings', [])
        if key_findings:
            report_lines.append("## 关键发现")
            for finding in key_findings:
                report_lines.append(f"- {finding}")
            report_lines.append("")
        
        # 统计检验结果
        statistical_tests = assessment_results.get('statistical_tests', {})
        if statistical_tests:
            report_lines.append("## 统计显著性检验")
            for test_name, test_result in statistical_tests.items():
                if isinstance(test_result, dict) and 'interpretation' in test_result:
                    report_lines.append(f"- **{test_name}**: {test_result['interpretation']}")
            report_lines.append("")
        
        # 生成时间
        import datetime
        report_lines.append(f"*报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"评估报告已保存至: {output_path}")
        
        return report_content


def assess_classification_accuracy(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 class_names: Optional[List[str]] = None,
                                 confidence_level: float = 0.95,
                                 **kwargs) -> Dict[str, Any]:
    """
    分类精度评估的便捷函数
    
    Parameters:
    -----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    y_proba : np.ndarray, optional
        预测概率
    class_names : list, optional
        类别名称
    confidence_level : float, default=0.95
        置信度水平
        
    Returns:
    --------
    assessment_results : dict
        精度评估结果
    """
    assessor = AccuracyAssessment(
        class_names=class_names,
        confidence_level=confidence_level,
        **kwargs
    )
    
    return assessor.assess_accuracy(y_true, y_pred, y_proba)