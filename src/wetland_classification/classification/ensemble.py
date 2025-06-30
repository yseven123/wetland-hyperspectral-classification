"""
集成学习分类器
==============

这个模块实现了多种集成学习算法，通过组合多个基分类器来提高分类性能。

支持的集成方法：
- 投票集成 (Voting Ensemble)
- 堆叠集成 (Stacking Ensemble)
- 加权融合 (Weighted Fusion)
- 动态选择集成 (Dynamic Selection)
- Bagging集成
- Boosting集成

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import pickle
from pathlib import Path
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from .base import BaseClassifier, ClassificationMetrics, handle_classification_errors
from .traditional import SVMClassifier, RandomForestClassifier, XGBoostClassifier
from .deep_learning import DeepLearningClassifier

# 设置日志
logger = logging.getLogger(__name__)


class VotingEnsemble(BaseClassifier):
    """
    投票集成分类器
    
    通过多个基分类器的投票来做出最终预测决策。
    支持硬投票和软投票两种方式。
    """
    
    def __init__(self,
                 base_classifiers: List[BaseClassifier],
                 voting: str = 'soft',
                 weights: Optional[List[float]] = None,
                 class_balance: bool = True,
                 cv_folds: int = 5,
                 optimize_weights: bool = False,
                 **kwargs):
        """
        初始化投票集成分类器
        
        Parameters:
        -----------
        base_classifiers : list
            基分类器列表
        voting : str, default='soft'
            投票方式 ('hard' 或 'soft')
        weights : list, optional
            分类器权重
        class_balance : bool, default=True
            是否考虑类别平衡
        cv_folds : int, default=5
            交叉验证折数
        optimize_weights : bool, default=False
            是否优化权重
        """
        super().__init__(**kwargs)
        
        self.base_classifiers = base_classifiers
        self.voting = voting
        self.weights = weights
        self.class_balance = class_balance
        self.cv_folds = cv_folds
        self.optimize_weights = optimize_weights
        
        self.n_classifiers = len(base_classifiers)
        self.classifier_scores = {}
        self.optimized_weights = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'VotingEnsemble':
        """
        训练投票集成分类器
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据
        y : np.ndarray
            训练标签数据
            
        Returns:
        --------
        self : VotingEnsemble
            返回自身实例
        """
        logger.info(f"开始训练投票集成分类器 - 基分类器数量: {self.n_classifiers}")
        start_time = time.time()
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        self.training_history['n_classifiers'] = self.n_classifiers
        
        # 训练基分类器
        logger.info("训练基分类器...")
        trained_classifiers = []
        
        for i, classifier in enumerate(self.base_classifiers):
            logger.info(f"训练第 {i+1}/{self.n_classifiers} 个分类器: {classifier.__class__.__name__}")
            
            # 训练分类器
            classifier.fit(X, y)
            trained_classifiers.append(classifier)
            
            # 评估分类器性能
            cv_scores = cross_val_score(
                classifier.model if hasattr(classifier, 'model') else classifier,
                X, y, cv=self.cv_folds, scoring='accuracy'
            )
            
            self.classifier_scores[i] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'classifier_name': classifier.__class__.__name__
            }
            
            logger.info(f"分类器 {i+1} CV精度: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.base_classifiers = trained_classifiers
        
        # 优化权重
        if self.optimize_weights:
            logger.info("优化分类器权重...")
            self.optimized_weights = self._optimize_weights(X, y)
            self.weights = self.optimized_weights
            logger.info(f"优化后权重: {self.weights}")
        elif self.weights is None:
            # 根据CV性能设置权重
            scores = [self.classifier_scores[i]['mean_cv_score'] for i in range(self.n_classifiers)]
            self.weights = np.array(scores) / np.sum(scores)
            logger.info(f"基于CV性能的权重: {self.weights}")
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['classifier_scores'] = self.classifier_scores
        self.training_history['weights'] = self.weights
        
        self.is_trained = True
        logger.info(f"投票集成训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """优化分类器权重"""
        from scipy.optimize import minimize
        
        def objective(weights):
            """优化目标函数"""
            # 归一化权重
            weights = weights / np.sum(weights)
            
            # 计算加权预测
            predictions = self._weighted_predict(X, weights)
            
            # 计算负准确率（用于最小化）
            accuracy = accuracy_score(y, predictions)
            return -accuracy
        
        # 初始权重
        initial_weights = np.ones(self.n_classifiers) / self.n_classifiers
        
        # 约束条件：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # 边界：权重非负
        bounds = [(0, 1) for _ in range(self.n_classifiers)]
        
        # 优化
        result = minimize(
            objective, initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def _weighted_predict(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """加权预测"""
        if self.voting == 'soft':
            # 软投票
            weighted_proba = np.zeros((X.shape[0], len(self.class_names)))
            
            for i, classifier in enumerate(self.base_classifiers):
                proba = classifier.predict_proba(X)
                weighted_proba += weights[i] * proba
            
            return np.argmax(weighted_proba, axis=1)
        else:
            # 硬投票
            predictions = np.zeros((X.shape[0], self.n_classifiers))
            
            for i, classifier in enumerate(self.base_classifiers):
                pred = classifier.predict(X)
                # 将类别标签转换为数值
                pred_numeric = np.array([list(self.class_names).index(str(p)) for p in pred])
                predictions[:, i] = pred_numeric
            
            # 加权投票
            weighted_votes = np.zeros((X.shape[0], len(self.class_names)))
            for i in range(X.shape[0]):
                for j in range(self.n_classifiers):
                    class_idx = int(predictions[i, j])
                    weighted_votes[i, class_idx] += weights[j]
            
            return np.argmax(weighted_votes, axis=1)
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        return self._weighted_predict(X, self.weights)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if self.voting == 'hard':
            raise ValueError("硬投票模式不支持概率预测")
        
        # 软投票概率预测
        weighted_proba = np.zeros((X.shape[0], len(self.class_names)))
        
        for i, classifier in enumerate(self.base_classifiers):
            proba = classifier.predict_proba(X)
            weighted_proba += self.weights[i] * proba
        
        return weighted_proba


class StackingEnsemble(BaseClassifier):
    """
    堆叠集成分类器
    
    使用元学习器来学习如何最好地组合基分类器的预测。
    """
    
    def __init__(self,
                 base_classifiers: List[BaseClassifier],
                 meta_classifier: Optional[BaseClassifier] = None,
                 cv_folds: int = 5,
                 use_probabilities: bool = True,
                 passthrough: bool = False,
                 **kwargs):
        """
        初始化堆叠集成分类器
        
        Parameters:
        -----------
        base_classifiers : list
            基分类器列表
        meta_classifier : BaseClassifier, optional
            元分类器，默认使用逻辑回归
        cv_folds : int, default=5
            交叉验证折数
        use_probabilities : bool, default=True
            是否使用概率作为元特征
        passthrough : bool, default=False
            是否将原始特征传递给元分类器
        """
        super().__init__(**kwargs)
        
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.passthrough = passthrough
        
        self.n_classifiers = len(base_classifiers)
        self.meta_features = None
        self.base_predictions = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'StackingEnsemble':
        """
        训练堆叠集成分类器
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据
        y : np.ndarray
            训练标签数据
            
        Returns:
        --------
        self : StackingEnsemble
            返回自身实例
        """
        logger.info(f"开始训练堆叠集成分类器 - 基分类器数量: {self.n_classifiers}")
        start_time = time.time()
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        
        # 生成元特征
        logger.info("生成元特征...")
        meta_features = self._generate_meta_features(X, y)
        
        # 训练基分类器（在全部数据上）
        logger.info("训练基分类器...")
        for i, classifier in enumerate(self.base_classifiers):
            logger.info(f"训练基分类器 {i+1}/{self.n_classifiers}: {classifier.__class__.__name__}")
            classifier.fit(X, y)
        
        # 设置默认元分类器
        if self.meta_classifier is None:
            from sklearn.linear_model import LogisticRegression
            self.meta_classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        
        # 训练元分类器
        logger.info("训练元分类器...")
        if hasattr(self.meta_classifier, 'fit'):
            # sklearn分类器
            self.meta_classifier.fit(meta_features, y)
        else:
            # 自定义分类器
            self.meta_classifier.fit(meta_features, y)
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['meta_feature_shape'] = meta_features.shape
        
        self.is_trained = True
        logger.info(f"堆叠集成训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """生成元特征"""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        if self.use_probabilities:
            meta_features = np.zeros((X.shape[0], self.n_classifiers * len(np.unique(y))))
        else:
            meta_features = np.zeros((X.shape[0], self.n_classifiers))
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for i, classifier in enumerate(self.base_classifiers):
                # 克隆分类器以避免在原始对象上训练
                clf_copy = clone(classifier) if hasattr(classifier, 'get_params') else classifier.__class__(**classifier.config)
                
                # 训练分类器
                clf_copy.fit(X_train, y_train)
                
                # 生成元特征
                if self.use_probabilities:
                    proba = clf_copy.predict_proba(X_val)
                    start_col = i * len(np.unique(y))
                    end_col = start_col + len(np.unique(y))
                    meta_features[val_idx, start_col:end_col] = proba
                else:
                    pred = clf_copy.predict(X_val)
                    # 转换为数值编码
                    pred_numeric = np.array([list(np.unique(y)).index(p) for p in pred])
                    meta_features[val_idx, i] = pred_numeric
        
        # 如果启用passthrough，添加原始特征
        if self.passthrough:
            meta_features = np.concatenate([meta_features, X], axis=1)
        
        return meta_features
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 生成元特征
        meta_features = self._predict_meta_features(X)
        
        # 元分类器预测
        if hasattr(self.meta_classifier, 'predict'):
            return self.meta_classifier.predict(meta_features)
        else:
            return self.meta_classifier.predict(meta_features)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 生成元特征
        meta_features = self._predict_meta_features(X)
        
        # 元分类器概率预测
        if hasattr(self.meta_classifier, 'predict_proba'):
            return self.meta_classifier.predict_proba(meta_features)
        else:
            return self.meta_classifier.predict_proba(meta_features)
    
    def _predict_meta_features(self, X: np.ndarray) -> np.ndarray:
        """为预测生成元特征"""
        if self.use_probabilities:
            meta_features = np.zeros((X.shape[0], self.n_classifiers * len(self.class_names)))
            
            for i, classifier in enumerate(self.base_classifiers):
                proba = classifier.predict_proba(X)
                start_col = i * len(self.class_names)
                end_col = start_col + len(self.class_names)
                meta_features[:, start_col:end_col] = proba
        else:
            meta_features = np.zeros((X.shape[0], self.n_classifiers))
            
            for i, classifier in enumerate(self.base_classifiers):
                pred = classifier.predict(X)
                # 转换为数值编码
                pred_numeric = np.array([list(self.class_names).index(str(p)) for p in pred])
                meta_features[:, i] = pred_numeric
        
        # 如果启用passthrough，添加原始特征
        if self.passthrough:
            meta_features = np.concatenate([meta_features, X], axis=1)
        
        return meta_features


class DynamicEnsemble(BaseClassifier):
    """
    动态选择集成分类器
    
    根据每个测试样本的局部特征动态选择最佳的基分类器。
    """
    
    def __init__(self,
                 base_classifiers: List[BaseClassifier],
                 selection_strategy: str = 'competence',
                 k_neighbors: int = 5,
                 competence_measure: str = 'accuracy',
                 **kwargs):
        """
        初始化动态选择集成分类器
        
        Parameters:
        -----------
        base_classifiers : list
            基分类器列表
        selection_strategy : str, default='competence'
            选择策略 ('competence', 'diversity', 'hybrid')
        k_neighbors : int, default=5
            近邻数量
        competence_measure : str, default='accuracy'
            胜任度度量 ('accuracy', 'f1', 'precision', 'recall')
        """
        super().__init__(**kwargs)
        
        self.base_classifiers = base_classifiers
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.competence_measure = competence_measure
        
        self.n_classifiers = len(base_classifiers)
        self.X_train = None
        self.y_train = None
        self.validation_predictions = None
        self.competence_matrix = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'DynamicEnsemble':
        """
        训练动态选择集成分类器
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据
        y : np.ndarray
            训练标签数据
            
        Returns:
        --------
        self : DynamicEnsemble
            返回自身实例
        """
        logger.info(f"开始训练动态选择集成分类器")
        start_time = time.time()
        
        # 保存训练数据用于动态选择
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        
        # 训练基分类器并计算胜任度
        logger.info("训练基分类器并计算胜任度...")
        self._train_and_evaluate(X, y)
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        
        self.is_trained = True
        logger.info(f"动态选择集成训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _train_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        """训练分类器并评估胜任度"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import NearestNeighbors
        
        # 交叉验证评估
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # 存储验证预测
        validation_predictions = np.zeros((X.shape[0], self.n_classifiers))
        validation_probabilities = np.zeros((X.shape[0], self.n_classifiers, len(self.class_names)))
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            for i, classifier in enumerate(self.base_classifiers):
                # 克隆并训练分类器
                clf_copy = clone(classifier) if hasattr(classifier, 'get_params') else classifier.__class__(**classifier.config)
                clf_copy.fit(X_train_fold, y_train_fold)
                
                # 验证集预测
                pred = clf_copy.predict(X_val_fold)
                proba = clf_copy.predict_proba(X_val_fold)
                
                # 转换为数值编码
                pred_numeric = np.array([list(self.class_names).index(str(p)) for p in pred])
                validation_predictions[val_idx, i] = pred_numeric
                validation_probabilities[val_idx, i] = proba
        
        # 训练所有基分类器（在全部数据上）
        for classifier in self.base_classifiers:
            classifier.fit(X, y)
        
        # 计算胜任度矩阵
        self.competence_matrix = self._compute_competence_matrix(
            validation_predictions, validation_probabilities, y
        )
        
        # 训练近邻模型
        self.nn_model = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn_model.fit(X)
    
    def _compute_competence_matrix(self, predictions: np.ndarray, 
                                  probabilities: np.ndarray, 
                                  y_true: np.ndarray) -> np.ndarray:
        """计算胜任度矩阵"""
        n_samples, n_classifiers = predictions.shape
        competence_matrix = np.zeros((n_samples, n_classifiers))
        
        y_true_numeric = np.array([list(self.class_names).index(str(label)) for label in y_true])
        
        for i in range(n_samples):
            for j in range(n_classifiers):
                if self.competence_measure == 'accuracy':
                    competence_matrix[i, j] = int(predictions[i, j] == y_true_numeric[i])
                elif self.competence_measure == 'confidence':
                    true_class = y_true_numeric[i]
                    competence_matrix[i, j] = probabilities[i, j, true_class]
                # 可以添加更多胜任度度量
        
        return competence_matrix
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        predictions = np.zeros(X.shape[0])
        
        for i, sample in enumerate(X):
            # 找到最近邻
            distances, indices = self.nn_model.kneighbors([sample])
            neighbor_indices = indices[0]
            
            # 计算每个分类器的胜任度
            classifier_competence = np.mean(self.competence_matrix[neighbor_indices], axis=0)
            
            # 选择最佳分类器
            best_classifier_idx = np.argmax(classifier_competence)
            
            # 使用最佳分类器预测
            pred = self.base_classifiers[best_classifier_idx].predict([sample])
            predictions[i] = list(self.class_names).index(str(pred[0]))
        
        # 转换回原始标签
        return np.array([self.class_names[int(p)] for p in predictions])
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        probabilities = np.zeros((X.shape[0], len(self.class_names)))
        
        for i, sample in enumerate(X):
            # 找到最近邻
            distances, indices = self.nn_model.kneighbors([sample])
            neighbor_indices = indices[0]
            
            # 计算每个分类器的胜任度
            classifier_competence = np.mean(self.competence_matrix[neighbor_indices], axis=0)
            
            # 加权组合所有分类器的概率预测
            weighted_proba = np.zeros(len(self.class_names))
            for j, classifier in enumerate(self.base_classifiers):
                proba = classifier.predict_proba([sample])[0]
                weighted_proba += classifier_competence[j] * proba
            
            # 归一化
            probabilities[i] = weighted_proba / np.sum(weighted_proba)
        
        return probabilities


class MultiClassifierSystem:
    """
    多分类器系统管理器
    
    统一管理和评估多个集成分类器的工具类。
    """
    
    def __init__(self):
        """初始化多分类器系统"""
        self.classifiers = {}
        self.evaluation_results = {}
        self.best_classifier = None
        self.best_score = 0.0
    
    def add_classifier(self, name: str, classifier: BaseClassifier):
        """添加分类器"""
        self.classifiers[name] = classifier
        logger.info(f"已添加分类器: {name}")
    
    def train_all(self, X: np.ndarray, y: np.ndarray):
        """训练所有分类器"""
        logger.info(f"开始训练 {len(self.classifiers)} 个分类器")
        
        for name, classifier in self.classifiers.items():
            logger.info(f"训练分类器: {name}")
            start_time = time.time()
            
            try:
                classifier.fit(X, y)
                training_time = time.time() - start_time
                logger.info(f"分类器 {name} 训练完成 - 耗时: {training_time:.2f}秒")
            except Exception as e:
                logger.error(f"分类器 {name} 训练失败: {e}")
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """评估所有分类器"""
        logger.info("开始评估所有分类器")
        
        for name, classifier in self.classifiers.items():
            if not classifier.is_trained:
                logger.warning(f"分类器 {name} 尚未训练，跳过评估")
                continue
            
            try:
                # 预测
                y_pred = classifier.predict(X_test)
                y_proba = classifier.predict_proba(X_test)
                
                # 计算指标
                metrics = ClassificationMetrics.calculate_all_metrics(
                    y_test, y_pred, y_proba, classifier.class_names
                )
                
                self.evaluation_results[name] = metrics
                
                # 更新最佳分类器
                if metrics['accuracy'] > self.best_score:
                    self.best_score = metrics['accuracy']
                    self.best_classifier = name
                
                logger.info(f"分类器 {name} 精度: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"分类器 {name} 评估失败: {e}")
        
        logger.info(f"最佳分类器: {self.best_classifier} (精度: {self.best_score:.4f})")
        
        return self.evaluation_results
    
    def plot_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制分类器性能比较图"""
        if not self.evaluation_results:
            raise ValueError("没有评估结果可绘制")
        
        # 准备数据
        classifiers = list(self.evaluation_results.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'kappa']
        
        data = []
        for classifier in classifiers:
            row = [self.evaluation_results[classifier].get(metric, 0) for metric in metrics]
            data.append(row)
        
        data = np.array(data)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(classifiers))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, data[:, i], width, label=metric)
        
        ax.set_xlabel('分类器')
        ax.set_ylabel('分数')
        ax.set_title('分类器性能比较')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_best_classifier(self) -> Tuple[str, BaseClassifier]:
        """获取最佳分类器"""
        if self.best_classifier is None:
            raise ValueError("尚未进行评估")
        
        return self.best_classifier, self.classifiers[self.best_classifier]
    
    def save_results(self, filepath: Union[str, Path]):
        """保存评估结果"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        results = {
            'evaluation_results': self.evaluation_results,
            'best_classifier': self.best_classifier,
            'best_score': self.best_score
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"评估结果已保存到: {filepath}")


def create_ensemble_classifier(ensemble_type: str, 
                             base_classifiers: List[BaseClassifier], 
                             **kwargs) -> BaseClassifier:
    """
    集成分类器工厂函数
    
    Parameters:
    -----------
    ensemble_type : str
        集成类型 ('voting', 'stacking', 'dynamic')
    base_classifiers : list
        基分类器列表
    **kwargs : dict
        集成参数
        
    Returns:
    --------
    ensemble : BaseClassifier
        创建的集成分类器实例
    """
    ensembles = {
        'voting': VotingEnsemble,
        'stacking': StackingEnsemble,
        'dynamic': DynamicEnsemble
    }
    
    if ensemble_type.lower() not in ensembles:
        raise ValueError(f"不支持的集成类型: {ensemble_type}")
    
    return ensembles[ensemble_type.lower()](base_classifiers, **kwargs)


def create_default_ensemble(config: Optional[Dict] = None) -> List[BaseClassifier]:
    """
    创建默认的基分类器组合
    
    Parameters:
    -----------
    config : dict, optional
        配置参数
        
    Returns:
    --------
    classifiers : list
        基分类器列表
    """
    config = config or {}
    
    classifiers = [
        SVMClassifier(kernel='rbf', **config.get('svm', {})),
        RandomForestClassifier(n_estimators=100, **config.get('rf', {})),
        XGBoostClassifier(n_estimators=100, **config.get('xgb', {}))
    ]
    
    # 如果配置中包含深度学习，添加深度学习分类器
    if config.get('include_deep_learning', False):
        classifiers.append(
            DeepLearningClassifier(
                model_type='hybrid',
                **config.get('deep_learning', {})
            )
        )
    
    return classifiers