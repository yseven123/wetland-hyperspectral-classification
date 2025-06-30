"""
分类器基类
===========

这个模块定义了所有分类器的基础接口和通用功能。

主要功能：
- 抽象基类定义
- 通用评估方法
- 模型保存和加载
- 分类报告生成
- 混淆矩阵计算

作者: 湿地遥感研究团队
日期: 2024
"""

import os
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import joblib
import json
from datetime import datetime
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """
    分类器抽象基类
    
    所有分类器都应该继承这个基类，并实现必要的抽象方法。
    提供了通用的训练、预测、评估和保存/加载功能。
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        初始化基础分类器
        
        Parameters:
        -----------
        config : dict, optional
            分类器配置参数
        random_state : int, default=42
            随机种子
        n_jobs : int, default=-1
            并行作业数量
        """
        self.config = config or {}
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.class_names = None
        self.feature_names = None
        self.model_info = {
            'created_at': datetime.now(),
            'model_type': self.__class__.__name__,
            'version': '1.0.0'
        }
        
        # 设置随机种子
        np.random.seed(random_state)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseClassifier':
        """
        训练分类器
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据
        y : np.ndarray
            训练标签数据
        **kwargs : dict
            其他训练参数
            
        Returns:
        --------
        self : BaseClassifier
            返回自身实例
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测分类结果
        
        Parameters:
        -----------
        X : np.ndarray
            待预测特征数据
            
        Returns:
        --------
        predictions : np.ndarray
            预测的类别标签
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率
        
        Parameters:
        -----------
        X : np.ndarray
            待预测特征数据
            
        Returns:
        --------
        probabilities : np.ndarray
            各类别的预测概率
        """
        pass
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray, 
                   X_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        训练并预测
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据
        y : np.ndarray
            训练标签数据
        X_test : np.ndarray, optional
            测试特征数据，如果为None则使用训练数据
            
        Returns:
        --------
        predictions : np.ndarray
            预测结果
        """
        self.fit(X, y)
        test_data = X_test if X_test is not None else X
        return self.predict(test_data)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                output_dict: bool = True) -> Union[Dict[str, float], float]:
        """
        评估分类器性能
        
        Parameters:
        -----------
        X : np.ndarray
            测试特征数据
        y_true : np.ndarray
            真实标签
        output_dict : bool, default=True
            是否返回详细评估结果字典
            
        Returns:
        --------
        results : dict or float
            评估结果
        """
        if not self.is_trained:
            raise ValueError("分类器尚未训练，请先调用fit方法")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # 计算多类AUC（如果是多类问题）
        try:
            if len(np.unique(y_true)) > 2:
                results['auc_macro'] = roc_auc_score(y_true, y_proba, 
                                                   multi_class='ovr', average='macro')
                results['auc_weighted'] = roc_auc_score(y_true, y_proba, 
                                                      multi_class='ovr', average='weighted')
            else:
                results['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception as e:
            logger.warning(f"无法计算AUC分数: {e}")
        
        # 计算平均精度分数
        try:
            results['average_precision'] = average_precision_score(y_true, y_proba, average='macro')
        except Exception as e:
            logger.warning(f"无法计算平均精度分数: {e}")
        
        return results if output_dict else results['accuracy']
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'accuracy',
                      return_train_score: bool = False) -> Dict[str, Any]:
        """
        交叉验证评估
        
        Parameters:
        -----------
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签数据
        cv : int, default=5
            交叉验证折数
        scoring : str, default='accuracy'
            评价指标
        return_train_score : bool, default=False
            是否返回训练分数
            
        Returns:
        --------
        cv_results : dict
            交叉验证结果
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring, 
                               n_jobs=self.n_jobs)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        if return_train_score:
            train_scores = cross_val_score(self.model, X, y, cv=skf, 
                                         scoring=scoring, n_jobs=self.n_jobs)
            results['train_scores'] = train_scores
            results['train_mean'] = train_scores.mean()
            results['train_std'] = train_scores.std()
        
        return results
    
    def confusion_matrix(self, X: np.ndarray, y_true: np.ndarray,
                        normalize: Optional[str] = None) -> np.ndarray:
        """
        计算混淆矩阵
        
        Parameters:
        -----------
        X : np.ndarray
            测试特征数据
        y_true : np.ndarray
            真实标签
        normalize : str, optional
            归一化方式 ('true', 'pred', 'all')
            
        Returns:
        --------
        cm : np.ndarray
            混淆矩阵
        """
        y_pred = self.predict(X)
        return confusion_matrix(y_true, y_pred, normalize=normalize)
    
    def classification_report(self, X: np.ndarray, y_true: np.ndarray,
                            output_dict: bool = False) -> Union[str, Dict]:
        """
        生成分类报告
        
        Parameters:
        -----------
        X : np.ndarray
            测试特征数据
        y_true : np.ndarray
            真实标签
        output_dict : bool, default=False
            是否返回字典格式
            
        Returns:
        --------
        report : str or dict
            分类报告
        """
        y_pred = self.predict(X)
        target_names = self.class_names if self.class_names else None
        
        return classification_report(y_true, y_pred, 
                                   target_names=target_names,
                                   output_dict=output_dict,
                                   zero_division=0)
    
    def plot_confusion_matrix(self, X: np.ndarray, y_true: np.ndarray,
                             normalize: Optional[str] = None,
                             figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制混淆矩阵热图
        
        Parameters:
        -----------
        X : np.ndarray
            测试特征数据
        y_true : np.ndarray
            真实标签
        normalize : str, optional
            归一化方式
        figsize : tuple, default=(8, 6)
            图形大小
        save_path : str, optional
            保存路径
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            图形对象
        """
        cm = self.confusion_matrix(X, y_true, normalize=normalize)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax,
                   xticklabels=self.class_names or range(cm.shape[1]),
                   yticklabels=self.class_names or range(cm.shape[0]))
        
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title(f'混淆矩阵 - {self.__class__.__name__}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns:
        --------
        importance : np.ndarray or None
            特征重要性分数
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            logger.warning("当前模型不支持特征重要性计算")
            return None
    
    def plot_feature_importance(self, top_k: int = 20,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        绘制特征重要性图
        
        Parameters:
        -----------
        top_k : int, default=20
            显示前k个重要特征
        figsize : tuple, default=(10, 8)
            图形大小
        save_path : str, optional
            保存路径
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or None
            图形对象
        """
        importance = self.get_feature_importance()
        if importance is None:
            return None
        
        # 获取前top_k个特征
        indices = np.argsort(importance)[::-1][:top_k]
        values = importance[indices]
        
        # 特征名称
        if self.feature_names:
            names = [self.feature_names[i] for i in indices]
        else:
            names = [f'特征{i}' for i in indices]
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_xlabel('特征')
        ax.set_ylabel('重要性')
        ax.set_title(f'特征重要性 - {self.__class__.__name__}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        保存模型
        
        Parameters:
        -----------
        filepath : str or Path
            保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备保存数据
        save_data = {
            'model': self.model,
            'config': self.config,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        
        # 保存模型
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"模型已保存到: {filepath}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'BaseClassifier':
        """
        加载模型
        
        Parameters:
        -----------
        filepath : str or Path
            模型文件路径
            
        Returns:
        --------
        classifier : BaseClassifier
            加载的分类器实例
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # 创建分类器实例
            classifier = cls(
                config=save_data['config'],
                random_state=save_data['random_state'],
                n_jobs=save_data['n_jobs']
            )
            
            # 恢复状态
            classifier.model = save_data['model']
            classifier.is_trained = save_data['is_trained']
            classifier.training_history = save_data['training_history']
            classifier.class_names = save_data['class_names']
            classifier.feature_names = save_data['feature_names']
            classifier.model_info = save_data['model_info']
            
            logger.info(f"模型已从 {filepath} 加载")
            return classifier
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def save_training_history(self, filepath: Union[str, Path]) -> None:
        """
        保存训练历史
        
        Parameters:
        -----------
        filepath : str or Path
            保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        history_data = {
            'model_type': self.__class__.__name__,
            'training_history': self.training_history,
            'model_info': self.model_info,
            'config': self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"训练历史已保存到: {filepath}")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"trained={self.is_trained}, "
                f"random_state={self.random_state})")
    
    def __str__(self) -> str:
        """详细字符串表示"""
        status = "已训练" if self.is_trained else "未训练"
        return (f"{self.__class__.__name__}\n"
                f"状态: {status}\n"
                f"随机种子: {self.random_state}\n"
                f"并行作业: {self.n_jobs}")


class ClassificationMetrics:
    """
    分类评估指标计算工具类
    
    提供各种分类性能评估指标的计算方法
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None,
                             class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        计算所有分类指标
        
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
            
        Returns:
        --------
        metrics : dict
            所有评估指标
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # 添加每个类别的详细指标
        if class_names:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(class_names):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # 添加AUC指标（如果提供了概率）
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) > 2:
                    metrics['auc_macro'] = roc_auc_score(y_true, y_proba, 
                                                       multi_class='ovr', average='macro')
                    metrics['auc_weighted'] = roc_auc_score(y_true, y_proba, 
                                                          multi_class='ovr', average='weighted')
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                
                metrics['average_precision'] = average_precision_score(y_true, y_proba, average='macro')
            except Exception as e:
                logger.warning(f"无法计算概率相关指标: {e}")
        
        return metrics
    
    @staticmethod
    def print_metrics_report(metrics: Dict[str, float], 
                           title: str = "分类性能报告") -> None:
        """
        打印格式化的指标报告
        
        Parameters:
        -----------
        metrics : dict
            评估指标字典
        title : str
            报告标题
        """
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        
        # 主要指标
        main_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'kappa']
        print("\n主要指标:")
        print("-" * 30)
        for metric in main_metrics:
            if metric in metrics:
                print(f"{metric:20s}: {metrics[metric]:.4f}")
        
        # AUC指标
        auc_metrics = [k for k in metrics.keys() if 'auc' in k or 'average_precision' in k]
        if auc_metrics:
            print("\nAUC相关指标:")
            print("-" * 30)
            for metric in auc_metrics:
                print(f"{metric:20s}: {metrics[metric]:.4f}")
        
        # 各类别详细指标
        class_metrics = [k for k in metrics.keys() if any(x in k for x in ['precision_', 'recall_', 'f1_']) 
                        and not any(x in k for x in ['macro', 'micro', 'weighted'])]
        if class_metrics:
            print("\n各类别详细指标:")
            print("-" * 30)
            for metric in sorted(class_metrics):
                print(f"{metric:20s}: {metrics[metric]:.4f}")
        
        print(f"{'='*50}\n")


# 错误处理装饰器
def handle_classification_errors(func):
    """分类器方法的错误处理装饰器"""
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"分类器 {self.__class__.__name__} 执行 {func.__name__} 时出错: {e}")
            raise
    return wrapper