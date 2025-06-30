"""
传统机器学习分类器
=================

这个模块实现了多种传统机器学习分类算法，特别针对高光谱湿地分类进行优化。

支持的算法：
- 支持向量机 (SVM)
- 随机森林 (RF)
- 极端梯度提升 (XGBoost)
- K近邻 (KNN)
- 朴素贝叶斯 (NB)

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
import xgboost as xgb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

from .base import BaseClassifier, handle_classification_errors

# 设置日志
logger = logging.getLogger(__name__)


class SVMClassifier(BaseClassifier):
    """
    支持向量机分类器
    
    针对高光谱数据优化的SVM分类器，支持多种核函数和自动参数调优。
    特别适合小样本、高维数据的分类任务。
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 probability: bool = True,
                 class_weight: Optional[str] = None,
                 cache_size: int = 2000,
                 auto_scale: bool = True,
                 feature_selection: bool = False,
                 n_features: Optional[int] = None,
                 param_search: bool = False,
                 search_cv: int = 3,
                 **kwargs):
        """
        初始化SVM分类器
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
        C : float, default=1.0
            正则化参数
        gamma : str or float, default='scale'
            核系数
        degree : int, default=3
            多项式核的度数
        probability : bool, default=True
            是否启用概率估计
        class_weight : str, optional
            类权重 ('balanced' 或 None)
        cache_size : int, default=2000
            缓存大小(MB)
        auto_scale : bool, default=True
            是否自动标准化特征
        feature_selection : bool, default=False
            是否进行特征选择
        n_features : int, optional
            选择的特征数量
        param_search : bool, default=False
            是否进行参数搜索
        search_cv : int, default=3
            参数搜索的交叉验证折数
        """
        super().__init__(**kwargs)
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.auto_scale = auto_scale
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.param_search = param_search
        self.search_cv = search_cv
        
        self.scaler = None
        self.feature_selector = None
        self.best_params = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVMClassifier':
        """
        训练SVM分类器
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据 (n_samples, n_features)
        y : np.ndarray
            训练标签数据 (n_samples,)
            
        Returns:
        --------
        self : SVMClassifier
            返回自身实例
        """
        logger.info(f"开始训练SVM分类器 - 核函数: {self.kernel}")
        start_time = time.time()
        
        # 数据检查
        if X.shape[0] != len(y):
            raise ValueError("特征数据和标签数据的样本数量不匹配")
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        
        # 数据预处理
        X_processed = X.copy()
        
        # 特征标准化
        if self.auto_scale:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
            logger.info("特征标准化完成")
        
        # 特征选择
        if self.feature_selection:
            if self.n_features is None:
                self.n_features = min(1000, X_processed.shape[1] // 2)
            
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            X_processed = self.feature_selector.fit_transform(X_processed, y)
            logger.info(f"特征选择完成 - 选择了 {X_processed.shape[1]} 个特征")
        
        # 参数搜索
        if self.param_search:
            logger.info("开始参数搜索...")
            best_model = self._parameter_search(X_processed, y)
            self.model = best_model
        else:
            # 使用指定参数创建模型
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                probability=self.probability,
                class_weight=self.class_weight,
                cache_size=self.cache_size,
                random_state=self.random_state
            )
            
            # 训练模型
            self.model.fit(X_processed, y)
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['kernel'] = self.kernel
        self.training_history['best_params'] = self.best_params
        
        self.is_trained = True
        logger.info(f"SVM训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _parameter_search(self, X: np.ndarray, y: np.ndarray) -> SVC:
        """参数搜索"""
        param_grid = {
            'linear': {'C': [0.1, 1, 10, 100]},
            'rbf': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'poly': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4]
            }
        }
        
        # 根据核函数选择参数网格
        if self.kernel in param_grid:
            params = param_grid[self.kernel]
        else:
            params = {'C': [0.1, 1, 10]}
        
        # 创建基础模型
        base_model = SVC(
            kernel=self.kernel,
            probability=self.probability,
            class_weight=self.class_weight,
            cache_size=self.cache_size,
            random_state=self.random_state
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model, params, cv=self.search_cv,
            scoring='accuracy', n_jobs=self.n_jobs,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        logger.info(f"最佳参数: {self.best_params}")
        
        return grid_search.best_estimator_
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if not self.probability:
            raise ValueError("模型创建时未启用概率估计")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """预处理特征数据"""
        X_processed = X.copy()
        
        if self.scaler:
            X_processed = self.scaler.transform(X_processed)
        
        if self.feature_selector:
            X_processed = self.feature_selector.transform(X_processed)
        
        return X_processed


class RandomForestClassifier(BaseClassifier):
    """
    随机森林分类器
    
    针对高光谱湿地分类优化的随机森林算法，具有良好的泛化能力和特征重要性分析。
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 class_weight: Optional[str] = None,
                 criterion: str = 'gini',
                 param_search: bool = False,
                 search_cv: int = 3,
                 **kwargs):
        """
        初始化随机森林分类器
        
        Parameters:
        -----------
        n_estimators : int, default=100
            树的数量
        max_depth : int, optional
            树的最大深度
        min_samples_split : int, default=2
            分裂内部节点所需的最小样本数
        min_samples_leaf : int, default=1
            叶子节点的最小样本数
        max_features : str, int or float, default='sqrt'
            寻找最佳分割时考虑的特征数量
        bootstrap : bool, default=True
            是否使用自助采样
        oob_score : bool, default=True
            是否使用袋外样本评估
        class_weight : str, optional
            类权重 ('balanced' 或 None)
        criterion : str, default='gini'
            分割质量衡量标准 ('gini' 或 'entropy')
        param_search : bool, default=False
            是否进行参数搜索
        search_cv : int, default=3
            参数搜索的交叉验证折数
        """
        super().__init__(**kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.criterion = criterion
        self.param_search = param_search
        self.search_cv = search_cv
        
        self.best_params = None
        self.oob_score_ = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestClassifier':
        """训练随机森林分类器"""
        logger.info("开始训练随机森林分类器")
        start_time = time.time()
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        
        # 参数搜索
        if self.param_search:
            logger.info("开始参数搜索...")
            self.model = self._parameter_search(X, y)
        else:
            # 使用指定参数创建模型
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                class_weight=self.class_weight,
                criterion=self.criterion,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            # 训练模型
            self.model.fit(X, y)
        
        # 记录OOB分数
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            self.oob_score_ = self.model.oob_score_
            logger.info(f"OOB分数: {self.oob_score_:.4f}")
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['oob_score'] = self.oob_score_
        self.training_history['best_params'] = self.best_params
        
        self.is_trained = True
        logger.info(f"随机森林训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _parameter_search(self, X: np.ndarray, y: np.ndarray):
        """参数搜索"""
        param_distributions = {
            'n_estimators': randint(50, 200),
            'max_depth': [None] + list(range(10, 31, 5)),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5]
        }
        
        # 创建基础模型
        base_model = RandomForestClassifier(
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            class_weight=self.class_weight,
            criterion=self.criterion,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            base_model, param_distributions, 
            n_iter=50, cv=self.search_cv,
            scoring='accuracy', n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        logger.info(f"最佳参数: {self.best_params}")
        
        return random_search.best_estimator_
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)


class XGBoostClassifier(BaseClassifier):
    """
    XGBoost分类器
    
    极端梯度提升分类器，在高光谱数据分类中表现优异。
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 objective: str = 'multi:softprob',
                 eval_metric: str = 'mlogloss',
                 early_stopping_rounds: Optional[int] = 10,
                 param_search: bool = False,
                 search_cv: int = 3,
                 **kwargs):
        """初始化XGBoost分类器"""
        super().__init__(**kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.param_search = param_search
        self.search_cv = search_cv
        
        self.best_params = None
        self.eval_results = {}
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, 
           X_val: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None,
           **kwargs) -> 'XGBoostClassifier':
        """训练XGBoost分类器"""
        logger.info("开始训练XGBoost分类器")
        start_time = time.time()
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        
        # 参数搜索
        if self.param_search:
            logger.info("开始参数搜索...")
            self.model = self._parameter_search(X, y)
        else:
            # 创建模型
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                objective=self.objective,
                eval_metric=self.eval_metric,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0
            )
            
            # 训练参数
            fit_params = {}
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
                if self.early_stopping_rounds:
                    fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            
            # 训练模型
            self.model.fit(X, y, **fit_params)
            
            # 记录评估结果
            if hasattr(self.model, 'evals_result_'):
                self.eval_results = self.model.evals_result_
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['eval_results'] = self.eval_results
        self.training_history['best_params'] = self.best_params
        
        self.is_trained = True
        logger.info(f"XGBoost训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _parameter_search(self, X: np.ndarray, y: np.ndarray):
        """参数搜索"""
        param_distributions = {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2)
        }
        
        # 创建基础模型
        base_model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0
        )
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=30, cv=self.search_cv,
            scoring='accuracy', n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        logger.info(f"最佳参数: {self.best_params}")
        
        return random_search.best_estimator_
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)


class KNNClassifier(BaseClassifier):
    """
    K近邻分类器
    
    基于距离的分类器，适合局部特征明显的高光谱数据。
    """
    
    def __init__(self,
                 n_neighbors: int = 5,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 p: int = 2,
                 metric: str = 'minkowski',
                 auto_scale: bool = True,
                 param_search: bool = False,
                 search_cv: int = 3,
                 **kwargs):
        """初始化KNN分类器"""
        super().__init__(**kwargs)
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.auto_scale = auto_scale
        self.param_search = param_search
        self.search_cv = search_cv
        
        self.scaler = None
        self.best_params = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'KNNClassifier':
        """训练KNN分类器"""
        logger.info("开始训练KNN分类器")
        start_time = time.time()
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        
        # 数据预处理
        X_processed = X.copy()
        if self.auto_scale:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
            logger.info("特征标准化完成")
        
        # 参数搜索
        if self.param_search:
            logger.info("开始参数搜索...")
            self.model = self._parameter_search(X_processed, y)
        else:
            # 创建模型
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                n_jobs=self.n_jobs
            )
            
            # 训练模型
            self.model.fit(X_processed, y)
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['best_params'] = self.best_params
        
        self.is_trained = True
        logger.info(f"KNN训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _parameter_search(self, X: np.ndarray, y: np.ndarray):
        """参数搜索"""
        param_grid = {
            'n_neighbors': range(3, 21, 2),
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'leaf_size': [20, 30, 50]
        }
        
        # 创建基础模型
        base_model = KNeighborsClassifier(
            algorithm=self.algorithm,
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.search_cv,
            scoring='accuracy', n_jobs=self.n_jobs,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        logger.info(f"最佳参数: {self.best_params}")
        
        return grid_search.best_estimator_
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """预处理特征数据"""
        X_processed = X.copy()
        
        if self.scaler:
            X_processed = self.scaler.transform(X_processed)
        
        return X_processed


class NaiveBayesClassifier(BaseClassifier):
    """
    朴素贝叶斯分类器
    
    基于概率的快速分类器，适合作为基准模型。
    """
    
    def __init__(self,
                 var_smoothing: float = 1e-9,
                 auto_scale: bool = True,
                 param_search: bool = False,
                 search_cv: int = 3,
                 **kwargs):
        """初始化朴素贝叶斯分类器"""
        super().__init__(**kwargs)
        
        self.var_smoothing = var_smoothing
        self.auto_scale = auto_scale
        self.param_search = param_search
        self.search_cv = search_cv
        
        self.scaler = None
        self.best_params = None
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'NaiveBayesClassifier':
        """训练朴素贝叶斯分类器"""
        logger.info("开始训练朴素贝叶斯分类器")
        start_time = time.time()
        
        # 记录数据信息
        self.class_names = [f"类别{i}" for i in np.unique(y)]
        self.training_history['n_samples'] = X.shape[0]
        self.training_history['n_features'] = X.shape[1]
        self.training_history['n_classes'] = len(self.class_names)
        
        # 数据预处理
        X_processed = X.copy()
        if self.auto_scale:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
            logger.info("特征标准化完成")
        
        # 参数搜索
        if self.param_search:
            logger.info("开始参数搜索...")
            self.model = self._parameter_search(X_processed, y)
        else:
            # 创建模型
            self.model = GaussianNB(var_smoothing=self.var_smoothing)
            
            # 训练模型
            self.model.fit(X_processed, y)
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['best_params'] = self.best_params
        
        self.is_trained = True
        logger.info(f"朴素贝叶斯训练完成 - 耗时: {training_time:.2f}秒")
        
        return self
    
    def _parameter_search(self, X: np.ndarray, y: np.ndarray):
        """参数搜索"""
        param_grid = {
            'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
        
        # 创建基础模型
        base_model = GaussianNB()
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.search_cv,
            scoring='accuracy', n_jobs=self.n_jobs,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        logger.info(f"最佳参数: {self.best_params}")
        
        return grid_search.best_estimator_
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """预处理特征数据"""
        X_processed = X.copy()
        
        if self.scaler:
            X_processed = self.scaler.transform(X_processed)
        
        return X_processed


def create_classifier(classifier_type: str, **kwargs) -> BaseClassifier:
    """
    分类器工厂函数
    
    Parameters:
    -----------
    classifier_type : str
        分类器类型 ('svm', 'rf', 'xgb', 'knn', 'nb')
    **kwargs : dict
        分类器参数
        
    Returns:
    --------
    classifier : BaseClassifier
        创建的分类器实例
    """
    classifiers = {
        'svm': SVMClassifier,
        'rf': RandomForestClassifier,
        'xgb': XGBoostClassifier,
        'knn': KNNClassifier,
        'nb': NaiveBayesClassifier
    }
    
    if classifier_type.lower() not in classifiers:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")
    
    return classifiers[classifier_type.lower()](**kwargs)