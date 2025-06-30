"""
湿地高光谱分类模块
=================

这个模块提供了完整的高光谱遥感分类功能，包括传统机器学习、深度学习和集成学习方法。

主要组件：
- BaseClassifier: 分类器基类
- 传统机器学习: SVM, RF, XGBoost, KNN, NB
- 深度学习: 3D-CNN, HybridSN, Transformer, ResNet
- 集成学习: Voting, Stacking, Dynamic Selection
- 工具类: 评估指标、多分类器系统管理

作者: 湿地遥感研究团队
日期: 2024
版本: 1.0.0
"""

from .base import (
    BaseClassifier,
    ClassificationMetrics,
    handle_classification_errors
)

from .traditional import (
    SVMClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
    KNNClassifier,
    NaiveBayesClassifier,
    create_classifier
)

from .deep_learning import (
    DeepLearningClassifier,
    CNN3D,
    HybridSN,
    SpectralTransformer,
    HyperspectralResNet,
    HyperspectralDataset,
    create_deep_classifier
)

from .ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    DynamicEnsemble,
    MultiClassifierSystem,
    create_ensemble_classifier,
    create_default_ensemble
)

# 版本信息
__version__ = "1.0.0"
__author__ = "湿地遥感研究团队"

# 所有可用的分类器类型
AVAILABLE_CLASSIFIERS = {
    # 传统机器学习
    'svm': SVMClassifier,
    'rf': RandomForestClassifier,
    'random_forest': RandomForestClassifier,
    'xgb': XGBoostClassifier,
    'xgboost': XGBoostClassifier,
    'knn': KNNClassifier,
    'nb': NaiveBayesClassifier,
    'naive_bayes': NaiveBayesClassifier,
    
    # 深度学习
    'cnn3d': lambda **kwargs: DeepLearningClassifier(model_type='cnn3d', **kwargs),
    '3d_cnn': lambda **kwargs: DeepLearningClassifier(model_type='cnn3d', **kwargs),
    'hybrid': lambda **kwargs: DeepLearningClassifier(model_type='hybrid', **kwargs),
    'hybridSN': lambda **kwargs: DeepLearningClassifier(model_type='hybrid', **kwargs),
    'transformer': lambda **kwargs: DeepLearningClassifier(model_type='transformer', **kwargs),
    'resnet': lambda **kwargs: DeepLearningClassifier(model_type='resnet', **kwargs),
    
    # 集成学习
    'voting': VotingEnsemble,
    'stacking': StackingEnsemble,
    'dynamic': DynamicEnsemble
}

# 导出的公共接口
__all__ = [
    # 基础类
    'BaseClassifier',
    'ClassificationMetrics',
    
    # 传统机器学习
    'SVMClassifier',
    'RandomForestClassifier', 
    'XGBoostClassifier',
    'KNNClassifier',
    'NaiveBayesClassifier',
    
    # 深度学习
    'DeepLearningClassifier',
    'CNN3D',
    'HybridSN',
    'SpectralTransformer',
    'HyperspectralResNet',
    'HyperspectralDataset',
    
    # 集成学习
    'VotingEnsemble',
    'StackingEnsemble',
    'DynamicEnsemble',
    'MultiClassifierSystem',
    
    # 工厂函数
    'create_classifier',
    'create_deep_classifier',
    'create_ensemble_classifier',
    'create_default_ensemble',
    
    # 工具函数
    'get_classifier',
    'list_available_classifiers',
    'create_multi_classifier_system',
    
    # 装饰器
    'handle_classification_errors',
    
    # 常量
    'AVAILABLE_CLASSIFIERS'
]


def get_classifier(classifier_type: str, **kwargs) -> BaseClassifier:
    """
    获取指定类型的分类器实例
    
    这是一个统一的分类器工厂函数，支持所有类型的分类器。
    
    Parameters:
    -----------
    classifier_type : str
        分类器类型，支持的类型包括：
        - 传统ML: 'svm', 'rf', 'xgb', 'knn', 'nb'
        - 深度学习: 'cnn3d', 'hybrid', 'transformer', 'resnet'
        - 集成学习: 'voting', 'stacking', 'dynamic'
    **kwargs : dict
        分类器参数
        
    Returns:
    --------
    classifier : BaseClassifier
        分类器实例
        
    Examples:
    ---------
    >>> # 创建SVM分类器
    >>> svm = get_classifier('svm', kernel='rbf', C=1.0)
    
    >>> # 创建随机森林分类器
    >>> rf = get_classifier('rf', n_estimators=100)
    
    >>> # 创建3D-CNN分类器
    >>> cnn = get_classifier('cnn3d', batch_size=64, num_epochs=100)
    
    >>> # 创建集成分类器
    >>> base_classifiers = [get_classifier('svm'), get_classifier('rf')]
    >>> ensemble = get_classifier('voting', base_classifiers=base_classifiers)
    """
    classifier_type = classifier_type.lower()
    
    if classifier_type not in AVAILABLE_CLASSIFIERS:
        available = ', '.join(sorted(AVAILABLE_CLASSIFIERS.keys()))
        raise ValueError(
            f"不支持的分类器类型: '{classifier_type}'. "
            f"可用类型: {available}"
        )
    
    classifier_class = AVAILABLE_CLASSIFIERS[classifier_type]
    
    # 对于集成分类器，需要特殊处理
    if classifier_type in ['voting', 'stacking', 'dynamic']:
        if 'base_classifiers' not in kwargs:
            raise ValueError(f"集成分类器 '{classifier_type}' 需要 'base_classifiers' 参数")
    
    return classifier_class(**kwargs)


def list_available_classifiers() -> dict:
    """
    列出所有可用的分类器类型
    
    Returns:
    --------
    classifiers : dict
        按类别组织的分类器字典
        
    Examples:
    ---------
    >>> classifiers = list_available_classifiers()
    >>> print("传统机器学习:", classifiers['traditional'])
    >>> print("深度学习:", classifiers['deep_learning'])
    >>> print("集成学习:", classifiers['ensemble'])
    """
    return {
        'traditional': [
            'svm', 'rf', 'random_forest', 'xgb', 'xgboost', 
            'knn', 'nb', 'naive_bayes'
        ],
        'deep_learning': [
            'cnn3d', '3d_cnn', 'hybrid', 'hybridSN', 
            'transformer', 'resnet'
        ],
        'ensemble': [
            'voting', 'stacking', 'dynamic'
        ]
    }


def create_multi_classifier_system(classifier_configs: dict) -> MultiClassifierSystem:
    """
    创建多分类器系统
    
    Parameters:
    -----------
    classifier_configs : dict
        分类器配置字典，格式：{name: {type: str, params: dict}}
        
    Returns:
    --------
    system : MultiClassifierSystem
        多分类器系统实例
        
    Examples:
    ---------
    >>> configs = {
    ...     'svm_rbf': {'type': 'svm', 'params': {'kernel': 'rbf'}},
    ...     'random_forest': {'type': 'rf', 'params': {'n_estimators': 100}},
    ...     'xgboost': {'type': 'xgb', 'params': {'n_estimators': 100}}
    ... }
    >>> system = create_multi_classifier_system(configs)
    """
    system = MultiClassifierSystem()
    
    for name, config in classifier_configs.items():
        classifier_type = config['type']
        params = config.get('params', {})
        
        classifier = get_classifier(classifier_type, **params)
        system.add_classifier(name, classifier)
    
    return system


def create_wetland_classifier_suite() -> MultiClassifierSystem:
    """
    创建专门针对湿地分类的分类器套件
    
    这个函数创建一个预配置的分类器系统，包含最适合湿地高光谱分类的算法组合。
    
    Returns:
    --------
    system : MultiClassifierSystem
        湿地分类器系统
        
    Examples:
    ---------
    >>> # 创建湿地分类器套件
    >>> suite = create_wetland_classifier_suite()
    >>> 
    >>> # 训练所有分类器
    >>> suite.train_all(X_train, y_train)
    >>> 
    >>> # 评估所有分类器
    >>> results = suite.evaluate_all(X_test, y_test)
    >>> 
    >>> # 获取最佳分类器
    >>> best_name, best_classifier = suite.get_best_classifier()
    """
    # 湿地分类优化配置
    configs = {
        # 传统机器学习 - 针对湿地优化
        'svm_rbf': {
            'type': 'svm',
            'params': {
                'kernel': 'rbf',
                'C': 100,
                'gamma': 'scale',
                'class_weight': 'balanced',
                'probability': True
            }
        },
        
        'random_forest': {
            'type': 'rf',
            'params': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'class_weight': 'balanced',
                'oob_score': True
            }
        },
        
        'xgboost': {
            'type': 'xgb',
            'params': {
                'n_estimators': 150,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        },
        
        # 深度学习 - 高光谱专用
        'hybrid_cnn': {
            'type': 'hybrid',
            'params': {
                'patch_size': 11,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 100,
                'early_stopping_patience': 15
            }
        },
        
        '3d_cnn': {
            'type': 'cnn3d',
            'params': {
                'patch_size': 9,
                'batch_size': 64,
                'learning_rate': 0.0005,
                'num_epochs': 80
            }
        }
    }
    
    return create_multi_classifier_system(configs)


# 便捷的分类器创建函数
def create_svm(kernel='rbf', **kwargs):
    """创建SVM分类器"""
    return get_classifier('svm', kernel=kernel, **kwargs)

def create_random_forest(n_estimators=100, **kwargs):
    """创建随机森林分类器"""
    return get_classifier('rf', n_estimators=n_estimators, **kwargs)

def create_xgboost(n_estimators=100, **kwargs):
    """创建XGBoost分类器"""
    return get_classifier('xgb', n_estimators=n_estimators, **kwargs)

def create_cnn3d(patch_size=11, **kwargs):
    """创建3D-CNN分类器"""
    return get_classifier('cnn3d', patch_size=patch_size, **kwargs)

def create_hybrid_cnn(patch_size=11, **kwargs):
    """创建HybridSN分类器"""
    return get_classifier('hybrid', patch_size=patch_size, **kwargs)


# 模块信息
def get_module_info():
    """获取模块信息"""
    return {
        'name': '湿地高光谱分类模块',
        'version': __version__,
        'author': __author__,
        'description': '完整的高光谱遥感分类工具包',
        'classifiers': {
            'traditional': len([k for k in AVAILABLE_CLASSIFIERS.keys() 
                              if k in ['svm', 'rf', 'xgb', 'knn', 'nb']]),
            'deep_learning': len([k for k in AVAILABLE_CLASSIFIERS.keys() 
                                if k in ['cnn3d', 'hybrid', 'transformer', 'resnet']]),
            'ensemble': len([k for k in AVAILABLE_CLASSIFIERS.keys() 
                           if k in ['voting', 'stacking', 'dynamic']])
        },
        'total_classifiers': len(set(AVAILABLE_CLASSIFIERS.values()))
    }


# 打印模块信息
if __name__ == "__main__":
    info = get_module_info()
    print(f"\n{info['name']} v{info['version']}")
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    print(f"\n支持的分类器:")
    print(f"  传统机器学习: {info['classifiers']['traditional']} 种")
    print(f"  深度学习: {info['classifiers']['deep_learning']} 种")
    print(f"  集成学习: {info['classifiers']['ensemble']} 种")
    print(f"  总计: {info['total_classifiers']} 种不同的分类器\n")
    
    # 显示可用分类器列表
    available = list_available_classifiers()
    for category, classifiers in available.items():
        print(f"{category.upper()}:")
        for classifier in classifiers:
            print(f"  - {classifier}")
        print()