"""
分类模块测试

测试传统机器学习、深度学习和集成学习分类算法。
验证各种分类器的训练、预测和评估功能。

Author: 湿地高光谱分类系统开发团队
Date: 2024
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 导入待测试的模块
try:
    from wetland_classification.classification import Classifier
    from wetland_classification.classification.traditional import (
        SVMClassifier, RandomForestClassifier, XGBoostClassifier
    )
    from wetland_classification.classification.deep_learning import (
        CNN3DClassifier, HybridSNClassifier, VisionTransformerClassifier
    )
    from wetland_classification.classification.ensemble import (
        VotingEnsemble, StackingEnsemble, BaggingEnsemble
    )
    from wetland_classification.evaluation.metrics import ClassificationMetrics
except ImportError:
    # 如果模块不存在，创建mock对象用于测试结构
    Classifier = Mock
    SVMClassifier = Mock
    RandomForestClassifier = Mock
    XGBoostClassifier = Mock
    CNN3DClassifier = Mock
    HybridSNClassifier = Mock
    VisionTransformerClassifier = Mock
    VotingEnsemble = Mock
    StackingEnsemble = Mock
    BaggingEnsemble = Mock
    ClassificationMetrics = Mock


class TestSVMClassifier:
    """SVM分类器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True
        }
        self.classifier = SVMClassifier(self.config)
        
    def test_svm_initialization(self):
        """测试SVM分类器初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'train')
        assert hasattr(self.classifier, 'predict')
        assert hasattr(self.classifier, 'predict_proba')
        
    def test_svm_training(self, sample_training_samples):
        """测试SVM训练"""
        # 准备训练数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        self.classifier.train(X_train, y_train)
        
        # 验证训练结果
        assert hasattr(self.classifier, 'model')
        assert self.classifier.is_trained
        assert self.classifier.n_classes == len(np.unique(y_train))
        
    def test_svm_prediction(self, sample_training_samples):
        """测试SVM预测"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        X_test = np.random.random((100, n_features))
        
        # 训练并预测
        self.classifier.train(X_train, y_train)
        predictions = self.classifier.predict(X_test)
        probabilities = self.classifier.predict_proba(X_test)
        
        # 验证预测结果
        assert predictions.shape == (100,)
        assert probabilities.shape == (100, self.classifier.n_classes)
        assert np.all(predictions >= 0)
        assert np.all(predictions < self.classifier.n_classes)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)  # 概率和为1
        
    def test_svm_cross_validation(self, sample_training_samples):
        """测试SVM交叉验证"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 执行交叉验证
        cv_scores = self.classifier.cross_validate(X, y, cv=3)
        
        # 验证交叉验证结果
        assert len(cv_scores) == 3
        assert np.all(cv_scores >= 0)
        assert np.all(cv_scores <= 1)
        
    def test_svm_parameter_optimization(self, sample_training_samples):
        """测试SVM参数优化"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto']
        }
        
        # 执行参数优化
        best_params, best_score = self.classifier.optimize_parameters(
            X, y, param_grid, cv=3
        )
        
        # 验证优化结果
        assert 'C' in best_params
        assert 'gamma' in best_params
        assert 0 <= best_score <= 1
        
    def test_svm_feature_importance(self, sample_training_samples):
        """测试SVM特征重要性"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        self.classifier.train(X, y)
        
        # 计算特征重要性（使用permutation importance）
        feature_importance = self.classifier.get_feature_importance(X, y)
        
        # 验证特征重要性
        assert len(feature_importance) == n_features
        assert np.all(feature_importance >= 0)


class TestRandomForestClassifier:
    """随机森林分类器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        self.classifier = RandomForestClassifier(self.config)
        
    def test_rf_initialization(self):
        """测试随机森林初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'train')
        assert hasattr(self.classifier, 'predict')
        
    def test_rf_training_and_prediction(self, sample_training_samples):
        """测试随机森林训练和预测"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        X_test = np.random.random((100, n_features))
        
        # 训练和预测
        self.classifier.train(X_train, y_train)
        predictions = self.classifier.predict(X_test)
        
        # 验证结果
        assert predictions.shape == (100,)
        assert np.all(predictions >= 0)
        assert np.all(predictions < len(np.unique(y_train)))
        
    def test_rf_feature_importance(self, sample_training_samples):
        """测试随机森林特征重要性"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        self.classifier.train(X, y)
        
        # 获取特征重要性
        importance = self.classifier.get_feature_importance()
        
        # 验证特征重要性
        assert len(importance) == n_features
        assert np.all(importance >= 0)
        assert np.abs(np.sum(importance) - 1.0) < 1e-6  # 重要性和为1
        
    def test_rf_out_of_bag_score(self, sample_training_samples):
        """测试随机森林袋外分数"""
        # 配置启用OOB
        config = self.config.copy()
        config['oob_score'] = True
        classifier = RandomForestClassifier(config)
        
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        classifier.train(X, y)
        
        # 获取OOB分数
        oob_score = classifier.get_oob_score()
        
        # 验证OOB分数
        assert 0 <= oob_score <= 1


class TestXGBoostClassifier:
    """XGBoost分类器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.classifier = XGBoostClassifier(self.config)
        
    def test_xgb_initialization(self):
        """测试XGBoost初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'train')
        assert hasattr(self.classifier, 'predict')
        
    def test_xgb_training_with_validation(self, sample_training_samples):
        """测试XGBoost带验证集的训练"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 分割训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练分类器
        training_history = self.classifier.train_with_validation(
            X_train, y_train, X_val, y_val, early_stopping_rounds=10
        )
        
        # 验证训练历史
        assert 'train_accuracy' in training_history
        assert 'val_accuracy' in training_history
        assert len(training_history['train_accuracy']) > 0
        
    def test_xgb_feature_importance(self, sample_training_samples):
        """测试XGBoost特征重要性"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        self.classifier.train(X, y)
        
        # 获取不同类型的特征重要性
        gain_importance = self.classifier.get_feature_importance(importance_type='gain')
        cover_importance = self.classifier.get_feature_importance(importance_type='cover')
        
        # 验证特征重要性
        assert len(gain_importance) == n_features
        assert len(cover_importance) == n_features
        assert np.all(gain_importance >= 0)
        assert np.all(cover_importance >= 0)


class TestCNN3DClassifier:
    """3D-CNN分类器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'input_shape': (11, 11, 200),  # patch_size x patch_size x bands
            'num_classes': 8,
            'conv_layers': [32, 64, 128],
            'kernel_size': (3, 3, 7),
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32
        }
        self.classifier = CNN3DClassifier(self.config)
        
    def test_cnn3d_initialization(self):
        """测试3D-CNN初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'build_model')
        assert hasattr(self.classifier, 'train')
        
    def test_cnn3d_model_building(self):
        """测试3D-CNN模型构建"""
        model = self.classifier.build_model()
        
        # 验证模型结构
        assert model is not None
        assert len(model.layers) > 0
        
        # 验证输入输出形状
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        assert input_shape[1:] == self.config['input_shape']
        assert output_shape[-1] == self.config['num_classes']
        
    def test_cnn3d_data_preparation(self, sample_hyperspectral_data, sample_training_samples):
        """测试3D-CNN数据准备"""
        # 准备patch数据
        patch_data, patch_labels = self.classifier.prepare_patch_data(
            sample_hyperspectral_data, 
            sample_training_samples,
            patch_size=11
        )
        
        # 验证patch数据
        expected_shape = (len(sample_training_samples['coordinates']), 11, 11, sample_hyperspectral_data.shape[2])
        assert patch_data.shape == expected_shape
        assert len(patch_labels) == len(sample_training_samples['coordinates'])
        
    def test_cnn3d_training_simulation(self, sample_hyperspectral_data, sample_training_samples):
        """测试3D-CNN训练模拟"""
        # 准备小数据集进行快速测试
        small_config = self.config.copy()
        small_config['epochs'] = 2
        small_config['batch_size'] = 8
        classifier = CNN3DClassifier(small_config)
        
        # 准备数据
        patch_data, patch_labels = classifier.prepare_patch_data(
            sample_hyperspectral_data[:50, :50, :50],  # 缩小数据规模
            {'coordinates': sample_training_samples['coordinates'][:20],
             'labels': sample_training_samples['labels'][:20]},
            patch_size=11
        )
        
        # 模拟训练过程
        training_history = classifier.train(patch_data, patch_labels, validation_split=0.2)
        
        # 验证训练历史
        assert 'loss' in training_history
        assert 'accuracy' in training_history
        assert len(training_history['loss']) == small_config['epochs']
        
    def test_cnn3d_prediction(self, sample_hyperspectral_data):
        """测试3D-CNN预测"""
        # 构建模型
        model = self.classifier.build_model()
        
        # 准备测试patch
        test_patches = np.random.random((10, 11, 11, 200))
        
        # 执行预测
        predictions = model.predict(test_patches)
        
        # 验证预测结果
        assert predictions.shape == (10, self.config['num_classes'])
        assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5)  # softmax输出


class TestHybridSNClassifier:
    """HybridSN分类器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'input_shape': (11, 11, 200),
            'num_classes': 8,
            'conv3d_filters': [8, 16, 32],
            'conv2d_filters': [64, 128],
            'kernel_size_3d': (3, 3, 7),
            'kernel_size_2d': (3, 3),
            'dropout_rate': 0.4,
            'learning_rate': 0.001
        }
        self.classifier = HybridSNClassifier(self.config)
        
    def test_hybridsn_initialization(self):
        """测试HybridSN初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'build_model')
        
    def test_hybridsn_model_architecture(self):
        """测试HybridSN模型架构"""
        model = self.classifier.build_model()
        
        # 验证模型包含3D和2D卷积层
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Conv3D' in layer_types
        assert 'Conv2D' in layer_types
        assert 'Dense' in layer_types
        
    def test_hybridsn_spectral_spatial_fusion(self):
        """测试HybridSN光谱-空间特征融合"""
        # 创建测试数据
        test_input = np.random.random((1, 11, 11, 200))
        
        # 构建模型
        model = self.classifier.build_model()
        
        # 获取中间层输出
        conv3d_output = self.classifier.get_conv3d_features(test_input)
        conv2d_output = self.classifier.get_conv2d_features(test_input)
        
        # 验证特征融合
        assert conv3d_output is not None
        assert conv2d_output is not None
        
        # 3D特征应该被reshape为2D特征
        assert len(conv2d_output.shape) == 4  # (batch, height, width, channels)


class TestVisionTransformerClassifier:
    """Vision Transformer分类器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'input_shape': (11, 11, 200),
            'num_classes': 8,
            'patch_size': (1, 1, 10),  # 光谱patch
            'num_heads': 8,
            'num_layers': 6,
            'd_model': 256,
            'mlp_ratio': 4,
            'dropout_rate': 0.1
        }
        self.classifier = VisionTransformerClassifier(self.config)
        
    def test_vit_initialization(self):
        """测试Vision Transformer初始化"""
        assert self.classifier is not None
        assert hasattr(self.classifier, 'build_model')
        
    def test_vit_patch_embedding(self):
        """测试Vision Transformer patch嵌入"""
        # 创建测试数据
        test_input = np.random.random((1, 11, 11, 200))
        
        # 获取patch嵌入
        patch_embeddings = self.classifier.create_patch_embeddings(test_input)
        
        # 验证patch嵌入
        assert patch_embeddings is not None
        # 验证嵌入维度符合d_model
        assert patch_embeddings.shape[-1] == self.config['d_model']
        
    def test_vit_attention_mechanism(self):
        """测试Vision Transformer注意力机制"""
        # 构建模型
        model = self.classifier.build_model()
        
        # 验证模型包含多头注意力层
        layer_types = [type(layer).__name__ for layer in model.layers]
        has_attention = any('Attention' in layer_type or 'MultiHead' in layer_type 
                          for layer_type in layer_types)
        assert has_attention
        
    def test_vit_positional_encoding(self):
        """测试Vision Transformer位置编码"""
        # 获取位置编码
        seq_length = 121  # 11*11 spatial patches
        pos_encoding = self.classifier.get_positional_encoding(
            seq_length, self.config['d_model']
        )
        
        # 验证位置编码
        assert pos_encoding.shape == (seq_length, self.config['d_model'])


class TestEnsembleMethods:
    """集成学习方法测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.base_classifiers = {
            'svm': SVMClassifier({'kernel': 'rbf', 'C': 1.0}),
            'rf': RandomForestClassifier({'n_estimators': 50}),
            'xgb': XGBoostClassifier({'n_estimators': 50})
        }
        
    def test_voting_ensemble(self, sample_training_samples):
        """测试投票集成"""
        config = {
            'voting': 'soft',  # 软投票
            'weights': [1, 1, 1]
        }
        
        ensemble = VotingEnsemble(self.base_classifiers, config)
        
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        X_test = np.random.random((50, n_features))
        
        # 训练集成模型
        ensemble.train(X_train, y_train)
        
        # 预测
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)
        
        # 验证结果
        assert predictions.shape == (50,)
        assert probabilities.shape == (50, len(np.unique(y_train)))
        
    def test_stacking_ensemble(self, sample_training_samples):
        """测试堆叠集成"""
        config = {
            'meta_classifier': 'logistic_regression',
            'cv_folds': 3
        }
        
        ensemble = StackingEnsemble(self.base_classifiers, config)
        
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        
        # 训练堆叠模型
        ensemble.train(X_train, y_train)
        
        # 验证元分类器被训练
        assert hasattr(ensemble, 'meta_classifier')
        assert ensemble.meta_classifier is not None
        
    def test_bagging_ensemble(self, sample_training_samples):
        """测试Bagging集成"""
        config = {
            'n_estimators': 5,
            'max_samples': 0.8,
            'max_features': 0.8,
            'bootstrap': True
        }
        
        # 使用单一分类器类型进行Bagging
        base_classifier = RandomForestClassifier({'n_estimators': 10})
        ensemble = BaggingEnsemble(base_classifier, config)
        
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        
        # 训练Bagging集成
        ensemble.train(X_train, y_train)
        
        # 验证集成分类器数量
        assert len(ensemble.estimators) == config['n_estimators']
        
    def test_ensemble_feature_importance(self, sample_training_samples):
        """测试集成模型特征重要性"""
        ensemble = VotingEnsemble(self.base_classifiers, {'voting': 'soft'})
        
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        
        # 训练集成模型
        ensemble.train(X_train, y_train)
        
        # 获取集成特征重要性
        ensemble_importance = ensemble.get_feature_importance(X_train, y_train)
        
        # 验证特征重要性
        assert len(ensemble_importance) == n_features
        assert np.all(ensemble_importance >= 0)


class TestClassificationMetrics:
    """分类评估指标测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.metrics = ClassificationMetrics()
        
    def test_basic_metrics_calculation(self):
        """测试基础指标计算"""
        # 模拟预测结果
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 1])
        
        # 计算各种指标
        accuracy = self.metrics.calculate_accuracy(y_true, y_pred)
        precision = self.metrics.calculate_precision(y_true, y_pred, average='weighted')
        recall = self.metrics.calculate_recall(y_true, y_pred, average='weighted')
        f1_score = self.metrics.calculate_f1_score(y_true, y_pred, average='weighted')
        
        # 验证指标范围
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1_score <= 1
        
    def test_confusion_matrix(self):
        """测试混淆矩阵"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
        
        # 计算混淆矩阵
        cm = self.metrics.calculate_confusion_matrix(y_true, y_pred)
        
        # 验证混淆矩阵
        assert cm.shape == (3, 3)  # 3个类别
        assert np.sum(cm) == len(y_true)  # 总和等于样本数
        
    def test_kappa_coefficient(self):
        """测试Kappa系数"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 1])
        
        # 计算Kappa系数
        kappa = self.metrics.calculate_kappa(y_true, y_pred)
        
        # 验证Kappa系数范围
        assert -1 <= kappa <= 1
        
    def test_class_specific_metrics(self):
        """测试类别特定指标"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 1])
        
        # 计算每个类别的指标
        class_metrics = self.metrics.calculate_class_metrics(y_true, y_pred)
        
        # 验证类别指标
        assert 'precision' in class_metrics
        assert 'recall' in class_metrics
        assert 'f1_score' in class_metrics
        assert 'support' in class_metrics
        
        # 验证每个类别都有指标
        n_classes = len(np.unique(y_true))
        assert len(class_metrics['precision']) == n_classes
        
    def test_probabilistic_metrics(self):
        """测试概率预测指标"""
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.3, 0.6, 0.1]
        ])
        
        # 计算对数损失
        log_loss = self.metrics.calculate_log_loss(y_true, y_proba)
        
        # 计算ROC AUC (多类别)
        auc_scores = self.metrics.calculate_multiclass_auc(y_true, y_proba)
        
        # 验证概率指标
        assert log_loss >= 0
        assert len(auc_scores) == 3  # 每个类别一个AUC分数
        assert np.all(auc_scores >= 0) and np.all(auc_scores <= 1)


class TestClassificationIntegration:
    """分类模块集成测试"""
    
    def test_complete_classification_pipeline(self, sample_hyperspectral_data, sample_training_samples):
        """测试完整的分类流水线"""
        # 1. 数据准备
        n_features = 100
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 2. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 3. 训练多个分类器
        classifiers = {
            'svm': SVMClassifier({'kernel': 'rbf', 'C': 1.0}),
            'rf': RandomForestClassifier({'n_estimators': 50}),
            'xgb': XGBoostClassifier({'n_estimators': 50})
        }
        
        results = {}
        for name, classifier in classifiers.items():
            # 训练分类器
            classifier.train(X_train, y_train)
            
            # 预测
            y_pred = classifier.predict(X_test)
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {'accuracy': accuracy, 'predictions': y_pred}
            
        # 4. 验证所有分类器都有合理的性能
        for name, result in results.items():
            assert 0 <= result['accuracy'] <= 1
            assert len(result['predictions']) == len(y_test)
            
    def test_model_persistence(self, sample_training_samples, temp_test_dir):
        """测试模型持久化"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        classifier = RandomForestClassifier({'n_estimators': 20})
        classifier.train(X_train, y_train)
        
        # 保存模型
        model_path = os.path.join(temp_test_dir, 'test_model.pkl')
        classifier.save_model(model_path)
        
        # 加载模型
        loaded_classifier = RandomForestClassifier.load_model(model_path)
        
        # 验证加载的模型
        assert loaded_classifier is not None
        
        # 验证预测一致性
        X_test = np.random.random((10, n_features))
        original_pred = classifier.predict(X_test)
        loaded_pred = loaded_classifier.predict(X_test)
        
        assert np.array_equal(original_pred, loaded_pred)
        
    def test_batch_prediction(self, sample_hyperspectral_data, sample_training_samples):
        """测试批量预测"""
        # 准备训练数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X_train = np.random.random((n_samples, n_features))
        y_train = np.array(sample_training_samples['labels'])
        
        # 训练分类器
        classifier = RandomForestClassifier({'n_estimators': 20})
        classifier.train(X_train, y_train)
        
        # 准备大批量测试数据
        large_X = np.random.random((10000, n_features))
        
        # 执行批量预测
        batch_predictions = classifier.predict_in_batches(large_X, batch_size=1000)
        
        # 验证批量预测结果
        assert len(batch_predictions) == 10000
        assert np.all(batch_predictions >= 0)
        assert np.all(batch_predictions < len(np.unique(y_train)))
        
    def test_cross_validation_comparison(self, sample_training_samples):
        """测试交叉验证比较"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 定义分类器
        classifiers = {
            'SVM': SVMClassifier({'kernel': 'rbf', 'C': 1.0}),
            'RF': RandomForestClassifier({'n_estimators': 20}),
            'XGB': XGBoostClassifier({'n_estimators': 20})
        }
        
        # 执行交叉验证比较
        cv_results = {}
        for name, classifier in classifiers.items():
            cv_scores = classifier.cross_validate(X, y, cv=3)
            cv_results[name] = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'scores': cv_scores
            }
            
        # 验证交叉验证结果
        for name, result in cv_results.items():
            assert 0 <= result['mean_score'] <= 1
            assert result['std_score'] >= 0
            assert len(result['scores']) == 3
            
    def test_classification_report_generation(self, sample_training_samples):
        """测试分类报告生成"""
        # 准备数据
        n_features = 50
        n_samples = len(sample_training_samples['coordinates'])
        X = np.random.random((n_samples, n_features))
        y = np.array(sample_training_samples['labels'])
        
        # 训练和预测
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        classifier = RandomForestClassifier({'n_estimators': 20})
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # 生成分类报告
        metrics = ClassificationMetrics()
        report = metrics.generate_classification_report(
            y_test, y_pred, 
            target_names=[f'湿地类型_{i}' for i in range(len(np.unique(y)))]
        )
        
        # 验证分类报告
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report
        
        # 验证每个类别的指标
        for class_name in [f'湿地类型_{i}' for i in range(len(np.unique(y)))]:
            if class_name in report:
                assert 'precision' in report[class_name]
                assert 'recall' in report[class_name]
                assert 'f1-score' in report[class_name]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])