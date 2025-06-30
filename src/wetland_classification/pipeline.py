#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主处理流水线
Main Processing Pipeline

集成数据处理、特征提取、分类和后处理的完整流水线

作者: Wetland Research Team
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .utils.logger import Logger
from .utils.io_utils import IOUtils
from .utils.visualization import VisualizationUtils

logger = logging.getLogger(__name__)


@dataclass
class PipelineResults:
    """流水线执行结果"""
    success: bool = False
    message: str = ""
    processing_time: float = 0.0
    
    # 数据信息
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_samples: int = 0
    
    # 分类结果
    classification_map: Optional[np.ndarray] = None
    probability_maps: Optional[np.ndarray] = None
    confidence_map: Optional[np.ndarray] = None
    
    # 精度评估
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # 景观分析
    landscape_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 文件路径
    output_files: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'success': self.success,
            'message': self.message,
            'processing_time': self.processing_time,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'num_samples': self.num_samples,
            'accuracy_metrics': self.accuracy_metrics,
            'landscape_metrics': self.landscape_metrics,
            'output_files': self.output_files
        }


class Pipeline:
    """湿地高光谱分类主处理流水线
    
    集成完整的数据处理流程：
    1. 数据加载和验证
    2. 预处理 (辐射校正、大气校正、几何校正、噪声去除)
    3. 特征提取 (光谱特征、植被指数、纹理特征、空间特征)
    4. 分类预测 (传统ML、深度学习、集成学习)
    5. 后处理 (空间滤波、形态学操作、一致性检查)
    6. 景观分析 (景观指数、连通性分析)
    7. 精度评估和结果输出
    """
    
    def __init__(self, config: Union[Config, str, Path]):
        """初始化流水线
        
        Args:
            config: 配置对象或配置文件路径
        """
        # 加载配置
        if isinstance(config, (str, Path)):
            self.config = Config.from_file(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError("config must be Config object or file path")
        
        # 初始化组件
        self._init_components()
        
        # 设置输出目录
        self._setup_output_dir()
        
        # 初始化日志
        self._setup_logging()
        
        # 验证环境
        self._validate_environment()
        
        logger.info("Pipeline initialized successfully")
    
    def run(self,
            input_path: Union[str, Path],
            ground_truth: Optional[Union[str, Path]] = None,
            output_dir: Optional[Union[str, Path]] = None,
            model_type: Optional[str] = None,
            **kwargs) -> PipelineResults:
        """执行完整的分类流水线
        
        Args:
            input_path: 输入高光谱数据路径
            ground_truth: 地面真值数据路径 (可选)
            output_dir: 输出目录 (可选)
            model_type: 模型类型 (可选)
            **kwargs: 其他参数
            
        Returns:
            PipelineResults: 处理结果
        """
        start_time = time.time()
        results = PipelineResults()
        
        try:
            logger.info("Starting classification pipeline")
            logger.info(f"Input: {input_path}")
            logger.info(f"Ground truth: {ground_truth}")
            logger.info(f"Model type: {model_type or self.config.get('classification.default_classifier')}")
            
            # 1. 数据加载和验证
            logger.info("Step 1: Loading and validating data")
            hyperspectral_data, metadata = self._load_and_validate_data(input_path)
            results.input_shape = hyperspectral_data.shape
            logger.info(f"Data shape: {hyperspectral_data.shape}")
            
            # 加载训练样本
            train_data, val_data, test_data = None, None, None
            if ground_truth:
                logger.info("Loading ground truth data")
                train_data, val_data, test_data = self._load_ground_truth(ground_truth, hyperspectral_data)
                results.num_samples = len(train_data[0]) if train_data else 0
                logger.info(f"Training samples: {results.num_samples}")
            
            # 2. 预处理
            logger.info("Step 2: Preprocessing")
            preprocessed_data = self._preprocess_data(hyperspectral_data, metadata)
            logger.info("Preprocessing completed")
            
            # 3. 特征提取
            logger.info("Step 3: Feature extraction")
            features = self._extract_features(preprocessed_data)
            logger.info(f"Features shape: {features.shape}")
            
            # 4. 分类预测
            logger.info("Step 4: Classification")
            classifier = self._get_classifier(model_type)
            
            if train_data is not None:
                # 训练模式
                logger.info("Training classifier")
                classifier = self._train_classifier(classifier, train_data, val_data, features)
                
                # 预测
                logger.info("Predicting classification")
                classification_map, probability_maps = self._predict_classification(
                    classifier, features
                )
                
                # 精度评估
                if test_data is not None:
                    logger.info("Evaluating accuracy")
                    results.accuracy_metrics, results.confusion_matrix, results.classification_report = \
                        self._evaluate_accuracy(classifier, test_data, features)
            else:
                # 预测模式 (使用预训练模型)
                logger.info("Loading pretrained model")
                classifier = self._load_pretrained_model(classifier)
                
                logger.info("Predicting classification")
                classification_map, probability_maps = self._predict_classification(
                    classifier, features
                )
            
            results.classification_map = classification_map
            results.probability_maps = probability_maps
            results.output_shape = classification_map.shape
            
            # 5. 后处理
            logger.info("Step 5: Post-processing")
            processed_map = self._postprocess_results(classification_map)
            results.classification_map = processed_map
            
            # 计算置信度
            if probability_maps is not None:
                results.confidence_map = self._calculate_confidence(probability_maps)
            
            # 6. 景观分析
            logger.info("Step 6: Landscape analysis")
            results.landscape_metrics = self._analyze_landscape(processed_map)
            
            # 7. 结果输出
            logger.info("Step 7: Saving results")
            output_dir = output_dir or self.output_dir
            results.output_files = self._save_results(
                processed_map, probability_maps, results.confidence_map,
                metadata, results.accuracy_metrics, results.landscape_metrics,
                output_dir
            )
            
            # 8. 可视化
            logger.info("Step 8: Generating visualizations")
            self._generate_visualizations(
                hyperspectral_data, processed_map, probability_maps,
                results.accuracy_metrics, results.landscape_metrics,
                output_dir
            )
            
            # 计算处理时间
            results.processing_time = time.time() - start_time
            results.success = True
            results.message = "Pipeline completed successfully"
            
            logger.info(f"Pipeline completed in {results.processing_time:.2f} seconds")
            self._print_summary(results)
            
        except Exception as e:
            results.processing_time = time.time() - start_time
            results.success = False
            results.message = f"Pipeline failed: {str(e)}"
            
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        return results
    
    def _init_components(self):
        """初始化组件"""
        from .data import DataLoader, DataValidator, DataAugmentation
        from .preprocessing import (
            RadiometricCorrector, AtmosphericCorrector,
            GeometricCorrector, NoiseReducer
        )
        from .features import (
            SpectralFeatureExtractor, VegetationIndexCalculator,
            TextureFeatureExtractor, SpatialFeatureExtractor
        )
        from .classification import (
            TraditionalClassifier, DeepLearningClassifier, EnsembleClassifier
        )
        from .postprocessing import (
            SpatialFilter, MorphologyProcessor, ConsistencyChecker
        )
        from .landscape import LandscapeMetrics, ConnectivityAnalyzer
        from .evaluation import MetricsCalculator, CrossValidator, UncertaintyAnalyzer
        
        # 数据组件
        self.data_loader = DataLoader(self.config)
        self.data_validator = DataValidator(self.config)
        self.data_augmentation = DataAugmentation(self.config)
        
        # 预处理组件
        self.radiometric_corrector = RadiometricCorrector(self.config)
        self.atmospheric_corrector = AtmosphericCorrector(self.config)
        self.geometric_corrector = GeometricCorrector(self.config)
        self.noise_reducer = NoiseReducer(self.config)
        
        # 特征提取组件
        self.spectral_extractor = SpectralFeatureExtractor(self.config)
        self.vegetation_calculator = VegetationIndexCalculator(self.config)
        self.texture_extractor = TextureFeatureExtractor(self.config)
        self.spatial_extractor = SpatialFeatureExtractor(self.config)
        
        # 分类组件
        self.traditional_classifier = TraditionalClassifier(self.config)
        self.deep_learning_classifier = DeepLearningClassifier(self.config)
        self.ensemble_classifier = EnsembleClassifier(self.config)
        
        # 后处理组件
        self.spatial_filter = SpatialFilter(self.config)
        self.morphology_processor = MorphologyProcessor(self.config)
        self.consistency_checker = ConsistencyChecker(self.config)
        
        # 景观分析组件
        self.landscape_metrics = LandscapeMetrics(self.config)
        self.connectivity_analyzer = ConnectivityAnalyzer(self.config)
        
        # 评估组件
        self.metrics_calculator = MetricsCalculator(self.config)
        self.cross_validator = CrossValidator(self.config)
        self.uncertainty_analyzer = UncertaintyAnalyzer(self.config)
        
        # 工具组件
        self.io_utils = IOUtils(self.config)
        self.viz_utils = VisualizationUtils(self.config)
    
    def _setup_output_dir(self):
        """设置输出目录"""
        base_dir = self.config.get('output.base_dir', 'output/')
        
        if self.config.get('output.create_timestamp_dir', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(base_dir) / f"classification_{timestamp}"
        else:
            self.output_dir = Path(base_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "pipeline.log"
        Logger.setup_logging(
            level=self.config.get('logging.level', 'INFO'),
            log_file=str(log_file)
        )
    
    def _validate_environment(self):
        """验证运行环境"""
        # 验证路径
        self.config.validate_paths()
        
        # 检查依赖
        try:
            import torch
            import sklearn
            import rasterio
            logger.info("Required dependencies available")
        except ImportError as e:
            logger.warning(f"Missing dependency: {e}")
    
    def _load_and_validate_data(self, input_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载和验证数据"""
        # 加载数据
        hyperspectral_data, metadata = self.data_loader.load_hyperspectral(input_path)
        
        # 验证数据
        is_valid, issues = self.data_validator.validate_hyperspectral(hyperspectral_data, metadata)
        
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")
        
        return hyperspectral_data, metadata
    
    def _load_ground_truth(self, ground_truth_path: Union[str, Path], 
                          hyperspectral_data: np.ndarray) -> Tuple[Tuple, Tuple, Tuple]:
        """加载地面真值数据"""
        # 加载样本数据
        samples = self.data_loader.load_samples(ground_truth_path)
        
        # 提取光谱和标签
        spectra, labels = self.data_loader.extract_spectra_and_labels(
            hyperspectral_data, samples
        )
        
        # 数据分割
        train_data, val_data, test_data = self.data_loader.split_data(
            spectra, labels, 
            train_ratio=self.config.get('data.samples.train_ratio', 0.7),
            val_ratio=self.config.get('data.samples.val_ratio', 0.15),
            test_ratio=self.config.get('data.samples.test_ratio', 0.15),
            stratify=self.config.get('data.samples.stratify', True)
        )
        
        return train_data, val_data, test_data
    
    def _preprocess_data(self, hyperspectral_data: np.ndarray, 
                        metadata: Dict[str, Any]) -> np.ndarray:
        """预处理数据"""
        processed_data = hyperspectral_data.copy()
        
        # 辐射校正
        if self.config.get('preprocessing.radiometric.enabled', True):
            processed_data = self.radiometric_corrector.correct(processed_data, metadata)
        
        # 大气校正
        if self.config.get('preprocessing.atmospheric.enabled', True):
            processed_data = self.atmospheric_corrector.correct(processed_data, metadata)
        
        # 几何校正
        if self.config.get('preprocessing.geometric.enabled', True):
            processed_data = self.geometric_corrector.correct(processed_data, metadata)
        
        # 噪声去除
        if self.config.get('preprocessing.noise_reduction.enabled', True):
            processed_data = self.noise_reducer.reduce_noise(processed_data)
        
        return processed_data
    
    def _extract_features(self, hyperspectral_data: np.ndarray) -> np.ndarray:
        """提取特征"""
        features_list = []
        
        # 光谱特征
        if self.config.get('features.spectral.enabled', True):
            spectral_features = self.spectral_extractor.extract(hyperspectral_data)
            features_list.append(spectral_features)
        
        # 植被指数
        if self.config.get('features.vegetation_indices.enabled', True):
            vegetation_indices = self.vegetation_calculator.calculate(hyperspectral_data)
            features_list.append(vegetation_indices)
        
        # 纹理特征
        if self.config.get('features.texture.enabled', True):
            texture_features = self.texture_extractor.extract(hyperspectral_data)
            features_list.append(texture_features)
        
        # 空间特征
        if self.config.get('features.spatial.enabled', True):
            spatial_features = self.spatial_extractor.extract(hyperspectral_data)
            features_list.append(spatial_features)
        
        # 合并特征
        if features_list:
            features = np.concatenate(features_list, axis=-1)
        else:
            features = hyperspectral_data
        
        return features
    
    def _get_classifier(self, model_type: Optional[str]):
        """获取分类器"""
        model_type = model_type or self.config.get('classification.default_classifier')
        
        if model_type in ['svm', 'random_forest', 'xgboost', 'knn']:
            return self.traditional_classifier
        elif model_type in ['cnn_3d', 'hybrid_cnn', 'vision_transformer', 'resnet_hs']:
            return self.deep_learning_classifier
        elif model_type in ['voting_ensemble', 'stacking_ensemble', 'weighted_ensemble']:
            return self.ensemble_classifier
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _train_classifier(self, classifier, train_data: Tuple, 
                         val_data: Tuple, features: np.ndarray):
        """训练分类器"""
        return classifier.train(train_data, val_data, features)
    
    def _load_pretrained_model(self, classifier):
        """加载预训练模型"""
        return classifier.load_pretrained()
    
    def _predict_classification(self, classifier, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分类预测"""
        return classifier.predict(features)
    
    def _evaluate_accuracy(self, classifier, test_data: Tuple, 
                          features: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
        """精度评估"""
        return self.metrics_calculator.evaluate(classifier, test_data, features)
    
    def _postprocess_results(self, classification_map: np.ndarray) -> np.ndarray:
        """后处理结果"""
        processed_map = classification_map.copy()
        
        # 空间滤波
        if self.config.get('postprocessing.spatial_filter.enabled', True):
            processed_map = self.spatial_filter.filter(processed_map)
        
        # 形态学操作
        if self.config.get('postprocessing.morphology.enabled', True):
            processed_map = self.morphology_processor.process(processed_map)
        
        # 一致性检查
        if self.config.get('postprocessing.consistency.enabled', True):
            processed_map = self.consistency_checker.check(processed_map)
        
        return processed_map
    
    def _calculate_confidence(self, probability_maps: np.ndarray) -> np.ndarray:
        """计算置信度"""
        # 使用最大概率作为置信度
        confidence_map = np.max(probability_maps, axis=-1)
        return confidence_map
    
    def _analyze_landscape(self, classification_map: np.ndarray) -> Dict[str, float]:
        """景观分析"""
        metrics = {}
        
        # 景观指数
        if self.config.get('landscape.metrics.enabled', True):
            landscape_metrics = self.landscape_metrics.calculate(classification_map)
            metrics.update(landscape_metrics)
        
        # 连通性分析
        if self.config.get('landscape.connectivity.enabled', True):
            connectivity_metrics = self.connectivity_analyzer.analyze(classification_map)
            metrics.update(connectivity_metrics)
        
        return metrics
    
    def _save_results(self, classification_map: np.ndarray, 
                     probability_maps: Optional[np.ndarray],
                     confidence_map: Optional[np.ndarray],
                     metadata: Dict[str, Any],
                     accuracy_metrics: Dict[str, float],
                     landscape_metrics: Dict[str, float],
                     output_dir: Path) -> Dict[str, str]:
        """保存结果"""
        output_files = {}
        
        # 保存分类结果
        classification_file = output_dir / "classification_map.tif"
        self.io_utils.save_raster(classification_map, classification_file, metadata)
        output_files['classification_map'] = str(classification_file)
        
        # 保存概率图
        if probability_maps is not None:
            probability_file = output_dir / "probability_maps.tif"
            self.io_utils.save_raster(probability_maps, probability_file, metadata)
            output_files['probability_maps'] = str(probability_file)
        
        # 保存置信度图
        if confidence_map is not None:
            confidence_file = output_dir / "confidence_map.tif"
            self.io_utils.save_raster(confidence_map, confidence_file, metadata)
            output_files['confidence_map'] = str(confidence_file)
        
        # 保存统计报告
        report = {
            'accuracy_metrics': accuracy_metrics,
            'landscape_metrics': landscape_metrics,
            'processing_config': self.config.to_dict()
        }
        
        report_file = output_dir / "classification_report.json"
        self.io_utils.save_json(report, report_file)
        output_files['report'] = str(report_file)
        
        return output_files
    
    def _generate_visualizations(self, hyperspectral_data: np.ndarray,
                               classification_map: np.ndarray,
                               probability_maps: Optional[np.ndarray],
                               accuracy_metrics: Dict[str, float],
                               landscape_metrics: Dict[str, float],
                               output_dir: Path):
        """生成可视化"""
        # 分类结果可视化
        self.viz_utils.plot_classification_map(
            classification_map, 
            save_path=output_dir / "classification_visualization.png"
        )
        
        # RGB复合图
        rgb_image = self.viz_utils.create_rgb_composite(hyperspectral_data)
        self.viz_utils.save_image(rgb_image, output_dir / "rgb_composite.png")
        
        # 精度评估图表
        if accuracy_metrics:
            self.viz_utils.plot_accuracy_metrics(
                accuracy_metrics,
                save_path=output_dir / "accuracy_metrics.png"
            )
        
        # 景观指数图表
        if landscape_metrics:
            self.viz_utils.plot_landscape_metrics(
                landscape_metrics,
                save_path=output_dir / "landscape_metrics.png"
            )
        
        # 概率分布图
        if probability_maps is not None:
            self.viz_utils.plot_probability_distribution(
                probability_maps,
                save_path=output_dir / "probability_distribution.png"
            )
    
    def _print_summary(self, results: PipelineResults):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("🌿 湿地高光谱分类结果摘要")
        print("="*60)
        
        print(f"📊 处理状态: {'✅ 成功' if results.success else '❌ 失败'}")
        print(f"⏱️  处理时间: {results.processing_time:.2f} 秒")
        
        if results.input_shape:
            print(f"📥 输入尺寸: {results.input_shape}")
        if results.output_shape:
            print(f"📤 输出尺寸: {results.output_shape}")
        if results.num_samples > 0:
            print(f"🎯 训练样本: {results.num_samples}")
        
        if results.accuracy_metrics:
            print("\n📈 精度指标:")
            for metric, value in results.accuracy_metrics.items():
                print(f"  {metric}: {value:.3f}")
        
        if results.landscape_metrics:
            print("\n🌍 景观指标:")
            for metric, value in list(results.landscape_metrics.items())[:5]:  # 显示前5个
                print(f"  {metric}: {value:.3f}")
        
        if results.output_files:
            print("\n📁 输出文件:")
            for file_type, file_path in results.output_files.items():
                print(f"  {file_type}: {file_path}")
        
        print("="*60)