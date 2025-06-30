# 湿地高光谱分类系统API文档
## Wetland Hyperspectral Classification System - API Reference

### 📋 目录

1. [核心模块](#核心模块)
2. [数据模块](#数据模块)
3. [预处理模块](#预处理模块)
4. [特征提取模块](#特征提取模块)
5. [分类模块](#分类模块)
6. [后处理模块](#后处理模块)
7. [景观分析模块](#景观分析模块)
8. [评估模块](#评估模块)
9. [工具模块](#工具模块)
10. [配置模块](#配置模块)

---

## 🎯 核心模块

### Pipeline

完整的处理流水线，提供一站式的高光谱数据分类解决方案。

```python
class Pipeline:
    """主处理流水线
    
    集成了数据加载、预处理、特征提取、分类和评估的完整流程。
    """
    
    def __init__(self, config: Config):
        """
        初始化流水线
        
        Args:
            config (Config): 配置对象
        """
        pass
    
    def run(self, 
            input_data: str, 
            ground_truth: Optional[str] = None,
            output_dir: str = 'output/') -> Dict[str, Any]:
        """
        运行完整的分类流程
        
        Args:
            input_data: 输入高光谱数据路径
            ground_truth: 可选的地面真实数据路径
            output_dir: 输出目录
            
        Returns:
            包含分类结果和评估指标的字典
            
        Example:
            >>> pipeline = Pipeline(config)
            >>> results = pipeline.run('data/scene.tif', 'data/labels.shp')
            >>> print(f"Accuracy: {results['accuracy']:.3f}")
        """
        pass
    
    def run_demo(self, demo_path: str = 'data/samples/demo_scene/') -> Dict[str, Any]:
        """
        运行演示流程
        
        Args:
            demo_path: 演示数据路径
            
        Returns:
            演示结果字典
        """
        pass
    
    @classmethod
    def from_config(cls, config_path: str) -> 'Pipeline':
        """
        从配置文件创建流水线
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置好的流水线实例
        """
        pass
```

---

## 📁 数据模块

### DataLoader

数据加载器，支持多种格式的高光谱数据和地面真实数据。

```python
class DataLoader:
    """数据加载器
    
    支持GeoTIFF, ENVI, HDF5等多种格式的高光谱数据加载，
    以及Shapefile, GeoJSON等地面真实数据格式。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化数据加载器
        
        Args:
            config: 可选的配置对象
        """
        pass
    
    def load_hyperspectral(self, path: str) -> Tuple[np.ndarray, Dict]:
        """
        加载高光谱数据
        
        Args:
            path: 数据文件路径
            
        Returns:
            (data, metadata): 数据数组和元数据字典
            data shape: (height, width, bands)
            
        Raises:
            DataLoadError: 当数据加载失败时
            
        Example:
            >>> loader = DataLoader()
            >>> data, meta = loader.load_hyperspectral('scene.tif')
            >>> print(f"Data shape: {data.shape}")
        """
        pass
    
    def load_ground_truth(self, path: str) -> np.ndarray:
        """
        加载地面真实标签
        
        Args:
            path: 标签文件路径（.shp, .geojson, .csv）
            
        Returns:
            标签数组，shape: (height, width)
            
        Example:
            >>> labels = loader.load_ground_truth('labels.shp')
            >>> unique_classes = np.unique(labels[labels > 0])
        """
        pass
    
    def load_scene_data(self, scene_dir: str) -> Dict[str, Any]:
        """
        加载完整场景数据
        
        Args:
            scene_dir: 场景数据目录
            
        Returns:
            包含高光谱数据、标签、元数据的字典
        """
        pass
```

### DataValidator

数据验证器，确保数据质量和格式正确性。

```python
class DataValidator:
    """数据验证器"""
    
    def validate_hyperspectral(self, data: np.ndarray) -> ValidationResult:
        """
        验证高光谱数据
        
        Args:
            data: 高光谱数据数组
            
        Returns:
            验证结果对象
            
        Example:
            >>> validator = DataValidator()
            >>> result = validator.validate_hyperspectral(data)
            >>> if not result.is_valid:
            ...     print(result.errors)
        """
        pass
    
    def validate_labels(self, labels: np.ndarray) -> ValidationResult:
        """
        验证标签数据
        
        Args:
            labels: 标签数组
            
        Returns:
            验证结果对象
        """
        pass
```

---

## 🔧 预处理模块

### Preprocessor

数据预处理器，提供辐射定标、大气校正、几何校正等功能。

```python
class Preprocessor:
    """数据预处理器
    
    提供完整的高光谱数据预处理功能，包括辐射定标、
    大气校正、几何校正和噪声去除。
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置
        """
        pass
    
    def process_all(self, data: np.ndarray, 
                   steps: List[str] = None) -> np.ndarray:
        """
        执行完整预处理流程
        
        Args:
            data: 原始高光谱数据
            steps: 预处理步骤列表，默认为全部步骤
            
        Returns:
            预处理后的数据
            
        Example:
            >>> preprocessor = Preprocessor()
            >>> processed = preprocessor.process_all(
            ...     raw_data, 
            ...     steps=['radiometric', 'atmospheric', 'geometric']
            ... )
        """
        pass
    
    def radiometric_calibration(self, data: np.ndarray) -> np.ndarray:
        """
        辐射定标
        
        将DN值转换为辐射亮度值
        
        Args:
            data: 原始DN值数据
            
        Returns:
            辐射定标后的数据
        """
        pass
    
    def atmospheric_correction(self, data: np.ndarray, 
                             method: str = 'FLAASH') -> np.ndarray:
        """
        大气校正
        
        Args:
            data: 辐射定标后的数据
            method: 大气校正方法 ('FLAASH', 'QUAC', 'ATCOR')
            
        Returns:
            大气校正后的反射率数据
        """
        pass
    
    def geometric_correction(self, data: np.ndarray) -> np.ndarray:
        """
        几何校正
        
        Args:
            data: 待校正的数据
            
        Returns:
            几何校正后的数据
        """
        pass
    
    def noise_reduction(self, data: np.ndarray, 
                       method: str = 'MNF') -> np.ndarray:
        """
        噪声去除
        
        Args:
            data: 待去噪的数据
            method: 去噪方法 ('MNF', 'PCA', 'Wavelet')
            
        Returns:
            去噪后的数据
        """
        pass
```

---

## 🎨 特征提取模块

### FeatureExtractor

特征提取器，支持光谱特征、植被指数、纹理特征等多种特征。

```python
class FeatureExtractor:
    """特征提取器
    
    提供光谱特征、植被指数、纹理特征、空间特征等
    多种特征提取功能。
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征提取器
        
        Args:
            config: 特征提取配置
        """
        pass
    
    def extract_all(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取所有特征
        
        Args:
            data: 预处理后的高光谱数据
            
        Returns:
            特征字典，键为特征类型，值为特征数组
            
        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_all(processed_data)
            >>> print(list(features.keys()))
            ['spectral', 'vegetation_indices', 'texture', 'spatial']
        """
        pass
    
    def extract_spectral_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取光谱特征
        
        Args:
            data: 高光谱数据 (H, W, Bands)
            
        Returns:
            光谱特征数组 (H, W, Features)
            
        Example:
            >>> spectral_features = extractor.extract_spectral_features(data)
            >>> print(f"Spectral features shape: {spectral_features.shape}")
        """
        pass
    
    def extract_vegetation_indices(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取植被指数
        
        Args:
            data: 高光谱数据
            
        Returns:
            植被指数字典
            
        Available indices:
            - NDVI: 归一化差异植被指数
            - NDWI: 归一化差异水分指数
            - EVI: 增强植被指数
            - SAVI: 土壤调节植被指数
            - PRI: 光化学反射指数
            - GNDVI: 绿度归一化差异植被指数
            
        Example:
            >>> indices = extractor.extract_vegetation_indices(data)
            >>> ndvi = indices['NDVI']
            >>> ndwi = indices['NDWI']
        """
        pass
    
    def extract_texture_features(self, data: np.ndarray, 
                               window_size: int = 5) -> np.ndarray:
        """
        提取纹理特征
        
        Args:
            data: 输入数据
            window_size: 纹理计算窗口大小
            
        Returns:
            纹理特征数组
            
        Features:
            - 对比度 (Contrast)
            - 相关性 (Correlation) 
            - 能量 (Energy)
            - 熵 (Entropy)
            - 均匀性 (Homogeneity)
            - 方差 (Variance)
        """
        pass
    
    def extract_spatial_features(self, data: np.ndarray) -> np.ndarray:
        """
        提取空间特征
        
        Args:
            data: 输入数据
            
        Returns:
            空间特征数组
            
        Features:
            - 边缘特征
            - 形态学特征
            - 局部二值模式
        """
        pass
```

### SpectralIndices

专门的光谱指数计算类。

```python
class SpectralIndices:
    """光谱指数计算器"""
    
    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        计算NDVI
        
        Args:
            nir: 近红外波段
            red: 红光波段
            
        Returns:
            NDVI数组
            
        Formula:
            NDVI = (NIR - Red) / (NIR + Red)
        """
        return (nir - red) / (nir + red + 1e-8)
    
    @staticmethod
    def ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """
        计算NDWI
        
        Args:
            nir: 近红外波段
            swir: 短波红外波段
            
        Returns:
            NDWI数组
            
        Formula:
            NDWI = (NIR - SWIR) / (NIR + SWIR)
        """
        return (nir - swir) / (nir + swir + 1e-8)
    
    @staticmethod
    def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        计算EVI
        
        Args:
            nir: 近红外波段
            red: 红光波段
            blue: 蓝光波段
            
        Returns:
            EVI数组
            
        Formula:
            EVI = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
        """
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
```

---

## 🤖 分类模块

### Classifier

通用分类器接口，支持多种机器学习和深度学习模型。

```python
class Classifier:
    """通用分类器
    
    支持SVM、Random Forest、XGBoost、3D-CNN、HybridSN等多种分类算法。
    """
    
    def __init__(self, model_type: str, **kwargs):
        """
        初始化分类器
        
        Args:
            model_type: 模型类型
                - 'svm': 支持向量机
                - 'random_forest': 随机森林
                - 'xgboost': 极端梯度提升
                - 'cnn_3d': 3D卷积神经网络
                - 'hybrid_cnn': 混合卷积网络
                - 'transformer': Vision Transformer
            **kwargs: 模型特定参数
            
        Example:
            >>> # SVM分类器
            >>> svm_clf = Classifier('svm', C=1.0, kernel='rbf')
            >>> 
            >>> # 深度学习分类器
            >>> cnn_clf = Classifier('cnn_3d', learning_rate=0.001, epochs=100)
        """
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'Classifier':
        """
        训练分类器
        
        Args:
            X: 训练特征 (N_samples, N_features) 或 (N_samples, H, W, Bands)
            y: 训练标签 (N_samples,)
            X_val: 可选的验证特征
            y_val: 可选的验证标签
            
        Returns:
            训练好的分类器实例
            
        Example:
            >>> classifier.fit(X_train, y_train, X_val, y_val)
            >>> print("Training completed!")
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            X: 测试特征
            
        Returns:
            预测标签数组
            
        Example:
            >>> predictions = classifier.predict(X_test)
            >>> accuracy = np.mean(predictions == y_test)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            X: 测试特征
            
        Returns:
            类别概率数组 (N_samples, N_classes)
            
        Example:
            >>> probabilities = classifier.predict_proba(X_test)
            >>> confidence = np.max(probabilities, axis=1)
        """
        pass
    
    def predict_scene(self, scene_path: str, 
                     output_path: str = None) -> np.ndarray:
        """
        预测整个场景
        
        Args:
            scene_path: 场景数据路径
            output_path: 可选的输出路径
            
        Returns:
            场景分类结果
            
        Example:
            >>> result = classifier.predict_scene('test_scene.tif', 'result.tif')
            >>> print(f"Classification completed. Unique classes: {np.unique(result)}")
        """
        pass
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Example:
            >>> classifier.save_model('models/best_model.pkl')
        """
        pass
    
    @classmethod
    def load_model(cls, path: str) -> 'Classifier':
        """
        加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            加载的分类器实例
            
        Example:
            >>> classifier = Classifier.load_model('models/best_model.pkl')
            >>> predictions = classifier.predict(X_test)
        """
        pass
```

### EnsembleClassifier

集成分类器，组合多个基分类器提高性能。

```python
class EnsembleClassifier:
    """集成分类器
    
    通过组合多个基分类器来提高分类性能和鲁棒性。
    """
    
    def __init__(self, classifiers: List[Tuple[str, str]], 
                 ensemble_method: str = 'voting'):
        """
        初始化集成分类器
        
        Args:
            classifiers: 基分类器列表 [(name, model_type), ...]
            ensemble_method: 集成方法 ('voting', 'stacking', 'weighted')
            
        Example:
            >>> ensemble = EnsembleClassifier([
            ...     ('svm', 'svm'),
            ...     ('rf', 'random_forest'),
            ...     ('cnn', 'hybrid_cnn')
            ... ], ensemble_method='voting')
        """
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练集成分类器
        
        Args:
            X: 训练特征
            y: 训练标签
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测
        
        Args:
            X: 测试特征
            
        Returns:
            集成预测结果
        """
        pass
```

---

## 🔬 后处理模块

### PostProcessor

后处理器，提供空间滤波、形态学操作等功能。

```python
class PostProcessor:
    """后处理器
    
    对分类结果进行后处理，包括空间滤波、形态学操作、
    一致性检查等，以提高分类结果的质量。
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        """
        初始化后处理器
        
        Args:
            config: 后处理配置
        """
        pass
    
    def spatial_filter(self, classification_map: np.ndarray,
                      filter_type: str = 'majority',
                      window_size: int = 3) -> np.ndarray:
        """
        空间滤波
        
        Args:
            classification_map: 分类结果图
            filter_type: 滤波类型 ('majority', 'median', 'mode')
            window_size: 滤波窗口大小
            
        Returns:
            滤波后的分类图
            
        Example:
            >>> processor = PostProcessor()
            >>> filtered = processor.spatial_filter(
            ...     classification_map, 
            ...     filter_type='majority', 
            ...     window_size=5
            ... )
        """
        pass
    
    def morphological_filter(self, classification_map: np.ndarray,
                           operation: str = 'opening',
                           kernel_size: int = 3) -> np.ndarray:
        """
        形态学滤波
        
        Args:
            classification_map: 分类结果图
            operation: 形态学操作 ('opening', 'closing', 'erosion', 'dilation')
            kernel_size: 核大小
            
        Returns:
            形态学滤波后的分类图
        """
        pass
    
    def remove_small_objects(self, classification_map: np.ndarray,
                           min_size: int = 10) -> np.ndarray:
        """
        移除小对象
        
        Args:
            classification_map: 分类结果图
            min_size: 最小对象大小（像素数）
            
        Returns:
            移除小对象后的分类图
        """
        pass
```

---

## 🌿 景观分析模块

### LandscapeAnalyzer

景观格局分析器，计算各种景观生态学指标。

```python
class LandscapeAnalyzer:
    """景观格局分析器
    
    计算景观生态学指标，包括斑块指标、类别指标和景观指标。
    """
    
    def __init__(self, config: Optional[LandscapeConfig] = None):
        """
        初始化景观分析器
        
        Args:
            config: 景观分析配置
        """
        pass
    
    def compute_metrics(self, classification_map: np.ndarray,
                       metrics: List[str] = None) -> Dict[str, float]:
        """
        计算景观指标
        
        Args:
            classification_map: 分类结果图
            metrics: 指标列表，如果为None则计算所有指标
            
        Returns:
            指标字典
            
        Available metrics:
            - 'patch_density': 斑块密度
            - 'edge_density': 边缘密度
            - 'largest_patch_index': 最大斑块指数
            - 'shannon_diversity': 香农多样性指数
            - 'simpson_diversity': 辛普森多样性指数
            - 'evenness': 均匀度指数
            - 'aggregation_index': 聚集指数
            - 'connectance': 连通度
            
        Example:
            >>> analyzer = LandscapeAnalyzer()
            >>> metrics = analyzer.compute_metrics(classification_map)
            >>> print(f"Shannon diversity: {metrics['shannon_diversity']:.3f}")
        """
        pass
    
    def patch_analysis(self, classification_map: np.ndarray) -> pd.DataFrame:
        """
        斑块分析
        
        Args:
            classification_map: 分类结果图
            
        Returns:
            斑块统计DataFrame
            
        Columns:
            - patch_id: 斑块ID
            - class_id: 类别ID
            - area: 面积
            - perimeter: 周长
            - shape_index: 形状指数
            - core_area: 核心区面积
        """
        pass
    
    def connectivity_analysis(self, classification_map: np.ndarray,
                            class_id: int,
                            distance_threshold: float = 100) -> Dict[str, Any]:
        """
        连通性分析
        
        Args:
            classification_map: 分类结果图
            class_id: 目标类别ID
            distance_threshold: 距离阈值（米）
            
        Returns:
            连通性分析结果
        """
        pass
```

---

## 📊 评估模块

### ModelEvaluator

模型评估器，提供全面的性能评估功能。

```python
class ModelEvaluator:
    """模型评估器
    
    提供分类精度评估、统计显著性检验、错误分析等功能。
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                class_names: List[str] = None) -> Dict[str, Any]:
        """
        全面评估
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            
        Returns:
            评估结果字典
            
        Metrics:
            - overall_accuracy: 总体精度
            - kappa: Kappa系数
            - f1_score: F1分数
            - precision: 精确率
            - recall: 召回率
            - confusion_matrix: 混淆矩阵
            - per_class_accuracy: 各类别精度
            
        Example:
            >>> evaluator = ModelEvaluator()
            >>> results = evaluator.evaluate(y_true, y_pred, class_names)
            >>> print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
            >>> print(f"Kappa: {results['kappa']:.3f}")
        """
        pass
    
    def cross_validation(self, classifier: Classifier,
                        X: np.ndarray, y: np.ndarray,
                        cv: int = 5,
                        stratified: bool = True) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            classifier: 分类器实例
            X: 特征数据
            y: 标签数据
            cv: 折数
            stratified: 是否分层抽样
            
        Returns:
            交叉验证结果
            
        Example:
            >>> cv_results = evaluator.cross_validation(
            ...     classifier, X, y, cv=5
            ... )
            >>> print(f"CV Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        """
        pass
    
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                        normalize: bool = False) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            normalize: 是否归一化
            
        Returns:
            混淆矩阵
        """
        pass
    
    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str] = None) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            
        Returns:
            分类报告字符串
        """
        pass
```

### UncertaintyAnalyzer

不确定性分析器，评估模型预测的不确定性。

```python
class UncertaintyAnalyzer:
    """不确定性分析器"""
    
    def monte_carlo_dropout(self, model, X: np.ndarray,
                           n_samples: int = 100) -> np.ndarray:
        """
        Monte Carlo Dropout不确定性估计
        
        Args:
            model: 深度学习模型
            X: 输入数据
            n_samples: 采样次数
            
        Returns:
            不确定性估计结果 (N_samples, N_classes, n_samples)
        """
        pass
    
    def predictive_entropy(self, predictions: np.ndarray) -> np.ndarray:
        """
        计算预测熵
        
        Args:
            predictions: 预测概率 (N_samples, N_classes)
            
        Returns:
            预测熵数组
        """
        pass
```

---

## 🛠️ 工具模块

### Visualizer

可视化工具，提供丰富的图表和地图展示功能。

```python
class Visualizer:
    """可视化工具
    
    提供分类结果可视化、精度评估图表、特征分析图等。
    """
    
    def plot_classification_map(self, classification_map: np.ndarray,
                              class_names: List[str] = None,
                              colors: List[str] = None,
                              title: str = 'Classification Map',
                              save_path: str = None):
        """
        绘制分类结果图
        
        Args:
            classification_map: 分类结果图
            class_names: 类别名称列表
            colors: 类别颜色列表
            title: 图表标题
            save_path: 保存路径
            
        Example:
            >>> visualizer = Visualizer()
            >>> visualizer.plot_classification_map(
            ...     result_map, 
            ...     class_names=['Water', 'Vegetation', 'Soil'],
            ...     save_path='classification_result.png'
            ... )
        """
        pass
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            class_names: List[str] = None,
                            normalize: bool = False,
                            title: str = 'Confusion Matrix',
                            save_path: str = None):
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称
            normalize: 是否归一化显示
            title: 图表标题
            save_path: 保存路径
        """
        pass
    
    def plot_spectral_signatures(self, spectra: np.ndarray,
                                class_labels: np.ndarray,
                                wavelengths: np.ndarray = None,
                                class_names: List[str] = None,
                                title: str = 'Spectral Signatures',
                                save_path: str = None):
        """
        绘制光谱特征曲线
        
        Args:
            spectra: 光谱数据 (N_samples, N_bands)
            class_labels: 类别标签
            wavelengths: 波长数组
            class_names: 类别名称
            title: 图表标题
            save_path: 保存路径
        """
        pass
    
    def plot_feature_importance(self, importance: np.ndarray,
                              feature_names: List[str] = None,
                              top_k: int = 20,
                              title: str = 'Feature Importance',
                              save_path: str = None):
        """
        绘制特征重要性图
        
        Args:
            importance: 特征重要性数组
            feature_names: 特征名称列表
            top_k: 显示前k个重要特征
            title: 图表标题
            save_path: 保存路径
        """
        pass
```

### Logger

日志管理器，提供结构化的日志记录功能。

```python
def get_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        配置好的日志记录器
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred: %s", error_message)
    """
    pass
```

### IOUtils

输入输出工具，提供文件操作的便利函数。

```python
class IOUtils:
    """输入输出工具类"""
    
    @staticmethod
    def save_geotiff(array: np.ndarray, 
                    output_path: str,
                    reference_path: str = None,
                    crs: str = None,
                    transform = None):
        """
        保存GeoTIFF文件
        
        Args:
            array: 要保存的数组
            output_path: 输出路径
            reference_path: 参考文件路径（用于获取地理信息）
            crs: 坐标系统
            transform: 仿射变换参数
            
        Example:
            >>> IOUtils.save_geotiff(
            ...     classification_result, 
            ...     'output/result.tif',
            ...     reference_path='input/scene.tif'
            ... )
        """
        pass
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径 (.yaml, .json)
            
        Returns:
            配置字典
        """
        pass
    
    @staticmethod
    def save_results(results: Dict[str, Any], 
                    output_dir: str,
                    format: str = 'json'):
        """
        保存结果
        
        Args:
            results: 结果字典
            output_dir: 输出目录
            format: 保存格式 ('json', 'pickle', 'yaml')
        """
        pass
```

---

## ⚙️ 配置模块

### Config

配置管理类，提供统一的配置管理接口。

```python
@dataclass
class Config:
    """主配置类
    
    统一管理系统的所有配置参数。
    """
    
    data: DataConfig
    preprocessing: PreprocessingConfig
    features: FeatureConfig
    classification: ClassificationConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象
            
        Example:
            >>> config = Config.from_file('config/config.yaml')
            >>> print(config.classification.model_type)
        """
        pass
    
    @classmethod
    def from_default(cls) -> 'Config':
        """
        创建默认配置
        
        Returns:
            默认配置对象
        """
        pass
    
    def save(self, output_path: str):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径
        """
        pass
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            是否有效
            
        Raises:
            ConfigValidationError: 配置无效时
        """
        pass
    
    def update(self, **kwargs):
        """
        更新配置参数
        
        Example:
            >>> config.update(
            ...     classification__model_type='hybrid_cnn',
            ...     features__vegetation_indices=['NDVI', 'EVI']
            ... )
        """
        pass
```

### 子配置类

```python
@dataclass
class DataConfig:
    """数据配置"""
    input_path: str = "data/raw/"
    output_path: str = "output/"
    file_format: str = "tif"
    cache_enabled: bool = True
    tile_size: int = 512
    overlap: int = 64

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    radiometric_calibration: bool = True
    atmospheric_correction: str = "FLAASH"
    geometric_correction: bool = True
    noise_reduction: str = "MNF"
    bad_bands: List[int] = None

@dataclass
class FeatureConfig:
    """特征提取配置"""
    spectral_features: bool = True
    vegetation_indices: List[str] = None
    texture_features: bool = True
    spatial_features: bool = True
    pca_components: int = 50

@dataclass
class ClassificationConfig:
    """分类配置"""
    model_type: str = "hybrid_cnn"
    training_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    
    # 模型特定参数
    svm_params: Dict[str, Any] = None
    rf_params: Dict[str, Any] = None
    cnn_params: Dict[str, Any] = None

@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = None
    cross_validation: int = 5
    confidence_interval: float = 0.95
    significance_level: float = 0.05
```

---

## 📝 使用示例

### 完整工作流示例

```python
from wetland_classification import Pipeline
from wetland_classification.config import Config

# 1. 加载配置
config = Config.from_file('config/config.yaml')

# 2. 创建流水线
pipeline = Pipeline(config)

# 3. 运行完整流程
results = pipeline.run(
    input_data='data/raw/wetland_scene.tif',
    ground_truth='data/raw/ground_truth.shp',
    output_dir='output/results/'
)

# 4. 查看结果
print(f"Classification completed!")
print(f"Overall Accuracy: {results['accuracy']:.3f}")
print(f"Kappa Coefficient: {results['kappa']:.3f}")
print(f"Output files saved to: {results['output_dir']}")
```

### 自定义工作流示例

```python
from wetland_classification.data import DataLoader
from wetland_classification.preprocessing import Preprocessor
from wetland_classification.features import FeatureExtractor
from wetland_classification.classification import Classifier
from wetland_classification.evaluation import ModelEvaluator

# 1. 数据加载
loader = DataLoader()
data, metadata = loader.load_hyperspectral('scene.tif')
labels = loader.load_ground_truth('labels.shp')

# 2. 数据预处理
preprocessor = Preprocessor()
processed_data = preprocessor.process_all(data)

# 3. 特征提取
extractor = FeatureExtractor()
features = extractor.extract_all(processed_data)

# 4. 模型训练
classifier = Classifier('hybrid_cnn')
classifier.fit(X_train, y_train, X_val, y_val)

# 5. 模型评估
evaluator = ModelEvaluator()
results = evaluator.evaluate(y_test, y_pred)
print(f"Test Accuracy: {results['overall_accuracy']:.3f}")
```

---

## 🔧 异常处理

### 自定义异常类

```python
class WetlandClassificationError(Exception):
    """基础异常类"""
    pass

class DataLoadError(WetlandClassificationError):
    """数据加载异常"""
    pass

class PreprocessingError(WetlandClassificationError):
    """预处理异常"""
    pass

class FeatureExtractionError(WetlandClassificationError):
    """特征提取异常"""
    pass

class ClassificationError(WetlandClassificationError):
    """分类异常"""
    pass

class ConfigurationError(WetlandClassificationError):
    """配置异常"""
    pass
```

---

## 📞 技术支持

### API相关问题

- 📧 **API支持**: api-support@example.com
- 📚 **在线文档**: https://api-docs.example.com
- 💻 **代码示例**: https://github.com/yourusername/wetland-examples
- 🐛 **Bug报告**: https://github.com/yourusername/wetland-hyperspectral-classification/issues

### 版本兼容性

当前API版本: **v1.0.0**

支持的Python版本: **3.8+**

主要依赖版本:
- NumPy: 1.21+
- Scikit-learn: 1.0+
- PyTorch: 1.12+
- Rasterio: 1.3+

---

*本API文档持续更新中，最后更新时间: 2024年6月30日*