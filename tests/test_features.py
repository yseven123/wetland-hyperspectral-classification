"""
特征提取模块测试

测试光谱特征、植被指数、纹理特征和空间特征提取功能。
验证各种特征提取算法的正确性和有效性。

Author: 湿地高光谱分类系统开发团队
Date: 2024
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.decomposition import PCA

# 导入待测试的模块
try:
    from wetland_classification.features import FeatureExtractor
    from wetland_classification.features.spectral import SpectralFeatures
    from wetland_classification.features.indices import VegetationIndices
    from wetland_classification.features.texture import TextureFeatures
    from wetland_classification.features.spatial import SpatialFeatures
except ImportError:
    # 如果模块不存在，创建mock对象用于测试结构
    FeatureExtractor = Mock
    SpectralFeatures = Mock
    VegetationIndices = Mock
    TextureFeatures = Mock
    SpatialFeatures = Mock


class TestSpectralFeatures:
    """光谱特征提取测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'pca_components': 20,
            'derivative_order': 1,
            'continuum_removal': True,
            'spectral_angle_threshold': 0.1
        }
        self.extractor = SpectralFeatures(self.config)
        
    def test_spectral_features_initialization(self):
        """测试光谱特征提取器初始化"""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'extract_pca_features')
        assert hasattr(self.extractor, 'extract_derivative_features')
        
    def test_pca_feature_extraction(self, sample_hyperspectral_data):
        """测试主成分分析特征提取"""
        # 重塑数据为2D矩阵 (pixels, bands)
        pixels = sample_hyperspectral_data.reshape(-1, sample_hyperspectral_data.shape[2])
        
        # 提取PCA特征
        pca_features, explained_variance = self.extractor.extract_pca_features(
            pixels, n_components=20
        )
        
        # 验证PCA特征
        assert pca_features.shape[1] == 20  # 20个主成分
        assert pca_features.shape[0] == pixels.shape[0]  # 像素数量不变
        assert len(explained_variance) == 20
        assert np.sum(explained_variance) <= 1.0  # 累计方差贡献率不超过1
        
        # 验证特征的方差降序排列
        assert np.all(explained_variance[:-1] >= explained_variance[1:])
        
    def test_derivative_spectral_features(self, sample_hyperspectral_data):
        """测试光谱导数特征"""
        # 提取一阶导数特征
        first_derivative = self.extractor.extract_derivative_features(
            sample_hyperspectral_data, order=1
        )
        
        # 验证导数特征形状
        expected_shape = (*sample_hyperspectral_data.shape[:2], 
                         sample_hyperspectral_data.shape[2] - 1)
        assert first_derivative.shape == expected_shape
        
        # 提取二阶导数特征
        second_derivative = self.extractor.extract_derivative_features(
            sample_hyperspectral_data, order=2
        )
        
        # 验证二阶导数特征形状
        expected_shape = (*sample_hyperspectral_data.shape[:2], 
                         sample_hyperspectral_data.shape[2] - 2)
        assert second_derivative.shape == expected_shape
        
    def test_continuum_removal(self, sample_hyperspectral_data):
        """测试连续统去除"""
        # 提取单个像素的光谱
        pixel_spectrum = sample_hyperspectral_data[50, 50, :]
        
        # 执行连续统去除
        cr_spectrum = self.extractor.continuum_removal(pixel_spectrum)
        
        # 验证连续统去除结果
        assert cr_spectrum.shape == pixel_spectrum.shape
        assert np.all(cr_spectrum <= 1.0)  # 连续统去除后值应该<=1
        assert np.all(cr_spectrum >= 0.0)  # 值应该非负
        
        # 验证吸收特征被增强
        # 连续统去除应该突出吸收带
        absorption_depth = 1 - np.min(cr_spectrum)
        original_depth = np.max(pixel_spectrum) - np.min(pixel_spectrum)
        assert absorption_depth > 0
        
    def test_spectral_angle_mapper(self, sample_hyperspectral_data):
        """测试光谱角度映射"""
        # 创建参考光谱
        reference_spectrum = np.mean(sample_hyperspectral_data.reshape(-1, sample_hyperspectral_data.shape[2]), axis=0)
        
        # 计算光谱角度
        sam_map = self.extractor.spectral_angle_mapper(
            sample_hyperspectral_data, reference_spectrum
        )
        
        # 验证SAM结果
        assert sam_map.shape == sample_hyperspectral_data.shape[:2]
        assert np.all(sam_map >= 0)  # 角度非负
        assert np.all(sam_map <= np.pi/2)  # 角度不超过90度
        
    def test_spectral_similarity_metrics(self, sample_hyperspectral_data):
        """测试光谱相似性度量"""
        # 选择两个像素光谱
        spectrum1 = sample_hyperspectral_data[25, 25, :]
        spectrum2 = sample_hyperspectral_data[75, 75, :]
        
        # 计算各种相似性度量
        correlation = self.extractor.spectral_correlation(spectrum1, spectrum2)
        euclidean_dist = self.extractor.spectral_euclidean_distance(spectrum1, spectrum2)
        spectral_angle = self.extractor.spectral_angle_distance(spectrum1, spectrum2)
        
        # 验证相似性度量
        assert -1 <= correlation <= 1  # 相关系数范围
        assert euclidean_dist >= 0     # 欧氏距离非负
        assert 0 <= spectral_angle <= np.pi/2  # 光谱角度范围
        
    def test_absorption_feature_detection(self, sample_hyperspectral_data):
        """测试吸收特征检测"""
        # 在光谱中模拟水分吸收带
        modified_data = sample_hyperspectral_data.copy()
        water_bands = [95, 120, 140]  # 模拟950nm, 1200nm, 1400nm
        for band in water_bands:
            if band < modified_data.shape[2]:
                modified_data[:, :, band] *= 0.7  # 降低反射率模拟吸收
        
        # 检测吸收特征
        absorption_features = self.extractor.detect_absorption_features(
            modified_data, wavelengths=np.linspace(400, 2500, modified_data.shape[2])
        )
        
        # 验证吸收特征检测
        assert 'absorption_bands' in absorption_features
        assert 'absorption_depths' in absorption_features
        assert len(absorption_features['absorption_bands']) > 0
        
    def test_red_edge_parameters(self, sample_hyperspectral_data):
        """测试红边参数提取"""
        # 模拟红边光谱特征
        wavelengths = np.linspace(400, 2500, sample_hyperspectral_data.shape[2])
        
        # 提取红边参数
        red_edge_params = self.extractor.extract_red_edge_parameters(
            sample_hyperspectral_data, wavelengths
        )
        
        # 验证红边参数
        assert 'red_edge_position' in red_edge_params
        assert 'red_edge_amplitude' in red_edge_params
        assert 'red_edge_slope' in red_edge_params
        
        # 验证参数合理性
        rep_map = red_edge_params['red_edge_position']
        assert rep_map.shape == sample_hyperspectral_data.shape[:2]
        assert np.all((rep_map >= 680) & (rep_map <= 780))  # 红边位置范围


class TestVegetationIndices:
    """植被指数测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        # 模拟波段中心波长
        self.wavelengths = np.linspace(400, 2500, 200)
        self.config = {
            'indices': ['NDVI', 'EVI', 'NDWI', 'MNDWI', 'SAVI', 'MSAVI'],
            'wavelengths': self.wavelengths
        }
        self.extractor = VegetationIndices(self.config)
        
    def test_vegetation_indices_initialization(self):
        """测试植被指数提取器初始化"""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'calculate_ndvi')
        assert hasattr(self.extractor, 'calculate_evi')
        
    def test_ndvi_calculation(self, sample_hyperspectral_data):
        """测试NDVI计算"""
        # 定义红光和近红外波段索引
        red_band = 60   # ~660nm
        nir_band = 80   # ~830nm
        
        # 计算NDVI
        ndvi = self.extractor.calculate_ndvi(
            sample_hyperspectral_data, red_band, nir_band
        )
        
        # 验证NDVI结果
        assert ndvi.shape == sample_hyperspectral_data.shape[:2]
        assert np.all(ndvi >= -1)  # NDVI范围-1到1
        assert np.all(ndvi <= 1)
        
        # 验证植被区域NDVI值
        vegetation_mask = ndvi > 0.3
        if np.any(vegetation_mask):
            vegetation_ndvi = ndvi[vegetation_mask]
            assert np.mean(vegetation_ndvi) > 0.3
            
    def test_evi_calculation(self, sample_hyperspectral_data):
        """测试EVI计算"""
        # 定义波段索引
        blue_band = 20   # ~480nm
        red_band = 60    # ~660nm  
        nir_band = 80    # ~830nm
        
        # 计算EVI
        evi = self.extractor.calculate_evi(
            sample_hyperspectral_data, blue_band, red_band, nir_band
        )
        
        # 验证EVI结果
        assert evi.shape == sample_hyperspectral_data.shape[:2]
        assert np.all(evi >= -1)  # EVI理论范围
        assert np.all(evi <= 1)
        
    def test_water_indices(self, sample_hyperspectral_data):
        """测试水体指数计算"""
        # 计算NDWI
        green_band = 40  # ~560nm
        nir_band = 80    # ~830nm
        
        ndwi = self.extractor.calculate_ndwi(
            sample_hyperspectral_data, green_band, nir_band
        )
        
        # 计算MNDWI
        swir_band = 120  # ~1640nm
        mndwi = self.extractor.calculate_mndwi(
            sample_hyperspectral_data, green_band, swir_band
        )
        
        # 验证水体指数
        assert ndwi.shape == sample_hyperspectral_data.shape[:2]
        assert mndwi.shape == sample_hyperspectral_data.shape[:2]
        assert np.all(ndwi >= -1) and np.all(ndwi <= 1)
        assert np.all(mndwi >= -1) and np.all(mndwi <= 1)
        
    def test_soil_adjusted_indices(self, sample_hyperspectral_data):
        """测试土壤调节植被指数"""
        red_band = 60
        nir_band = 80
        
        # 计算SAVI
        savi = self.extractor.calculate_savi(
            sample_hyperspectral_data, red_band, nir_band, soil_factor=0.5
        )
        
        # 计算MSAVI
        msavi = self.extractor.calculate_msavi(
            sample_hyperspectral_data, red_band, nir_band
        )
        
        # 验证土壤调节指数
        assert savi.shape == sample_hyperspectral_data.shape[:2]
        assert msavi.shape == sample_hyperspectral_data.shape[:2]
        
        # SAVI应该在合理范围内
        assert np.all(savi >= -1.5) and np.all(savi <= 1.5)
        
    def test_custom_vegetation_index(self, sample_hyperspectral_data):
        """测试自定义植被指数"""
        # 定义自定义指数公式：(NIR - Red) / (NIR + Red + 0.5)
        formula = {
            'numerator': [('nir', 1), ('red', -1)],
            'denominator': [('nir', 1), ('red', 1), ('constant', 0.5)]
        }
        
        band_mapping = {'red': 60, 'nir': 80}
        
        # 计算自定义指数
        custom_index = self.extractor.calculate_custom_index(
            sample_hyperspectral_data, formula, band_mapping
        )
        
        # 验证自定义指数
        assert custom_index.shape == sample_hyperspectral_data.shape[:2]
        assert np.all(np.isfinite(custom_index))  # 确保没有无穷值
        
    def test_batch_indices_calculation(self, sample_hyperspectral_data):
        """测试批量指数计算"""
        # 定义要计算的指数列表
        indices_config = {
            'NDVI': {'red_band': 60, 'nir_band': 80},
            'NDWI': {'green_band': 40, 'nir_band': 80},
            'EVI': {'blue_band': 20, 'red_band': 60, 'nir_band': 80}
        }
        
        # 批量计算指数
        indices_stack = self.extractor.calculate_multiple_indices(
            sample_hyperspectral_data, indices_config
        )
        
        # 验证批量计算结果
        assert indices_stack.shape == (*sample_hyperspectral_data.shape[:2], len(indices_config))
        
        # 验证每个指数
        for i, index_name in enumerate(indices_config.keys()):
            index_values = indices_stack[:, :, i]
            assert np.all(np.isfinite(index_values))
            
    def test_temporal_indices_analysis(self, sample_hyperspectral_data):
        """测试时间序列指数分析"""
        # 模拟多时相数据
        temporal_data = np.stack([
            sample_hyperspectral_data,
            sample_hyperspectral_data * 1.1,  # 模拟植被生长
            sample_hyperspectral_data * 0.9   # 模拟植被衰减
        ], axis=0)
        
        # 计算时间序列NDVI
        temporal_ndvi = self.extractor.calculate_temporal_ndvi(
            temporal_data, red_band=60, nir_band=80
        )
        
        # 分析NDVI变化趋势
        ndvi_trend = self.extractor.analyze_ndvi_trend(temporal_ndvi)
        
        # 验证时间序列分析
        assert temporal_ndvi.shape == (3, *sample_hyperspectral_data.shape[:2])
        assert 'trend_slope' in ndvi_trend
        assert 'trend_r2' in ndvi_trend
        assert ndvi_trend['trend_slope'].shape == sample_hyperspectral_data.shape[:2]


class TestTextureFeatures:
    """纹理特征测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'window_size': 3,
            'distances': [1, 2],
            'angles': [0, 45, 90, 135],
            'glcm_properties': ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        }
        self.extractor = TextureFeatures(self.config)
        
    def test_texture_features_initialization(self):
        """测试纹理特征提取器初始化"""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'extract_glcm_features')
        assert hasattr(self.extractor, 'extract_lbp_features')
        
    def test_glcm_texture_features(self, sample_hyperspectral_data):
        """测试GLCM纹理特征"""
        # 选择一个波段进行纹理分析
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 提取GLCM特征
        glcm_features = self.extractor.extract_glcm_features(
            band_image, distances=[1], angles=[0, 90]
        )
        
        # 验证GLCM特征
        expected_features = len(self.config['glcm_properties']) * len([1]) * len([0, 90])
        assert glcm_features.shape[2] == expected_features
        assert glcm_features.shape[:2] == band_image.shape
        
        # 验证特征值范围
        assert np.all(glcm_features >= 0)  # GLCM特征通常非负
        
    def test_local_binary_pattern(self, sample_hyperspectral_data):
        """测试局部二值模式"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 提取LBP特征
        lbp_features = self.extractor.extract_lbp_features(
            band_image, radius=1, n_points=8
        )
        
        # 验证LBP特征
        assert lbp_features.shape == band_image.shape
        assert np.all(lbp_features >= 0)
        assert np.all(lbp_features <= 255)  # LBP值范围0-255
        
    def test_gabor_filter_features(self, sample_hyperspectral_data):
        """测试Gabor滤波器特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 定义Gabor参数
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, 45, 90, 135]
        
        # 提取Gabor特征
        gabor_features = self.extractor.extract_gabor_features(
            band_image, frequencies, orientations
        )
        
        # 验证Gabor特征
        expected_features = len(frequencies) * len(orientations) * 2  # 实部和虚部
        assert gabor_features.shape[2] == expected_features
        assert gabor_features.shape[:2] == band_image.shape
        
    def test_wavelet_texture_features(self, sample_hyperspectral_data):
        """测试小波纹理特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 提取小波特征
        wavelet_features = self.extractor.extract_wavelet_features(
            band_image, wavelet='db4', levels=3
        )
        
        # 验证小波特征
        assert 'approximation' in wavelet_features
        assert 'details' in wavelet_features
        assert len(wavelet_features['details']) == 3  # 3个层次
        
    def test_multispectral_texture(self, sample_hyperspectral_data):
        """测试多光谱纹理特征"""
        # 选择多个波段进行纹理分析
        selected_bands = [20, 50, 80, 120]  # 选择代表性波段
        
        # 提取多光谱纹理
        multispectral_texture = self.extractor.extract_multispectral_texture(
            sample_hyperspectral_data, selected_bands
        )
        
        # 验证多光谱纹理
        expected_features = len(selected_bands) * len(self.config['glcm_properties'])
        assert multispectral_texture.shape[2] >= expected_features
        
    def test_texture_energy_measures(self, sample_hyperspectral_data):
        """测试纹理能量度量"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 计算各种纹理能量度量
        energy_measures = self.extractor.calculate_texture_energy(band_image)
        
        # 验证能量度量
        assert 'entropy' in energy_measures
        assert 'energy' in energy_measures
        assert 'variance' in energy_measures
        assert 'uniformity' in energy_measures
        
        # 验证度量值范围
        assert np.all(energy_measures['energy'] >= 0)
        assert np.all(energy_measures['entropy'] >= 0)


class TestSpatialFeatures:
    """空间特征测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'window_size': 5,
            'edge_operators': ['sobel', 'canny', 'laplacian'],
            'morphology_kernels': ['disk', 'square'],
            'scale_levels': [1, 2, 4]
        }
        self.extractor = SpatialFeatures(self.config)
        
    def test_spatial_features_initialization(self):
        """测试空间特征提取器初始化"""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'extract_edge_features')
        assert hasattr(self.extractor, 'extract_morphological_features')
        
    def test_edge_detection_features(self, sample_hyperspectral_data):
        """测试边缘检测特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 提取边缘特征
        edge_features = self.extractor.extract_edge_features(band_image)
        
        # 验证边缘特征
        assert 'sobel_magnitude' in edge_features
        assert 'sobel_direction' in edge_features
        assert 'canny_edges' in edge_features
        
        # 验证特征形状
        for feature_name, feature_map in edge_features.items():
            assert feature_map.shape == band_image.shape
            
    def test_morphological_features(self, sample_hyperspectral_data):
        """测试形态学特征"""
        # 选择一个波段并二值化
        band_image = sample_hyperspectral_data[:, :, 50]
        binary_image = band_image > np.mean(band_image)
        
        # 提取形态学特征
        morph_features = self.extractor.extract_morphological_features(binary_image)
        
        # 验证形态学特征
        assert 'opening' in morph_features
        assert 'closing' in morph_features
        assert 'gradient' in morph_features
        
        # 验证特征类型
        for feature_name, feature_map in morph_features.items():
            assert isinstance(feature_map, np.ndarray)
            assert feature_map.shape == binary_image.shape
            
    def test_multiscale_features(self, sample_hyperspectral_data):
        """测试多尺度特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 提取多尺度特征
        multiscale_features = self.extractor.extract_multiscale_features(
            band_image, scales=[1, 2, 4]
        )
        
        # 验证多尺度特征
        assert len(multiscale_features) == 3  # 3个尺度
        
        for scale, features in multiscale_features.items():
            assert isinstance(features, dict)
            assert 'mean' in features
            assert 'std' in features
            assert 'gradient' in features
            
    def test_spatial_autocorrelation(self, sample_hyperspectral_data):
        """测试空间自相关特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 计算空间自相关
        autocorr_features = self.extractor.calculate_spatial_autocorrelation(
            band_image, distances=[1, 2, 3]
        )
        
        # 验证自相关特征
        assert 'moran_i' in autocorr_features
        assert 'geary_c' in autocorr_features
        
        # Moran's I应该在-1到1之间
        moran_values = autocorr_features['moran_i']
        assert np.all(moran_values >= -1)
        assert np.all(moran_values <= 1)
        
    def test_fractal_dimension(self, sample_hyperspectral_data):
        """测试分形维数特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 计算分形维数
        fractal_dim = self.extractor.calculate_fractal_dimension(band_image)
        
        # 验证分形维数
        assert fractal_dim.shape == band_image.shape
        assert np.all(fractal_dim >= 2)  # 2D分形维数下界
        assert np.all(fractal_dim <= 3)  # 2D分形维数上界
        
    def test_spatial_context_features(self, sample_hyperspectral_data):
        """测试空间上下文特征"""
        # 选择一个波段
        band_image = sample_hyperspectral_data[:, :, 50]
        
        # 提取空间上下文特征
        context_features = self.extractor.extract_spatial_context(
            band_image, window_size=5
        )
        
        # 验证上下文特征
        assert 'local_mean' in context_features
        assert 'local_std' in context_features
        assert 'local_range' in context_features
        assert 'neighborhood_similarity' in context_features
        
        # 验证特征形状
        for feature_name, feature_map in context_features.items():
            assert feature_map.shape == band_image.shape


class TestFeatureIntegration:
    """特征提取模块集成测试"""
    
    def test_complete_feature_extraction_pipeline(self, sample_hyperspectral_data):
        """测试完整的特征提取流水线"""
        # 初始化特征提取器
        config = {
            'spectral': {'pca_components': 10, 'derivative_order': 1},
            'vegetation_indices': ['NDVI', 'EVI', 'NDWI'],
            'texture': {'window_size': 3, 'glcm_properties': ['contrast', 'energy']},
            'spatial': {'edge_operators': ['sobel'], 'window_size': 3}
        }
        
        feature_extractor = FeatureExtractor(config)
        
        # 提取所有特征
        all_features = feature_extractor.extract_all_features(sample_hyperspectral_data)
        
        # 验证特征提取结果
        assert 'spectral' in all_features
        assert 'indices' in all_features
        assert 'texture' in all_features
        assert 'spatial' in all_features
        
        # 验证特征维度
        h, w = sample_hyperspectral_data.shape[:2]
        for feature_type, features in all_features.items():
            if isinstance(features, np.ndarray):
                assert features.shape[:2] == (h, w)
                
    def test_feature_selection_and_ranking(self, sample_hyperspectral_data, sample_training_samples):
        """测试特征选择和排序"""
        # 提取特征
        feature_extractor = FeatureExtractor({})
        
        # 模拟提取的特征
        n_features = 50
        features = np.random.random((len(sample_training_samples['coordinates']), n_features))
        labels = np.array(sample_training_samples['labels'])
        
        # 执行特征选择
        selected_features, feature_importance = feature_extractor.select_features(
            features, labels, method='mutual_info', n_features=20
        )
        
        # 验证特征选择结果
        assert selected_features.shape[1] == 20  # 选择了20个特征
        assert len(feature_importance) == n_features
        assert np.all(feature_importance >= 0)  # 重要性分数非负
        
    def test_feature_scaling_and_normalization(self, sample_hyperspectral_data):
        """测试特征缩放和标准化"""
        # 提取一些特征
        features = sample_hyperspectral_data.reshape(-1, sample_hyperspectral_data.shape[2])
        
        feature_extractor = FeatureExtractor({})
        
        # 测试不同的标准化方法
        normalized_features = feature_extractor.normalize_features(
            features, method='z_score'
        )
        
        # 验证Z-score标准化
        assert normalized_features.shape == features.shape
        assert np.abs(np.mean(normalized_features, axis=0)).max() < 1e-10  # 均值接近0
        assert np.abs(np.std(normalized_features, axis=0) - 1).max() < 1e-10  # 标准差接近1
        
        # 测试MinMax标准化
        minmax_features = feature_extractor.normalize_features(
            features, method='minmax'
        )
        
        assert np.all(minmax_features >= 0)  # 最小值为0
        assert np.all(minmax_features <= 1)  # 最大值为1
        
    def test_feature_quality_assessment(self, sample_hyperspectral_data):
        """测试特征质量评估"""
        feature_extractor = FeatureExtractor({})
        
        # 模拟一些特征
        good_features = np.random.normal(0, 1, (1000, 10))  # 正态分布特征
        bad_features = np.ones((1000, 5))  # 常数特征
        
        features = np.concatenate([good_features, bad_features], axis=1)
        
        # 评估特征质量
        quality_report = feature_extractor.assess_feature_quality(features)
        
        # 验证质量评估
        assert 'variance' in quality_report
        assert 'correlation_matrix' in quality_report
        assert 'redundant_features' in quality_report
        
        # 常数特征应该被识别为低质量
        low_variance_indices = quality_report['redundant_features']
        assert len(low_variance_indices) >= 5  # 至少识别出5个常数特征
        
    def test_dimensionality_reduction(self, sample_hyperspectral_data):
        """测试维度降低"""
        # 重塑数据
        features = sample_hyperspectral_data.reshape(-1, sample_hyperspectral_data.shape[2])
        
        feature_extractor = FeatureExtractor({})
        
        # 测试PCA降维
        reduced_features_pca = feature_extractor.reduce_dimensionality(
            features, method='pca', n_components=20
        )
        
        # 验证PCA降维
        assert reduced_features_pca.shape[1] == 20
        assert reduced_features_pca.shape[0] == features.shape[0]
        
        # 测试t-SNE降维
        if features.shape[0] > 100:  # t-SNE需要足够的样本
            sample_indices = np.random.choice(features.shape[0], 100, replace=False)
            sample_features = features[sample_indices]
            
            reduced_features_tsne = feature_extractor.reduce_dimensionality(
                sample_features, method='tsne', n_components=2
            )
            
            # 验证t-SNE降维
            assert reduced_features_tsne.shape == (100, 2)
            
    def test_feature_visualization(self, sample_hyperspectral_data):
        """测试特征可视化"""
        feature_extractor = FeatureExtractor({})
        
        # 提取一些特征用于可视化
        features = sample_hyperspectral_data[:, :, [20, 50, 80]]  # 选择3个波段
        
        # 生成特征可视化
        vis_config = {
            'feature_maps': True,
            'histograms': True,
            'correlation_plot': True,
            'save_path': None  # 不保存文件，只返回图像对象
        }
        
        visualization_results = feature_extractor.visualize_features(
            features, vis_config
        )
        
        # 验证可视化结果
        assert 'feature_maps' in visualization_results
        assert 'histograms' in visualization_results
        assert 'correlation_plot' in visualization_results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])