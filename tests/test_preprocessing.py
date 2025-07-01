"""
预处理模块测试

测试辐射定标、大气校正、几何校正和噪声去除等预处理功能。
确保高光谱数据的预处理流程正确可靠。

Author: 湿地高光谱分类系统开发团队
Date: 2024
"""

import pytest
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock

# 导入待测试的模块
try:
    from wetland_classification.preprocessing import (
        RadiometricCorrector, AtmosphericCorrector, 
        GeometricCorrector, NoiseReducer
    )
    from wetland_classification.preprocessing.radiometric import (
        dark_object_subtraction, gain_offset_correction
    )
    from wetland_classification.preprocessing.atmospheric import (
        flaash_correction, dark_spectrum_fitting
    )
    from wetland_classification.preprocessing.noise_reduction import (
        minimum_noise_fraction, savitzky_golay_filter
    )
except ImportError:
    # 如果模块不存在，创建mock对象用于测试结构
    RadiometricCorrector = Mock
    AtmosphericCorrector = Mock
    GeometricCorrector = Mock
    NoiseReducer = Mock


class TestRadiometricCorrector:
    """辐射定标校正测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'method': 'gain_offset',
            'gain_values': np.ones(200),  # 假设200个波段
            'offset_values': np.zeros(200),
            'scale_factor': 0.0001,
            'dark_object_threshold': 0.01
        }
        self.corrector = RadiometricCorrector(self.config)
        
    def test_radiometric_corrector_initialization(self):
        """测试辐射定标校正器初始化"""
        assert self.corrector is not None
        assert hasattr(self.corrector, 'correct')
        assert hasattr(self.corrector, 'gain_values')
        assert hasattr(self.corrector, 'offset_values')
        
    def test_gain_offset_correction(self, sample_hyperspectral_data):
        """测试增益偏移校正"""
        # 模拟原始DN值数据
        dn_data = (sample_hyperspectral_data * 10000).astype(np.uint16)
        
        # 执行增益偏移校正
        corrected_data = self.corrector.gain_offset_correction(dn_data)
        
        # 验证校正结果
        assert corrected_data.shape == dn_data.shape
        assert corrected_data.dtype == np.float32
        assert np.all(corrected_data >= 0)  # 反射率应该非负
        assert np.all(corrected_data <= 1)  # 反射率通常不超过1
        
    def test_dark_object_subtraction(self, sample_hyperspectral_data):
        """测试暗目标扣除法"""
        # 在数据中添加一些暗目标像素
        dark_data = sample_hyperspectral_data.copy()
        dark_data[0:10, 0:10, :] = 0.005  # 设置暗目标区域
        
        # 执行暗目标扣除
        corrected_data = self.corrector.dark_object_subtraction(dark_data)
        
        # 验证校正结果
        assert corrected_data.shape == dark_data.shape
        
        # 验证暗目标区域的反射率被调整
        dark_region_mean = np.mean(corrected_data[0:10, 0:10, :])
        assert dark_region_mean >= 0
        
    def test_radiometric_calibration_metadata(self, sample_hyperspectral_data):
        """测试辐射定标元数据处理"""
        # 模拟元数据
        metadata = {
            'gain_values': np.random.uniform(0.8, 1.2, 200),
            'offset_values': np.random.uniform(-50, 50, 200),
            'acquisition_date': '2024-06-15',
            'solar_zenith_angle': 30.5,
            'solar_azimuth_angle': 150.2
        }
        
        # 使用元数据进行校正
        corrected_data, cal_metadata = self.corrector.correct_with_metadata(
            sample_hyperspectral_data, metadata
        )
        
        # 验证校正结果和元数据
        assert corrected_data.shape == sample_hyperspectral_data.shape
        assert 'calibration_method' in cal_metadata
        assert 'processing_date' in cal_metadata
        
    def test_cross_calibration(self, sample_hyperspectral_data):
        """测试交叉定标"""
        # 模拟参考数据
        reference_data = sample_hyperspectral_data * 0.95 + 0.02
        
        # 执行交叉定标
        calibrated_data = self.corrector.cross_calibration(
            sample_hyperspectral_data, reference_data
        )
        
        # 验证定标结果
        assert calibrated_data.shape == sample_hyperspectral_data.shape
        
        # 验证定标后的数据与参考数据更接近
        original_diff = np.mean(np.abs(sample_hyperspectral_data - reference_data))
        calibrated_diff = np.mean(np.abs(calibrated_data - reference_data))
        assert calibrated_diff < original_diff
        
    def test_quality_assessment(self, sample_hyperspectral_data):
        """测试定标质量评估"""
        corrected_data = self.corrector.correct(sample_hyperspectral_data)
        
        # 生成质量评估报告
        quality_report = self.corrector.assess_quality(
            sample_hyperspectral_data, corrected_data
        )
        
        # 验证质量报告
        assert 'snr_improvement' in quality_report
        assert 'spectral_consistency' in quality_report
        assert 'radiometric_accuracy' in quality_report
        assert isinstance(quality_report['snr_improvement'], float)


class TestAtmosphericCorrector:
    """大气校正测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'method': 'flaash',
            'atmospheric_model': 'mid_latitude_summer',
            'aerosol_model': 'rural',
            'visibility': 40.0,  # km
            'water_vapor': 2.5,  # cm
            'solar_zenith': 30.0,
            'solar_azimuth': 180.0,
            'sensor_altitude': 705.0  # km
        }
        self.corrector = AtmosphericCorrector(self.config)
        
    def test_atmospheric_corrector_initialization(self):
        """测试大气校正器初始化"""
        assert self.corrector is not None
        assert hasattr(self.corrector, 'correct')
        assert hasattr(self.corrector, 'atmospheric_model')
        
    def test_flaash_correction(self, sample_hyperspectral_data):
        """测试FLAASH大气校正"""
        # 模拟表观反射率数据
        apparent_reflectance = sample_hyperspectral_data * 0.3 + 0.1
        
        # 执行FLAASH校正
        surface_reflectance = self.corrector.flaash_correction(apparent_reflectance)
        
        # 验证校正结果
        assert surface_reflectance.shape == apparent_reflectance.shape
        assert np.all(surface_reflectance >= 0)
        assert np.all(surface_reflectance <= 1)
        
        # 验证大气影响被移除（地表反射率应该更稳定）
        apparent_std = np.std(apparent_reflectance, axis=(0, 1))
        surface_std = np.std(surface_reflectance, axis=(0, 1))
        # 在某些波段，标准差应该减小
        assert np.mean(surface_std) <= np.mean(apparent_std) * 1.1
        
    def test_dark_spectrum_fitting(self, sample_hyperspectral_data):
        """测试暗光谱拟合法"""
        # 模拟包含大气散射的数据
        atmospheric_data = sample_hyperspectral_data + np.random.normal(0, 0.05, 
                                                                      sample_hyperspectral_data.shape)
        
        # 执行暗光谱拟合校正
        corrected_data = self.corrector.dark_spectrum_fitting(atmospheric_data)
        
        # 验证校正结果
        assert corrected_data.shape == atmospheric_data.shape
        
        # 验证暗像素区域的光谱特征
        dark_pixels = np.where(np.mean(atmospheric_data, axis=2) < np.percentile(
            np.mean(atmospheric_data, axis=2), 5))
        if len(dark_pixels[0]) > 0:
            dark_spectrum = np.mean(corrected_data[dark_pixels], axis=0)
            assert np.all(dark_spectrum >= 0)
            
    def test_atmospheric_parameter_estimation(self, sample_hyperspectral_data):
        """测试大气参数估算"""
        # 估算大气参数
        estimated_params = self.corrector.estimate_atmospheric_parameters(
            sample_hyperspectral_data
        )
        
        # 验证估算的参数
        assert 'visibility' in estimated_params
        assert 'water_vapor' in estimated_params
        assert 'aerosol_optical_depth' in estimated_params
        
        # 验证参数的合理性
        assert 5 <= estimated_params['visibility'] <= 100  # km
        assert 0 <= estimated_params['water_vapor'] <= 10   # cm
        assert 0 <= estimated_params['aerosol_optical_depth'] <= 2
        
    def test_adjacency_effect_correction(self, sample_hyperspectral_data):
        """测试邻域效应校正"""
        # 执行邻域效应校正
        corrected_data = self.corrector.correct_adjacency_effect(
            sample_hyperspectral_data, kernel_size=3
        )
        
        # 验证校正结果
        assert corrected_data.shape == sample_hyperspectral_data.shape
        
        # 验证边缘效应被减弱
        edge_difference = self.corrector.calculate_edge_contrast(corrected_data)
        original_difference = self.corrector.calculate_edge_contrast(sample_hyperspectral_data)
        assert edge_difference <= original_difference
        
    def test_correction_validation(self, sample_hyperspectral_data):
        """测试大气校正验证"""
        corrected_data = self.corrector.correct(sample_hyperspectral_data)
        
        # 验证校正质量
        validation_metrics = self.corrector.validate_correction(
            sample_hyperspectral_data, corrected_data
        )
        
        # 验证评估指标
        assert 'spectral_stability' in validation_metrics
        assert 'atmospheric_removal_efficiency' in validation_metrics
        assert 'spatial_consistency' in validation_metrics
        
        # 指标值应该在合理范围内
        assert 0 <= validation_metrics['atmospheric_removal_efficiency'] <= 1


class TestGeometricCorrector:
    """几何校正测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'method': 'polynomial',
            'order': 2,
            'resampling': 'bilinear',
            'target_crs': 'EPSG:4326',
            'pixel_size': 30.0  # meters
        }
        self.corrector = GeometricCorrector(self.config)
        
    def test_geometric_corrector_initialization(self):
        """测试几何校正器初始化"""
        assert self.corrector is not None
        assert hasattr(self.corrector, 'correct')
        assert hasattr(self.corrector, 'register_images')
        
    def test_polynomial_correction(self, sample_hyperspectral_data):
        """测试多项式几何校正"""
        # 模拟控制点
        control_points = {
            'source_points': [(50, 50), (150, 50), (50, 150), (150, 150)],
            'target_points': [(51, 49), (149, 51), (49, 151), (151, 149)]
        }
        
        # 执行多项式校正
        corrected_data, transform_matrix = self.corrector.polynomial_correction(
            sample_hyperspectral_data, control_points
        )
        
        # 验证校正结果
        assert corrected_data.shape[2] == sample_hyperspectral_data.shape[2]  # 波段数不变
        assert transform_matrix.shape == (3, 3)  # 变换矩阵
        
    def test_image_registration(self, sample_hyperspectral_data):
        """测试图像配准"""
        # 创建模拟的参考图像（稍微偏移）
        reference_data = np.roll(sample_hyperspectral_data, shift=(2, 3), axis=(0, 1))
        
        # 执行图像配准
        registered_data, registration_params = self.corrector.register_images(
            sample_hyperspectral_data, reference_data
        )
        
        # 验证配准结果
        assert registered_data.shape == sample_hyperspectral_data.shape
        assert 'translation_x' in registration_params
        assert 'translation_y' in registration_params
        assert 'rotation_angle' in registration_params
        
    def test_resampling_methods(self, sample_hyperspectral_data):
        """测试不同重采样方法"""
        resampling_methods = ['nearest', 'bilinear', 'cubic']
        
        for method in resampling_methods:
            config = self.config.copy()
            config['resampling'] = method
            corrector = GeometricCorrector(config)
            
            # 执行重采样
            resampled_data = corrector.resample(
                sample_hyperspectral_data, scale_factor=0.5
            )
            
            # 验证重采样结果
            expected_shape = (
                sample_hyperspectral_data.shape[0] // 2,
                sample_hyperspectral_data.shape[1] // 2,
                sample_hyperspectral_data.shape[2]
            )
            assert resampled_data.shape == expected_shape
            
    def test_coordinate_transformation(self, sample_hyperspectral_data):
        """测试坐标系统转换"""
        # 模拟源坐标系和目标坐标系
        source_crs = 'EPSG:32633'  # UTM Zone 33N
        target_crs = 'EPSG:4326'   # WGS84
        
        # 模拟地理变换参数
        geotransform = [116.0, 0.0001, 0, 40.0, 0, -0.0001]
        
        # 执行坐标转换
        transformed_data, new_geotransform = self.corrector.transform_coordinates(
            sample_hyperspectral_data, geotransform, source_crs, target_crs
        )
        
        # 验证转换结果
        assert transformed_data.shape[2] == sample_hyperspectral_data.shape[2]
        assert len(new_geotransform) == 6
        
    def test_orthorectification(self, sample_hyperspectral_data):
        """测试正射校正"""
        # 模拟DEM数据
        dem_data = np.random.uniform(0, 1000, sample_hyperspectral_data.shape[:2])
        
        # 模拟传感器参数
        sensor_params = {
            'altitude': 705000,  # meters
            'look_angle': 0,     # degrees
            'azimuth_angle': 180 # degrees
        }
        
        # 执行正射校正
        ortho_data = self.corrector.orthorectification(
            sample_hyperspectral_data, dem_data, sensor_params
        )
        
        # 验证正射校正结果
        assert ortho_data.shape == sample_hyperspectral_data.shape
        
    def test_accuracy_assessment(self, sample_hyperspectral_data):
        """测试几何校正精度评估"""
        # 模拟校正前后的控制点
        before_points = [(50, 50), (150, 50), (50, 150)]
        after_points = [(50.5, 49.8), (149.7, 50.3), (49.9, 150.2)]
        
        # 计算几何精度
        accuracy_metrics = self.corrector.assess_geometric_accuracy(
            before_points, after_points
        )
        
        # 验证精度指标
        assert 'rmse' in accuracy_metrics
        assert 'mean_error' in accuracy_metrics
        assert 'max_error' in accuracy_metrics
        assert accuracy_metrics['rmse'] >= 0


class TestNoiseReducer:
    """噪声去除测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = {
            'method': 'mnf',
            'components': 50,
            'window_size': 3,
            'threshold': 0.95
        }
        self.reducer = NoiseReducer(self.config)
        
    def test_noise_reducer_initialization(self):
        """测试噪声去除器初始化"""
        assert self.reducer is not None
        assert hasattr(self.reducer, 'reduce_noise')
        assert hasattr(self.reducer, 'estimate_noise')
        
    def test_minimum_noise_fraction(self, sample_hyperspectral_data):
        """测试最小噪声分离变换"""
        # 在数据中添加噪声
        noisy_data = sample_hyperspectral_data + np.random.normal(
            0, 0.05, sample_hyperspectral_data.shape
        )
        
        # 执行MNF变换
        denoised_data, mnf_components = self.reducer.minimum_noise_fraction(
            noisy_data, n_components=50
        )
        
        # 验证去噪结果
        assert denoised_data.shape == noisy_data.shape
        assert mnf_components.shape[1] == 50  # 50个主成分
        
        # 验证噪声被减少
        original_noise = np.std(noisy_data - sample_hyperspectral_data)
        residual_noise = np.std(denoised_data - sample_hyperspectral_data)
        assert residual_noise < original_noise
        
    def test_savitzky_golay_filter(self, sample_hyperspectral_data):
        """测试Savitzky-Golay滤波"""
        # 在光谱维度添加噪声
        noisy_spectra = sample_hyperspectral_data + np.random.normal(
            0, 0.02, sample_hyperspectral_data.shape
        )
        
        # 执行S-G滤波
        filtered_data = self.reducer.savitzky_golay_filter(
            noisy_spectra, window_length=5, polyorder=2
        )
        
        # 验证滤波结果
        assert filtered_data.shape == noisy_spectra.shape
        
        # 验证光谱平滑效果
        original_smoothness = self.reducer.calculate_spectral_smoothness(sample_hyperspectral_data)
        filtered_smoothness = self.reducer.calculate_spectral_smoothness(filtered_data)
        assert filtered_smoothness >= original_smoothness
        
    def test_spatial_filtering(self, sample_hyperspectral_data):
        """测试空间滤波"""
        # 添加椒盐噪声
        noisy_data = sample_hyperspectral_data.copy()
        noise_mask = np.random.random(noisy_data.shape[:2]) < 0.05
        noisy_data[noise_mask] = np.random.random((np.sum(noise_mask), noisy_data.shape[2]))
        
        # 执行空间中值滤波
        filtered_data = self.reducer.spatial_median_filter(
            noisy_data, kernel_size=3
        )
        
        # 验证滤波结果
        assert filtered_data.shape == noisy_data.shape
        
        # 验证椒盐噪声被移除
        noise_reduction = np.mean(np.abs(filtered_data - sample_hyperspectral_data))
        original_noise = np.mean(np.abs(noisy_data - sample_hyperspectral_data))
        assert noise_reduction < original_noise
        
    def test_spectral_angle_mapper_denoising(self, sample_hyperspectral_data):
        """测试光谱角度映射去噪"""
        # 创建参考光谱库
        reference_spectra = np.mean(sample_hyperspectral_data.reshape(-1, sample_hyperspectral_data.shape[2]), axis=0)
        
        # 添加噪声
        noisy_data = sample_hyperspectral_data + np.random.normal(0, 0.03, sample_hyperspectral_data.shape)
        
        # 执行SAM去噪
        denoised_data = self.reducer.sam_denoising(
            noisy_data, reference_spectra, threshold=0.1
        )
        
        # 验证去噪结果
        assert denoised_data.shape == noisy_data.shape
        
    def test_noise_estimation(self, sample_hyperspectral_data):
        """测试噪声估算"""
        # 添加已知的噪声
        noise_level = 0.05
        noisy_data = sample_hyperspectral_data + np.random.normal(
            0, noise_level, sample_hyperspectral_data.shape
        )
        
        # 估算噪声水平
        estimated_noise = self.reducer.estimate_noise_level(noisy_data)
        
        # 验证噪声估算精度
        assert 'noise_variance' in estimated_noise
        assert 'snr_per_band' in estimated_noise
        
        # 估算的噪声方差应该接近真实值
        estimated_std = np.sqrt(estimated_noise['noise_variance'])
        assert abs(estimated_std - noise_level) < 0.02
        
    def test_adaptive_filtering(self, sample_hyperspectral_data):
        """测试自适应滤波"""
        # 创建不同区域有不同噪声水平的数据
        noisy_data = sample_hyperspectral_data.copy()
        
        # 区域1：低噪声
        noisy_data[:50, :50, :] += np.random.normal(0, 0.01, (50, 50, sample_hyperspectral_data.shape[2]))
        
        # 区域2：高噪声
        noisy_data[50:, 50:, :] += np.random.normal(0, 0.05, (50, 50, sample_hyperspectral_data.shape[2]))
        
        # 执行自适应滤波
        adaptive_filtered = self.reducer.adaptive_filter(noisy_data)
        
        # 验证自适应滤波结果
        assert adaptive_filtered.shape == noisy_data.shape
        
        # 验证不同区域采用了不同的滤波强度
        low_noise_region = adaptive_filtered[:50, :50, :]
        high_noise_region = adaptive_filtered[50:, 50:, :]
        
        # 高噪声区域应该有更强的平滑效果
        low_smoothness = self.reducer.calculate_spectral_smoothness(low_noise_region)
        high_smoothness = self.reducer.calculate_spectral_smoothness(high_noise_region)
        assert high_smoothness >= low_smoothness


class TestPreprocessingIntegration:
    """预处理模块集成测试"""
    
    def test_complete_preprocessing_pipeline(self, sample_hyperspectral_data):
        """测试完整的预处理流水线"""
        # 1. 初始化所有预处理器
        radiometric_config = {'method': 'gain_offset', 'gain_values': np.ones(200), 'offset_values': np.zeros(200)}
        atmospheric_config = {'method': 'flaash', 'atmospheric_model': 'mid_latitude_summer'}
        geometric_config = {'method': 'polynomial', 'order': 2}
        noise_config = {'method': 'mnf', 'components': 50}
        
        radiometric_corrector = RadiometricCorrector(radiometric_config)
        atmospheric_corrector = AtmosphericCorrector(atmospheric_config)
        geometric_corrector = GeometricCorrector(geometric_config)
        noise_reducer = NoiseReducer(noise_config)
        
        # 2. 执行完整流水线
        # 模拟原始DN值数据
        raw_data = (sample_hyperspectral_data * 10000).astype(np.uint16)
        
        # 辐射定标
        calibrated_data = radiometric_corrector.correct(raw_data)
        
        # 大气校正
        surface_reflectance = atmospheric_corrector.correct(calibrated_data)
        
        # 几何校正
        georeferenced_data, _ = geometric_corrector.correct(surface_reflectance)
        
        # 噪声去除
        final_data = noise_reducer.reduce_noise(georeferenced_data)
        
        # 3. 验证流水线结果
        assert final_data.shape[2] == sample_hyperspectral_data.shape[2]  # 波段数保持不变
        assert np.all(final_data >= 0)  # 反射率非负
        assert np.all(final_data <= 1.2)  # 反射率在合理范围内
        
    def test_preprocessing_quality_metrics(self, sample_hyperspectral_data):
        """测试预处理质量评估"""
        # 模拟预处理前后的数据
        preprocessed_data = sample_hyperspectral_data * 0.95 + 0.02
        
        # 计算质量指标
        quality_metrics = {
            'spectral_fidelity': self._calculate_spectral_fidelity(sample_hyperspectral_data, preprocessed_data),
            'spatial_integrity': self._calculate_spatial_integrity(sample_hyperspectral_data, preprocessed_data),
            'noise_reduction': self._calculate_noise_reduction(sample_hyperspectral_data, preprocessed_data)
        }
        
        # 验证质量指标
        assert 0 <= quality_metrics['spectral_fidelity'] <= 1
        assert 0 <= quality_metrics['spatial_integrity'] <= 1
        assert quality_metrics['noise_reduction'] >= 0
        
    def test_preprocessing_metadata_tracking(self, sample_hyperspectral_data):
        """测试预处理元数据跟踪"""
        # 初始化预处理器
        processor = RadiometricCorrector({'method': 'gain_offset'})
        
        # 执行预处理并跟踪元数据
        processed_data, metadata = processor.correct_with_metadata(
            sample_hyperspectral_data, track_metadata=True
        )
        
        # 验证元数据
        assert 'processing_steps' in metadata
        assert 'processing_date' in metadata
        assert 'input_data_info' in metadata
        assert 'processing_parameters' in metadata
        
    def test_preprocessing_error_handling(self, sample_hyperspectral_data):
        """测试预处理错误处理"""
        corrector = RadiometricCorrector({'method': 'gain_offset'})
        
        # 测试无效输入数据
        invalid_data = np.random.random((10, 10))  # 缺少波段维度
        
        with pytest.raises(ValueError):
            corrector.correct(invalid_data)
            
        # 测试数据类型错误
        wrong_type_data = sample_hyperspectral_data.astype(np.int8)
        
        with pytest.warns(UserWarning):
            result = corrector.correct(wrong_type_data)
            assert result is not None
            
    def test_memory_efficient_preprocessing(self, sample_hyperspectral_data):
        """测试内存高效的预处理"""
        # 模拟大数据处理
        large_data = np.random.random((1000, 1000, 100)).astype(np.float32)
        
        processor = NoiseReducer({'method': 'mnf', 'chunk_size': 100})
        
        # 执行分块处理
        processed_chunks = list(processor.process_in_chunks(large_data, chunk_size=100))
        
        # 验证分块处理结果
        assert len(processed_chunks) > 1
        
        # 重建完整数据
        reconstructed_data = np.concatenate(processed_chunks, axis=0)
        assert reconstructed_data.shape == large_data.shape
        
    def _calculate_spectral_fidelity(self, original, processed):
        """计算光谱保真度"""
        correlation = np.corrcoef(
            original.reshape(-1, original.shape[2]).T,
            processed.reshape(-1, processed.shape[2]).T
        )
        return np.mean(np.diag(correlation[:original.shape[2], original.shape[2]:]))
        
    def _calculate_spatial_integrity(self, original, processed):
        """计算空间完整性"""
        # 计算空间相关性
        original_edges = np.abs(np.gradient(np.mean(original, axis=2)))
        processed_edges = np.abs(np.gradient(np.mean(processed, axis=2)))
        
        correlation = np.corrcoef(original_edges.flatten(), processed_edges.flatten())[0, 1]
        return max(0, correlation)
        
    def _calculate_noise_reduction(self, original, processed):
        """计算噪声减少程度"""
        original_noise = np.std(original, axis=(0, 1))
        processed_noise = np.std(processed, axis=(0, 1))
        
        noise_reduction = np.mean((original_noise - processed_noise) / original_noise)
        return max(0, noise_reduction)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])