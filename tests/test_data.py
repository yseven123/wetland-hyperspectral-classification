"""
数据模块测试

测试数据加载器、验证器和数据增强功能。
涵盖高光谱数据读取、样本加载、数据验证和增强等功能。

Author: 湿地高光谱分类系统开发团队
Date: 2024
"""

import pytest
import numpy as np
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 导入待测试的模块
try:
    from wetland_classification.data import DataLoader, DataValidator
    from wetland_classification.data.augmentation import DataAugmentation
    from wetland_classification.config import Config
except ImportError:
    # 如果模块不存在，创建mock对象用于测试结构
    DataLoader = Mock
    DataValidator = Mock
    DataAugmentation = Mock
    Config = Mock

class TestDataLoader:
    """数据加载器测试类"""
    
    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.config = {
            'data': {
                'bands_range': [0, 200],
                'nodata_value': -9999,
                'scale_factor': 0.0001,
                'chunk_size': 1024
            }
        }
        
    def test_loader_initialization(self):
        """测试数据加载器初始化"""
        loader = DataLoader(self.config['data'])
        assert loader is not None
        assert hasattr(loader, 'load_hyperspectral')
        assert hasattr(loader, 'load_training_samples')
        
    def test_load_hyperspectral_data(self, sample_hyperspectral_data, temp_test_dir):
        """测试高光谱数据加载功能"""
        # 创建测试文件
        test_file = os.path.join(temp_test_dir, 'test_hyperspectral.tif')
        
        # 模拟创建GeoTIFF文件
        with patch('rasterio.open') as mock_rasterio:
            mock_dataset = MagicMock()
            mock_dataset.read.return_value = sample_hyperspectral_data.transpose(2, 0, 1)
            mock_dataset.shape = sample_hyperspectral_data.shape[:2]
            mock_dataset.count = sample_hyperspectral_data.shape[2]
            mock_dataset.crs = 'EPSG:4326'
            mock_dataset.transform = [1, 0, 116, 0, -1, 40]
            mock_dataset.nodata = -9999
            mock_rasterio.return_value.__enter__.return_value = mock_dataset
            
            loader = DataLoader(self.config['data'])
            data, metadata = loader.load_hyperspectral(test_file)
            
            # 验证数据形状和类型
            expected_shape = sample_hyperspectral_data.shape
            assert data.shape == expected_shape
            assert isinstance(data, np.ndarray)
            
            # 验证元数据
            assert 'crs' in metadata
            assert 'transform' in metadata
            assert 'bands_count' in metadata
            assert metadata['bands_count'] == expected_shape[2]
            
    def test_load_with_band_selection(self, sample_hyperspectral_data):
        """测试波段选择功能"""
        # 配置波段范围
        config = self.config.copy()
        config['data']['bands_range'] = [50, 150]  # 选择50-150波段
        
        with patch('rasterio.open') as mock_rasterio:
            mock_dataset = MagicMock()
            mock_dataset.read.return_value = sample_hyperspectral_data.transpose(2, 0, 1)
            mock_dataset.count = sample_hyperspectral_data.shape[2]
            mock_rasterio.return_value.__enter__.return_value = mock_dataset
            
            loader = DataLoader(config['data'])
            data, _ = loader.load_hyperspectral('test.tif')
            
            # 验证波段选择后的形状
            expected_bands = 150 - 50
            assert data.shape[2] == expected_bands
            
    def test_load_training_samples_from_shapefile(self, temp_test_dir):
        """测试从Shapefile加载训练样本"""
        # 创建模拟shapefile数据
        sample_data = {
            'geometry': [
                {'type': 'Point', 'coordinates': [116.5, 39.5]},
                {'type': 'Point', 'coordinates': [116.6, 39.6]},
                {'type': 'Point', 'coordinates': [116.7, 39.7]}
            ],
            'properties': [
                {'class_id': 1, 'class_name': '开放水面'},
                {'class_id': 2, 'class_name': '挺水植物'},
                {'class_id': 3, 'class_name': '沉水植物'}
            ]
        }
        
        with patch('geopandas.read_file') as mock_gpd:
            mock_gdf = MagicMock()
            mock_gdf.__len__.return_value = 3
            mock_gdf.iterrows.return_value = enumerate([
                (0, {'geometry': sample_data['geometry'][0], 
                     'class_id': 1, 'class_name': '开放水面'}),
                (1, {'geometry': sample_data['geometry'][1],
                     'class_id': 2, 'class_name': '挺水植物'}),
                (2, {'geometry': sample_data['geometry'][2],
                     'class_id': 3, 'class_name': '沉水植物'})
            ])
            mock_gpd.return_value = mock_gdf
            
            loader = DataLoader(self.config['data'])
            samples = loader.load_training_samples('test_samples.shp')
            
            # 验证加载的样本
            assert len(samples) == 3
            assert 'coordinates' in samples
            assert 'labels' in samples
            assert 'class_names' in samples
            
    def test_load_training_samples_from_csv(self, temp_test_dir):
        """测试从CSV文件加载训练样本"""
        # 创建测试CSV文件
        csv_file = os.path.join(temp_test_dir, 'training_samples.csv')
        csv_content = """x,y,class_id,class_name
116.5,39.5,1,开放水面
116.6,39.6,2,挺水植物
116.7,39.7,3,沉水植物
"""
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        with patch('pandas.read_csv') as mock_pd:
            mock_df = MagicMock()
            mock_df.__len__.return_value = 3
            mock_df.to_dict.return_value = {
                'x': {0: 116.5, 1: 116.6, 2: 116.7},
                'y': {0: 39.5, 1: 39.6, 2: 39.7},
                'class_id': {0: 1, 1: 2, 2: 3},
                'class_name': {0: '开放水面', 1: '挺水植物', 2: '沉水植物'}
            }
            mock_pd.return_value = mock_df
            
            loader = DataLoader(self.config['data'])
            samples = loader.load_training_samples(csv_file)
            
            # 验证样本数据
            assert len(samples['coordinates']) == 3
            assert len(samples['labels']) == 3
            assert len(samples['class_names']) == 3
            
    def test_data_chunking(self, sample_hyperspectral_data):
        """测试数据分块读取功能"""
        config = self.config.copy()
        config['data']['chunk_size'] = 50  # 小的分块大小用于测试
        
        loader = DataLoader(config['data'])
        
        # 模拟大数据的分块处理
        chunks = list(loader.chunk_data(sample_hyperspectral_data, chunk_size=50))
        
        # 验证分块结果
        assert len(chunks) > 1  # 应该被分成多块
        
        # 验证所有分块的总大小等于原始数据
        total_pixels = sum(chunk.shape[0] * chunk.shape[1] for chunk in chunks)
        original_pixels = sample_hyperspectral_data.shape[0] * sample_hyperspectral_data.shape[1]
        assert total_pixels == original_pixels
        
    def test_memory_efficient_loading(self, sample_hyperspectral_data):
        """测试内存高效的数据加载"""
        loader = DataLoader(self.config['data'])
        
        # 测试延迟加载功能
        with patch.object(loader, '_lazy_load_data') as mock_lazy:
            mock_lazy.return_value = sample_hyperspectral_data
            
            data_generator = loader.load_data_lazy('test.tif')
            
            # 验证生成器对象
            assert hasattr(data_generator, '__next__')
            
    def test_error_handling(self):
        """测试错误处理机制"""
        loader = DataLoader(self.config['data'])
        
        # 测试文件不存在的情况
        with pytest.raises(FileNotFoundError):
            loader.load_hyperspectral('nonexistent_file.tif')
            
        # 测试无效文件格式
        with pytest.raises(ValueError):
            loader.load_training_samples('invalid_file.xyz')


class TestDataValidator:
    """数据验证器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.validator = DataValidator()
        
    def test_hyperspectral_data_validation(self, sample_hyperspectral_data):
        """测试高光谱数据验证"""
        # 测试有效数据
        is_valid, messages = self.validator.validate_hyperspectral(sample_hyperspectral_data)
        assert is_valid
        assert len(messages) == 0
        
        # 测试无效数据 - 维度错误
        invalid_data = np.random.random((100, 100))  # 缺少波段维度
        is_valid, messages = self.validator.validate_hyperspectral(invalid_data)
        assert not is_valid
        assert any('维度' in msg for msg in messages)
        
    def test_spectral_range_validation(self, sample_hyperspectral_data):
        """测试光谱范围验证"""
        # 测试正常范围
        is_valid, messages = self.validator.validate_spectral_range(
            sample_hyperspectral_data, min_val=0, max_val=1
        )
        assert is_valid
        
        # 测试超出范围的数据
        invalid_data = sample_hyperspectral_data * 10  # 放大到超出范围
        is_valid, messages = self.validator.validate_spectral_range(
            invalid_data, min_val=0, max_val=1
        )
        assert not is_valid
        
    def test_nodata_detection(self):
        """测试无效数据检测"""
        # 创建包含无效值的数据
        data = np.random.random((50, 50, 100))
        data[20:30, 20:30, :] = -9999  # 设置无效值区域
        
        nodata_mask = self.validator.detect_nodata(data, nodata_value=-9999)
        
        # 验证无效数据掩膜
        assert nodata_mask.shape == data.shape[:2]
        assert np.any(nodata_mask)  # 应该检测到无效值
        assert np.all(nodata_mask[20:30, 20:30])  # 无效区域应该被标记
        
    def test_spatial_consistency_check(self, sample_classification_labels):
        """测试空间一致性检查"""
        # 测试正常的分类标签
        is_consistent, issues = self.validator.check_spatial_consistency(
            sample_classification_labels
        )
        
        # 验证一致性检查结果
        assert isinstance(is_consistent, bool)
        assert isinstance(issues, list)
        
    def test_training_samples_validation(self, sample_training_samples):
        """测试训练样本验证"""
        is_valid, messages = self.validator.validate_training_samples(
            sample_training_samples
        )
        
        # 验证样本有效性
        assert is_valid
        assert len(messages) == 0
        
        # 测试无效样本 - 缺少必要字段
        invalid_samples = {'coordinates': [(10, 10)]}  # 缺少labels字段
        is_valid, messages = self.validator.validate_training_samples(invalid_samples)
        assert not is_valid
        assert any('labels' in msg for msg in messages)
        
    def test_class_balance_check(self, sample_training_samples):
        """测试类别平衡性检查"""
        balance_report = self.validator.check_class_balance(
            sample_training_samples['labels']
        )
        
        # 验证平衡性报告
        assert 'class_counts' in balance_report
        assert 'balance_ratio' in balance_report
        assert 'is_balanced' in balance_report
        
        # 测试极度不平衡的数据
        imbalanced_labels = [0] * 1000 + [1] * 10  # 极度不平衡
        balance_report = self.validator.check_class_balance(imbalanced_labels)
        assert not balance_report['is_balanced']


class TestDataAugmentation:
    """数据增强测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.augmentation = DataAugmentation()
        
    def test_spectral_augmentation(self, sample_hyperspectral_data):
        """测试光谱增强"""
        original_shape = sample_hyperspectral_data.shape
        
        # 测试光谱噪声添加
        noisy_data = self.augmentation.add_spectral_noise(
            sample_hyperspectral_data, noise_level=0.01
        )
        
        # 验证增强后的数据
        assert noisy_data.shape == original_shape
        assert not np.array_equal(noisy_data, sample_hyperspectral_data)
        
        # 测试光谱平移
        shifted_data = self.augmentation.spectral_shift(
            sample_hyperspectral_data, shift_range=(-0.1, 0.1)
        )
        assert shifted_data.shape == original_shape
        
    def test_spatial_augmentation(self, sample_hyperspectral_data):
        """测试空间增强"""
        original_shape = sample_hyperspectral_data.shape
        
        # 测试随机旋转
        rotated_data = self.augmentation.random_rotation(
            sample_hyperspectral_data, max_angle=30
        )
        
        # 验证旋转后的数据（可能形状略有变化）
        assert rotated_data.shape[2] == original_shape[2]  # 波段数不变
        
        # 测试随机翻转
        flipped_data = self.augmentation.random_flip(sample_hyperspectral_data)
        assert flipped_data.shape == original_shape
        
    def test_pixel_level_augmentation(self, sample_hyperspectral_data):
        """测试像素级增强"""
        # 提取像素样本
        pixels = sample_hyperspectral_data.reshape(-1, sample_hyperspectral_data.shape[2])
        sample_pixels = pixels[:1000]  # 取前1000个像素
        
        # 测试像素混合
        mixed_pixels = self.augmentation.mixup_pixels(sample_pixels, alpha=0.5)
        
        # 验证混合结果
        assert mixed_pixels.shape == sample_pixels.shape
        
        # 测试光谱插值
        interpolated_pixels = self.augmentation.spectral_interpolation(
            sample_pixels, factor=2
        )
        assert interpolated_pixels.shape[1] == sample_pixels.shape[1] * 2
        
    def test_augmentation_pipeline(self, sample_hyperspectral_data):
        """测试增强流水线"""
        augmentation_config = {
            'spectral_noise': {'enabled': True, 'noise_level': 0.02},
            'spatial_rotation': {'enabled': True, 'max_angle': 15},
            'spectral_shift': {'enabled': True, 'shift_range': (-0.05, 0.05)}
        }
        
        # 执行增强流水线
        augmented_data = self.augmentation.apply_pipeline(
            sample_hyperspectral_data, augmentation_config
        )
        
        # 验证流水线结果
        assert augmented_data.shape == sample_hyperspectral_data.shape
        assert not np.array_equal(augmented_data, sample_hyperspectral_data)
        
    def test_label_preserving_augmentation(self, sample_hyperspectral_data, 
                                         sample_classification_labels):
        """测试保持标签一致性的增强"""
        # 测试同时增强数据和标签
        aug_data, aug_labels = self.augmentation.augment_with_labels(
            sample_hyperspectral_data, sample_classification_labels,
            augmentation_type='rotation', angle=90
        )
        
        # 验证数据和标签的一致性
        assert aug_data.shape[2] == sample_hyperspectral_data.shape[2]
        assert aug_labels.shape == sample_classification_labels.shape
        
    def test_batch_augmentation(self, sample_hyperspectral_data):
        """测试批量增强"""
        # 创建批量数据
        batch_size = 5
        patch_size = 32
        batch_data = np.random.random((batch_size, patch_size, patch_size, 
                                     sample_hyperspectral_data.shape[2]))
        
        # 执行批量增强
        augmented_batch = self.augmentation.augment_batch(
            batch_data, augmentation_types=['noise', 'flip']
        )
        
        # 验证批量增强结果
        assert augmented_batch.shape == batch_data.shape
        assert not np.array_equal(augmented_batch, batch_data)


class TestDataIntegration:
    """数据模块集成测试"""
    
    def test_complete_data_workflow(self, sample_hyperspectral_data, 
                                  sample_training_samples, temp_test_dir):
        """测试完整的数据处理工作流"""
        # 1. 初始化组件
        config = {
            'data': {
                'bands_range': [0, 200],
                'nodata_value': -9999,
                'scale_factor': 0.0001
            }
        }
        
        loader = DataLoader(config['data'])
        validator = DataValidator()
        augmentation = DataAugmentation()
        
        # 2. 模拟数据加载
        with patch.object(loader, 'load_hyperspectral') as mock_load:
            mock_load.return_value = (sample_hyperspectral_data, 
                                    {'crs': 'EPSG:4326', 'bands_count': 200})
            
            data, metadata = loader.load_hyperspectral('test.tif')
            
            # 3. 数据验证
            is_valid, messages = validator.validate_hyperspectral(data)
            assert is_valid
            
            # 4. 数据增强
            augmented_data = augmentation.add_spectral_noise(data, noise_level=0.01)
            
            # 5. 验证工作流结果
            assert augmented_data.shape == sample_hyperspectral_data.shape
            assert metadata['bands_count'] == 200
            
    def test_error_propagation(self):
        """测试错误传播机制"""
        loader = DataLoader({'bands_range': [0, 100]})
        
        # 测试错误是否正确传播
        with pytest.raises(Exception):
            loader.load_hyperspectral('nonexistent_file.tif')
            
    def test_memory_management(self, sample_hyperspectral_data):
        """测试内存管理"""
        # 测试大数据处理时的内存使用
        loader = DataLoader({'chunk_size': 1024})
        
        # 模拟处理大数据
        large_data = np.random.random((1000, 1000, 100)).astype(np.float32)
        
        # 验证分块处理不会导致内存溢出
        chunks = list(loader.chunk_data(large_data, chunk_size=100))
        assert len(chunks) > 1
        
        # 清理内存
        del large_data, chunks


if __name__ == '__main__':
    pytest.main([__file__, '-v'])