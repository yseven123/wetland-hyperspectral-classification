"""
湿地高光谱后处理模块
==================

这个模块提供了完整的分类结果后处理功能，用于提高分类精度和空间一致性。

主要组件：
- 空间滤波: 多种滤波算法去除分类噪声
- 形态学操作: 连通性分析、区域处理、边界优化
- 一致性处理: 时序和空间一致性检查与修正
- 统一流水线: 完整的后处理工作流

作者: 湿地遥感研究团队
日期: 2024
版本: 1.0.0
"""

from .spatial_filter import (
    SpatialFilter,
    MedianFilter,
    GaussianFilter,
    BilateralFilter,
    MorphologicalFilter,
    AdaptiveFilter,
    MarkovRandomFieldFilter,
    SpatialFilterPipeline,
    create_wetland_filter_pipeline,
    create_filter
)

from .morphology import (
    MorphologyProcessor,
    ConnectedComponentAnalyzer,
    RegionProcessor,
    HoleFillingProcessor,
    SkeletonProcessor,
    create_morphology_pipeline,
    apply_morphology_pipeline
)

from .consistency import (
    ConsistencyProcessor,
    TemporalConsistencyProcessor,
    SpatialConsistencyProcessor,
    create_consistency_pipeline,
    apply_consistency_check
)

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# 版本信息
__version__ = "1.0.0"
__author__ = "湿地遥感研究团队"

# 设置日志
logger = logging.getLogger(__name__)

# 所有可用的处理器类型
AVAILABLE_PROCESSORS = {
    # 空间滤波器
    'spatial_filters': {
        'median': MedianFilter,
        'gaussian': GaussianFilter,
        'bilateral': BilateralFilter,
        'morphological': MorphologicalFilter,
        'adaptive': AdaptiveFilter,
        'mrf': MarkovRandomFieldFilter
    },
    
    # 形态学处理器
    'morphology_processors': {
        'connected_components': ConnectedComponentAnalyzer,
        'region_processor': RegionProcessor,
        'hole_filling': HoleFillingProcessor,
        'skeleton': SkeletonProcessor
    },
    
    # 一致性处理器
    'consistency_processors': {
        'temporal': TemporalConsistencyProcessor,
        'spatial': SpatialConsistencyProcessor
    }
}

# 导出的公共接口
__all__ = [
    # 空间滤波
    'SpatialFilter',
    'MedianFilter',
    'GaussianFilter', 
    'BilateralFilter',
    'MorphologicalFilter',
    'AdaptiveFilter',
    'MarkovRandomFieldFilter',
    'SpatialFilterPipeline',
    
    # 形态学操作
    'MorphologyProcessor',
    'ConnectedComponentAnalyzer',
    'RegionProcessor',
    'HoleFillingProcessor',
    'SkeletonProcessor',
    
    # 一致性处理
    'ConsistencyProcessor',
    'TemporalConsistencyProcessor',
    'SpatialConsistencyProcessor',
    
    # 流水线和工厂函数
    'PostProcessingPipeline',
    'create_processor',
    'create_postprocessing_pipeline',
    'create_wetland_postprocessing_suite',
    
    # 便捷函数
    'apply_spatial_filtering',
    'apply_morphology_processing',
    'apply_consistency_check',
    'quick_postprocess',
    
    # 工具函数
    'evaluate_postprocessing_quality',
    'visualize_postprocessing_results',
    'compare_postprocessing_methods',
    
    # 常量
    'AVAILABLE_PROCESSORS'
]


class PostProcessingPipeline:
    """
    后处理流水线
    
    统一管理和执行多个后处理步骤的完整流水线。
    """
    
    def __init__(self, 
                 processors: Optional[List] = None,
                 enable_logging: bool = True,
                 save_intermediate: bool = False):
        """
        初始化后处理流水线
        
        Parameters:
        -----------
        processors : list, optional
            处理器列表
        enable_logging : bool, default=True
            是否启用详细日志
        save_intermediate : bool, default=False
            是否保存中间结果
        """
        self.processors = processors or []
        self.enable_logging = enable_logging
        self.save_intermediate = save_intermediate
        
        self.processing_history = []
        self.intermediate_results = {}
        self.quality_metrics = {}
    
    def add_processor(self, processor, name: Optional[str] = None):
        """
        添加处理器
        
        Parameters:
        -----------
        processor : object
            处理器实例
        name : str, optional
            处理器名称
        """
        if name is None:
            name = processor.__class__.__name__
        
        self.processors.append({
            'processor': processor,
            'name': name
        })
        
        if self.enable_logging:
            logger.info(f"已添加处理器: {name}")
    
    def process(self, 
                classification_map: Union[np.ndarray, List[np.ndarray]],
                auxiliary_data: Optional[Dict[str, np.ndarray]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行后处理流水线
        
        Parameters:
        -----------
        classification_map : np.ndarray or list
            分类图像或时序分类图像列表
        auxiliary_data : dict, optional
            辅助数据
            
        Returns:
        --------
        results : dict
            处理结果和统计信息
        """
        if self.enable_logging:
            logger.info(f"开始后处理流水线 - 处理器数量: {len(self.processors)}")
        
        start_time = time.time()
        
        # 初始化结果
        results = {
            'original': classification_map.copy() if isinstance(classification_map, np.ndarray) 
                       else [img.copy() for img in classification_map],
            'steps': [],
            'statistics': {},
            'quality_metrics': {}
        }
        
        current_data = classification_map
        self.processing_history = []
        self.intermediate_results = {}
        
        # 执行每个处理器
        for i, processor_info in enumerate(self.processors):
            processor = processor_info['processor']
            name = processor_info['name']
            
            if self.enable_logging:
                logger.info(f"执行处理器 {i+1}/{len(self.processors)}: {name}")
            
            step_start = time.time()
            
            try:
                # 根据处理器类型调用相应方法
                if hasattr(processor, 'process'):
                    if isinstance(processor, (TemporalConsistencyProcessor, SpatialConsistencyProcessor)):
                        # 一致性处理器
                        if hasattr(processor, 'check_consistency'):
                            consistency_result = processor.check_consistency(
                                current_data, auxiliary_data=auxiliary_data, **kwargs
                            )
                            step_result = processor.correct_inconsistency(current_data, **kwargs)
                            
                            # 保存一致性检查结果
                            results['statistics'][f'{name}_consistency'] = consistency_result
                        else:
                            step_result = processor.process(current_data, **kwargs)
                    else:
                        # 其他处理器
                        step_result = processor.process(current_data, **kwargs)
                        
                elif hasattr(processor, 'apply'):
                    # 滤波器
                    if isinstance(current_data, list):
                        # 对时序数据的每个时相分别处理
                        step_result = []
                        for img in current_data:
                            filtered = processor.apply(img, **kwargs)
                            step_result.append(filtered)
                    else:
                        step_result = processor.apply(current_data, **kwargs)
                        
                else:
                    logger.warning(f"处理器 {name} 没有可识别的处理方法")
                    step_result = current_data
                
                # 计算处理时间
                step_time = time.time() - step_start
                
                # 计算变化统计
                change_stats = self._calculate_change_statistics(current_data, step_result)
                
                # 记录步骤信息
                step_info = {
                    'processor_name': name,
                    'step_number': i + 1,
                    'processing_time': step_time,
                    'change_statistics': change_stats
                }
                
                results['steps'].append(step_info)
                self.processing_history.append(step_info)
                
                # 保存中间结果
                if self.save_intermediate:
                    self.intermediate_results[f'step_{i+1}_{name}'] = (
                        step_result.copy() if isinstance(step_result, np.ndarray)
                        else [img.copy() for img in step_result]
                    )
                
                # 更新当前数据
                current_data = step_result
                
                if self.enable_logging:
                    logger.info(f"{name} 完成 - 耗时: {step_time:.2f}秒, 变化像素: {change_stats.get('changed_pixels', 0)}")
                
            except Exception as e:
                logger.error(f"处理器 {name} 执行失败: {e}")
                # 继续使用原数据
                step_result = current_data
        
        # 最终结果
        results['final'] = current_data
        
        # 总体统计
        total_time = time.time() - start_time
        results['total_processing_time'] = total_time
        results['intermediate_results'] = self.intermediate_results
        
        # 质量评估
        if isinstance(classification_map, np.ndarray) and isinstance(current_data, np.ndarray):
            quality_metrics = self._evaluate_processing_quality(
                classification_map, current_data
            )
            results['quality_metrics'] = quality_metrics
            self.quality_metrics = quality_metrics
        
        if self.enable_logging:
            logger.info(f"后处理流水线完成 - 总耗时: {total_time:.2f}秒")
        
        return results
    
    def _calculate_change_statistics(self, 
                                   original: Union[np.ndarray, List[np.ndarray]], 
                                   processed: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
        """计算变化统计"""
        try:
            if isinstance(original, list) and isinstance(processed, list):
                # 时序数据
                total_changed = 0
                total_pixels = 0
                
                for orig_img, proc_img in zip(original, processed):
                    changed = np.sum(orig_img != proc_img)
                    total_changed += changed
                    total_pixels += orig_img.size
                
                change_ratio = total_changed / total_pixels if total_pixels > 0 else 0
                
                return {
                    'changed_pixels': total_changed,
                    'total_pixels': total_pixels,
                    'change_ratio': change_ratio,
                    'data_type': 'temporal'
                }
                
            elif isinstance(original, np.ndarray) and isinstance(processed, np.ndarray):
                # 单一图像
                changed_pixels = np.sum(original != processed)
                total_pixels = original.size
                change_ratio = changed_pixels / total_pixels
                
                return {
                    'changed_pixels': changed_pixels,
                    'total_pixels': total_pixels,
                    'change_ratio': change_ratio,
                    'data_type': 'single'
                }
            else:
                return {'changed_pixels': 0, 'total_pixels': 0, 'change_ratio': 0.0}
                
        except Exception as e:
            logger.warning(f"计算变化统计失败: {e}")
            return {'changed_pixels': 0, 'total_pixels': 0, 'change_ratio': 0.0}
    
    def _evaluate_processing_quality(self, 
                                   original: np.ndarray, 
                                   processed: np.ndarray) -> Dict[str, float]:
        """评估处理质量"""
        try:
            from scipy import ndimage
            from skimage import measure
            
            # 空间平滑度（邻域一致性）
            def calculate_smoothness(img):
                # 计算每个像素与其邻域的一致性
                kernel = np.ones((3, 3))
                kernel[1, 1] = 0
                
                smoothness_map = np.zeros_like(img, dtype=float)
                for i in range(1, img.shape[0] - 1):
                    for j in range(1, img.shape[1] - 1):
                        center = img[i, j]
                        neighborhood = img[i-1:i+2, j-1:j+2]
                        neighborhood = neighborhood[kernel == 1]
                        smoothness_map[i, j] = np.sum(neighborhood == center) / len(neighborhood)
                
                return np.mean(smoothness_map)
            
            original_smoothness = calculate_smoothness(original)
            processed_smoothness = calculate_smoothness(processed)
            
            # 连通组件数量变化
            orig_components = []
            proc_components = []
            
            for class_id in np.unique(original):
                if class_id == 0:
                    continue
                
                # 原始图像的连通组件
                orig_mask = original == class_id
                _, orig_num = ndimage.label(orig_mask)
                orig_components.append(orig_num)
                
                # 处理后图像的连通组件
                proc_mask = processed == class_id
                _, proc_num = ndimage.label(proc_mask)
                proc_components.append(proc_num)
            
            avg_orig_components = np.mean(orig_components) if orig_components else 0
            avg_proc_components = np.mean(proc_components) if proc_components else 0
            
            # 边界长度变化（复杂度）
            def calculate_boundary_length(img):
                # 使用Sobel算子检测边界
                from scipy import ndimage
                sobel_x = ndimage.sobel(img.astype(float), axis=0)
                sobel_y = ndimage.sobel(img.astype(float), axis=1)
                boundary_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                return np.sum(boundary_magnitude > 0)
            
            orig_boundary_length = calculate_boundary_length(original)
            proc_boundary_length = calculate_boundary_length(processed)
            
            quality_metrics = {
                'original_smoothness': original_smoothness,
                'processed_smoothness': processed_smoothness,
                'smoothness_improvement': processed_smoothness - original_smoothness,
                'original_avg_components': avg_orig_components,
                'processed_avg_components': avg_proc_components,
                'component_reduction': avg_orig_components - avg_proc_components,
                'original_boundary_length': orig_boundary_length,
                'processed_boundary_length': proc_boundary_length,
                'boundary_simplification': (orig_boundary_length - proc_boundary_length) / orig_boundary_length
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            return {}
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        if not self.processing_history:
            return {'message': '尚未执行任何处理'}
        
        total_time = sum(step['processing_time'] for step in self.processing_history)
        total_changes = sum(step['change_statistics'].get('changed_pixels', 0) 
                          for step in self.processing_history)
        
        summary = {
            'total_steps': len(self.processing_history),
            'total_processing_time': total_time,
            'total_pixel_changes': total_changes,
            'processing_steps': [step['processor_name'] for step in self.processing_history],
            'quality_metrics': self.quality_metrics
        }
        
        return summary
    
    def visualize_pipeline_results(self, 
                                 results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """可视化流水线结果"""
        if isinstance(results['original'], list):
            # 时序数据可视化
            return self._visualize_temporal_results(results, save_path)
        else:
            # 单一图像可视化
            return self._visualize_single_results(results, save_path)
    
    def _visualize_single_results(self, 
                                results: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """可视化单一图像处理结果"""
        original = results['original']
        final = results['final']
        steps = results['steps']
        
        # 创建子图
        n_intermediates = min(4, len(self.intermediate_results))  # 最多显示4个中间结果
        n_cols = 3 + n_intermediates
        
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        if n_cols == 1:
            axes = axes.reshape(2, 1)
        
        # 原始图像
        im1 = axes[0, 0].imshow(original, cmap='tab20')
        axes[0, 0].set_title('原始分类结果')
        axes[0, 0].axis('off')
        
        # 中间结果
        intermediate_items = list(self.intermediate_results.items())[:n_intermediates]
        for i, (step_name, step_result) in enumerate(intermediate_items):
            col_idx = i + 1
            axes[0, col_idx].imshow(step_result, cmap='tab20')
            axes[0, col_idx].set_title(step_name.split('_')[-1])
            axes[0, col_idx].axis('off')
        
        # 最终结果
        im2 = axes[0, -1].imshow(final, cmap='tab20')
        axes[0, -1].set_title('最终结果')
        axes[0, -1].axis('off')
        
        # 变化图
        change_map = (original != final).astype(int)
        axes[1, 0].imshow(change_map, cmap='Reds')
        axes[1, 0].set_title(f'变化像素 ({np.sum(change_map)})')
        axes[1, 0].axis('off')
        
        # 处理时间统计
        if len(steps) > 0:
            step_names = [step['processor_name'] for step in steps]
            step_times = [step['processing_time'] for step in steps]
            
            axes[1, 1].bar(range(len(step_names)), step_times)
            axes[1, 1].set_xticks(range(len(step_names)))
            axes[1, 1].set_xticklabels(step_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('处理时间 (秒)')
            axes[1, 1].set_title('各步骤处理时间')
        
        # 变化率统计
        if len(steps) > 0:
            change_ratios = [step['change_statistics'].get('change_ratio', 0) for step in steps]
            
            axes[1, 2].plot(range(len(step_names)), change_ratios, 'o-')
            axes[1, 2].set_xticks(range(len(step_names)))
            axes[1, 2].set_xticklabels(step_names, rotation=45, ha='right')
            axes[1, 2].set_ylabel('变化率')
            axes[1, 2].set_title('各步骤变化率')
        
        # 隐藏多余的子图
        for col in range(n_intermediates + 1, n_cols - 1):
            axes[1, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _visualize_temporal_results(self, 
                                  results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """可视化时序数据处理结果"""
        original_seq = results['original']
        final_seq = results['final']
        
        n_periods = len(original_seq)
        fig, axes = plt.subplots(3, n_periods, figsize=(4*n_periods, 12))
        
        if n_periods == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(n_periods):
            # 原始图像
            axes[0, i].imshow(original_seq[i], cmap='tab20')
            axes[0, i].set_title(f'原始 T{i+1}')
            axes[0, i].axis('off')
            
            # 处理后图像
            axes[1, i].imshow(final_seq[i], cmap='tab20')
            axes[1, i].set_title(f'处理后 T{i+1}')
            axes[1, i].axis('off')
            
            # 变化图
            change_map = (original_seq[i] != final_seq[i]).astype(int)
            axes[2, i].imshow(change_map, cmap='Reds')
            axes[2, i].set_title(f'变化 T{i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_processor(processor_type: str, category: str, **kwargs):
    """
    创建处理器实例
    
    Parameters:
    -----------
    processor_type : str
        处理器类型
    category : str
        处理器类别 ('spatial_filters', 'morphology_processors', 'consistency_processors')
    **kwargs : dict
        处理器参数
        
    Returns:
    --------
    processor : object
        处理器实例
    """
    if category not in AVAILABLE_PROCESSORS:
        available_categories = list(AVAILABLE_PROCESSORS.keys())
        raise ValueError(f"不支持的处理器类别: {category}. 可用类别: {available_categories}")
    
    category_processors = AVAILABLE_PROCESSORS[category]
    
    if processor_type not in category_processors:
        available_types = list(category_processors.keys())
        raise ValueError(f"类别 {category} 中不支持的处理器类型: {processor_type}. 可用类型: {available_types}")
    
    processor_class = category_processors[processor_type]
    return processor_class(**kwargs)


def create_postprocessing_pipeline(config: Optional[Dict[str, Any]] = None) -> PostProcessingPipeline:
    """
    创建后处理流水线
    
    Parameters:
    -----------
    config : dict, optional
        流水线配置
        
    Returns:
    --------
    pipeline : PostProcessingPipeline
        后处理流水线
    """
    config = config or {}
    
    pipeline = PostProcessingPipeline(
        enable_logging=config.get('enable_logging', True),
        save_intermediate=config.get('save_intermediate', False)
    )
    
    # 添加默认处理器
    processors_config = config.get('processors', [])
    
    if not processors_config:
        # 使用默认配置
        processors_config = [
            {'type': 'median', 'category': 'spatial_filters', 'params': {'filter_size': 3}},
            {'type': 'connected_components', 'category': 'morphology_processors', 'params': {'min_size': 10}},
            {'type': 'hole_filling', 'category': 'morphology_processors', 'params': {'min_hole_size': 5}},
            {'type': 'spatial', 'category': 'consistency_processors', 'params': {'homogeneity_threshold': 0.7}}
        ]
    
    for proc_config in processors_config:
        processor = create_processor(
            proc_config['type'],
            proc_config['category'],
            **proc_config.get('params', {})
        )
        pipeline.add_processor(processor, proc_config.get('name'))
    
    return pipeline


def create_wetland_postprocessing_suite() -> PostProcessingPipeline:
    """
    创建专门针对湿地分类的后处理套件
    
    Returns:
    --------
    pipeline : PostProcessingPipeline
        湿地后处理流水线
    """
    logger.info("创建湿地专用后处理套件")
    
    pipeline = PostProcessingPipeline(enable_logging=True, save_intermediate=True)
    
    # 1. 噪声去除 - 中值滤波
    median_filter = create_processor('median', 'spatial_filters', filter_size=3, iterations=1)
    pipeline.add_processor(median_filter, 'noise_removal')
    
    # 2. 连通组件分析 - 移除小斑块
    cc_analyzer = create_processor('connected_components', 'morphology_processors', 
                                 min_size=15, analyze_properties=True)
    pipeline.add_processor(cc_analyzer, 'small_patch_removal')
    
    # 3. 空洞填充 - 填补内部空洞
    hole_filler = create_processor('hole_filling', 'morphology_processors',
                                 min_hole_size=8, fill_strategy='majority')
    pipeline.add_processor(hole_filler, 'hole_filling')
    
    # 4. 区域处理 - 边界平滑
    region_processor = create_processor('region_processor', 'morphology_processors',
                                      boundary_smooth=True, merge_threshold=0.1)
    pipeline.add_processor(region_processor, 'boundary_smoothing')
    
    # 5. 自适应滤波 - 局部优化
    adaptive_filter = create_processor('adaptive', 'spatial_filters',
                                     window_size=5, threshold_ratio=0.6)
    pipeline.add_processor(adaptive_filter, 'adaptive_optimization')
    
    # 6. 空间一致性检查
    spatial_consistency = create_processor('spatial', 'consistency_processors',
                                         homogeneity_threshold=0.8, neighborhood_size=3)
    pipeline.add_processor(spatial_consistency, 'spatial_consistency')
    
    # 7. 轻微高斯平滑 - 最终平滑
    gaussian_filter = create_processor('gaussian', 'spatial_filters', sigma=0.8)
    pipeline.add_processor(gaussian_filter, 'final_smoothing')
    
    logger.info("湿地后处理套件创建完成")
    
    return pipeline


# 便捷函数
def apply_spatial_filtering(classification_map: np.ndarray,
                          filter_type: str = 'median',
                          **kwargs) -> np.ndarray:
    """
    应用空间滤波
    
    Parameters:
    -----------
    classification_map : np.ndarray
        分类图像
    filter_type : str, default='median'
        滤波器类型
    **kwargs : dict
        滤波器参数
        
    Returns:
    --------
    filtered_map : np.ndarray
        滤波后的图像
    """
    filter_obj = create_processor(filter_type, 'spatial_filters', **kwargs)
    return filter_obj.apply(classification_map)


def apply_morphology_processing(classification_map: np.ndarray,
                              processor_type: str = 'connected_components',
                              **kwargs) -> np.ndarray:
    """
    应用形态学处理
    
    Parameters:
    -----------
    classification_map : np.ndarray
        分类图像
    processor_type : str, default='connected_components'
        处理器类型
    **kwargs : dict
        处理器参数
        
    Returns:
    --------
    processed_map : np.ndarray
        处理后的图像
    """
    processor = create_processor(processor_type, 'morphology_processors', **kwargs)
    result = processor.process(classification_map, **kwargs)
    
    # 根据处理器类型返回相应结果
    if isinstance(result, dict):
        return result.get('final', result.get('processed_map', classification_map))
    else:
        return result


def quick_postprocess(classification_map: Union[np.ndarray, List[np.ndarray]],
                     method: str = 'standard',
                     **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    """
    快速后处理
    
    Parameters:
    -----------
    classification_map : np.ndarray or list
        分类图像或时序分类图像列表
    method : str, default='standard'
        处理方法 ('standard', 'wetland', 'conservative', 'aggressive')
    **kwargs : dict
        其他参数
        
    Returns:
    --------
    processed_map : np.ndarray or list
        处理后的分类图像
    """
    logger.info(f"执行快速后处理 - 方法: {method}")
    
    if method == 'wetland':
        pipeline = create_wetland_postprocessing_suite()
    elif method == 'conservative':
        # 保守处理：只进行基本的噪声去除
        pipeline = PostProcessingPipeline()
        median_filter = create_processor('median', 'spatial_filters', filter_size=3)
        pipeline.add_processor(median_filter)
    elif method == 'aggressive':
        # 激进处理：更多的平滑和约束
        pipeline = PostProcessingPipeline()
        # 添加多个处理步骤
        for filter_size in [3, 5]:
            gaussian_filter = create_processor('gaussian', 'spatial_filters', sigma=filter_size*0.3)
            pipeline.add_processor(gaussian_filter)
    else:
        # 标准处理
        pipeline = create_postprocessing_pipeline()
    
    results = pipeline.process(classification_map, **kwargs)
    return results['final']


def evaluate_postprocessing_quality(original: np.ndarray,
                                  processed: np.ndarray,
                                  ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    评估后处理质量
    
    Parameters:
    -----------
    original : np.ndarray
        原始分类图像
    processed : np.ndarray
        处理后分类图像
    ground_truth : np.ndarray, optional
        真实标签
        
    Returns:
    --------
    metrics : dict
        质量评估指标
    """
    pipeline = PostProcessingPipeline()
    quality_metrics = pipeline._evaluate_processing_quality(original, processed)
    
    if ground_truth is not None:
        # 添加精度相关指标
        from sklearn.metrics import accuracy_score, classification_report
        
        original_accuracy = accuracy_score(ground_truth.flatten(), original.flatten())
        processed_accuracy = accuracy_score(ground_truth.flatten(), processed.flatten())
        
        quality_metrics.update({
            'original_accuracy': original_accuracy,
            'processed_accuracy': processed_accuracy,
            'accuracy_improvement': processed_accuracy - original_accuracy
        })
    
    return quality_metrics


def visualize_postprocessing_results(original: np.ndarray,
                                   processed: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    可视化后处理结果
    
    Parameters:
    -----------
    original : np.ndarray
        原始分类图像
    processed : np.ndarray
        处理后分类图像
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图形对象
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    im1 = axes[0].imshow(original, cmap='tab20')
    axes[0].set_title('原始分类结果')
    axes[0].axis('off')
    
    # 处理后图像
    im2 = axes[1].imshow(processed, cmap='tab20')
    axes[1].set_title('后处理结果')
    axes[1].axis('off')
    
    # 变化图
    change_map = (original != processed).astype(int)
    im3 = axes[2].imshow(change_map, cmap='Reds')
    axes[2].set_title(f'变化像素 ({np.sum(change_map)})')
    axes[2].axis('off')
    
    # 添加颜色条
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_postprocessing_methods(classification_map: np.ndarray,
                                 methods: List[str] = None,
                                 ground_truth: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    比较不同后处理方法
    
    Parameters:
    -----------
    classification_map : np.ndarray
        原始分类图像
    methods : list, optional
        要比较的方法列表
    ground_truth : np.ndarray, optional
        真实标签
    save_path : str, optional
        保存路径
        
    Returns:
    --------
    comparison_results : dict
        比较结果
    """
    if methods is None:
        methods = ['conservative', 'standard', 'wetland', 'aggressive']
    
    logger.info(f"比较 {len(methods)} 种后处理方法")
    
    results = {'original': classification_map}
    quality_metrics = {}
    processing_times = {}
    
    for method in methods:
        logger.info(f"测试方法: {method}")
        start_time = time.time()
        
        processed = quick_postprocess(classification_map, method=method)
        processing_time = time.time() - start_time
        
        results[method] = processed
        processing_times[method] = processing_time
        
        # 评估质量
        metrics = evaluate_postprocessing_quality(classification_map, processed, ground_truth)
        quality_metrics[method] = metrics
    
    # 创建比较图
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods + 1), 10))
    
    # 原始图像
    axes[0, 0].imshow(classification_map, cmap='tab20')
    axes[0, 0].set_title('原始分类结果')
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')  # 空白
    
    # 各种方法的结果
    for i, method in enumerate(methods):
        col_idx = i + 1
        
        # 处理结果
        axes[0, col_idx].imshow(results[method], cmap='tab20')
        axes[0, col_idx].set_title(f'{method}\n({processing_times[method]:.2f}s)')
        axes[0, col_idx].axis('off')
        
        # 变化图
        change_map = (classification_map != results[method]).astype(int)
        axes[1, col_idx].imshow(change_map, cmap='Reds')
        axes[1, col_idx].set_title(f'变化: {np.sum(change_map)} 像素')
        axes[1, col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    comparison_results = {
        'results': results,
        'quality_metrics': quality_metrics,
        'processing_times': processing_times,
        'comparison_figure': fig
    }
    
    return comparison_results


# 模块信息
def get_module_info():
    """获取模块信息"""
    total_processors = sum(len(processors) for processors in AVAILABLE_PROCESSORS.values())
    
    return {
        'name': '湿地高光谱后处理模块',
        'version': __version__,
        'author': __author__,
        'description': '完整的分类结果后处理工具包',
        'processor_categories': list(AVAILABLE_PROCESSORS.keys()),
        'total_processors': total_processors,
        'spatial_filters': len(AVAILABLE_PROCESSORS['spatial_filters']),
        'morphology_processors': len(AVAILABLE_PROCESSORS['morphology_processors']),
        'consistency_processors': len(AVAILABLE_PROCESSORS['consistency_processors'])
    }


# 打印模块信息
if __name__ == "__main__":
    info = get_module_info()
    print(f"\n{info['name']} v{info['version']}")
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    print(f"\n支持的处理器:")
    print(f"  空间滤波器: {info['spatial_filters']} 种")
    print(f"  形态学处理器: {info['morphology_processors']} 种")
    print(f"  一致性处理器: {info['consistency_processors']} 种")
    print(f"  总计: {info['total_processors']} 种处理器\n")
    
    # 显示各类别的具体处理器
    for category, processors in AVAILABLE_PROCESSORS.items():
        print(f"{category.upper()}:")
        for proc_name in processors.keys():
            print(f"  - {proc_name}")
        print()