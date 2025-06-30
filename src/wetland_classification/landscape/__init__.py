"""
湿地高光谱景观分析模块
===================

这个模块提供了完整的景观生态学分析功能，专门针对湿地生态系统设计。

主要组件：
- 景观格局指数: 50+种经典指数计算 (斑块、类别、景观级)
- 连通性分析: 结构与功能连通性评估
- 生态网络分析: 图论方法的网络结构分析
- 空间模式识别: 景观格局模式识别
- 生态系统服务: 湿地生态服务功能评估

理论基础：
- 景观生态学理论
- 图论和网络分析
- 保护生物学原理
- 生态系统服务理论

作者: 湿地遥感研究团队
日期: 2024
版本: 1.0.0
"""

from .metrics import (
    LandscapeMetricsCalculator,
    WetlandMetricsCalculator,
    calculate_landscape_metrics
)

from .connectivity import (
    ConnectivityAnalyzer,
    ResistanceMapper,
    EcologicalNetworkAnalyzer,
    analyze_landscape_connectivity
)

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 版本信息
__version__ = "1.0.0"
__author__ = "湿地遥感研究团队"

# 设置日志
logger = logging.getLogger(__name__)

# 导出的公共接口
__all__ = [
    # 景观指数
    'LandscapeMetricsCalculator',
    'WetlandMetricsCalculator',
    'calculate_landscape_metrics',
    
    # 连通性分析
    'ConnectivityAnalyzer',
    'ResistanceMapper',
    'EcologicalNetworkAnalyzer',
    'analyze_landscape_connectivity',
    
    # 统一分析框架
    'LandscapeAnalyzer',
    'WetlandLandscapeAnalyzer',
    
    # 便捷函数
    'quick_landscape_analysis',
    'compare_landscape_patterns',
    'analyze_temporal_changes',
    'create_landscape_report',
    
    # 可视化工具
    'plot_landscape_metrics',
    'plot_connectivity_network',
    'plot_resistance_map',
    'visualize_landscape_analysis',
    
    # 工具函数
    'export_metrics_to_excel',
    'create_metrics_summary',
    'validate_landscape_data',
    
    # 常量和配置
    'LANDSCAPE_METRICS_CATEGORIES',
    'WETLAND_CLASS_MAPPING',
    'DEFAULT_RESISTANCE_VALUES'
]

# 景观指数类别
LANDSCAPE_METRICS_CATEGORIES = {
    'area_metrics': [
        'CA', 'PLAND', 'LPI', 'TE', 'ED', 'LSI'
    ],
    'shape_metrics': [
        'MSI', 'AWMSI', 'SHAPE', 'FRAC', 'GYRATE'
    ],
    'core_metrics': [
        'TCA', 'CPLAND', 'CORE', 'CAI', 'DCAD'
    ],
    'aggregation_metrics': [
        'AI', 'IJI', 'COHESION', 'DIVISION', 'SPLIT'
    ],
    'diversity_metrics': [
        'PR', 'PRD', 'SHDI', 'SIDI', 'SHEI', 'SIEI'
    ],
    'connectivity_metrics': [
        'PROX', 'SIMI', 'ENN', 'CONNECT'
    ]
}

# 湿地类别映射
WETLAND_CLASS_MAPPING = {
    1: 'water',
    2: 'shallow_water', 
    3: 'wetland_vegetation',
    4: 'mudflat',
    5: 'dry_land',
    6: 'agriculture',
    7: 'urban',
    8: 'forest'
}

# 默认阻力值
DEFAULT_RESISTANCE_VALUES = {
    'wetland_bird': {
        'water': 1.0,
        'shallow_water': 1.2,
        'wetland_vegetation': 1.5,
        'mudflat': 2.0,
        'grassland': 3.0,
        'forest': 4.0,
        'agriculture': 5.0,
        'urban': 10.0,
        'road': 15.0
    },
    'amphibian': {
        'water': 1.0,
        'shallow_water': 1.1,
        'wetland_vegetation': 1.3,
        'mudflat': 1.8,
        'grassland': 4.0,
        'forest': 2.0,
        'agriculture': 8.0,
        'urban': 15.0,
        'road': 25.0
    },
    'general': {
        'water': 1.0,
        'shallow_water': 1.5,
        'wetland_vegetation': 2.0,
        'mudflat': 2.5,
        'grassland': 3.0,
        'forest': 3.5,
        'agriculture': 5.0,
        'urban': 10.0,
        'road': 15.0
    }
}


class LandscapeAnalyzer:
    """
    景观分析器
    
    统一的景观生态学分析框架，集成指数计算和连通性分析。
    """
    
    def __init__(self, 
                 pixel_size: float = 1.0,
                 max_distance: float = 1000.0,
                 species_type: str = 'general',
                 **kwargs):
        """
        初始化景观分析器
        
        Parameters:
        -----------
        pixel_size : float, default=1.0
            像元大小（米）
        max_distance : float, default=1000.0
            最大连通距离（米）
        species_type : str, default='general'
            目标物种类型
        """
        self.pixel_size = pixel_size
        self.max_distance = max_distance
        self.species_type = species_type
        self.config = kwargs
        
        # 初始化子模块
        self.metrics_calculator = LandscapeMetricsCalculator(
            pixel_size=pixel_size, **kwargs
        )
        self.connectivity_analyzer = ConnectivityAnalyzer(
            pixel_size=pixel_size, max_distance=max_distance, **kwargs
        )
        self.resistance_mapper = ResistanceMapper(
            species_type=species_type, **kwargs
        )
        self.network_analyzer = EcologicalNetworkAnalyzer(**kwargs)
    
    def comprehensive_analysis(self, 
                             classification_map: np.ndarray,
                             habitat_classes: List[int],
                             class_mapping: Optional[Dict[int, str]] = None,
                             include_connectivity: bool = True,
                             include_network: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        综合景观分析
        
        Parameters:
        -----------
        classification_map : np.ndarray
            分类图像
        habitat_classes : list
            栖息地类别列表
        class_mapping : dict, optional
            类别映射
        include_connectivity : bool, default=True
            是否包含连通性分析
        include_network : bool, default=True
            是否包含网络分析
            
        Returns:
        --------
        results : dict
            综合分析结果
        """
        logger.info("开始综合景观分析")
        
        results = {
            'metadata': {
                'image_shape': classification_map.shape,
                'pixel_size': self.pixel_size,
                'habitat_classes': habitat_classes,
                'species_type': self.species_type,
                'analysis_timestamp': pd.Timestamp.now()
            }
        }
        
        # 1. 景观指数计算
        logger.info("计算景观指数")
        landscape_metrics = self.metrics_calculator.calculate_all_metrics(
            classification_map, 
            class_names=class_mapping,
            levels=['patch', 'class', 'landscape']
        )
        results['landscape_metrics'] = landscape_metrics
        
        # 2. 连通性分析
        if include_connectivity:
            logger.info("分析景观连通性")
            
            # 创建阻力地图
            if class_mapping is None:
                class_mapping = WETLAND_CLASS_MAPPING
            
            resistance_map = self.resistance_mapper.create_resistance_map(
                classification_map, class_mapping
            )
            
            # 结构连通性
            structural_connectivity = self.connectivity_analyzer.analyze_structural_connectivity(
                classification_map, habitat_classes
            )
            
            # 功能连通性
            functional_connectivity = self.connectivity_analyzer.analyze_functional_connectivity(
                classification_map, resistance_map, habitat_classes
            )
            
            results['connectivity_analysis'] = {
                'structural': structural_connectivity,
                'functional': functional_connectivity,
                'resistance_map': resistance_map
            }
        
        # 3. 生态网络分析
        if include_network and include_connectivity:
            logger.info("分析生态网络")
            
            connectivity_matrix = results['connectivity_analysis']['functional']['functional_connectivity']
            patches = results['connectivity_analysis']['functional']['patches']
            
            if len(patches) > 0:
                network_metrics = self.network_analyzer.analyze_network_structure(
                    connectivity_matrix, patches
                )
                
                keystone_patches = self.network_analyzer.identify_keystone_patches(
                    connectivity_matrix, patches
                )
                
                robustness_analysis = self.network_analyzer.simulate_network_robustness(
                    connectivity_matrix, patches, removal_strategy='degree'
                )
                
                results['network_analysis'] = {
                    'network_metrics': network_metrics,
                    'keystone_patches': keystone_patches,
                    'robustness_analysis': robustness_analysis
                }
        
        # 4. 综合评估
        results['synthesis'] = self._synthesize_results(results)
        
        logger.info("综合景观分析完成")
        
        return results
    
    def _synthesize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """综合分析结果"""
        synthesis = {}
        
        # 景观完整性评分
        if 'landscape_metrics' in results:
            landscape_metrics = results['landscape_metrics'].get('landscape_metrics', {})
            
            # 基于多个指标计算完整性分数
            integrity_components = {
                'patch_density': 1 - min(landscape_metrics.get('PD', 0) / 10, 1),  # 斑块密度越低越好
                'largest_patch': landscape_metrics.get('LPI', 0) / 100,  # 最大斑块指数
                'connectivity': 1 - landscape_metrics.get('LSI', 1),  # 景观形状指数越低越好
                'diversity': min(landscape_metrics.get('SHDI', 0) / 2, 1)  # Shannon多样性
            }
            
            integrity_score = np.mean(list(integrity_components.values()))
            
            synthesis['landscape_integrity'] = {
                'overall_score': integrity_score,
                'components': integrity_components,
                'interpretation': self._interpret_integrity_score(integrity_score)
            }
        
        # 连通性评估
        if 'connectivity_analysis' in results:
            structural = results['connectivity_analysis']['structural']['connectivity_metrics']
            functional = results['connectivity_analysis']['functional']['functional_metrics']
            
            connectivity_score = (
                structural.get('connectance', 0) + 
                functional.get('avg_functional_connectivity', 0)
            ) / 2
            
            synthesis['connectivity_assessment'] = {
                'connectivity_score': connectivity_score,
                'structural_connectance': structural.get('connectance', 0),
                'functional_connectivity': functional.get('avg_functional_connectivity', 0),
                'interpretation': self._interpret_connectivity_score(connectivity_score)
            }
        
        # 生态健康状况
        health_indicators = []
        
        if 'landscape_integrity' in synthesis:
            health_indicators.append(synthesis['landscape_integrity']['overall_score'])
        
        if 'connectivity_assessment' in synthesis:
            health_indicators.append(synthesis['connectivity_assessment']['connectivity_score'])
        
        if health_indicators:
            ecological_health = np.mean(health_indicators)
            synthesis['ecological_health'] = {
                'overall_health': ecological_health,
                'status': self._classify_health_status(ecological_health),
                'recommendations': self._generate_recommendations(ecological_health, results)
            }
        
        return synthesis
    
    def _interpret_integrity_score(self, score: float) -> str:
        """解释完整性分数"""
        if score >= 0.8:
            return "景观完整性优秀，生态系统结构良好"
        elif score >= 0.6:
            return "景观完整性良好，存在轻微破碎化"
        elif score >= 0.4:
            return "景观完整性一般，破碎化程度中等"
        elif score >= 0.2:
            return "景观完整性较差，破碎化严重"
        else:
            return "景观完整性很差，生态系统高度破碎"
    
    def _interpret_connectivity_score(self, score: float) -> str:
        """解释连通性分数"""
        if score >= 0.8:
            return "连通性优秀，物种迁移通道畅通"
        elif score >= 0.6:
            return "连通性良好，大部分区域连通"
        elif score >= 0.4:
            return "连通性一般，存在连通障碍"
        elif score >= 0.2:
            return "连通性较差，迁移阻力较大"
        else:
            return "连通性很差，生态孤岛效应明显"
    
    def _classify_health_status(self, health_score: float) -> str:
        """分类生态健康状况"""
        if health_score >= 0.8:
            return "健康"
        elif health_score >= 0.6:
            return "亚健康"
        elif health_score >= 0.4:
            return "一般"
        elif health_score >= 0.2:
            return "不健康"
        else:
            return "严重不健康"
    
    def _generate_recommendations(self, health_score: float, results: Dict[str, Any]) -> List[str]:
        """生成管理建议"""
        recommendations = []
        
        if health_score < 0.6:
            recommendations.append("加强生态系统保护，减少人为干扰")
            recommendations.append("实施生态修复工程，恢复退化湿地")
        
        if 'connectivity_analysis' in results:
            connectivity_score = results.get('synthesis', {}).get('connectivity_assessment', {}).get('connectivity_score', 0)
            
            if connectivity_score < 0.5:
                recommendations.append("建设生态廊道，改善栖息地连通性")
                recommendations.append("减少道路和建筑对生态廊道的阻隔")
        
        if 'network_analysis' in results:
            keystone_patches = results['network_analysis']['keystone_patches']
            if keystone_patches:
                recommendations.append(f"重点保护关键石斑块（共{len(keystone_patches)}个）")
        
        if not recommendations:
            recommendations.append("继续维护当前良好的生态状况")
            recommendations.append("建立长期监测体系，跟踪变化趋势")
        
        return recommendations


class WetlandLandscapeAnalyzer(LandscapeAnalyzer):
    """
    湿地景观分析器
    
    专门针对湿地生态系统的景观分析工具。
    """
    
    def __init__(self, **kwargs):
        """初始化湿地景观分析器"""
        super().__init__(species_type='wetland_bird', **kwargs)
        
        # 使用湿地专用指数计算器
        self.metrics_calculator = WetlandMetricsCalculator(
            pixel_size=self.pixel_size, **kwargs
        )
    
    def wetland_comprehensive_analysis(self, 
                                     classification_map: np.ndarray,
                                     water_level_data: Optional[np.ndarray] = None,
                                     **kwargs) -> Dict[str, Any]:
        """
        湿地综合分析
        
        Parameters:
        -----------
        classification_map : np.ndarray
            湿地分类图像
        water_level_data : np.ndarray, optional
            水位数据
            
        Returns:
        --------
        results : dict
            湿地综合分析结果
        """
        logger.info("开始湿地综合分析")
        
        # 定义湿地栖息地类别
        wetland_habitat_classes = [1, 2, 3, 4]  # 水体、浅水、湿地植被、泥滩
        
        # 标准综合分析
        results = self.comprehensive_analysis(
            classification_map, 
            wetland_habitat_classes,
            class_mapping=WETLAND_CLASS_MAPPING,
            **kwargs
        )
        
        # 湿地专用指数
        wetland_specific_metrics = self.metrics_calculator.calculate_wetland_specific_metrics(
            classification_map, water_level_data
        )
        results['wetland_specific_metrics'] = wetland_specific_metrics
        
        # 湿地功能评估
        results['wetland_functions'] = self._assess_wetland_functions(
            classification_map, results
        )
        
        # 湿地健康评估
        results['wetland_health'] = self._assess_wetland_health(
            results, wetland_specific_metrics
        )
        
        logger.info("湿地综合分析完成")
        
        return results
    
    def _assess_wetland_functions(self, 
                                classification_map: np.ndarray,
                                results: Dict[str, Any]) -> Dict[str, float]:
        """评估湿地功能"""
        functions = {}
        
        # 获取各类型面积
        unique_classes, counts = np.unique(classification_map, return_counts=True)
        total_pixels = classification_map.size
        
        area_ratios = {}
        for class_id, count in zip(unique_classes, counts):
            area_ratios[class_id] = count / total_pixels
        
        # 1. 水源涵养功能
        water_area_ratio = area_ratios.get(1, 0) + area_ratios.get(2, 0)  # 水体+浅水
        functions['water_conservation'] = min(water_area_ratio * 2, 1.0)  # 标准化到[0,1]
        
        # 2. 水质净化功能
        vegetation_area_ratio = area_ratios.get(3, 0)  # 湿地植被
        functions['water_purification'] = min(vegetation_area_ratio * 3, 1.0)
        
        # 3. 生物多样性支持功能
        landscape_metrics = results.get('landscape_metrics', {}).get('landscape_metrics', {})
        diversity_index = landscape_metrics.get('SHDI', 0)
        functions['biodiversity_support'] = min(diversity_index / 2, 1.0)
        
        # 4. 防洪调蓄功能
        total_wetland_ratio = (
            area_ratios.get(1, 0) + area_ratios.get(2, 0) + 
            area_ratios.get(3, 0) + area_ratios.get(4, 0)
        )
        functions['flood_control'] = min(total_wetland_ratio * 1.5, 1.0)
        
        # 5. 气候调节功能
        if 'connectivity_analysis' in results:
            connectivity_score = results['synthesis']['connectivity_assessment']['connectivity_score']
            functions['climate_regulation'] = connectivity_score
        else:
            functions['climate_regulation'] = total_wetland_ratio
        
        # 6. 碳储存功能
        # 基于湿地植被面积和连通性
        vegetation_quality = vegetation_area_ratio * functions.get('biodiversity_support', 0.5)
        functions['carbon_storage'] = min(vegetation_quality * 2, 1.0)
        
        return functions
    
    def _assess_wetland_health(self, 
                             results: Dict[str, Any],
                             wetland_specific_metrics: Dict[str, float]) -> Dict[str, Any]:
        """评估湿地健康状况"""
        health_indicators = {}
        
        # 1. 结构健康
        landscape_integrity = results.get('synthesis', {}).get('landscape_integrity', {}).get('overall_score', 0)
        health_indicators['structural_health'] = landscape_integrity
        
        # 2. 功能健康
        connectivity_score = results.get('synthesis', {}).get('connectivity_assessment', {}).get('connectivity_score', 0)
        health_indicators['functional_health'] = connectivity_score
        
        # 3. 湿地完整性
        wetland_integrity = wetland_specific_metrics.get('WETLAND_INTEGRITY', 0)
        health_indicators['wetland_integrity'] = wetland_integrity
        
        # 4. 水体连通性
        water_connectivity = wetland_specific_metrics.get('WATER_CONNECTIVITY', 0)
        health_indicators['water_connectivity'] = water_connectivity
        
        # 5. 破碎化程度（越低越好）
        fragmentation = wetland_specific_metrics.get('WETLAND_FRAGMENTATION', 1)
        health_indicators['fragmentation_resistance'] = 1 - fragmentation
        
        # 综合健康评分
        overall_health = np.mean(list(health_indicators.values()))
        
        # 健康等级
        if overall_health >= 0.8:
            health_grade = "优秀"
            health_description = "湿地生态系统健康状况优秀"
        elif overall_health >= 0.6:
            health_grade = "良好"
            health_description = "湿地生态系统健康状况良好"
        elif overall_health >= 0.4:
            health_grade = "一般"
            health_description = "湿地生态系统健康状况一般"
        elif overall_health >= 0.2:
            health_grade = "较差"
            health_description = "湿地生态系统健康状况较差"
        else:
            health_grade = "很差"
            health_description = "湿地生态系统健康状况很差"
        
        return {
            'health_indicators': health_indicators,
            'overall_health': overall_health,
            'health_grade': health_grade,
            'health_description': health_description,
            'key_issues': self._identify_key_issues(health_indicators),
            'improvement_priorities': self._prioritize_improvements(health_indicators)
        }
    
    def _identify_key_issues(self, health_indicators: Dict[str, float]) -> List[str]:
        """识别关键问题"""
        issues = []
        
        if health_indicators.get('structural_health', 0) < 0.5:
            issues.append("景观结构破碎化严重")
        
        if health_indicators.get('functional_health', 0) < 0.5:
            issues.append("生态功能连通性不足")
        
        if health_indicators.get('water_connectivity', 0) < 0.5:
            issues.append("水体连通性较差")
        
        if health_indicators.get('fragmentation_resistance', 0) < 0.5:
            issues.append("湿地破碎化程度高")
        
        if not issues:
            issues.append("整体健康状况良好")
        
        return issues
    
    def _prioritize_improvements(self, health_indicators: Dict[str, float]) -> List[str]:
        """改进优先级"""
        priorities = []
        
        # 按健康指标排序，最差的优先改进
        sorted_indicators = sorted(health_indicators.items(), key=lambda x: x[1])
        
        for indicator, score in sorted_indicators[:3]:  # 取前3个最差的
            if score < 0.6:
                if indicator == 'structural_health':
                    priorities.append("优先恢复湿地景观结构完整性")
                elif indicator == 'functional_health':
                    priorities.append("加强生态廊道建设，提高功能连通性")
                elif indicator == 'water_connectivity':
                    priorities.append("改善水系连通，恢复水文过程")
                elif indicator == 'fragmentation_resistance':
                    priorities.append("减少湿地分割，扩大连续栖息地面积")
        
        if not priorities:
            priorities.append("维护现有良好状态，建立长期监测机制")
        
        return priorities


# 便捷函数
def quick_landscape_analysis(classification_map: np.ndarray,
                            analysis_type: str = 'wetland',
                            pixel_size: float = 1.0,
                            **kwargs) -> Dict[str, Any]:
    """
    快速景观分析
    
    Parameters:
    -----------
    classification_map : np.ndarray
        分类图像
    analysis_type : str, default='wetland'
        分析类型 ('general', 'wetland')
    pixel_size : float, default=1.0
        像元大小
        
    Returns:
    --------
    results : dict
        分析结果
    """
    if analysis_type == 'wetland':
        analyzer = WetlandLandscapeAnalyzer(pixel_size=pixel_size, **kwargs)
        return analyzer.wetland_comprehensive_analysis(classification_map, **kwargs)
    else:
        analyzer = LandscapeAnalyzer(pixel_size=pixel_size, **kwargs)
        habitat_classes = list(range(1, int(np.max(classification_map)) + 1))
        return analyzer.comprehensive_analysis(classification_map, habitat_classes, **kwargs)


def compare_landscape_patterns(classification_maps: List[np.ndarray],
                             dates: Optional[List[str]] = None,
                             pixel_size: float = 1.0,
                             analysis_type: str = 'wetland') -> Dict[str, Any]:
    """
    比较多期景观格局
    
    Parameters:
    -----------
    classification_maps : list
        多期分类图像列表
    dates : list, optional
        对应日期列表
    pixel_size : float, default=1.0
        像元大小
    analysis_type : str, default='wetland'
        分析类型
        
    Returns:
    --------
    comparison_results : dict
        比较结果
    """
    logger.info(f"比较 {len(classification_maps)} 期景观格局")
    
    if dates is None:
        dates = [f"Period_{i+1}" for i in range(len(classification_maps))]
    
    # 分析每期数据
    period_results = {}
    for i, (classification_map, date) in enumerate(zip(classification_maps, dates)):
        logger.info(f"分析第 {i+1} 期: {date}")
        
        results = quick_landscape_analysis(
            classification_map, 
            analysis_type=analysis_type, 
            pixel_size=pixel_size
        )
        period_results[date] = results
    
    # 提取关键指标进行比较
    comparison_metrics = _extract_comparison_metrics(period_results)
    
    # 变化趋势分析
    trend_analysis = _analyze_trends(comparison_metrics, dates)
    
    # 变化驱动力分析
    change_drivers = _analyze_change_drivers(classification_maps, dates)
    
    return {
        'period_results': period_results,
        'comparison_metrics': comparison_metrics,
        'trend_analysis': trend_analysis,
        'change_drivers': change_drivers,
        'summary': _create_comparison_summary(trend_analysis)
    }


def _extract_comparison_metrics(period_results: Dict[str, Dict]) -> pd.DataFrame:
    """提取比较指标"""
    metrics_data = []
    
    for date, results in period_results.items():
        row = {'date': date}
        
        # 景观级指标
        landscape_metrics = results.get('landscape_metrics', {}).get('landscape_metrics', {})
        for metric, value in landscape_metrics.items():
            row[f'landscape_{metric}'] = value
        
        # 湿地专用指标
        wetland_metrics = results.get('wetland_specific_metrics', {})
        for metric, value in wetland_metrics.items():
            row[f'wetland_{metric}'] = value
        
        # 综合指标
        synthesis = results.get('synthesis', {})
        if 'landscape_integrity' in synthesis:
            row['integrity_score'] = synthesis['landscape_integrity']['overall_score']
        
        if 'connectivity_assessment' in synthesis:
            row['connectivity_score'] = synthesis['connectivity_assessment']['connectivity_score']
        
        if 'ecological_health' in synthesis:
            row['health_score'] = synthesis['ecological_health']['overall_health']
        
        metrics_data.append(row)
    
    return pd.DataFrame(metrics_data)


def _analyze_trends(comparison_metrics: pd.DataFrame, dates: List[str]) -> Dict[str, Any]:
    """分析变化趋势"""
    trends = {}
    
    # 数值列
    numeric_columns = comparison_metrics.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'date']
    
    for column in numeric_columns:
        values = comparison_metrics[column].values
        
        if len(values) > 1:
            # 计算变化率
            change_rate = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
            
            # 趋势方向
            if change_rate > 0.1:
                trend_direction = "显著增加"
            elif change_rate < -0.1:
                trend_direction = "显著减少"
            elif abs(change_rate) <= 0.05:
                trend_direction = "基本稳定"
            else:
                trend_direction = "轻微变化"
            
            # 计算变化幅度
            change_magnitude = abs(change_rate)
            
            trends[column] = {
                'initial_value': values[0],
                'final_value': values[-1],
                'change_rate': change_rate,
                'change_magnitude': change_magnitude,
                'trend_direction': trend_direction,
                'values': values.tolist()
            }
    
    return trends


def _analyze_change_drivers(classification_maps: List[np.ndarray], dates: List[str]) -> Dict[str, Any]:
    """分析变化驱动力"""
    change_drivers = {}
    
    # 土地利用变化矩阵
    if len(classification_maps) >= 2:
        from_map = classification_maps[0]
        to_map = classification_maps[-1]
        
        # 计算变化区域
        change_mask = from_map != to_map
        change_ratio = np.sum(change_mask) / change_mask.size
        
        # 主要变化类型
        unique_from = np.unique(from_map[change_mask])
        unique_to = np.unique(to_map[change_mask])
        
        change_drivers['change_ratio'] = change_ratio
        change_drivers['changed_area_ratio'] = change_ratio
        change_drivers['main_source_classes'] = unique_from.tolist()
        change_drivers['main_target_classes'] = unique_to.tolist()
        
        # 变化热点区域
        # 这里可以添加更复杂的空间分析
        
    return change_drivers


def _create_comparison_summary(trend_analysis: Dict[str, Any]) -> Dict[str, str]:
    """创建比较摘要"""
    summary = {}
    
    # 关键指标趋势
    key_metrics = ['integrity_score', 'connectivity_score', 'health_score']
    
    for metric in key_metrics:
        if metric in trend_analysis:
            trend = trend_analysis[metric]
            summary[metric] = f"{trend['trend_direction']} (变化率: {trend['change_rate']:.2%})"
    
    return summary


def analyze_temporal_changes(classification_maps: List[np.ndarray],
                           dates: List[str],
                           pixel_size: float = 1.0) -> Dict[str, Any]:
    """
    时序变化分析
    
    Parameters:
    -----------
    classification_maps : list
        时序分类图像
    dates : list
        日期列表
    pixel_size : float, default=1.0
        像元大小
        
    Returns:
    --------
    temporal_results : dict
        时序分析结果
    """
    return compare_landscape_patterns(classification_maps, dates, pixel_size, 'wetland')


def create_landscape_report(analysis_results: Dict[str, Any],
                          output_path: str,
                          include_plots: bool = True) -> str:
    """
    创建景观分析报告
    
    Parameters:
    -----------
    analysis_results : dict
        分析结果
    output_path : str
        输出路径
    include_plots : bool, default=True
        是否包含图表
        
    Returns:
    --------
    report_path : str
        报告文件路径
    """
    logger.info(f"创建景观分析报告: {output_path}")
    
    # 这里可以实现详细的报告生成逻辑
    # 包括Markdown、PDF或HTML格式的报告
    
    report_content = _generate_report_content(analysis_results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"报告已保存: {output_path}")
    
    return output_path


def _generate_report_content(analysis_results: Dict[str, Any]) -> str:
    """生成报告内容"""
    report = """
# 湿地景观分析报告

## 1. 概述

本报告基于高光谱遥感数据，对湿地生态系统的景观格局特征进行了全面分析。

## 2. 分析结果

### 2.1 景观指数

"""
    
    # 添加具体的分析结果
    if 'landscape_metrics' in analysis_results:
        report += "#### 主要景观指数：\n\n"
        
        landscape_metrics = analysis_results['landscape_metrics'].get('landscape_metrics', {})
        for metric, value in landscape_metrics.items():
            report += f"- {metric}: {value:.4f}\n"
    
    if 'synthesis' in analysis_results:
        synthesis = analysis_results['synthesis']
        
        report += "\n### 2.2 综合评估\n\n"
        
        if 'ecological_health' in synthesis:
            health = synthesis['ecological_health']
            report += f"**生态健康状况**: {health['status']}\n\n"
            report += f"**综合评分**: {health['overall_health']:.3f}\n\n"
            
            if 'recommendations' in health:
                report += "**管理建议**:\n\n"
                for rec in health['recommendations']:
                    report += f"- {rec}\n"
    
    report += "\n---\n生成时间: " + str(pd.Timestamp.now())
    
    return report


# 可视化函数
def plot_landscape_metrics(metrics_df: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """绘制景观指数图表"""
    # 实现景观指数可视化
    pass


def plot_connectivity_network(connectivity_matrix: np.ndarray,
                             patches: List[Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
    """绘制连通性网络图"""
    # 实现连通性网络可视化
    pass


def visualize_landscape_analysis(analysis_results: Dict[str, Any],
                               save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """可视化景观分析结果"""
    # 实现综合可视化
    pass


# 工具函数
def export_metrics_to_excel(analysis_results: Dict[str, Any],
                          filepath: str) -> None:
    """导出指标到Excel"""
    # 实现Excel导出功能
    pass


def validate_landscape_data(classification_map: np.ndarray) -> Dict[str, Any]:
    """验证景观数据"""
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # 检查数据类型
    if not isinstance(classification_map, np.ndarray):
        validation_results['errors'].append("输入必须是numpy数组")
        validation_results['is_valid'] = False
    
    # 检查维度
    if classification_map.ndim != 2:
        validation_results['errors'].append("输入必须是2D数组")
        validation_results['is_valid'] = False
    
    # 检查数据范围
    if np.any(classification_map < 0):
        validation_results['warnings'].append("存在负值")
    
    return validation_results


# 模块信息
def get_module_info():
    """获取模块信息"""
    return {
        'name': '湿地高光谱景观分析模块',
        'version': __version__,
        'author': __author__,
        'description': '完整的景观生态学分析工具包',
        'capabilities': {
            'landscape_metrics': len(LANDSCAPE_METRICS_CATEGORIES),
            'connectivity_analysis': True,
            'network_analysis': True,
            'wetland_specialized': True,
            'temporal_analysis': True
        },
        'supported_metrics': sum(len(metrics) for metrics in LANDSCAPE_METRICS_CATEGORIES.values()),
        'supported_species': list(DEFAULT_RESISTANCE_VALUES.keys())
    }


if __name__ == "__main__":
    info = get_module_info()
    print(f"\n{info['name']} v{info['version']}")
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    print(f"\n功能特性:")
    print(f"  支持指标: {info['supported_metrics']} 种")
    print(f"  连通性分析: {info['capabilities']['connectivity_analysis']}")
    print(f"  网络分析: {info['capabilities']['network_analysis']}")
    print(f"  湿地专业化: {info['capabilities']['wetland_specialized']}")
    print(f"  时序分析: {info['capabilities']['temporal_analysis']}")
    print(f"\n支持物种类型: {', '.join(info['supported_species'])}")
    print(f"\n指数类别:")
    for category, metrics in LANDSCAPE_METRICS_CATEGORIES.items():
        print(f"  {category}: {len(metrics)} 种指数")
    print()