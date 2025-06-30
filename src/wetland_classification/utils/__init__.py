"""
湿地高光谱分类系统 - 工具模块
Wetland Hyperspectral Classification System - Utils Module

工具模块提供了项目所需的各种通用功能，包括：
- 输入输出工具 (io_utils)
- 可视化工具 (visualization)  
- 日志系统 (logger)
- 辅助函数 (helpers)

Author: Wetland Classification Team
Date: 2024
License: MIT
"""

from .io_utils import (
    # 数据读取函数
    load_hyperspectral_data,
    load_ground_truth,
    load_config,
    save_classification_result,
    save_metrics,
    save_model,
    load_model,
    
    # 格式转换函数
    convert_to_geotiff,
    convert_to_envi,
    export_to_shapefile,
    
    # 数据验证函数
    validate_hyperspectral_data,
    validate_ground_truth,
    check_spatial_alignment,
)

from .visualization import (
    # 光谱可视化
    plot_spectral_signature,
    plot_spectral_curves,
    plot_spectral_comparison,
    
    # 分类结果可视化
    plot_classification_map,
    plot_confusion_matrix,
    plot_accuracy_curves,
    plot_feature_importance,
    
    # 统计图表
    plot_class_distribution,
    plot_training_history,
    plot_validation_curves,
    
    # 景观分析可视化
    plot_landscape_metrics,
    plot_connectivity_map,
    
    # 导出功能
    save_publication_figure,
    create_report_figures,
)

from .logger import (
    # 日志记录器
    get_logger,
    setup_logging,
    log_system_info,
    log_data_info,
    log_processing_step,
    log_model_performance,
    
    # 进度跟踪
    ProgressTracker,
    create_progress_bar,
)

from .helpers import (
    # 数组操作
    reshape_for_sklearn,
    reshape_for_pytorch,
    normalize_data,
    standardize_data,
    
    # 时间工具
    format_duration,
    get_timestamp,
    Timer,
    
    # 文件工具
    ensure_dir,
    get_file_size,
    clean_temp_files,
    
    # 内存管理
    get_memory_usage,
    check_memory_availability,
    clear_cache,
    
    # 数学工具
    calculate_class_weights,
    compute_spatial_distance,
    apply_smoothing_filter,
    
    # 验证工具
    validate_parameters,
    check_data_consistency,
    verify_model_compatibility,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "Wetland Classification Team"

# 导出的主要类和函数
__all__ = [
    # IO工具
    'load_hyperspectral_data',
    'load_ground_truth', 
    'save_classification_result',
    'save_metrics',
    'save_model',
    'load_model',
    
    # 可视化工具
    'plot_spectral_signature',
    'plot_classification_map',
    'plot_confusion_matrix',
    'plot_training_history',
    'save_publication_figure',
    
    # 日志工具
    'get_logger',
    'setup_logging',
    'ProgressTracker',
    
    # 辅助工具
    'reshape_for_sklearn',
    'normalize_data',
    'Timer',
    'ensure_dir',
    'calculate_class_weights',
    'validate_parameters',
]