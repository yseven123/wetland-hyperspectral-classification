"""
评估模块 (evaluation)
====================

这个模块提供了全面的分类结果评估功能，包括：

1. **精度评估 (metrics.py)**
   - 分类精度指标计算
   - 混淆矩阵分析
   - 类别特定的评估指标
   - ROC和PR曲线分析

2. **交叉验证 (validation.py)**
   - K折交叉验证
   - 分层交叉验证
   - 时序交叉验证
   - 空间交叉验证
   - 模型性能稳定性评估

3. **不确定性分析 (uncertainty.py)**
   - 预测不确定性量化
   - 模型不确定性评估
   - 空间不确定性分析
   - 贝叶斯不确定性估计
   - 集成模型不确定性

主要特性：
- 支持多类别分类评估
- 提供空间感知的验证策略
- 全面的不确定性量化方法
- 丰富的可视化功能
- 标准化的评估报告生成

使用示例：
    ```python
    from wetland_classification.evaluation import (
        ClassificationEvaluator,
        CrossValidator,
        UncertaintyAnalyzer
    )
    
    # 创建评估器
    evaluator = ClassificationEvaluator()
    validator = CrossValidator()
    uncertainty_analyzer = UncertaintyAnalyzer()
    
    # 评估分类结果
    metrics = evaluator.evaluate_classification(y_true, y_pred)
    cv_results = validator.cross_validate(model, X, y)
    uncertainty = uncertainty_analyzer.analyze_uncertainty(predictions, probabilities)
    ```

作者: 湿地遥感团队
日期: 2024
版本: 1.0.0
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import logging

# 设置模块日志
logger = logging.getLogger(__name__)

try:
    # 导入精度评估模块
    from .metrics import (
        ClassificationEvaluator,
        MultiClassMetrics,
        ConfusionMatrixAnalyzer,
        ROCAnalyzer,
        ClassSpecificEvaluator,
        calculate_classification_metrics,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_precision_recall_curves,
        generate_classification_report
    )
    logger.info("精度评估模块加载成功")
    
except ImportError as e:
    logger.warning(f"精度评估模块加载失败: {e}")
    # 提供占位符类
    class ClassificationEvaluator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("精度评估模块未正确安装")

try:
    # 导入交叉验证模块
    from .validation import (
        CrossValidator,
        SpatialCrossValidator,
        TemporalCrossValidator,
        StratifiedValidator,
        ModelValidator,
        ValidationStrategy,
        create_spatial_folds,
        create_temporal_folds,
        evaluate_model_stability,
        hyperparameter_validation
    )
    logger.info("交叉验证模块加载成功")
    
except ImportError as e:
    logger.warning(f"交叉验证模块加载失败: {e}")
    # 提供占位符类
    class CrossValidator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("交叉验证模块未正确安装")

try:
    # 导入不确定性分析模块
    from .uncertainty import (
        UncertaintyAnalyzer,
        SpatialUncertaintyAnalyzer,
        calculate_ensemble_uncertainty,
        load_uncertainty_config,
        save_uncertainty_results
    )
    logger.info("不确定性分析模块加载成功")
    
except ImportError as e:
    logger.warning(f"不确定性分析模块加载失败: {e}")
    # 提供占位符类
    class UncertaintyAnalyzer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("不确定性分析模块未正确安装")

# 模块版本信息
__version__ = "1.0.0"
__author__ = "湿地遥感团队"
__email__ = "wetland.remote.sensing@example.com"

# 导出的主要类和函数
__all__ = [
    # 精度评估相关
    'ClassificationEvaluator',
    'MultiClassMetrics', 
    'ConfusionMatrixAnalyzer',
    'ROCAnalyzer',
    'ClassSpecificEvaluator',
    'calculate_classification_metrics',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'generate_classification_report',
    
    # 交叉验证相关
    'CrossValidator',
    'SpatialCrossValidator',
    'TemporalCrossValidator', 
    'StratifiedValidator',
    'ModelValidator',
    'ValidationStrategy',
    'create_spatial_folds',
    'create_temporal_folds',
    'evaluate_model_stability',
    'hyperparameter_validation',
    
    # 不确定性分析相关
    'UncertaintyAnalyzer',
    'SpatialUncertaintyAnalyzer',
    'calculate_ensemble_uncertainty',
    'load_uncertainty_config',
    'save_uncertainty_results',
    
    # 工具函数
    'create_comprehensive_evaluator',
    'run_complete_evaluation',
    'compare_models',
    'generate_evaluation_report'
]

# 模块级配置
DEFAULT_CONFIG = {
    'metrics': {
        'include_per_class': True,
        'include_confusion_matrix': True,
        'include_roc_analysis': True,
        'confidence_level': 0.95
    },
    'validation': {
        'cv_folds': 5,
        'spatial_buffer': 1000,  # 米
        'stratify': True,
        'shuffle': True,
        'random_state': 42
    },
    'uncertainty': {
        'methods': ['entropy', 'confidence', 'ensemble'],
        'bootstrap_samples': 100,
        'confidence_level': 0.95,
        'spatial_analysis': True
    }
}


def create_comprehensive_evaluator(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    创建综合评估器
    
    这个函数创建并返回一套完整的评估工具，包括精度评估、
    交叉验证和不确定性分析器。
    
    Args:
        config: 可选的配置字典，用于定制评估器行为
        
    Returns:
        包含所有评估器的字典
        
    Example:
        ```python
        evaluators = create_comprehensive_evaluator()
        
        # 使用不同的评估器
        metrics = evaluators['classifier'].evaluate_classification(y_true, y_pred)
        cv_results = evaluators['validator'].cross_validate(model, X, y)
        uncertainty = evaluators['uncertainty'].analyze_uncertainty(pred, prob)
        ```
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    logger.info("创建综合评估器")
    
    evaluators = {}
    
    try:
        # 创建分类评估器
        evaluators['classifier'] = ClassificationEvaluator(config.get('metrics', {}))
        
        # 创建交叉验证器
        evaluators['validator'] = CrossValidator(config.get('validation', {}))
        
        # 创建不确定性分析器
        evaluators['uncertainty'] = UncertaintyAnalyzer(config.get('uncertainty', {}))
        
        # 创建空间不确定性分析器
        evaluators['spatial_uncertainty'] = SpatialUncertaintyAnalyzer(
            config.get('uncertainty', {})
        )
        
        logger.info("综合评估器创建成功")
        return evaluators
        
    except Exception as e:
        logger.error(f"创建综合评估器失败: {str(e)}")
        raise


def run_complete_evaluation(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray] = None,
                          model: Optional[Any] = None,
                          X: Optional[np.ndarray] = None,
                          spatial_coords: Optional[np.ndarray] = None,
                          class_labels: Optional[List[str]] = None,
                          config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    运行完整的评估流程
    
    这个函数执行全面的模型评估，包括精度指标计算、交叉验证
    和不确定性分析。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        y_prob: 预测概率 (可选)
        model: 训练好的模型 (用于交叉验证)
        X: 特征数据 (用于交叉验证)
        spatial_coords: 空间坐标 (用于空间验证和不确定性分析)
        class_labels: 类别标签名称
        config: 评估配置
        
    Returns:
        包含所有评估结果的字典
        
    Example:
        ```python
        results = run_complete_evaluation(
            y_true=y_test,
            y_pred=predictions, 
            y_prob=probabilities,
            model=trained_model,
            X=X_test,
            spatial_coords=coords,
            class_labels=['水体', '植被', '土壤', '建筑']
        )
        
        print(f"总体精度: {results['metrics']['overall_accuracy']:.3f}")
        print(f"Kappa系数: {results['metrics']['kappa']:.3f}")
        ```
    """
    logger.info("开始完整评估流程")
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    results = {
        'metrics': {},
        'validation': {},
        'uncertainty': {},
        'summary': {}
    }
    
    try:
        # 1. 精度评估
        logger.info("执行精度评估")
        evaluator = ClassificationEvaluator(config.get('metrics', {}))
        results['metrics'] = evaluator.evaluate_classification(
            y_true=y_true,
            y_pred=y_pred, 
            y_prob=y_prob,
            class_labels=class_labels
        )
        
        # 2. 交叉验证 (如果提供了模型和特征数据)
        if model is not None and X is not None:
            logger.info("执行交叉验证")
            if spatial_coords is not None:
                validator = SpatialCrossValidator(config.get('validation', {}))
            else:
                validator = CrossValidator(config.get('validation', {}))
            
            results['validation'] = validator.cross_validate(
                model=model,
                X=X,
                y=y_true,
                spatial_coords=spatial_coords
            )
        
        # 3. 不确定性分析 (如果提供了概率)
        if y_prob is not None:
            logger.info("执行不确定性分析")
            uncertainty_analyzer = UncertaintyAnalyzer(config.get('uncertainty', {}))
            results['uncertainty'] = uncertainty_analyzer.analyze_uncertainty(
                predictions=y_pred,
                probabilities=y_prob,
                spatial_coords=spatial_coords
            )
        
        # 4. 生成总结
        results['summary'] = _generate_evaluation_summary(results)
        
        logger.info("完整评估流程完成")
        return results
        
    except Exception as e:
        logger.error(f"完整评估流程失败: {str(e)}")
        raise


def compare_models(models_results: Dict[str, Dict],
                  metric: str = 'overall_accuracy') -> pd.DataFrame:
    """
    比较多个模型的性能
    
    Args:
        models_results: 模型结果字典，格式为 {model_name: evaluation_results}
        metric: 用于比较的主要指标
        
    Returns:
        包含模型比较结果的DataFrame
        
    Example:
        ```python
        model_results = {
            'SVM': svm_evaluation_results,
            'RF': rf_evaluation_results, 
            'CNN': cnn_evaluation_results
        }
        
        comparison = compare_models(model_results, metric='kappa')
        print(comparison)
        ```
    """
    import pandas as pd
    
    logger.info(f"比较模型性能，基于指标: {metric}")
    
    comparison_data = []
    
    for model_name, results in models_results.items():
        row = {'Model': model_name}
        
        # 添加基本指标
        if 'metrics' in results:
            metrics = results['metrics']
            row.update({
                'Overall_Accuracy': metrics.get('overall_accuracy', np.nan),
                'Kappa': metrics.get('kappa', np.nan),
                'F1_Macro': metrics.get('f1_macro', np.nan),
                'F1_Weighted': metrics.get('f1_weighted', np.nan)
            })
        
        # 添加交叉验证结果
        if 'validation' in results:
            cv = results['validation']
            row.update({
                'CV_Mean': cv.get('cv_scores', {}).get('mean', np.nan),
                'CV_Std': cv.get('cv_scores', {}).get('std', np.nan)
            })
        
        # 添加不确定性信息
        if 'uncertainty' in results:
            unc = results['uncertainty']
            if 'total' in unc:
                row['Mean_Uncertainty'] = np.mean(unc['total'])
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 按指定指标排序
    if metric in comparison_df.columns:
        comparison_df = comparison_df.sort_values(metric, ascending=False)
    
    logger.info("模型比较完成")
    return comparison_df


def generate_evaluation_report(evaluation_results: Dict[str, Any],
                             output_path: Optional[str] = None,
                             format: str = 'html') -> str:
    """
    生成详细的评估报告
    
    Args:
        evaluation_results: 评估结果字典
        output_path: 输出路径
        format: 报告格式 ('html', 'pdf', 'markdown')
        
    Returns:
        报告内容字符串
        
    Example:
        ```python
        report = generate_evaluation_report(
            evaluation_results,
            output_path='evaluation_report.html',
            format='html'
        )
        ```
    """
    logger.info(f"生成评估报告，格式: {format}")
    
    if format == 'html':
        return _generate_html_report(evaluation_results, output_path)
    elif format == 'markdown':
        return _generate_markdown_report(evaluation_results, output_path)
    elif format == 'pdf':
        return _generate_pdf_report(evaluation_results, output_path)
    else:
        raise ValueError(f"不支持的报告格式: {format}")


def _generate_evaluation_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """生成评估总结"""
    summary = {}
    
    # 基本统计
    if 'metrics' in results:
        metrics = results['metrics']
        summary['performance'] = {
            'overall_accuracy': metrics.get('overall_accuracy', 0),
            'kappa': metrics.get('kappa', 0),
            'f1_weighted': metrics.get('f1_weighted', 0)
        }
        
        # 性能等级
        oa = metrics.get('overall_accuracy', 0)
        if oa >= 0.9:
            summary['performance_level'] = 'Excellent'
        elif oa >= 0.8:
            summary['performance_level'] = 'Good'
        elif oa >= 0.7:
            summary['performance_level'] = 'Fair'
        else:
            summary['performance_level'] = 'Poor'
    
    # 稳定性评估
    if 'validation' in results:
        cv_std = results['validation'].get('cv_scores', {}).get('std', 0)
        summary['stability'] = 'High' if cv_std < 0.05 else 'Medium' if cv_std < 0.1 else 'Low'
    
    # 可靠性评估
    if 'uncertainty' in results:
        if 'total' in results['uncertainty']:
            mean_uncertainty = np.mean(results['uncertainty']['total'])
            summary['reliability'] = 'High' if mean_uncertainty < 0.3 else 'Medium' if mean_uncertainty < 0.6 else 'Low'
    
    return summary


def _generate_html_report(results: Dict[str, Any], output_path: Optional[str]) -> str:
    """生成HTML格式报告"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>湿地高光谱分类评估报告</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>湿地高光谱分类评估报告</h1>
            <p>生成时间: {timestamp}</p>
        </div>
    """.format(timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 添加评估结果内容
    if 'metrics' in results:
        html_content += """
        <div class="section">
            <h2>分类精度指标</h2>
            <div class="metric">总体精度: {:.3f}</div>
            <div class="metric">Kappa系数: {:.3f}</div>
            <div class="metric">F1分数: {:.3f}</div>
        </div>
        """.format(
            results['metrics'].get('overall_accuracy', 0),
            results['metrics'].get('kappa', 0),
            results['metrics'].get('f1_weighted', 0)
        )
    
    html_content += """
    </body>
    </html>
    """
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML报告已保存至: {output_path}")
    
    return html_content


def _generate_markdown_report(results: Dict[str, Any], output_path: Optional[str]) -> str:
    """生成Markdown格式报告"""
    import pandas as pd
    
    md_content = f"""# 湿地高光谱分类评估报告

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估概览

"""
    
    # 添加基本指标
    if 'metrics' in results:
        metrics = results['metrics']
        md_content += f"""
### 分类精度指标

| 指标 | 数值 |
|------|------|
| 总体精度 | {metrics.get('overall_accuracy', 0):.3f} |
| Kappa系数 | {metrics.get('kappa', 0):.3f} |
| F1分数(加权) | {metrics.get('f1_weighted', 0):.3f} |
| F1分数(宏平均) | {metrics.get('f1_macro', 0):.3f} |

"""
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logger.info(f"Markdown报告已保存至: {output_path}")
    
    return md_content


def _generate_pdf_report(results: Dict[str, Any], output_path: Optional[str]) -> str:
    """生成PDF格式报告"""
    logger.warning("PDF报告生成需要额外的依赖包，当前返回HTML格式")
    return _generate_html_report(results, output_path)


# 模块初始化日志
logger.info(f"评估模块 v{__version__} 初始化完成")
logger.info("可用功能: 精度评估、交叉验证、不确定性分析")

# 检查必要依赖
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    logger.info("所有必要依赖包已加载")
except ImportError as e:
    logger.warning(f"部分依赖包缺失: {e}")
    logger.warning("某些功能可能无法正常使用")