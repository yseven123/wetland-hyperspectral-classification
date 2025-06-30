"""
日志系统模块
Logger Module

提供完整的日志记录和进度跟踪功能，包括：
- 多级别日志记录
- 文件和控制台输出
- 进度条和状态跟踪
- 系统信息记录
- 性能监控
"""

import os
import sys
import logging
import time
import psutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(
    log_dir: Union[str, Path] = "logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        max_file_size: 最大文件大小（字节）
        backup_count: 备份文件数量
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 创建格式化器
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 文件处理器
    if log_to_file:
        from logging.handlers import RotatingFileHandler
        
        log_file = log_dir / f"wetland_classification_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger对象
    """
    return logging.getLogger(name)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    记录系统信息
    
    Args:
        logger: 日志记录器，如果为None则使用根记录器
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("=" * 50)
    logger.info("系统信息")
    logger.info("=" * 50)
    
    # 操作系统信息
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"CPU架构: {platform.machine()}")
    logger.info(f"处理器: {platform.processor()}")
    
    # CPU信息
    logger.info(f"CPU核心数: {psutil.cpu_count(logical=False)} (物理)")
    logger.info(f"CPU线程数: {psutil.cpu_count(logical=True)} (逻辑)")
    
    # 内存信息
    memory = psutil.virtual_memory()
    logger.info(f"总内存: {memory.total / (1024**3):.2f} GB")
    logger.info(f"可用内存: {memory.available / (1024**3):.2f} GB")
    logger.info(f"内存使用率: {memory.percent:.1f}%")
    
    # GPU信息 (如果可用)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            logger.info("CUDA不可用")
    except ImportError:
        logger.info("PyTorch未安装，无法检测GPU信息")
    
    # Python环境信息
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    
    logger.info("=" * 50)


def log_data_info(
    data: np.ndarray,
    data_name: str = "数据",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    记录数据信息
    
    Args:
        data: 数据数组
        data_name: 数据名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"{data_name}信息:")
    logger.info(f"  形状: {data.shape}")
    logger.info(f"  数据类型: {data.dtype}")
    logger.info(f"  数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
    logger.info(f"  均值: {np.mean(data):.6f}")
    logger.info(f"  标准差: {np.std(data):.6f}")
    
    # 检查特殊值
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    if nan_count > 0:
        logger.warning(f"  包含 {nan_count} 个NaN值")
    if inf_count > 0:
        logger.warning(f"  包含 {inf_count} 个无穷值")
    
    # 内存占用
    memory_mb = data.nbytes / (1024 * 1024)
    logger.info(f"  内存占用: {memory_mb:.2f} MB")


def log_processing_step(
    step_name: str,
    start_time: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    记录处理步骤
    
    Args:
        step_name: 步骤名称
        start_time: 开始时间，如果为None则记录开始
        logger: 日志记录器
        
    Returns:
        当前时间戳
    """
    if logger is None:
        logger = logging.getLogger()
    
    current_time = time.time()
    
    if start_time is None:
        logger.info(f"开始执行: {step_name}")
        return current_time
    else:
        duration = current_time - start_time
        logger.info(f"完成执行: {step_name} (耗时: {duration:.2f}秒)")
        return current_time


def log_model_performance(
    metrics: Dict[str, float],
    model_name: str = "模型",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    记录模型性能指标
    
    Args:
        metrics: 性能指标字典
        model_name: 模型名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"{model_name}性能指标:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(
        self,
        total: int,
        description: str = "处理中",
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10  # 每隔多少个项目记录一次
    ):
        self.total = total
        self.description = description
        self.logger = logger or logging.getLogger()
        self.log_interval = log_interval
        self.start_time = time.time()
        self.completed = 0
        
        # 创建进度条
        self.pbar = tqdm(
            total=total,
            desc=description,
            unit="items",
            ncols=80
        )
        
        self.logger.info(f"开始{description}: 总计 {total} 项")
    
    def update(self, n: int = 1, message: Optional[str] = None) -> None:
        """
        更新进度
        
        Args:
            n: 完成的项目数
            message: 额外信息
        """
        self.completed += n
        self.pbar.update(n)
        
        # 定期记录进度
        if self.completed % self.log_interval == 0 or self.completed >= self.total:
            elapsed_time = time.time() - self.start_time
            progress_percent = (self.completed / self.total) * 100
            
            if self.completed < self.total:
                eta = (elapsed_time / self.completed) * (self.total - self.completed)
                eta_str = f", 预计剩余: {eta:.1f}秒"
            else:
                eta_str = ""
            
            log_msg = (
                f"进度: {self.completed}/{self.total} "
                f"({progress_percent:.1f}%), "
                f"已耗时: {elapsed_time:.1f}秒{eta_str}"
            )
            
            if message:
                log_msg += f" - {message}"
            
            self.logger.info(log_msg)
    
    def close(self, message: Optional[str] = None) -> None:
        """
        关闭进度跟踪器
        
        Args:
            message: 完成信息
        """
        self.pbar.close()
        
        total_time = time.time() - self.start_time
        avg_time = total_time / self.total if self.total > 0 else 0
        
        final_msg = (
            f"完成{self.description}: {self.completed}/{self.total}, "
            f"总耗时: {total_time:.2f}秒, "
            f"平均: {avg_time:.3f}秒/项"
        )
        
        if message:
            final_msg += f" - {message}"
        
        self.logger.info(final_msg)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_progress_bar(
    total: int,
    description: str = "处理中",
    **kwargs
) -> tqdm:
    """
    创建简单的进度条
    
    Args:
        total: 总项目数
        description: 描述
        **kwargs: tqdm的额外参数
        
    Returns:
        tqdm进度条对象
    """
    return tqdm(
        total=total,
        desc=description,
        unit="items",
        ncols=80,
        **kwargs
    )


class Timer:
    """计时器上下文管理器"""
    
    def __init__(
        self,
        name: str = "操作",
        logger: Optional[logging.Logger] = None,
        auto_log: bool = True
    ):
        self.name = name
        self.logger = logger or logging.getLogger()
        self.auto_log = auto_log
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
        if self.auto_log:
            self.logger.info(f"开始{self.name}")
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            耗时（秒）
        """
        self.end_time = time.time()
        duration = self.elapsed_time
        
        if self.auto_log:
            self.logger.info(f"完成{self.name}，耗时: {duration:.3f}秒")
        
        return duration
    
    @property
    def elapsed_time(self) -> float:
        """获取已耗时"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextmanager
def log_memory_usage(
    operation_name: str = "操作",
    logger: Optional[logging.Logger] = None
):
    """
    记录内存使用情况的上下文管理器
    
    Args:
        operation_name: 操作名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()
    
    # 记录操作前内存
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    logger.info(f"开始{operation_name}，当前内存使用: {memory_before:.2f} MB")
    
    try:
        yield
    finally:
        # 记录操作后内存
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = memory_after - memory_before
        
        logger.info(
            f"完成{operation_name}，内存使用: {memory_after:.2f} MB "
            f"(变化: {memory_diff:+.2f} MB)"
        )


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger()
        self.metrics = {}
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: Union[int, float]) -> None:
        """记录指标"""
        self.metrics[name] = value
        self.logger.debug(f"记录指标 {name}: {value}")
    
    def log_metrics(self) -> None:
        """记录所有指标"""
        total_time = time.time() - self.start_time
        
        self.logger.info("性能监控报告:")
        self.logger.info(f"  总运行时间: {total_time:.2f}秒")
        
        for name, value in self.metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {name}: {value:.4f}")
            else:
                self.logger.info(f"  {name}: {value}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    
    def log_memory_status(self) -> None:
        """记录内存状态"""
        memory = self.get_memory_usage()
        self.logger.info(
            f"内存使用: RSS={memory['rss_mb']:.2f}MB, "
            f"VMS={memory['vms_mb']:.2f}MB, "
            f"使用率={memory['percent']:.1f}%"
        )


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()


def setup_default_logging():
    """设置默认日志配置"""
    setup_logging(
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )


# 在模块导入时自动设置默认日志
if not logging.getLogger().handlers:
    setup_default_logging()