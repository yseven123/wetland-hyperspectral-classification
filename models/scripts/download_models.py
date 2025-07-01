#!/usr/bin/env python3
"""
湿地高光谱分类系统 - 预训练模型下载脚本

功能:
- 自动下载预训练模型
- 验证文件完整性
- 管理模型版本
- 支持断点续传

作者: 湿地分类系统开发团队
日期: 2024-12-01
"""

import os
import json
import hashlib
import argparse
import requests
import urllib.request
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """预训练模型下载器"""
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化下载器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = Path(models_dir)
        self.pretrained_dir = self.models_dir / "pretrained"
        self.registry_file = self.pretrained_dir / "model_registry.json"
        self.checksums_file = self.pretrained_dir / "checksums.md5"
        
        # 创建目录
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        
        # 备用下载源
        self.mirror_urls = [
            "https://github.com/yourusername/wetland-hyperspectral-classification/releases/download",
            "https://pan.baidu.com/s/xxxxx",  # 百度网盘
            "https://drive.google.com/xxxxx",  # Google Drive
        ]
        
    def load_registry(self) -> Dict:
        """加载模型注册表"""
        if not self.registry_file.exists():
            logger.error(f"模型注册表不存在: {self.registry_file}")
            return {}
            
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_md5(self, file_path: Path) -> str:
        """计算文件MD5校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """验证文件校验和"""
        if not file_path.exists():
            return False
            
        actual_checksum = self.calculate_md5(file_path)
        expected = expected_checksum.replace('md5:', '')
        
        if actual_checksum == expected:
            logger.info(f"✓ 文件校验成功: {file_path.name}")
            return True
        else:
            logger.warning(f"✗ 文件校验失败: {file_path.name}")
            logger.warning(f"  期望: {expected}")
            logger.warning(f"  实际: {actual_checksum}")
            return False
    
    def download_with_progress(self, url: str, file_path: Path, resume: bool = True) -> bool:
        """
        带进度条的文件下载
        
        Args:
            url: 下载链接
            file_path: 保存路径
            resume: 是否支持断点续传
            
        Returns:
            下载是否成功
        """
        # 创建目录
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查是否需要断点续传
        resume_byte_pos = 0
        if resume and file_path.exists():
            resume_byte_pos = file_path.stat().st_size
            
        # 设置请求头
        headers = {}
        if resume_byte_pos:
            headers['Range'] = f'bytes={resume_byte_pos}-'
            
        try:
            # 发送请求
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # 获取文件总大小
            total_size = int(response.headers.get('content-length', 0))
            if resume_byte_pos:
                total_size += resume_byte_pos
                
            # 打开文件进行写入
            mode = 'ab' if resume_byte_pos else 'wb'
            with open(file_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=resume_byte_pos,
                    unit='B',
                    unit_scale=True,
                    desc=file_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            logger.info(f"✓ 下载完成: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ 下载失败: {file_path.name} - {str(e)}")
            return False
    
    def download_model(self, model_id: str, force: bool = False) -> bool:
        """
        下载指定模型
        
        Args:
            model_id: 模型ID
            force: 强制重新下载
            
        Returns:
            下载是否成功
        """
        registry = self.load_registry()
        
        if model_id not in registry['models']:
            logger.error(f"模型不存在: {model_id}")
            return False
            
        model_info = registry['models'][model_id]
        file_path = self.pretrained_dir / model_info['file_path']
        
        # 检查文件是否已存在且校验正确
        if not force and file_path.exists():
            if self.verify_checksum(file_path, model_info['checksum']):
                logger.info(f"模型已存在且校验正确: {model_id}")
                return True
        
        # 下载模型
        logger.info(f"开始下载模型: {model_id}")
        logger.info(f"文件大小: {model_info['file_size']}")
        
        download_url = model_info['download_url']
        success = self.download_with_progress(download_url, file_path)
        
        if success:
            # 验证下载的文件
            if self.verify_checksum(file_path, model_info['checksum']):
                logger.info(f"✓ 模型下载并验证成功: {model_id}")
                return True
            else:
                logger.error(f"✗ 模型文件校验失败: {model_id}")
                file_path.unlink()  # 删除损坏的文件
                return False
        else:
            return False
    
    def download_by_type(self, model_type: str, force: bool = False) -> List[str]:
        """
        按类型下载模型
        
        Args:
            model_type: 模型类型 (traditional_ml, deep_learning, ensemble)
            force: 强制重新下载
            
        Returns:
            成功下载的模型列表
        """
        registry = self.load_registry()
        models = registry['models']
        
        target_models = [
            model_id for model_id, model_info in models.items()
            if model_info['type'] == model_type
        ]
        
        if not target_models:
            logger.error(f"未找到类型为 {model_type} 的模型")
            return []
        
        logger.info(f"找到 {len(target_models)} 个 {model_type} 类型的模型")
        
        successful_downloads = []
        for model_id in target_models:
            if self.download_model(model_id, force):
                successful_downloads.append(model_id)
                
        return successful_downloads
    
    def download_all(self, force: bool = False) -> List[str]:
        """
        下载所有模型
        
        Args:
            force: 强制重新下载
            
        Returns:
            成功下载的模型列表
        """
        registry = self.load_registry()
        models = list(registry['models'].keys())
        
        logger.info(f"开始下载所有模型，共 {len(models)} 个")
        
        successful_downloads = []
        for model_id in models:
            if self.download_model(model_id, force):
                successful_downloads.append(model_id)
                
        return successful_downloads
    
    def list_models(self, model_type: Optional[str] = None) -> None:
        """列出可用模型"""
        registry = self.load_registry()
        models = registry['models']
        
        if model_type:
            models = {
                k: v for k, v in models.items() 
                if v['type'] == model_type
            }
        
        print(f"\n可用模型 ({len(models)} 个):")
        print("-" * 80)
        print(f"{'模型ID':<25} {'类型':<15} {'算法':<15} {'精度':<8} {'大小':<10}")
        print("-" * 80)
        
        for model_id, model_info in models.items():
            accuracy = model_info['performance']['overall_accuracy']
            file_size = model_info['file_size']
            algorithm = model_info['algorithm']
            model_type = model_info['type']
            
            print(f"{model_id:<25} {model_type:<15} {algorithm:<15} {accuracy:<8.3f} {file_size:<10}")
    
    def check_models(self) -> Dict[str, bool]:
        """检查已下载模型的完整性"""
        registry = self.load_registry()
        models = registry['models']
        
        results = {}
        print(f"\n检查模型完整性:")
        print("-" * 60)
        
        for model_id, model_info in models.items():
            file_path = self.pretrained_dir / model_info['file_path']
            
            if file_path.exists():
                is_valid = self.verify_checksum(file_path, model_info['checksum'])
                results[model_id] = is_valid
                status = "✓ 正常" if is_valid else "✗ 损坏"
                print(f"{model_id:<30} {status}")
            else:
                results[model_id] = False
                print(f"{model_id:<30} ✗ 未下载")
                
        return results
    
    def clean_cache(self) -> None:
        """清理下载缓存"""
        cache_files = [
            "*.tmp",
            "*.download",
            "*.part"
        ]
        
        for pattern in cache_files:
            for file_path in self.pretrained_dir.rglob(pattern):
                file_path.unlink()
                logger.info(f"删除缓存文件: {file_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="湿地高光谱分类系统 - 预训练模型下载工具"
    )
    
    parser.add_argument(
        '--models-dir',
        default='models',
        help='模型存储目录 (默认: models)'
    )
    
    # 操作类型
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        '--all',
        action='store_true',
        help='下载所有模型'
    )
    
    group.add_argument(
        '--type',
        choices=['traditional_ml', 'deep_learning', 'ensemble', 'compressed'],
        help='按类型下载模型'
    )
    
    group.add_argument(
        '--model',
        help='下载指定模型 (使用模型ID)'
    )
    
    group.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用模型'
    )
    
    group.add_argument(
        '--check',
        action='store_true',
        help='检查已下载模型的完整性'
    )
    
    group.add_argument(
        '--clean',
        action='store_true',
        help='清理下载缓存'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新下载'
    )
    
    parser.add_argument(
        '--list-type',
        help='列出指定类型的模型'
    )
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = ModelDownloader(args.models_dir)
    
    try:
        if args.all:
            # 下载所有模型
            successful = downloader.download_all(args.force)
            print(f"\n成功下载 {len(successful)} 个模型")
            
        elif args.type:
            # 按类型下载
            successful = downloader.download_by_type(args.type, args.force)
            print(f"\n成功下载 {len(successful)} 个 {args.type} 类型的模型")
            
        elif args.model:
            # 下载指定模型
            success = downloader.download_model(args.model, args.force)
            if success:
                print(f"\n✓ 模型 {args.model} 下载成功")
            else:
                print(f"\n✗ 模型 {args.model} 下载失败")
                
        elif args.list:
            # 列出所有模型
            downloader.list_models()
            
        elif args.list_type:
            # 列出指定类型的模型
            downloader.list_models(args.list_type)
            
        elif args.check:
            # 检查模型完整性
            results = downloader.check_models()
            valid_count = sum(results.values())
            total_count = len(results)
            print(f"\n检查完成: {valid_count}/{total_count} 个模型正常")
            
        elif args.clean:
            # 清理缓存
            downloader.clean_cache()
            print("\n缓存清理完成")
            
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()