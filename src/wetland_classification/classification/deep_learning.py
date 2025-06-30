"""
深度学习分类器
==============

这个模块实现了多种深度学习分类算法，专门针对高光谱遥感数据设计。

支持的算法：
- 3D-CNN: 三维卷积神经网络
- HybridSN: 3D-2D混合卷积网络
- Vision Transformer: 视觉Transformer
- ResNet: 残差网络
- DenseNet: 密集连接网络

作者: 湿地遥感研究团队
日期: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from .base import BaseClassifier, handle_classification_errors

# 设置日志
logger = logging.getLogger(__name__)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


class HyperspectralDataset(Dataset):
    """高光谱数据集类"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 patch_size: int = 11, 
                 transform=None):
        """
        初始化数据集
        
        Parameters:
        -----------
        X : np.ndarray
            光谱特征数据
        y : np.ndarray
            标签数据
        patch_size : int
            空间窗口大小
        transform : callable, optional
            数据变换函数
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.patch_size = patch_size
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class CNN3D(nn.Module):
    """
    3D卷积神经网络
    
    专门针对高光谱数据的三维卷积网络，能够同时提取光谱和空间特征。
    """
    
    def __init__(self, 
                 input_channels: int,
                 num_classes: int,
                 patch_size: int = 11,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[Tuple] = [(7, 3, 3), (5, 3, 3), (3, 3, 3)],
                 dropout_rate: float = 0.5):
        """
        初始化3D-CNN网络
        
        Parameters:
        -----------
        input_channels : int
            输入光谱波段数
        num_classes : int
            分类类别数
        patch_size : int
            空间窗口大小
        conv_channels : list
            卷积层通道数
        kernel_sizes : list
            卷积核大小
        dropout_rate : float
            Dropout比率
        """
        super(CNN3D, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        # 3D卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for i, out_channels in enumerate(conv_channels):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else (3, 3, 3)
            
            conv_block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2, 2),
                nn.Dropout3d(dropout_rate * 0.5)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # 计算全连接层输入维度
        self.feature_size = self._get_feature_size()
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def _get_feature_size(self):
        """计算特征维度"""
        # 创建虚拟输入来计算维度
        x = torch.zeros(1, 1, self.input_channels, self.patch_size, self.patch_size)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch_size, channels, height, width)
        # 转换为3D: (batch_size, 1, channels, height, width)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        # 3D卷积
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class HybridSN(nn.Module):
    """
    混合光谱网络 (HybridSN)
    
    结合3D和2D卷积的混合网络，先用3D卷积提取光谱-空间特征，
    再用2D卷积提取空间特征。
    """
    
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 patch_size: int = 11,
                 spectral_size: int = 30,
                 dropout_rate: float = 0.4):
        """
        初始化HybridSN网络
        
        Parameters:
        -----------
        input_channels : int
            输入光谱波段数
        num_classes : int
            分类类别数
        patch_size : int
            空间窗口大小
        spectral_size : int
            3D卷积输出的光谱维度
        dropout_rate : float
            Dropout比率
        """
        super(HybridSN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.spectral_size = spectral_size
        
        # 3D卷积层 - 光谱-空间特征提取
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, (5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # 计算3D卷积后的维度
        conv3d_output_channels = 32
        conv3d_output_spectral = input_channels - 7 - 5 - 3 + 3  # 考虑padding
        conv3d_output_spatial = patch_size  # 由于padding=(0,1,1)，空间维度保持不变
        
        # 重塑层：将3D特征映射转换为2D
        self.reshape_channels = conv3d_output_channels * conv3d_output_spectral
        
        # 2D卷积层 - 空间特征提取
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(self.reshape_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch_size, channels, height, width)
        # 转换为3D: (batch_size, 1, channels, height, width)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        # 3D卷积
        x = self.conv3d_1(x)  # (B, 8, C', H, W)
        x = self.conv3d_2(x)  # (B, 16, C'', H, W)
        x = self.conv3d_3(x)  # (B, 32, C''', H, W)
        
        # 重塑为2D: (B, 32*C''', H, W)
        batch_size = x.size(0)
        x = x.view(batch_size, self.reshape_channels, x.size(3), x.size(4))
        
        # 2D卷积
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        
        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class SpectralTransformer(nn.Module):
    """
    光谱Transformer
    
    基于注意力机制的光谱特征学习网络。
    """
    
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout_rate: float = 0.1):
        """
        初始化光谱Transformer
        
        Parameters:
        -----------
        input_channels : int
            输入光谱波段数
        num_classes : int
            分类类别数
        d_model : int
            模型维度
        nhead : int
            注意力头数
        num_layers : int
            Transformer层数
        dropout_rate : float
            Dropout比率
        """
        super(SpectralTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, input_channels, d_model))
        
        # 光谱嵌入
        self.spectral_embedding = nn.Linear(1, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch_size, channels, height, width)
        # 提取中心像素的光谱
        center = x.size(-1) // 2
        x = x[:, :, center, center]  # (batch_size, channels)
        
        # 光谱嵌入
        x = x.unsqueeze(-1)  # (batch_size, channels, 1)
        x = self.spectral_embedding(x)  # (batch_size, channels, d_model)
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # Transformer编码
        x = self.transformer(x)  # (batch_size, channels, d_model)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 分类
        x = self.classifier(x)
        
        return x


class ResNetBlock(nn.Module):
    """ResNet残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HyperspectralResNet(nn.Module):
    """
    高光谱ResNet
    
    基于残差连接的深度网络，适合深层特征学习。
    """
    
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 layers: List[int] = [2, 2, 2, 2],
                 dropout_rate: float = 0.5):
        """
        初始化高光谱ResNet
        
        Parameters:
        -----------
        input_channels : int
            输入光谱波段数
        num_classes : int
            分类类别数
        layers : list
            各层的残差块数量
        dropout_rate : float
            Dropout比率
        """
        super(HyperspectralResNet, self).__init__()
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 残差层
        self.layer1 = self._make_layer(64, layers[0], 1)
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """构建残差层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch_size, channels, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        return x


class DeepLearningClassifier(BaseClassifier):
    """
    深度学习分类器基类
    
    所有深度学习分类器的基础类，提供训练、预测等通用功能。
    """
    
    def __init__(self,
                 model_type: str = 'cnn3d',
                 patch_size: int = 11,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 10,
                 optimizer_type: str = 'adam',
                 scheduler_type: str = 'step',
                 weight_decay: float = 1e-4,
                 device: Optional[str] = None,
                 auto_scale: bool = True,
                 validation_split: float = 0.2,
                 **kwargs):
        """
        初始化深度学习分类器
        
        Parameters:
        -----------
        model_type : str, default='cnn3d'
            模型类型 ('cnn3d', 'hybrid', 'transformer', 'resnet')
        patch_size : int, default=11
            空间窗口大小
        batch_size : int, default=64
            批量大小
        learning_rate : float, default=0.001
            学习率
        num_epochs : int, default=100
            训练轮数
        early_stopping_patience : int, default=10
            早停耐心值
        optimizer_type : str, default='adam'
            优化器类型
        scheduler_type : str, default='step'
            学习率调度器类型
        weight_decay : float, default=1e-4
            权重衰减
        device : str, optional
            计算设备
        auto_scale : bool, default=True
            是否自动标准化
        validation_split : float, default=0.2
            验证集比例
        """
        super().__init__(**kwargs)
        
        self.model_type = model_type
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.weight_decay = weight_decay
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.auto_scale = auto_scale
        self.validation_split = validation_split
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.label_encoder = None
        self.training_history = defaultdict(list)
    
    @handle_classification_errors
    def fit(self, X: np.ndarray, y: np.ndarray, 
           X_val: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None,
           **kwargs) -> 'DeepLearningClassifier':
        """
        训练深度学习分类器
        
        Parameters:
        -----------
        X : np.ndarray
            训练特征数据
        y : np.ndarray
            训练标签数据
        X_val : np.ndarray, optional
            验证特征数据
        y_val : np.ndarray, optional
            验证标签数据
            
        Returns:
        --------
        self : DeepLearningClassifier
            返回自身实例
        """
        logger.info(f"开始训练深度学习分类器 - 模型类型: {self.model_type}")
        start_time = time.time()
        
        # 数据预处理
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # 划分验证集
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_processed, 
                test_size=self.validation_split,
                stratify=y_processed,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X_processed, y_processed
            X_val, y_val = self._preprocess_data(X_val, y_val, fit_transform=False)
        
        # 创建数据加载器
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        # 创建模型
        self._create_model(X_train.shape[1], len(np.unique(y_processed)))
        
        # 创建优化器和调度器
        self._create_optimizer()
        self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss, train_accuracy = self._train_epoch(train_loader)
            
            # 验证
            val_loss, val_accuracy = self._validate_epoch(val_loader)
            
            # 学习率调度
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 早停检查
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                epochs_without_improvement += 1
            
            # 日志输出
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
            
            # 早停
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"早停触发 - 在第 {epoch+1} 轮停止训练")
                break
        
        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        # 记录训练信息
        training_time = time.time() - start_time
        self.training_history['training_time'] = training_time
        self.training_history['best_val_accuracy'] = best_val_accuracy
        
        self.is_trained = True
        logger.info(f"深度学习训练完成 - 耗时: {training_time:.2f}秒")
        logger.info(f"最佳验证精度: {best_val_accuracy:.4f}")
        
        return self
    
    def _preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                        fit_transform: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理"""
        # 特征标准化
        if self.auto_scale:
            if fit_transform:
                self.scaler = StandardScaler()
                # 重塑为2D进行标准化
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.fit_transform(X_reshaped)
                X_processed = X_scaled.reshape(X.shape)
            else:
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.transform(X_reshaped)
                X_processed = X_scaled.reshape(X.shape)
        else:
            X_processed = X.copy()
        
        # 标签编码
        if fit_transform:
            self.label_encoder = LabelEncoder()
            y_processed = self.label_encoder.fit_transform(y)
            self.class_names = [f"类别{i}" for i in self.label_encoder.classes_]
        else:
            y_processed = self.label_encoder.transform(y)
        
        return X_processed, y_processed
    
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, 
                          shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        dataset = HyperspectralDataset(X, y, self.patch_size)
        return DataLoader(dataset, batch_size=self.batch_size, 
                         shuffle=shuffle, num_workers=0)
    
    def _create_model(self, input_channels: int, num_classes: int):
        """创建模型"""
        if self.model_type == 'cnn3d':
            self.model = CNN3D(input_channels, num_classes, self.patch_size)
        elif self.model_type == 'hybrid':
            self.model = HybridSN(input_channels, num_classes, self.patch_size)
        elif self.model_type == 'transformer':
            self.model = SpectralTransformer(input_channels, num_classes)
        elif self.model_type == 'resnet':
            self.model = HyperspectralResNet(input_channels, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.model = self.model.to(self.device)
        logger.info(f"模型已创建并移动到设备: {self.device}")
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif self.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @handle_classification_errors
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 预处理
        X_processed, _ = self._preprocess_data(X, np.zeros(len(X)), fit_transform=False)
        
        # 创建数据加载器
        dataset = HyperspectralDataset(X_processed, np.zeros(len(X_processed)), self.patch_size)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 预测
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
        
        # 反编码标签
        predictions = self.label_encoder.inverse_transform(predictions)
        
        return np.array(predictions)
    
    @handle_classification_errors
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 预处理
        X_processed, _ = self._preprocess_data(X, np.zeros(len(X)), fit_transform=False)
        
        # 创建数据加载器
        dataset = HyperspectralDataset(X_processed, np.zeros(len(X_processed)), self.patch_size)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # 预测
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                proba = F.softmax(outputs, dim=1)
                probabilities.extend(proba.cpu().numpy())
        
        return np.array(probabilities)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制训练历史"""
        if not self.training_history:
            raise ValueError("没有训练历史可绘制")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        axes[0, 0].plot(self.training_history['train_loss'], label='训练损失', color='blue')
        axes[0, 0].plot(self.training_history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 精度曲线
        axes[0, 1].plot(self.training_history['train_accuracy'], label='训练精度', color='blue')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='验证精度', color='red')
        axes[0, 1].set_title('精度曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 0].plot(self.training_history['learning_rate'], color='green')
        axes[1, 0].set_title('学习率曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 模型信息
        axes[1, 1].axis('off')
        info_text = f"""
        模型类型: {self.model_type}
        批量大小: {self.batch_size}
        学习率: {self.learning_rate}
        训练轮数: {len(self.training_history['train_loss'])}
        最佳验证精度: {self.training_history.get('best_val_accuracy', 'N/A'):.4f}
        训练时间: {self.training_history.get('training_time', 0):.2f}秒
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_deep_classifier(model_type: str, **kwargs) -> DeepLearningClassifier:
    """
    深度学习分类器工厂函数
    
    Parameters:
    -----------
    model_type : str
        模型类型 ('cnn3d', 'hybrid', 'transformer', 'resnet')
    **kwargs : dict
        分类器参数
        
    Returns:
    --------
    classifier : DeepLearningClassifier
        创建的深度学习分类器实例
    """
    supported_types = ['cnn3d', 'hybrid', 'transformer', 'resnet']
    
    if model_type.lower() not in supported_types:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {supported_types}")
    
    kwargs['model_type'] = model_type.lower()
    return DeepLearningClassifier(**kwargs)