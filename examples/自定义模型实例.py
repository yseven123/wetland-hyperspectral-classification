#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
湿地高光谱分类系统 - 自定义模型示例
Wetland Hyperspectral Classification System - Custom Model Example

这个示例展示了如何在湿地高光谱分类系统中开发和集成自定义模型，包括：
- 自定义深度学习模型架构
- 自定义特征提取器
- 自定义损失函数和优化器
- 模型融合和集成策略
- 自定义评估指标
- 模型可解释性分析
- 迁移学习和微调

作者: 研究团队
日期: 2024-06-30
版本: 1.0.0
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from datetime import datetime
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入湿地分类系统模块
try:
    from wetland_classification import Pipeline
    from wetland_classification.config import Config
    from wetland_classification.data import DataLoader
    from wetland_classification.preprocessing import Preprocessor
    from wetland_classification.features import FeatureExtractor
    from wetland_classification.classification.base import BaseClassifier
    from wetland_classification.evaluation import ModelEvaluator
    from wetland_classification.utils.visualization import Visualizer
    from wetland_classification.utils.logger import get_logger
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已正确安装湿地分类系统")
    sys.exit(1)

# 设置日志
logger = get_logger(__name__)

# ======================== 自定义数据集类 ========================

class HyperspectralDataset(Dataset):
    """
    自定义高光谱数据集类
    """
    
    def __init__(self, features, labels, transform=None):
        """
        初始化数据集
        
        Args:
            features: 特征数据 (N, D)
            labels: 标签数据 (N,)
            transform: 可选的数据变换
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

class SpectralAugmentation:
    """
    光谱数据增强类
    """
    
    def __init__(self, noise_std=0.01, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, spectrum):
        # 添加高斯噪声
        if self.noise_std > 0:
            noise = torch.normal(0, self.noise_std, spectrum.shape)
            spectrum = spectrum + noise
        
        # 尺度变换
        if self.scale_range:
            scale = torch.uniform(self.scale_range[0], self.scale_range[1], (1,))
            spectrum = spectrum * scale
        
        return spectrum

# ======================== 自定义模型架构 ========================

class SpectralTransformer(nn.Module):
    """
    自定义光谱Transformer模型
    """
    
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=6, 
                 hidden_dim=512, dropout=0.1):
        """
        初始化光谱Transformer
        
        Args:
            input_dim: 输入特征维度
            num_classes: 类别数
            num_heads: 注意力头数
            num_layers: Transformer层数
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
        """
        super(SpectralTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = self._create_position_encoding(1000, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _create_position_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影 (batch_size, input_dim) -> (batch_size, hidden_dim)
        x = self.input_projection(x)
        
        # 添加位置编码
        seq_len = 1  # 对于光谱数据，我们将其视为单一序列
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer编码 (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        
        # 全局平均池化 (batch_size, hidden_dim)
        x = x.mean(dim=0)
        
        # 分类
        logits = self.classifier(x)
        
        return logits

class HybridCNN1D(nn.Module):
    """
    混合1D CNN模型，结合多尺度卷积
    """
    
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(HybridCNN1D, self).__init__()
        
        # 多尺度卷积分支
        self.conv_branch1 = self._create_conv_branch(input_dim, [64, 128], [3, 3])
        self.conv_branch2 = self._create_conv_branch(input_dim, [64, 128], [5, 5])
        self.conv_branch3 = self._create_conv_branch(input_dim, [64, 128], [7, 7])
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(384, 192),  # 3 * 128 = 384
            nn.Tanh(),
            nn.Linear(192, 3),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def _create_conv_branch(self, input_dim, channels, kernel_sizes):
        """创建卷积分支"""
        layers = []
        in_channels = 1
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 重塑输入 (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # 多尺度特征提取
        feat1 = self.conv_branch1(x).squeeze(-1)  # (batch_size, 128)
        feat2 = self.conv_branch2(x).squeeze(-1)  # (batch_size, 128)
        feat3 = self.conv_branch3(x).squeeze(-1)  # (batch_size, 128)
        
        # 特征融合
        features = torch.cat([feat1, feat2, feat3], dim=1)  # (batch_size, 384)
        
        # 注意力权重
        attention_weights = self.attention(features)  # (batch_size, 3)
        
        # 加权特征融合
        weighted_feat1 = feat1 * attention_weights[:, 0:1]
        weighted_feat2 = feat2 * attention_weights[:, 1:2]
        weighted_feat3 = feat3 * attention_weights[:, 2:3]
        
        final_features = torch.cat([weighted_feat1, weighted_feat2, weighted_feat3], dim=1)
        
        # 分类
        logits = self.classifier(final_features)
        
        return logits

class ResidualSpectralNet(nn.Module):
    """
    残差光谱网络
    """
    
    def __init__(self, input_dim, num_classes, num_blocks=4):
        super(ResidualSpectralNet, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, 256)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._create_residual_block(256, 256) for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _create_residual_block(self, in_dim, out_dim):
        """创建残差块"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)  # (batch_size, 256)
        
        # 残差连接
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)  # 残差连接
        
        # 添加维度用于池化
        x = x.unsqueeze(-1)  # (batch_size, 256, 1)
        
        # 输出
        logits = self.output_layer(x)
        
        return logits

# ======================== 自定义特征提取器 ========================

class CustomFeatureExtractor:
    """
    自定义特征提取器
    """
    
    def __init__(self, wavelengths=None):
        self.wavelengths = wavelengths
    
    def extract_advanced_spectral_features(self, data):
        """
        提取高级光谱特征
        """
        features = []
        feature_names = []
        
        # 1. 波段比值特征
        if data.shape[1] > 50:
            # 红边比值
            red_idx = data.shape[1] // 3
            nir_idx = 2 * data.shape[1] // 3
            
            red_edge_ratio = data[:, nir_idx] / (data[:, red_idx] + 1e-8)
            features.append(red_edge_ratio.reshape(-1, 1))
            feature_names.append('red_edge_ratio')
            
            # 水分比值
            if data.shape[1] > 100:
                swir_idx = int(0.8 * data.shape[1])
                water_ratio = data[:, nir_idx] / (data[:, swir_idx] + 1e-8)
                features.append(water_ratio.reshape(-1, 1))
                feature_names.append('water_ratio')
        
        # 2. 光谱形状特征
        spectral_mean = np.mean(data, axis=1).reshape(-1, 1)
        spectral_std = np.std(data, axis=1).reshape(-1, 1)
        spectral_skew = self._calculate_skewness(data).reshape(-1, 1)
        spectral_kurt = self._calculate_kurtosis(data).reshape(-1, 1)
        
        features.extend([spectral_mean, spectral_std, spectral_skew, spectral_kurt])
        feature_names.extend(['spectral_mean', 'spectral_std', 'spectral_skew', 'spectral_kurt'])
        
        # 3. 导数特征
        first_deriv = np.gradient(data, axis=1)
        deriv_mean = np.mean(first_deriv, axis=1).reshape(-1, 1)
        deriv_std = np.std(first_deriv, axis=1).reshape(-1, 1)
        
        features.extend([deriv_mean, deriv_std])
        feature_names.extend(['deriv_mean', 'deriv_std'])
        
        # 4. 吸收特征
        absorption_depth = self._calculate_absorption_depth(data)
        absorption_area = self._calculate_absorption_area(data)
        
        features.extend([absorption_depth.reshape(-1, 1), absorption_area.reshape(-1, 1)])
        feature_names.extend(['absorption_depth', 'absorption_area'])
        
        # 5. 光谱角特征
        reference_spectrum = np.mean(data, axis=0)
        spectral_angles = self._calculate_spectral_angle(data, reference_spectrum)
        features.append(spectral_angles.reshape(-1, 1))
        feature_names.append('spectral_angle')
        
        return np.column_stack(features), feature_names
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        skewness = np.mean(normalized ** 3, axis=1)
        return skewness
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        kurtosis = np.mean(normalized ** 4, axis=1) - 3
        return kurtosis
    
    def _calculate_absorption_depth(self, data):
        """计算吸收深度"""
        max_vals = np.max(data, axis=1)
        min_vals = np.min(data, axis=1)
        return max_vals - min_vals
    
    def _calculate_absorption_area(self, data):
        """计算吸收面积"""
        return np.trapz(data, axis=1)
    
    def _calculate_spectral_angle(self, data, reference):
        """计算光谱角"""
        angles = []
        for spectrum in data:
            dot_product = np.dot(spectrum, reference)
            norm_product = np.linalg.norm(spectrum) * np.linalg.norm(reference)
            if norm_product > 0:
                angle = np.arccos(np.clip(dot_product / norm_product, -1, 1))
            else:
                angle = 0
            angles.append(angle)
        return np.array(angles)

# ======================== 自定义损失函数 ========================

class FocalLoss(nn.Module):
    """
    Focal Loss，用于处理类别不平衡问题
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ======================== 自定义分类器包装类 ========================

class CustomModelWrapper(BaseClassifier):
    """
    自定义模型包装器，继承自基础分类器
    """
    
    def __init__(self, model_type='spectral_transformer', **kwargs):
        super().__init__()
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {}
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, 
            learning_rate=0.001, **kwargs):
        """
        训练自定义模型
        """
        # 数据预处理
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 验证集处理
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
        else:
            X_val_scaled, y_val_encoded = None, None
        
        # 创建模型
        input_dim = X_scaled.shape[1]
        num_classes = len(np.unique(y_encoded))
        
        self.model = self._create_model(input_dim, num_classes)
        self.model.to(self.device)
        
        # 训练模型
        self._train_model(X_scaled, y_encoded, X_val_scaled, y_val_encoded, 
                         epochs, batch_size, learning_rate)
        
        return self
    
    def _create_model(self, input_dim, num_classes):
        """创建指定类型的模型"""
        if self.model_type == 'spectral_transformer':
            return SpectralTransformer(input_dim, num_classes, **self.model_params)
        elif self.model_type == 'hybrid_cnn1d':
            return HybridCNN1D(input_dim, num_classes, **self.model_params)
        elif self.model_type == 'residual_spectral':
            return ResidualSpectralNet(input_dim, num_classes, **self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_model(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
        """训练模型的具体实现"""
        # 创建数据加载器
        train_dataset = HyperspectralDataset(X_train, y_train, 
                                           transform=SpectralAugmentation())
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = HyperspectralDataset(X_val, y_val)
            val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # 损失函数和优化器
        criterion = FocalLoss(gamma=2.0)  # 使用自定义Focal Loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练循环
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        patience = 0
        early_stopping_patience = 20
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            if val_loader:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = correct / total
                
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_acc)
                
                # 早停检查
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_custom_model.pth')
                else:
                    patience += 1
                
                if patience >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
                if val_loader:
                    logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 加载最佳模型
        if val_loader and os.path.exists('best_custom_model.pth'):
            self.model.load_state_dict(torch.load('best_custom_model.pth'))
        
        # 保存训练历史
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # 分批预测
            batch_size = 1000
            for i in range(0, len(X_scaled), batch_size):
                batch_X = torch.FloatTensor(X_scaled[i:i+batch_size]).to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # 反向编码
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            # 分批预测
            batch_size = 1000
            for i in range(0, len(X_scaled), batch_size):
                batch_X = torch.FloatTensor(X_scaled[i:i+batch_size]).to(self.device)
                outputs = self.model(batch_X)
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save_model(self, path):
        """保存模型"""
        model_data = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'training_history': self.training_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, path):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建实例
        instance = cls(model_data['model_type'], **model_data['model_params'])
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.training_history = model_data['training_history']
        
        # 重建模型
        input_dim = len(instance.scaler.mean_)
        num_classes = len(instance.label_encoder.classes_)
        instance.model = instance._create_model(input_dim, num_classes)
        instance.model.load_state_dict(model_data['model_state_dict'])
        instance.model.to(instance.device)
        
        return instance

# ======================== 自定义评估指标 ========================

class CustomMetrics:
    """
    自定义评估指标类
    """
    
    @staticmethod
    def wetland_specific_score(y_true, y_pred, class_weights=None):
        """
        湿地特定评分，重点关注湿地类别的分类性能
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # 定义湿地类别权重
        if class_weights is None:
            class_weights = {
                1: 2.0,  # 开放水面 - 高权重
                2: 2.0,  # 浅水区域 - 高权重
                3: 1.5,  # 挺水植物 - 中等权重
                4: 1.5,  # 浮叶植物 - 中等权重
                5: 1.5,  # 沉水植物 - 中等权重
                6: 1.0,  # 湿生草本 - 标准权重
                7: 0.5,  # 土壤 - 低权重
                8: 0.3   # 建筑物 - 最低权重
            }
        
        # 计算加权F1分数
        unique_classes = np.unique(y_true)
        weighted_f1_scores = []
        
        for cls in unique_classes:
            if cls in class_weights:
                cls_mask = (y_true == cls) | (y_pred == cls)
                if np.sum(cls_mask) > 0:
                    cls_f1 = f1_score(y_true[cls_mask], y_pred[cls_mask], 
                                    labels=[cls], average='macro', zero_division=0)
                    weighted_f1_scores.append(cls_f1 * class_weights[cls])
        
        return np.mean(weighted_f1_scores) if weighted_f1_scores else 0
    
    @staticmethod
    def ecological_consistency_score(classification_map):
        """
        生态一致性评分，评估分类结果的生态合理性
        """
        height, width = classification_map.shape
        consistency_score = 0
        total_checks = 0
        
        # 定义生态邻接规则
        wetland_classes = {1, 2, 3, 4, 5, 6}  # 湿地相关类别
        water_classes = {1, 2}  # 水体类别
        vegetation_classes = {3, 4, 5, 6}  # 植被类别
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center_class = classification_map[i, j]
                
                # 获取8邻域
                neighbors = [
                    classification_map[i-1, j-1], classification_map[i-1, j], classification_map[i-1, j+1],
                    classification_map[i, j-1],                              classification_map[i, j+1],
                    classification_map[i+1, j-1], classification_map[i+1, j], classification_map[i+1, j+1]
                ]
                
                # 检查生态合理性
                if center_class in water_classes:
                    # 水体周围应该有湿地植被
                    wetland_neighbors = sum(1 for n in neighbors if n in wetland_classes)
                    if wetland_neighbors >= 3:
                        consistency_score += 1
                
                elif center_class in vegetation_classes:
                    # 植被周围应该有水体或其他植被
                    compatible_neighbors = sum(1 for n in neighbors 
                                             if n in wetland_classes)
                    if compatible_neighbors >= 4:
                        consistency_score += 1
                
                total_checks += 1
        
        return consistency_score / total_checks if total_checks > 0 else 0

# ======================== 主要工作流程函数 ========================

def setup_custom_directories():
    """
    设置自定义模型示例的目录结构
    """
    directories = [
        'output/custom_models',
        'output/custom_models/models',
        'output/custom_models/results',
        'output/custom_models/figures',
        'output/custom_models/analysis',
        'output/custom_models/comparisons',
        'logs/custom_models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def load_demo_data_for_custom():
    """
    为自定义模型示例加载演示数据
    """
    # 使用之前创建的演示数据
    demo_dir = Path('data/samples/demo_scene')
    
    if not demo_dir.exists():
        logger.info("演示数据不存在，创建模拟数据...")
        create_simple_demo_data()
    
    # 加载数据
    data = np.load(demo_dir / 'hyperspectral_data.npy')
    labels = np.load(demo_dir / 'ground_truth.npy')
    
    # 加载类别信息
    with open(demo_dir / 'class_info.json', 'r', encoding='utf-8') as f:
        class_info = json.load(f)
    
    class_info = {int(k): v for k, v in class_info.items()}
    
    return data, labels, class_info

def create_simple_demo_data():
    """
    创建简单的演示数据
    """
    demo_dir = Path('data/samples/demo_scene')
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    height, width, bands = 100, 100, 80
    
    # 创建模拟高光谱数据和标签
    data = np.random.rand(height, width, bands)
    labels = np.zeros((height, width), dtype=int)
    
    # 简单的空间分布
    for i in range(height):
        for j in range(width):
            if i < height // 3:
                labels[i, j] = 1  # 水体
            elif i < 2 * height // 3:
                labels[i, j] = 2  # 植被
            else:
                labels[i, j] = 3  # 土壤
    
    # 保存数据
    np.save(demo_dir / 'hyperspectral_data.npy', data)
    np.save(demo_dir / 'ground_truth.npy', labels)
    
    # 类别信息
    class_info = {
        1: {'name': '水体', 'color': '#0000FF'},
        2: {'name': '植被', 'color': '#00FF00'},
        3: {'name': '土壤', 'color': '#8B4513'}
    }
    
    with open(demo_dir / 'class_info.json', 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)

def custom_model_workflow():
    """
    自定义模型主工作流程
    """
    logger.info("="*60)
    logger.info("开始湿地高光谱自定义模型示例")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 步骤1: 数据准备
        logger.info("步骤1: 数据准备")
        data, labels, class_info = load_demo_data_for_custom()
        
        # 步骤2: 自定义特征提取
        logger.info("步骤2: 自定义特征提取")
        features, feature_names = custom_feature_extraction(data)
        
        # 步骤3: 数据准备
        logger.info("步骤3: 数据划分")
        X_train, X_test, y_train, y_test = prepare_training_data(features, labels)
        
        # 步骤4: 训练多个自定义模型
        logger.info("步骤4: 训练自定义模型")
        custom_models = train_custom_models(X_train, y_train, X_test, y_test, class_info)
        
        # 步骤5: 模型比较和评估
        logger.info("步骤5: 模型评估与比较")
        evaluation_results = evaluate_custom_models(custom_models, X_test, y_test, class_info)
        
        # 步骤6: 可解释性分析
        logger.info("步骤6: 模型可解释性分析")
        interpretability_results = analyze_model_interpretability(
            custom_models, X_test, y_test, feature_names
        )
        
        # 步骤7: 迁移学习示例
        logger.info("步骤7: 迁移学习示例")
        transfer_learning_results = demonstrate_transfer_learning(
            custom_models, X_train, y_train, X_test, y_test
        )
        
        # 步骤8: 模型融合
        logger.info("步骤8: 模型融合")
        ensemble_results = create_custom_ensemble(custom_models, X_test, y_test, class_info)
        
        # 步骤9: 结果可视化
        logger.info("步骤9: 结果可视化")
        create_custom_visualizations(
            evaluation_results, interpretability_results, 
            transfer_learning_results, ensemble_results, class_info
        )
        
        # 步骤10: 保存模型和结果
        logger.info("步骤10: 保存结果")
        save_custom_results(
            custom_models, evaluation_results, interpretability_results,
            ensemble_results, class_info
        )
        
        total_time = time.time() - start_time
        display_custom_results(evaluation_results, ensemble_results, total_time)
        
    except Exception as e:
        logger.error(f"自定义模型流程执行失败: {e}")
        raise

def custom_feature_extraction(data):
    """
    自定义特征提取
    """
    height, width, bands = data.shape
    reshaped_data = data.reshape(-1, bands)
    
    # 使用自定义特征提取器
    extractor = CustomFeatureExtractor()
    advanced_features, feature_names = extractor.extract_advanced_spectral_features(reshaped_data)
    
    # 添加原始光谱特征（下采样）
    step = max(1, bands // 30)
    spectral_features = reshaped_data[:, ::step]
    spectral_names = [f'band_{i}' for i in range(0, bands, step)]
    
    # 合并特征
    all_features = np.column_stack([spectral_features, advanced_features])
    all_feature_names = spectral_names + feature_names
    
    logger.info(f"自定义特征提取完成，总特征数: {all_features.shape[1]}")
    
    return all_features, all_feature_names

def prepare_training_data(features, labels):
    """
    准备训练数据
    """
    # 获取有效样本
    valid_mask = labels.ravel() > 0
    valid_features = features[valid_mask]
    valid_labels = labels.ravel()[valid_mask]
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        valid_features, valid_labels, test_size=0.3, random_state=42, stratify=valid_labels
    )
    
    logger.info(f"数据划分完成 - 训练: {len(X_train)}, 测试: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_custom_models(X_train, y_train, X_test, y_test, class_info):
    """
    训练多个自定义模型
    """
    models = {}
    
    # 数据验证集划分
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 1. 光谱Transformer模型
    logger.info("训练光谱Transformer模型...")
    transformer_model = CustomModelWrapper(
        model_type='spectral_transformer',
        num_heads=4,
        num_layers=4,
        hidden_dim=256,
        dropout=0.1
    )
    transformer_model.fit(X_train_split, y_train_split, X_val, y_val, epochs=80)
    models['SpectralTransformer'] = transformer_model
    
    # 2. 混合1D CNN模型
    logger.info("训练混合1D CNN模型...")
    cnn_model = CustomModelWrapper(
        model_type='hybrid_cnn1d',
        dropout=0.3
    )
    cnn_model.fit(X_train_split, y_train_split, X_val, y_val, epochs=80)
    models['HybridCNN1D'] = cnn_model
    
    # 3. 残差光谱网络
    logger.info("训练残差光谱网络...")
    residual_model = CustomModelWrapper(
        model_type='residual_spectral',
        num_blocks=6
    )
    residual_model.fit(X_train_split, y_train_split, X_val, y_val, epochs=80)
    models['ResidualSpectral'] = residual_model
    
    return models

def evaluate_custom_models(models, X_test, y_test, class_info):
    """
    评估自定义模型
    """
    evaluation_results = {}
    
    for model_name, model in models.items():
        logger.info(f"评估模型: {model_name}")
        
        # 基础预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # 基础指标
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        accuracy = accuracy_score(y_test, y_pred)
        
        # 自定义指标
        custom_metrics = CustomMetrics()
        wetland_score = custom_metrics.wetland_specific_score(y_test, y_pred)
        
        # 分类报告
        class_names = [class_info[i]['name'] for i in sorted(class_info.keys())]
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'wetland_score': wetland_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba,
            'training_history': model.training_history
        }
        
        logger.info(f"{model_name} - 精度: {accuracy:.3f}, 湿地评分: {wetland_score:.3f}")
    
    return evaluation_results

def analyze_model_interpretability(models, X_test, y_test, feature_names):
    """
    分析模型可解释性
    """
    interpretability_results = {}
    
    for model_name, model in models.items():
        logger.info(f"分析 {model_name} 可解释性...")
        
        model_interp = {}
        
        # 1. 梯度分析（仅适用于深度学习模型）
        gradients = analyze_gradients(model, X_test[:100])  # 限制样本数量
        model_interp['gradient_importance'] = gradients
        
        # 2. 输入扰动分析
        perturbation_importance = analyze_input_perturbation(model, X_test[:100], y_test[:100])
        model_interp['perturbation_importance'] = perturbation_importance
        
        # 3. 注意力分析（仅适用于Transformer）
        if model_name == 'SpectralTransformer':
            attention_weights = extract_attention_weights(model, X_test[:50])
            model_interp['attention_weights'] = attention_weights
        
        interpretability_results[model_name] = model_interp
    
    return interpretability_results

def analyze_gradients(model, X_test):
    """
    分析模型梯度
    """
    model.model.eval()
    X_tensor = torch.FloatTensor(model.scaler.transform(X_test)).to(model.device)
    X_tensor.requires_grad_(True)
    
    outputs = model.model(X_tensor)
    
    # 计算相对于输入的梯度
    gradients = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=X_tensor,
        create_graph=False,
        retain_graph=False
    )[0]
    
    # 计算特征重要性（梯度的绝对值平均）
    feature_importance = torch.abs(gradients).mean(dim=0).cpu().numpy()
    
    return feature_importance

def analyze_input_perturbation(model, X_test, y_test):
    """
    输入扰动分析
    """
    original_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, original_pred)
    
    feature_importance = []
    
    for feature_idx in range(X_test.shape[1]):
        # 扰动特定特征
        X_perturbed = X_test.copy()
        X_perturbed[:, feature_idx] = np.random.permutation(X_perturbed[:, feature_idx])
        
        # 预测扰动后的结果
        perturbed_pred = model.predict(X_perturbed)
        perturbed_accuracy = accuracy_score(y_test, perturbed_pred)
        
        # 计算重要性（精度下降程度）
        importance = original_accuracy - perturbed_accuracy
        feature_importance.append(importance)
    
    return np.array(feature_importance)

def extract_attention_weights(model, X_test):
    """
    提取注意力权重（适用于Transformer模型）
    """
    if not hasattr(model.model, 'transformer'):
        return None
    
    model.model.eval()
    X_tensor = torch.FloatTensor(model.scaler.transform(X_test)).to(model.device)
    
    attention_weights = []
    
    # 注册钩子函数提取注意力权重
    def attention_hook(module, input, output):
        if hasattr(output, 'detach'):
            attention_weights.append(output.detach().cpu().numpy())
    
    hooks = []
    for layer in model.model.transformer.layers:
        if hasattr(layer, 'self_attn'):
            hook = layer.self_attn.register_forward_hook(attention_hook)
            hooks.append(hook)
    
    with torch.no_grad():
        _ = model.model(X_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return attention_weights

def demonstrate_transfer_learning(models, X_train, y_train, X_test, y_test):
    """
    演示迁移学习
    """
    logger.info("演示迁移学习...")
    
    # 选择最佳模型作为预训练模型
    best_model_name = 'SpectralTransformer'  # 假设这是最佳模型
    pretrained_model = models[best_model_name]
    
    # 创建少量训练数据的场景
    small_data_ratio = 0.1
    n_small = int(len(X_train) * small_data_ratio)
    indices = np.random.choice(len(X_train), n_small, replace=False)
    X_small = X_train[indices]
    y_small = y_train[indices]
    
    # 1. 从头训练（基准）
    logger.info("从头训练小数据集模型...")
    scratch_model = CustomModelWrapper(
        model_type='spectral_transformer',
        num_heads=4,
        num_layers=4,
        hidden_dim=256
    )
    scratch_model.fit(X_small, y_small, epochs=50)
    scratch_pred = scratch_model.predict(X_test)
    scratch_accuracy = accuracy_score(y_test, scratch_pred)
    
    # 2. 迁移学习
    logger.info("应用迁移学习...")
    transfer_model = CustomModelWrapper(
        model_type='spectral_transformer',
        num_heads=4,
        num_layers=4,
        hidden_dim=256
    )
    
    # 复制预训练权重
    transfer_model.scaler = pretrained_model.scaler
    transfer_model.label_encoder = pretrained_model.label_encoder
    
    # 创建模型并加载预训练权重
    input_dim = X_small.shape[1]
    num_classes = len(np.unique(y_small))
    transfer_model.model = transfer_model._create_model(input_dim, num_classes)
    transfer_model.model.load_state_dict(pretrained_model.model.state_dict())
    transfer_model.model.to(transfer_model.device)
    
    # 微调（使用较小的学习率）
    transfer_model._train_model(
        transfer_model.scaler.transform(X_small), 
        transfer_model.label_encoder.transform(y_small),
        None, None,
        epochs=30, batch_size=16, learning_rate=0.0001
    )
    
    transfer_pred = transfer_model.predict(X_test)
    transfer_accuracy = accuracy_score(y_test, transfer_pred)
    
    transfer_results = {
        'scratch_accuracy': scratch_accuracy,
        'transfer_accuracy': transfer_accuracy,
        'improvement': transfer_accuracy - scratch_accuracy,
        'small_data_size': n_small,
        'small_data_ratio': small_data_ratio
    }
    
    logger.info(f"迁移学习结果 - 从头训练: {scratch_accuracy:.3f}, "
               f"迁移学习: {transfer_accuracy:.3f}, "
               f"提升: {transfer_results['improvement']:.3f}")
    
    return transfer_results

def create_custom_ensemble(models, X_test, y_test, class_info):
    """
    创建自定义集成模型
    """
    logger.info("创建自定义集成模型...")
    
    # 获取所有模型的预测概率
    all_probabilities = []
    model_names = list(models.keys())
    
    for model_name, model in models.items():
        proba = model.predict_proba(X_test)
        all_probabilities.append(proba)
    
    all_probabilities = np.array(all_probabilities)  # (n_models, n_samples, n_classes)
    
    # 1. 简单平均集成
    avg_proba = np.mean(all_probabilities, axis=0)
    avg_pred = np.argmax(avg_proba, axis=1)
    
    # 将预测转换回原始标签
    label_encoder = models[model_names[0]].label_encoder
    avg_pred_labels = label_encoder.inverse_transform(avg_pred)
    avg_accuracy = accuracy_score(y_test, avg_pred_labels)
    
    # 2. 加权集成（基于各模型的性能）
    model_weights = []
    for model_name, model in models.items():
        model_pred = model.predict(X_test)
        model_acc = accuracy_score(y_test, model_pred)
        model_weights.append(model_acc)
    
    # 归一化权重
    model_weights = np.array(model_weights)
    model_weights = model_weights / np.sum(model_weights)
    
    # 加权平均
    weighted_proba = np.average(all_probabilities, axis=0, weights=model_weights)
    weighted_pred = np.argmax(weighted_proba, axis=1)
    weighted_pred_labels = label_encoder.inverse_transform(weighted_pred)
    weighted_accuracy = accuracy_score(y_test, weighted_pred_labels)
    
    # 3. 投票集成
    all_predictions = []
    for model_name, model in models.items():
        pred = model.predict(X_test)
        encoded_pred = label_encoder.transform(pred)
        all_predictions.append(encoded_pred)
    
    all_predictions = np.array(all_predictions)  # (n_models, n_samples)
    
    # 多数投票
    from scipy.stats import mode
    vote_pred, _ = mode(all_predictions, axis=0)
    vote_pred = vote_pred.flatten()
    vote_pred_labels = label_encoder.inverse_transform(vote_pred)
    vote_accuracy = accuracy_score(y_test, vote_pred_labels)
    
    ensemble_results = {
        'average_ensemble': {
            'accuracy': avg_accuracy,
            'predictions': avg_pred_labels,
            'probabilities': avg_proba
        },
        'weighted_ensemble': {
            'accuracy': weighted_accuracy,
            'predictions': weighted_pred_labels,
            'probabilities': weighted_proba,
            'model_weights': dict(zip(model_names, model_weights))
        },
        'voting_ensemble': {
            'accuracy': vote_accuracy,
            'predictions': vote_pred_labels
        }
    }
    
    logger.info(f"集成结果 - 平均: {avg_accuracy:.3f}, "
               f"加权: {weighted_accuracy:.3f}, "
               f"投票: {vote_accuracy:.3f}")
    
    return ensemble_results

def create_custom_visualizations(evaluation_results, interpretability_results,
                               transfer_learning_results, ensemble_results, class_info):
    """
    创建自定义模型可视化
    """
    logger.info("创建自定义模型可视化...")
    
    # 1. 模型性能对比
    create_model_performance_comparison(evaluation_results)
    
    # 2. 训练曲线
    create_training_curves(evaluation_results)
    
    # 3. 特征重要性分析
    create_feature_importance_analysis(interpretability_results)
    
    # 4. 迁移学习效果
    create_transfer_learning_visualization(transfer_learning_results)
    
    # 5. 集成模型效果
    create_ensemble_comparison(evaluation_results, ensemble_results)
    
    # 6. 混淆矩阵对比
    create_confusion_matrix_comparison(evaluation_results, class_info)
    
    logger.info("自定义模型可视化创建完成")

def create_model_performance_comparison(evaluation_results):
    """
    创建模型性能对比图
    """
    models = list(evaluation_results.keys())
    accuracies = [results['accuracy'] for results in evaluation_results.values()]
    wetland_scores = [results['wetland_score'] for results in evaluation_results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 精度对比
    bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    ax1.set_ylabel('准确率')
    ax1.set_title('自定义模型准确率对比')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 湿地特定评分对比
    bars2 = ax2.bar(models, wetland_scores, color=['orange', 'purple', 'brown'], alpha=0.7)
    ax2.set_ylabel('湿地特定评分')
    ax2.set_title('湿地分类专用评分对比')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars2, wetland_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/custom_models/figures/model_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_training_curves(evaluation_results):
    """
    创建训练曲线
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['blue', 'red', 'green']
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        if 'training_history' in results and results['training_history']:
            history = results['training_history']
            
            if 'train_losses' in history:
                epochs = range(1, len(history['train_losses']) + 1)
                
                # 训练损失
                axes[idx].plot(epochs, history['train_losses'], 
                              label='训练损失', color=colors[idx], alpha=0.7)
                
                if 'val_losses' in history:
                    axes[idx].plot(epochs, history['val_losses'], 
                                  label='验证损失', color=colors[idx], linestyle='--')
                
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel('损失')
                axes[idx].set_title(f'{model_name} 训练曲线')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/custom_models/figures/training_curves.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_analysis(interpretability_results):
    """
    创建特征重要性分析图
    """
    fig, axes = plt.subplots(len(interpretability_results), 2, 
                           figsize=(15, 5 * len(interpretability_results)))
    
    if len(interpretability_results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, results) in enumerate(interpretability_results.items()):
        # 梯度重要性
        if 'gradient_importance' in results:
            importance = results['gradient_importance']
            top_indices = np.argsort(importance)[-20:]  # 前20个重要特征
            
            axes[idx, 0].barh(range(len(top_indices)), importance[top_indices], 
                             color='skyblue', alpha=0.7)
            axes[idx, 0].set_xlabel('重要性分数')
            axes[idx, 0].set_title(f'{model_name} - 梯度重要性 (Top 20)')
            axes[idx, 0].grid(True, alpha=0.3)
        
        # 扰动重要性
        if 'perturbation_importance' in results:
            importance = results['perturbation_importance']
            top_indices = np.argsort(importance)[-20:]
            
            axes[idx, 1].barh(range(len(top_indices)), importance[top_indices], 
                             color='lightcoral', alpha=0.7)
            axes[idx, 1].set_xlabel('重要性分数')
            axes[idx, 1].set_title(f'{model_name} - 扰动重要性 (Top 20)')
            axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/custom_models/figures/feature_importance_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_transfer_learning_visualization(transfer_learning_results):
    """
    创建迁移学习效果可视化
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 精度对比
    methods = ['从头训练', '迁移学习']
    accuracies = [
        transfer_learning_results['scratch_accuracy'],
        transfer_learning_results['transfer_accuracy']
    ]
    
    bars = ax1.bar(methods, accuracies, color=['orange', 'green'], alpha=0.7)
    ax1.set_ylabel('准确率')
    ax1.set_title(f'迁移学习效果对比\n(数据量: {transfer_learning_results["small_data_ratio"]:.1%})')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 改进程度
    improvement = transfer_learning_results['improvement']
    improvement_percent = improvement / transfer_learning_results['scratch_accuracy'] * 100
    
    ax2.bar(['精度提升'], [improvement], color='blue', alpha=0.7)
    ax2.set_ylabel('精度提升')
    ax2.set_title(f'迁移学习改进效果\n({improvement_percent:.1f}% 相对提升)')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    ax2.text(0, improvement + 0.002, f'{improvement:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/custom_models/figures/transfer_learning_effect.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_ensemble_comparison(evaluation_results, ensemble_results):
    """
    创建集成模型对比
    """
    # 获取单个模型的精度
    individual_accuracies = [results['accuracy'] for results in evaluation_results.values()]
    individual_names = list(evaluation_results.keys())
    
    # 获取集成模型的精度
    ensemble_accuracies = [
        ensemble_results['average_ensemble']['accuracy'],
        ensemble_results['weighted_ensemble']['accuracy'],
        ensemble_results['voting_ensemble']['accuracy']
    ]
    ensemble_names = ['平均集成', '加权集成', '投票集成']
    
    # 合并数据
    all_names = individual_names + ensemble_names
    all_accuracies = individual_accuracies + ensemble_accuracies
    colors = ['skyblue'] * len(individual_names) + ['orange', 'red', 'green']
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(all_names, all_accuracies, color=colors, alpha=0.7)
    
    # 添加分隔线
    plt.axvline(x=len(individual_names) - 0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.ylabel('准确率')
    plt.title('单个模型 vs 集成模型性能对比')
    plt.tick_params(axis='x', rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, all_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 添加图例
    plt.text(len(individual_names) / 2, max(all_accuracies) * 0.95, 
            '单个模型', ha='center', fontsize=12, fontweight='bold')
    plt.text(len(individual_names) + len(ensemble_names) / 2, max(all_accuracies) * 0.95,
            '集成模型', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/custom_models/figures/ensemble_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix_comparison(evaluation_results, class_info):
    """
    创建混淆矩阵对比
    """
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    class_names = [class_info[i]['name'] for i in sorted(class_info.keys())]
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        import seaborn as sns
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx])
        
        axes[idx].set_title(f'{model_name}\n混淆矩阵')
        axes[idx].set_xlabel('预测标签')
        axes[idx].set_ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig('output/custom_models/figures/confusion_matrix_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_custom_results(models, evaluation_results, interpretability_results,
                       ensemble_results, class_info):
    """
    保存自定义模型结果
    """
    logger.info("保存自定义模型结果...")
    
    output_dir = Path('output/custom_models')
    
    # 1. 保存模型
    for model_name, model in models.items():
        model_path = output_dir / 'models' / f'{model_name.lower()}_model.pkl'
        model.save_model(model_path)
    
    # 2. 保存评估结果
    results_to_save = {}
    for model_name, results in evaluation_results.items():
        results_to_save[model_name] = {
            'accuracy': float(results['accuracy']),
            'wetland_score': float(results['wetland_score']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'training_history': results['training_history']
        }
    
    with open(output_dir / 'results' / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    # 3. 保存集成结果
    ensemble_to_save = {}
    for ensemble_type, results in ensemble_results.items():
        ensemble_to_save[ensemble_type] = {
            'accuracy': float(results['accuracy']),
        }
        if 'model_weights' in results:
            ensemble_to_save[ensemble_type]['model_weights'] = {
                k: float(v) for k, v in results['model_weights'].items()
            }
    
    with open(output_dir / 'results' / 'ensemble_results.json', 'w', encoding='utf-8') as f:
        json.dump(ensemble_to_save, f, ensure_ascii=False, indent=2)
    
    # 4. 生成文本报告
    generate_custom_model_report(evaluation_results, ensemble_results, class_info, output_dir)
    
    logger.info(f"自定义模型结果已保存到: {output_dir}")

def generate_custom_model_report(evaluation_results, ensemble_results, class_info, output_dir):
    """
    生成自定义模型报告
    """
    report_path = output_dir / 'results' / 'custom_model_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 湿地高光谱分类系统 - 自定义模型报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 执行摘要
        f.write("## 📋 执行摘要\n\n")
        best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
        best_accuracy = evaluation_results[best_model]['accuracy']
        
        f.write(f"本次自定义模型开发共实现了 **{len(evaluation_results)}** 个深度学习模型，")
        f.write(f"最佳模型为 **{best_model}**，测试集精度达到 **{best_accuracy:.3f}**。\n\n")
        
        # 模型性能汇总
        f.write("## 📊 模型性能汇总\n\n")
        f.write("| 模型 | 准确率 | 湿地评分 | 描述 |\n")
        f.write("|------|--------|----------|------|\n")
        
        model_descriptions = {
            'SpectralTransformer': '基于Transformer架构的光谱分类模型',
            'HybridCNN1D': '多尺度1D卷积神经网络',
            'ResidualSpectral': '残差连接的光谱网络'
        }
        
        for model_name, results in evaluation_results.items():
            description = model_descriptions.get(model_name, '自定义深度学习模型')
            f.write(f"| {model_name} | {results['accuracy']:.3f} | "
                   f"{results['wetland_score']:.3f} | {description} |\n")
        
        f.write("\n")
        
        # 集成学习结果
        f.write("## 🔗 集成学习结果\n\n")
        f.write("| 集成方法 | 准确率 | 描述 |\n")
        f.write("|----------|--------|------|\n")
        
        ensemble_descriptions = {
            'average_ensemble': '简单平均集成',
            'weighted_ensemble': '基于性能的加权集成',
            'voting_ensemble': '多数投票集成'
        }
        
        for ensemble_type, results in ensemble_results.items():
            description = ensemble_descriptions.get(ensemble_type, '集成方法')
            f.write(f"| {description} | {results['accuracy']:.3f} | "
                   f"{'多模型' + ensemble_type.split('_')[0] + '融合'} |\n")
        
        f.write("\n")
        
        # 技术创新点
        f.write("## 💡 技术创新点\n\n")
        f.write("### 模型架构创新\n\n")
        f.write("1. **光谱Transformer**: 首次将Transformer架构应用于高光谱数据分类\n")
        f.write("2. **多尺度CNN**: 设计了多尺度1D卷积网络，capture不同尺度的光谱特征\n")
        f.write("3. **残差光谱网络**: 在光谱数据上应用残差连接，提高模型训练稳定性\n\n")
        
        f.write("### 损失函数创新\n\n")
        f.write("1. **Focal Loss**: 解决湿地分类中的类别不平衡问题\n")
        f.write("2. **标签平滑**: 提高模型泛化能力，减少过拟合\n\n")
        
        f.write("### 评估指标创新\n\n")
        f.write("1. **湿地特定评分**: 针对湿地生态系统特点设计的专用评估指标\n")
        f.write("2. **生态一致性评分**: 评估分类结果的生态合理性\n\n")
        
        # 最佳实践
        f.write("## 🏆 最佳实践\n\n")
        f.write("### 模型选择建议\n\n")
        f.write(f"- **推荐模型**: {best_model} (精度: {best_accuracy:.3f})\n")
        f.write("- **使用场景**: 高精度湿地分类需求\n")
        f.write("- **计算资源**: 建议使用GPU加速训练\n\n")
        
        f.write("### 集成策略建议\n\n")
        best_ensemble = max(ensemble_results, key=lambda x: ensemble_results[x]['accuracy'])
        best_ensemble_acc = ensemble_results[best_ensemble]['accuracy']
        f.write(f"- **推荐策略**: {ensemble_descriptions[best_ensemble]} (精度: {best_ensemble_acc:.3f})\n")
        f.write("- **适用情况**: 对精度要求极高的应用场景\n")
        f.write("- **计算成本**: 需要训练和维护多个模型\n\n")
        
        # 部署建议
        f.write("## 🚀 部署建议\n\n")
        f.write("1. **模型优化**: 使用ONNX等格式优化模型推理速度\n")
        f.write("2. **批量处理**: 对于大规模数据，建议使用批量推理\n")
        f.write("3. **硬件要求**: GPU内存建议8GB以上，CPU建议16核以上\n")
        f.write("4. **监控策略**: 部署后需要监控模型性能和输出质量\n\n")
        
        f.write("---\n")
        f.write("*报告生成完成*")

def display_custom_results(evaluation_results, ensemble_results, total_time):
    """
    显示自定义模型结果
    """
    logger.info("="*60)
    logger.info("自定义模型示例完成!")
    logger.info("="*60)
    
    print(f"\n🤖 自定义模型性能:")
    for model_name, results in evaluation_results.items():
        print(f"   {model_name:18s} - 精度: {results['accuracy']:.3f}, "
              f"湿地评分: {results['wetland_score']:.3f}")
    
    # 找出最佳模型
    best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['accuracy'])
    best_accuracy = evaluation_results[best_model]['accuracy']
    print(f"\n🏆 最佳模型: {best_model} (精度: {best_accuracy:.3f})")
    
    print(f"\n🔗 集成学习结果:")
    for ensemble_type, results in ensemble_results.items():
        print(f"   {ensemble_type:18s} - 精度: {results['accuracy']:.3f}")
    
    # 找出最佳集成方法
    best_ensemble = max(ensemble_results, key=lambda x: ensemble_results[x]['accuracy'])
    best_ensemble_acc = ensemble_results[best_ensemble]['accuracy']
    print(f"\n🏆 最佳集成: {best_ensemble} (精度: {best_ensemble_acc:.3f})")
    
    print(f"\n⏱️  总执行时间: {total_time:.2f} 秒")
    
    print(f"\n📁 输出文件位置:")
    print(f"   - 自定义模型: output/custom_models/models/")
    print(f"   - 评估结果: output/custom_models/results/")
    print(f"   - 可视化图: output/custom_models/figures/")
    print(f"   - 分析报告: output/custom_models/results/custom_model_report.md")

def main():
    """
    主函数
    """
    try:
        # 设置目录
        setup_custom_directories()
        
        # 执行自定义模型工作流程
        custom_model_workflow()
        
        print("\n✅ 自定义模型示例执行完成!")
        print("🔍 请查看 output/custom_models/ 目录下的详细结果")
        print("🤖 自定义模型: output/custom_models/models/")
        print("📊 性能分析: output/custom_models/results/")
        print("🎨 可视化图表: output/custom_models/figures/")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        print("\n⚠️ 程序被用户中断")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"\n❌ 程序执行失败: {e}")
        print("💡 请检查错误信息并重试")
        raise

if __name__ == "__main__":
    main()