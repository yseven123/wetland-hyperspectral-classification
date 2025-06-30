# Processed Data Directory
## 处理后数据目录

此目录用于存放经过预处理、特征提取和数据增强后的数据文件。

### 📁 目录结构

```
processed/
├── preprocessed/          # 预处理后数据
│   ├── radiometric/       # 辐射定标后数据
│   ├── atmospheric/       # 大气校正后数据
│   ├── geometric/         # 几何校正后数据
│   └── denoised/          # 去噪后数据
├── features/              # 特征数据
│   ├── spectral/          # 光谱特征
│   ├── indices/           # 植被指数
│   ├── texture/           # 纹理特征
│   └── spatial/           # 空间特征
├── patches/               # 图像块数据
│   ├── training/          # 训练图像块
│   ├── validation/        # 验证图像块
│   └── testing/           # 测试图像块
└── augmented/             # 数据增强后数据
    ├── rotated/           # 旋转增强
    ├── flipped/           # 翻转增强
    ├── noisy/             # 噪声增强
    └── brightness/        # 亮度增强
```

### 📋 文件命名规范

#### 预处理数据
- 格式: `YYYY-MM-DD_SITE_PROCESS_VERSION.ext`
- 示例: `2024-03-15_Dongting_ATM_v1.tif` (大气校正)
- 示例: `2024-03-15_Dongting_GEO_v1.tif` (几何校正)

#### 特征数据
- 格式: `YYYY-MM-DD_SITE_FEATURE_TYPE.ext`
- 示例: `2024-03-15_Dongting_NDVI.tif`
- 示例: `2024-03-15_Dongting_GLCM.npy`

#### 图像块数据
- 格式: `YYYY-MM-DD_SITE_PATCH_SIZE_ID.npy`
- 示例: `2024-03-15_Dongting_PATCH_64x64_001.npy`

### 🔧 处理流程说明

#### 1. 预处理流程
```
原始数据 → 辐射定标 → 大气校正 → 几何校正 → 噪声去除
```

#### 2. 特征提取流程
```
预处理数据 → 光谱特征 → 植被指数 → 纹理特征 → 空间特征
```

#### 3. 数据分割流程
```
特征数据 → 图像块分割 → 训练/验证/测试集划分
```

#### 4. 数据增强流程
```
训练数据 → 几何变换 → 光谱变换 → 噪声添加
```

### 📊 数据格式说明

#### 支持的文件格式
- **图像数据**: `.tif`, `.img`, `.hdf5`
- **数组数据**: `.npy`, `.npz`
- **特征数据**: `.pkl`, `.h5`
- **标签数据**: `.npy`, `.csv`

#### 数据类型
- **光谱数据**: float32, 归一化到[0,1]
- **特征数据**: float32, 标准化处理
- **标签数据**: int32, 类别编码

### 🚀 使用说明

```python
import numpy as np
import rasterio
from wetland_classification.data import ProcessedDataLoader

# 加载预处理数据
loader = ProcessedDataLoader()
preprocessed_data = loader.load_preprocessed('processed/preprocessed/scene_atm.tif')

# 加载特征数据
features = loader.load_features('processed/features/')

# 加载图像块数据
patches, labels = loader.load_patches('processed/patches/training/')

# 加载增强数据
augmented_data = loader.load_augmented('processed/augmented/')
```

### 📈 质量控制

#### 数据验证检查点
- [ ] 数据维度正确性
- [ ] 数值范围合理性
- [ ] 空值和异常值
- [ ] 文件完整性
- [ ] 格式一致性

#### 质量指标
- **数据完整率**: > 99%
- **处理成功率**: > 95%
- **特征有效性**: 相关性 > 0.3
- **增强多样性**: 变换覆盖率 > 80%

### ⚠️ 注意事项

1. **存储空间**: 处理后数据占用空间通常是原始数据的2-5倍
2. **版本管理**: 使用版本号标识不同处理参数的结果
3. **临时文件**: 定期清理处理过程中的临时文件
4. **并行处理**: 大数据量时建议使用并行处理
5. **内存管理**: 注意内存使用，避免内存溢出

### 🔄 数据更新流程

1. **增量更新**: 仅处理新增的原始数据
2. **全量更新**: 重新处理所有原始数据
3. **参数更新**: 更新处理参数后重新处理
4. **版本标记**: 为每次更新创建版本标记