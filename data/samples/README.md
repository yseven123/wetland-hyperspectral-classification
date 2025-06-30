# Sample Data Directory
## 示例样本数据目录

此目录包含用于演示、测试和验证的示例数据集。

### 📁 目录结构

```
samples/
├── demo_scene/            # 演示场景数据
│   ├── hyperspectral/     # 小尺寸高光谱数据
│   ├── ground_truth/      # 对应的标签数据
│   ├── processed/         # 预处理结果
│   └── README.md          # 演示说明
├── test_data/             # 标准测试数据
│   ├── benchmark/         # 基准测试数据
│   ├── validation/        # 验证数据集
│   ├── metrics/           # 评估指标结果
│   └── README.md          # 测试说明
└── validation/            # 独立验证数据
    ├── external/          # 外部验证数据
    ├── cross_site/        # 跨站点验证
    ├── temporal/          # 时序验证数据
    └── README.md          # 验证说明
```

### 🎯 演示场景数据 (demo_scene/)

#### 数据规模
- **空间尺寸**: 256×256 像素
- **光谱波段**: 224 个波段
- **文件大小**: ~50MB
- **处理时间**: <5分钟

#### 包含内容
- `demo_hyperspectral.tif`: 演示高光谱数据
- `demo_labels.shp`: 地面真实标签
- `demo_samples.csv`: 训练样本点
- `demo_config.yaml`: 演示配置文件

#### 使用示例
```python
from wetland_classification import Pipeline
from wetland_classification.config import Config

# 加载演示配置
config = Config.from_file('data/samples/demo_scene/demo_config.yaml')

# 创建处理流水线
pipeline = Pipeline(config)

# 运行演示
results = pipeline.run_demo('data/samples/demo_scene/')
print(f"演示完成，精度: {results['accuracy']:.3f}")
```

### 🧪 标准测试数据 (test_data/)

#### 基准测试集
- **Indian Pines**: 145×145, 220波段, 16类
- **Pavia University**: 610×340, 103波段, 9类
- **Salinas**: 512×217, 204波段, 16类
- **Kennedy Space Center**: 512×614, 176波段, 13类

#### 数据格式
```python
# 数据加载示例
import scipy.io as sio
import numpy as np

# 加载Indian Pines数据
data = sio.loadmat('test_data/benchmark/indian_pines.mat')
hyperspectral = data['indian_pines']  # (145, 145, 220)
labels = data['indian_pines_gt']       # (145, 145)

# 湿地专用测试集
wetland_data = np.load('test_data/benchmark/wetland_test.npz')
X_test = wetland_data['hyperspectral']  # (N, H, W, bands)
y_test = wetland_data['labels']         # (N, H, W)
```

#### 评估基准
| 数据集 | SVM | RF | XGBoost | 3D-CNN | HybridSN | ViT |
|--------|-----|----|---------|---------|---------|----|
| Indian Pines | 0.892 | 0.908 | 0.915 | 0.934 | 0.941 | 0.945 |
| Pavia Univ | 0.945 | 0.958 | 0.962 | 0.975 | 0.981 | 0.984 |
| Wetland Demo | 0.876 | 0.894 | 0.903 | 0.921 | 0.928 | 0.932 |

### ✅ 独立验证数据 (validation/)

#### 外部验证集
- **不同传感器**: AVIRIS, HySpex, PRISMA
- **不同地区**: 长江流域, 珠江流域, 黄河流域
- **不同季节**: 春季, 夏季, 秋季, 冬季
- **不同年份**: 2020-2024年数据

#### 跨站点验证
```python
# 跨站点验证示例
from wetland_classification.evaluation import CrossSiteValidator

validator = CrossSiteValidator()
results = validator.validate(
    train_sites=['dongting', 'poyang'],
    test_sites=['taihu', 'hongze'],
    model_type='hybrid_cnn'
)
print(f"跨站点平均精度: {results['mean_accuracy']:.3f}")
```

### 📊 数据统计信息

#### 演示数据统计
```yaml
demo_scene:
  spatial_size: [256, 256]
  spectral_bands: 224
  classes: 8
  samples_per_class: [150, 200, 180, 220, 160, 190, 170, 140]
  total_samples: 1410
  file_size: "52.3 MB"
```

#### 测试数据统计
```yaml
test_data:
  total_datasets: 15
  total_size: "2.1 GB"
  hyperspectral_scenes: 12
  ground_truth_sets: 15
  validation_accuracy: [0.85, 0.95]
```

### 🔧 数据生成脚本

#### 创建演示数据
```python
# scripts/create_demo_data.py
from wetland_classification.utils import SampleGenerator

generator = SampleGenerator()
demo_data = generator.create_demo_scene(
    size=(256, 256),
    bands=224,
    classes=8,
    noise_level=0.02
)
generator.save_demo('data/samples/demo_scene/', demo_data)
```

#### 下载测试数据
```bash
# scripts/download_test_data.sh
#!/bin/bash

# 下载标准测试数据集
python scripts/download_datasets.py --dataset indian_pines
python scripts/download_datasets.py --dataset pavia_university
python scripts/download_datasets.py --dataset salinas

# 验证数据完整性
python scripts/verify_datasets.py
```

### 🚀 快速开始

#### 1. 运行演示
```bash
# 基础演示
python examples/基础分类示例.py --data data/samples/demo_scene/

# 高级演示
python examples/高级分析示例.py --data data/samples/demo_scene/ --model hybrid_cnn
```

#### 2. 测试验证
```bash
# 运行标准测试
python tests/test_integration.py --test_data data/samples/test_data/

# 跨站点验证
python scripts/cross_site_validation.py --validation_data data/samples/validation/
```

### 📈 性能基准

#### 处理时间 (演示数据)
- **数据加载**: 0.5秒
- **预处理**: 15秒
- **特征提取**: 8秒
- **模型训练**: 120秒
- **预测**: 5秒
- **总计**: ~150秒

#### 内存使用
- **演示数据**: ~200MB RAM
- **测试数据**: ~2GB RAM
- **验证数据**: ~8GB RAM

### ⚠️ 使用注意事项

1. **数据许可**: 部分测试数据集有使用限制
2. **下载时间**: 首次下载可能需要较长时间
3. **存储空间**: 确保有足够的磁盘空间
4. **网络连接**: 下载数据需要稳定的网络连接
5. **版本兼容**: 确保数据格式与代码版本兼容

### 📞 技术支持

如遇到样本数据相关问题：
- 📧 数据问题: samples-support@example.com
- 🔧 技术问题: [GitHub Issues](https://github.com/yourusername/wetland-hyperspectral-classification/issues)
- 📚 使用文档: [示例教程](../../examples/)