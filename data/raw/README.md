# Raw Data Directory
## 原始数据目录

此目录用于存放未经处理的原始高光谱数据和相关文件。

### 📁 目录结构

```
raw/
├── hyperspectral/          # 高光谱数据文件
│   ├── *.tif              # GeoTIFF格式数据
│   ├── *.hdr              # ENVI头文件
│   ├── *.bil              # 波段交叉存储
│   └── *.bsq              # 波段顺序存储
├── ground_truth/          # 地面真实标签
│   ├── *.shp              # Shapefile格式
│   ├── *.geojson          # GeoJSON格式
│   └── *.csv              # CSV格式样本点
├── auxiliary/             # 辅助数据
│   ├── dem/               # 数字高程模型
│   ├── weather/           # 气象数据
│   ├── soil/              # 土壤数据
│   └── hydrology/         # 水文数据
└── metadata/              # 元数据文件
    ├── sensor_info/       # 传感器信息
    ├── flight_logs/       # 飞行日志
    └── acquisition_params/ # 采集参数
```

### 📋 文件命名规范

#### 高光谱数据
- 格式: `YYYY-MM-DD_SITE_SENSOR_FLIGHT.ext`
- 示例: `2024-03-15_Dongting_AVIRIS_F01.tif`

#### 地面真实数据
- 格式: `YYYY-MM-DD_SITE_GT_VERSION.ext`
- 示例: `2024-03-15_Dongting_GT_v1.shp`

#### 辅助数据
- 格式: `YYYY-MM-DD_SITE_DATATYPE.ext`
- 示例: `2024-03-15_Dongting_DEM.tif`

### 🚀 使用说明

1. **数据放置**: 将原始数据文件按类型放入相应子目录
2. **命名规范**: 严格按照命名规范重命名文件
3. **元数据**: 确保每个数据文件都有对应的元数据
4. **备份**: 原始数据应保持只读，并定期备份

### ⚠️ 注意事项

- 不要修改此目录中的任何文件
- 确保文件权限设置为只读
- 定期检查文件完整性
- 维护详细的数据清单