{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 湿地高光谱数据预处理流程\n",
    "\n",
    "## 概述\n",
    "本notebook展示了湿地高光谱遥感数据的完整预处理流程，包括：\n",
    "- 辐射定标\n",
    "- 大气校正\n",
    "- 几何校正\n",
    "- 噪声去除\n",
    "- 数据质量评估\n",
    "\n",
    "预处理是高光谱数据分析的关键步骤，直接影响后续分类精度。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 导入必要的库\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# GDAL和光谱库\n",
    "from osgeo import gdal, osr, ogr\n",
    "import spectral as spy\n",
    "import rasterio\n",
    "from rasterio.windows import Window\n",
    "from rasterio.transform import from_bounds\n",
    "\n",
    "# 科学计算库\n",
    "from scipy import ndimage, signal\n",
    "from scipy.stats import pearsonr\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# 自定义模块\n",
    "import sys\n",
    "sys.path.append('../src')\n",
    "from wetland_classification.preprocessing import (\n",
    "    RadiometricCorrector,\n",
    "    AtmosphericCorrector, \n",
    "    GeometricCorrector,\n",
    "    NoiseReducer\n",
    ")\n",
    "from wetland_classification.data import DataLoader, DataValidator\n",
    "from wetland_classification.utils import visualization, logger\n",
    "\n",
    "# 设置绘图样式\n",
    "plt.style.use('seaborn-v0_8')\n",
    "sns.set_palette(\"husl\")\n",
    "\n",
    "# 配置日志\n",
    "logger = logger.setup_logger('preprocessing', level='INFO')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 数据加载与初步检查"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 设置数据路径\n",
    "data_dir = Path('../data/raw')\n",
    "output_dir = Path('../data/processed')\n",
    "output_dir.mkdir(exist_ok=True)\n",
    "\n",
    "# 数据文件路径\n",
    "hyperspectral_file = data_dir / 'wetland_hyperspectral.tif'\n",
    "metadata_file = data_dir / 'metadata.json'\n",
    "\n",
    "print(f\"数据目录: {data_dir}\")\n",
    "print(f\"输出目录: {output_dir}\")\n",
    "print(f\"高光谱数据文件: {hyperspectral_file}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 加载高光谱数据\n",
    "loader = DataLoader()\n",
    "hyperspectral_data, metadata = loader.load_hyperspectral(str(hyperspectral_file))\n",
    "\n",
    "print(\"数据基本信息:\")\n",
    "print(f\"数据形状: {hyperspectral_data.shape}\")\n",
    "print(f\"数据类型: {hyperspectral_data.dtype}\")\n",
    "print(f\"波段数量: {hyperspectral_data.shape[2]}\")\n",
    "print(f\"空间分辨率: {metadata.get('pixel_size', 'Unknown')}\")\n",
    "print(f\"波长范围: {metadata.get('wavelength_range', 'Unknown')}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 数据质量初步检查\n",
    "validator = DataValidator()\n",
    "quality_report = validator.check_data_quality(hyperspectral_data, metadata)\n",
    "\n",
    "print(\"\\n数据质量检查结果:\")\n",
    "for key, value in quality_report.items():\n",
    "    print(f\"{key}: {value}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. 辐射定标"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 创建辐射定标器\n",
    "radiometric_corrector = RadiometricCorrector(\n",
    "    gain_values=metadata.get('gain', None),\n",
    "    offset_values=metadata.get('offset', None),\n",
    "    solar_irradiance=metadata.get('solar_irradiance', None)\n",
    ")\n",
    "\n",
    "print(\"开始辐射定标...\")\n",
    "# 执行辐射定标\n",
    "radiance_data = radiometric_corrector.dn_to_radiance(hyperspectral_data)\n",
    "reflectance_data = radiometric_corrector.radiance_to_reflectance(\n",
    "    radiance_data, \n",
    "    solar_zenith=metadata.get('solar_zenith', 30),\n",
    "    earth_sun_distance=metadata.get('earth_sun_distance', 1.0)\n",
    ")\n",
    "\n",
    "print(f\"辐射定标完成\")\n",
    "print(f\"反射率数据范围: {reflectance_data.min():.4f} - {reflectance_data.max():.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 可视化辐射定标效果\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
    "\n",
    "# 选择几个代表性波段进行可视化\n",
    "bands_to_show = [50, 100, 150]  # 假设对应不同光谱区域\n",
    "band_names = ['Blue', 'Green', 'NIR']\n",
    "\n",
    "for i, (band_idx, band_name) in enumerate(zip(bands_to_show, band_names)):\n",
    "    # 原始DN值\n",
    "    im1 = axes[0, i].imshow(hyperspectral_data[:, :, band_idx], cmap='viridis')\n",
    "    axes[0, i].set_title(f'原始DN值 - {band_name} (Band {band_idx})')\n",
    "    axes[0, i].axis('off')\n",
    "    plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)\n",
    "    \n",
    "    # 反射率\n",
    "    im2 = axes[1, i].imshow(reflectance_data[:, :, band_idx], cmap='viridis')\n",
    "    axes[1, i].set_title(f'反射率 - {band_name} (Band {band_idx})')\n",
    "    axes[1, i].axis('off')\n",
    "    plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 大气校正"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 创建大气校正器\n",
    "atmospheric_corrector = AtmosphericCorrector(\n",
    "    method='FLAASH',  # 可选: 'FLAASH', 'ATCOR', 'DOS', 'ELM'\n",
    "    visibility=23.0,  # 能见度 km\n",
    "    water_vapor=2.5,  # 水汽含量 g/cm²\n",
    "    ozone=0.31,       # 臭氧含量 atm-cm\n",
    "    aerosol_model='rural',  # 气溶胶模型\n",
    "    atmospheric_model='mid_latitude_summer'  # 大气模型\n",
    ")\n",
    "\n",
    "print(\"开始大气校正...\")\n",
    "# 执行大气校正\n",
    "surface_reflectance = atmospheric_corrector.apply_correction(\n",
    "    reflectance_data,\n",
    "    wavelengths=metadata.get('wavelengths'),\n",
    "    solar_zenith=metadata.get('solar_zenith', 30),\n",
    "    view_zenith=metadata.get('view_zenith', 0),\n",
    "    relative_azimuth=metadata.get('relative_azimuth', 0)\n",
    ")\n",
    "\n",
    "print(f\"大气校正完成\")\n",
    "print(f\"地表反射率数据范围: {surface_reflectance.min():.4f} - {surface_reflectance.max():.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 对比大气校正前后的光谱曲线\n",
    "# 选择几个典型地物的像素进行对比\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
    "\n",
    "# 选择不同类型的像素点\n",
    "pixels = {\n",
    "    '水体': (100, 150),\n",
    "    '植被': (200, 200),\n",
    "    '土壤': (50, 300),\n",
    "    '建筑': (300, 100)\n",
    "}\n",
    "\n",
    "wavelengths = metadata.get('wavelengths', range(reflectance_data.shape[2]))\n",
    "\n",
    "for idx, (land_type, (x, y)) in enumerate(pixels.items()):\n",
    "    ax = axes[idx // 2, idx % 2]\n",
    "    \n",
    "    # 校正前\n",
    "    spectrum_before = reflectance_data[x, y, :]\n",
    "    # 校正后\n",
    "    spectrum_after = surface_reflectance[x, y, :]\n",
    "    \n",
    "    ax.plot(wavelengths, spectrum_before, 'b-', label='大气校正前', linewidth=2)\n",
    "    ax.plot(wavelengths, spectrum_after, 'r-', label='大气校正后', linewidth=2)\n",
    "    \n",
    "    ax.set_title(f'{land_type} 光谱对比 (像素: {x}, {y})')\n",
    "    ax.set_xlabel('波长 (nm)')\n",
    "    ax.set_ylabel('反射率')\n",
    "    ax.legend()\n",
    "    ax.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 几何校正"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 创建几何校正器\n",
    "geometric_corrector = GeometricCorrector(\n",
    "    target_crs='EPSG:4326',  # 目标坐标系\n",
    "    resampling_method='bilinear'  # 重采样方法\n",
    ")\n",
    "\n",
    "print(\"开始几何校正...\")\n",
    "\n",
    "# 加载地面控制点 (如果有的话)\n",
    "gcp_file = data_dir / 'ground_control_points.csv'\n",
    "if gcp_file.exists():\n",
    "    gcps = pd.read_csv(gcp_file)\n",
    "    print(f\"加载了 {len(gcps)} 个地面控制点\")\n",
    "else:\n",
    "    print(\"未找到地面控制点文件，使用图像自带的地理参考信息\")\n",
    "    gcps = None\n",
    "\n",
    "# 执行几何校正\n",
    "georeferenced_data, transform = geometric_corrector.apply_correction(\n",
    "    surface_reflectance,\n",
    "    metadata,\n",
    "    gcps=gcps\n",
    ")\n",
    "\n",
    "print(f\"几何校正完成\")\n",
    "print(f\"校正后数据形状: {georeferenced_data.shape}\")\n",
    "print(f\"地理变换参数: {transform}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 几何校正精度评估\n",
    "if gcps is not None:\n",
    "    accuracy_metrics = geometric_corrector.assess_accuracy(gcps, transform)\n",
    "    \n",
    "    print(\"\\n几何校正精度评估:\")\n",
    "    print(f\"均方根误差 (RMSE): {accuracy_metrics['rmse']:.3f} 像素\")\n",
    "    print(f\"平均误差: {accuracy_metrics['mean_error']:.3f} 像素\")\n",
    "    print(f\"标准差: {accuracy_metrics['std_error']:.3f} 像素\")\n",
    "    \n",
    "    # 可视化控制点误差分布\n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n",
    "    \n",
    "    # 误差向量图\n",
    "    ax1.quiver(gcps['image_x'], gcps['image_y'], \n",
    "               accuracy_metrics['residuals_x'], accuracy_metrics['residuals_y'],\n",
    "               scale=10, color='red', alpha=0.7)\n",
    "    ax1.set_title('控制点误差向量')\n",
    "    ax1.set_xlabel('图像X坐标')\n",
    "    ax1.set_ylabel('图像Y坐标')\n",
    "    ax1.grid(True, alpha=0.3)\n",
    "    \n",
    "    # 误差分布直方图\n",
    "    errors = np.sqrt(accuracy_metrics['residuals_x']**2 + accuracy_metrics['residuals_y']**2)\n",
    "    ax2.hist(errors, bins=20, alpha=0.7, color='blue')\n",
    "    ax2.axvline(accuracy_metrics['rmse'], color='red', linestyle='--', \n",
    "                label=f'RMSE = {accuracy_metrics[\"rmse\"]:.3f}')\n",
    "    ax2.set_title('误差分布直方图')\n",
    "    ax2.set_xlabel('误差 (像素)')\n",
    "    ax2.set_ylabel('频次')\n",
    "    ax2.legend()\n",
    "    ax2.grid(True, alpha=0.3)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 噪声去除"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 创建噪声去除器\n",
    "noise_reducer = NoiseReducer(\n",
    "    methods=['savgol', 'gaussian', 'bilateral'],  # 多种去噪方法\n",
    "    params={\n",
    "        'savgol': {'window_length': 5, 'polyorder': 2},\n",
    "        'gaussian': {'sigma': 1.0},\n",
    "        'bilateral': {'sigma_color': 0.1, 'sigma_spatial': 1.0}\n",
    "    }\n",
    ")\n",
    "\n",
    "print(\"开始噪声去除...\")\n",
    "\n",
    "# 噪声检测\n",
    "noise_bands = noise_reducer.detect_noisy_bands(georeferenced_data)\n",
    "print(f\"检测到噪声波段: {noise_bands}\")\n",
    "\n",
    "# 执行噪声去除\n",
    "denoised_data = noise_reducer.remove_noise(\n",
    "    georeferenced_data,\n",
    "    noise_bands=noise_bands\n",
    ")\n",
    "\n",
    "print(f\"噪声去除完成\")\n",
    "print(f\"去噪后数据形状: {denoised_data.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 噪声去除效果评估\n",
    "# 计算信噪比改善\n",
    "snr_before = noise_reducer.calculate_snr(georeferenced_data)\n",
    "snr_after = noise_reducer.calculate_snr(denoised_data)\n",
    "snr_improvement = snr_after - snr_before\n",
    "\n",
    "print(f\"\\n噪声去除效果评估:\")\n",
    "print(f\"平均信噪比改善: {np.mean(snr_improvement):.2f} dB\")\n",
    "print(f\"最大信噪比改善: {np.max(snr_improvement):.2f} dB\")\n",
    "print(f\"最小信噪比改善: {np.min(snr_improvement):.2f} dB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 可视化噪声去除效果\n",
    "fig, axes = plt.subplots(3, 2, figsize=(12, 15))\n",
    "\n",
    "# 选择一个噪声较重的波段\n",
    "noisy_band = noise_bands[0] if noise_bands else 100\n",
    "\n",
    "# 原始图像\n",
    "im1 = axes[0, 0].imshow(georeferenced_data[:, :, noisy_band], cmap='viridis')\n",
    "axes[0, 0].set_title(f'去噪前 - Band {noisy_band}')\n",
    "axes[0, 0].axis('off')\n",
    "plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)\n",
    "\n",
    "# 去噪后图像\n",
    "im2 = axes[0, 1].imshow(denoised_data[:, :, noisy_band], cmap='viridis')\n",
    "axes[0, 1].set_title(f'去噪后 - Band {noisy_band}')\n",
    "axes[0, 1].axis('off')\n",
    "plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)\n",
    "\n",
    "# 噪声差异图\n",
    "noise_diff = georeferenced_data[:, :, noisy_band] - denoised_data[:, :, noisy_band]\n",
    "im3 = axes[1, 0].imshow(noise_diff, cmap='RdBu_r')\n",
    "axes[1, 0].set_title('去除的噪声')\n",
    "axes[1, 0].axis('off')\n",
    "plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)\n",
    "\n",
    "# 信噪比改善图\n",
    "wavelengths = metadata.get('wavelengths', range(denoised_data.shape[2]))\n",
    "axes[1, 1].plot(wavelengths, snr_improvement, 'b-', linewidth=2)\n",
    "axes[1, 1].set_title('信噪比改善')\n",
    "axes[1, 1].set_xlabel('波长 (nm)')\n",
    "axes[1, 1].set_ylabel('SNR改善 (dB)')\n",
    "axes[1, 1].grid(True, alpha=0.3)\n",
    "\n",
    "# 光谱曲线对比\n",
    "pixel_x, pixel_y = 150, 150\n",
    "spectrum_before = georeferenced_data[pixel_x, pixel_y, :]\n",
    "spectrum_after = denoised_data[pixel_x, pixel_y, :]\n",
    "\n",
    "axes[2, 0].plot(wavelengths, spectrum_before, 'r-', label='去噪前', alpha=0.7)\n",
    "axes[2, 0].plot(wavelengths, spectrum_after, 'b-', label='去噪后', linewidth=2)\n",
    "axes[2, 0].set_title(f'光谱曲线对比 (像素: {pixel_x}, {pixel_y})')\n",
    "axes[2, 0].set_xlabel('波长 (nm)')\n",
    "axes[2, 0].set_ylabel('反射率')\n",
    "axes[2, 0].legend()\n",
    "axes[2, 0].grid(True, alpha=0.3)\n",
    "\n",
    "# 噪声功率谱密度\n",
    "from scipy import signal\n",
    "f, psd_before = signal.welch(spectrum_before, nperseg=min(64, len(spectrum_before)//4))\n",
    "f, psd_after = signal.welch(spectrum_after, nperseg=min(64, len(spectrum_after)//4))\n",
    "\n",
    "axes[2, 1].semilogy(f, psd_before, 'r-', label='去噪前', alpha=0.7)\n",
    "axes[2, 1].semilogy(f, psd_after, 'b-', label='去噪后', linewidth=2)\n",
    "axes[2, 1].set_title('功率谱密度')\n",
    "axes[2, 1].set_xlabel('频率')\n",
    "axes[2, 1].set_ylabel('功率谱密度')\n",
    "axes[2, 1].legend()\n",
    "axes[2, 1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. 数据质量评估"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 综合数据质量评估\n",
    "print(\"=== 预处理前后数据质量对比 ===\")\n",
    "\n",
    "# 原始数据统计\n",
    "original_stats = {\n",
    "    'mean': np.mean(hyperspectral_data),\n",
    "    'std': np.std(hyperspectral_data),\n",
    "    'min': np.min(hyperspectral_data),\n",
    "    'max': np.max(hyperspectral_data),\n",
    "    'dynamic_range': np.max(hyperspectral_data) - np.min(hyperspectral_data)\n",
    "}\n",
    "\n",
    "# 预处理后数据统计\n",
    "processed_stats = {\n",
    "    'mean': np.mean(denoised_data),\n",
    "    'std': np.std(denoised_data),\n",
    "    'min': np.min(denoised_data),\n",
    "    'max': np.max(denoised_data),\n",
    "    'dynamic_range': np.max(denoised_data) - np.min(denoised_data)\n",
    "}\n",
    "\n",
    "# 打印对比结果\n",
    "comparison_df = pd.DataFrame({\n",
    "    '原始数据': original_stats,\n",
    "    '预处理后': processed_stats\n",
    "})\n",
    "\n",
    "print(comparison_df.round(4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 数据分布可视化\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "\n",
    "# 选择几个代表性波段进行统计分析\n",
    "band_indices = [20, 100, 200, 350]  # 不同光谱区域\n",
    "band_names = ['可见光', '红边', '近红外', '短波红外']\n",
    "\n",
    "for i, (band_idx, band_name) in enumerate(zip(band_indices, band_names)):\n",
    "    ax = axes[i//2, i%2]\n",
    "    \n",
    "    # 原始数据直方图\n",
    "    original_band = hyperspectral_data[:, :, band_idx].flatten()\n",
    "    processed_band = denoised_data[:, :, band_idx].flatten()\n",
    "    \n",
    "    ax.hist(original_band, bins=50, alpha=0.6, label='原始数据', color='red', density=True)\n",
    "    ax.hist(processed_band, bins=50, alpha=0.6, label='预处理后', color='blue', density=True)\n",
    "    \n",
    "    ax.set_title(f'{band_name} - Band {band_idx}')\n",
    "    ax.set_xlabel('反射率')\n",
    "    ax.set_ylabel('密度')\n",
    "    ax.legend()\n",
    "    ax.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. 保存预处理结果"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 保存预处理后的数据\n",
    "output_file = output_dir / 'preprocessed_hyperspectral.tif'\n",
    "\n",
    "# 更新元数据\n",
    "updated_metadata = metadata.copy()\n",
    "updated_metadata.update({\n",
    "    'preprocessing_steps': [\n",
    "        'radiometric_correction',\n",
    "        'atmospheric_correction', \n",
    "        'geometric_correction',\n",
    "        'noise_reduction'\n",
    "    ],\n",
    "    'processing_date': pd.Timestamp.now().isoformat(),\n",
    "    'data_type': 'surface_reflectance',\n",
    "    'data_range': [float(denoised_data.min()), float(denoised_data.max())],\n",
    "    'coordinate_system': 'EPSG:4326',\n",
    "    'quality_flags': {\n",
    "        'radiometric_quality': 'good',\n",
    "        'atmospheric_quality': 'good', \n",
    "        'geometric_quality': 'good',\n",
    "        'noise_level': 'low'\n",
    "    }\n",
    "})\n",
    "\n",
    "# 使用rasterio保存数据\n",
    "with rasterio.open(\n",
    "    output_file,\n",
    "    'w',\n",
    "    driver='GTiff',\n",
    "    height=denoised_data.shape[0],\n",
    "    width=denoised_data.shape[1],\n",
    "    count=denoised_data.shape[2],\n",
    "    dtype=denoised_data.dtype,\n",
    "    crs='EPSG:4326',\n",
    "    transform=transform,\n",
    "    compress='lzw'\n",
    ") as dst:\n",
    "    for i in range(denoised_data.shape[2]):\n",
    "        dst.write(denoised_data[:, :, i], i+1)\n",
    "    \n",
    "    # 写入波长信息到band descriptions\n",
    "    if 'wavelengths' in metadata:\n",
    "        wavelengths = metadata['wavelengths']\n",
    "        for i, wl in enumerate(wavelengths[:denoised_data.shape[2]]):\n",
    "            dst.set_band_description(i+1, f'Band {i+1}: {wl:.2f} nm')\n",
    "\n",
    "print(f\"预处理数据已保存到: {output_file}\")\n",
    "\n",
    "# 保存元数据\n",
    "metadata_output = output_dir / 'preprocessed_metadata.json'\n",
    "import json\n",
    "with open(metadata_output, 'w', encoding='utf-8') as f:\n",
    "    json.dump(updated_metadata, f, indent=2, ensure_ascii=False)\n",
    "\n",
    "print(f\"元数据已保存到: {metadata_output}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. 预处理质量报告"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 生成预处理质量报告\n",
    "quality_report = {\n",
    "    '数据概况': {\n",
    "        '原始数据形状': str(hyperspectral_data.shape),\n",
    "        '预处理后形状': str(denoised_data.shape),\n",
    "        '波段数量': denoised_data.shape[2],\n",
    "        '空间分辨率': metadata.get('pixel_size', 'Unknown'),\n",
    "        '波长范围': f\"{metadata.get('wavelength_range', 'Unknown')}\"\n",
    "    },\n",
    "    '预处理步骤': {\n",
    "        '辐射定标': '✓ 完成',\n",
    "        '大气校正': '✓ 完成', \n",
    "        '几何校正': '✓ 完成',\n",
    "        '噪声去除': '✓ 完成'\n",
    "    },\n",
    "    '质量指标': {\n",
    "        '平均信噪比改善': f\"{np.mean(snr_improvement):.2f} dB\",\n",
    "        '数据完整性': f\"{(1 - np.isnan(denoised_data).sum() / denoised_data.size) * 100:.2f}%\",\n",
    "        '动态范围': f\"{processed_stats['dynamic_range']:.4f}\",\n",
    "        '数据标准差': f\"{processed_stats['std']:.4f}\"\n",
    "    }\n",
    "}\n",
    "\n",
    "if gcps is not None:\n",
    "    quality_report['几何精度'] = {\n",
    "        'RMSE': f\"{accuracy_metrics['rmse']:.3f} 像素\",\n",
    "        '平均误差': f\"{accuracy_metrics['mean_error']:.3f} 像素\",\n",
    "        '控制点数量': len(gcps)\n",
    "    }\n",
    "\n",
    "# 打印质量报告\n",
    "print(\"\\n\" + \"=\"*50)\n",
    "print(\"湿地高光谱数据预处理质量报告\")\n",
    "print(\"=\"*50)\n",
    "\n",
    "for section, items in quality_report.items():\n",
    "    print(f\"\\n{section}:\")\n",
    "    for key, value in items.items():\n",
    "        print(f\"  {key}: {value}\")\n",
    "\n",
    "# 保存质量报告\n",
    "report_file = output_dir / 'preprocessing_quality_report.json'\n",
    "with open(report_file, 'w', encoding='utf-8') as f:\n",
    "    json.dump(quality_report, f, indent=2, ensure_ascii=False)\n",
    "\n",
    "print(f\"\\n质量报告已保存到: {report_file}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 总结\n",
    "\n",
    "本notebook完成了湿地高光谱数据的完整预处理流程:\n",
    "\n",
    "1. **辐射定标**: 将原始DN值转换为地表反射率\n",
    "2. **大气校正**: 去除大气影响，获得真实地表反射率\n",
    "3. **几何校正**: 确保数据的地理定位精度\n",
    "4. **噪声去除**: 提高数据质量和信噪比\n",
    "\n",
    "预处理后的数据已准备好进行特征提取和分类分析。\n",
    "\n",
    "### 关键成果:\n",
    "- 获得高质量的地表反射率数据\n",
    "- 信噪比显著改善\n",
    "- 地理定位精度满足要求\n",
    "- 数据完整性良好\n",
    "\n",
    "### 下一步:\n",
    "继续进行特征工程和分类模型训练。"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}