#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
湿地高光谱分类系统安装配置
Wetland Hyperspectral Classification System Setup Configuration
"""

from setuptools import setup, find_packages
import os
import re

# 读取版本信息
def get_version():
    version_file = os.path.join('src', 'wetland_classification', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# 读取长描述
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements.txt
def get_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# 开发依赖
dev_requirements = [
    'pytest>=6.2.0',
    'pytest-cov>=3.0.0',
    'pytest-mock>=3.6.0',
    'black>=22.0.0',
    'isort>=5.10.0',
    'flake8>=4.0.0',
    'mypy>=0.910',
    'pre-commit>=2.15.0',
    'sphinx>=4.3.0',
    'sphinx-rtd-theme>=1.0.0',
    'jupyter>=1.0.0',
    'jupyterlab>=3.2.0',
]

# 可选依赖
optional_requirements = {
    'gpu': [
        'cupy-cuda11x>=10.0.0',
        'rapids-cudf>=22.02',
        'rapids-cuml>=22.02',
    ],
    'visualization': [
        'plotly>=5.5.0',
        'bokeh>=2.4.0',
        'folium>=0.12.0',
        'ipywidgets>=7.6.0',
    ],
    'docs': [
        'sphinx>=4.3.0',
        'sphinx-rtd-theme>=1.0.0',
        'sphinxcontrib-bibtex>=2.4.0',
        'myst-parser>=0.17.0',
    ],
    'dev': dev_requirements,
}

# 所有可选依赖
optional_requirements['all'] = [
    req for reqs in optional_requirements.values() for req in reqs
]

setup(
    name='wetland-hyperspectral-classification',
    version=get_version(),
    author='Wetland Research Team',
    author_email='wetland.research@example.com',
    description='基于深度学习与机器学习的高光谱遥感湿地生态系统精细化分类与景观格局分析系统',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/wetland-hyperspectral-classification',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/wetland-hyperspectral-classification/issues',
        'Documentation': 'https://yourusername.github.io/wetland-hyperspectral-classification/',
        'Source': 'https://github.com/yourusername/wetland-hyperspectral-classification',
    },
    
    # 包配置
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Python版本要求
    python_requires='>=3.8',
    
    # 依赖配置
    install_requires=get_requirements(),
    extras_require=optional_requirements,
    
    # 包数据
    package_data={
        'wetland_classification': [
            'config/*.yaml',
            'data/samples/*',
            'models/pretrained/*',
        ],
    },
    include_package_data=True,
    
    # 入口点
    entry_points={
        'console_scripts': [
            'wetland-classify=wetland_classification.cli:main',
            'wetland-preprocess=wetland_classification.preprocessing.cli:main',
            'wetland-train=wetland_classification.training.cli:main',
            'wetland-evaluate=wetland_classification.evaluation.cli:main',
        ],
    },
    
    # 分类器
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # 关键词
    keywords=[
        'hyperspectral', 'remote sensing', 'wetland', 'classification',
        'deep learning', 'machine learning', 'ecology', 'GIS',
        'landscape analysis', 'ecosystem mapping'
    ],
    
    # ZIP安全
    zip_safe=False,
    
    # 测试套件
    test_suite='tests',
    tests_require=dev_requirements,
    
    # 最低要求
    obsoletes=['wetland_classification<1.0'],
)