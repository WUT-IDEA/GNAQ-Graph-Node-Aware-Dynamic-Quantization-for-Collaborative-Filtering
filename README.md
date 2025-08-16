# GNAQ: A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering

本项目是CIKM'25论文《A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering (GNAQ)》的官方实现，专注于通过图神经网络量化技术实现高效的协同过滤推荐系统。

## 🎯 项目概述

GNAQ是一种轻量级图嵌入框架，专为大规模协同过滤任务设计。该框架通过创新的量化策略和组合嵌入技术，在保持推荐精度的同时显著降低了存储和计算开销。

## 📊 数据集

本项目支持三个广泛使用的推荐系统基准数据集：
- Gowalla
- Yelp2020
- Amazon-Book

### 数据格式
每个数据集包含以下文件：
- `train.txt`：训练集交互数据
- `test.txt`：测试集交互数据  
- `user_list.txt`：用户ID映射
- `item_list.txt`：物品ID映射

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone [repository-url]
cd GNAQ-master

# 安装依赖
pip install -r requirements.txt
```

### 基础运行

#### 单GPU运行示例
```bash
# Gowalla数据集 - 256维嵌入
python engine.py --dataset_name gowalla --latent_dim 256 --device_id 0

# Yelp2020数据集 - 128维嵌入  
python engine.py --dataset_name yelp2020 --latent_dim 128 --device_id 0

# Amazon-Book数据集 - 64维嵌入
python engine.py --dataset_name amazon-book --latent_dim 64 --device_id 0
```

#### 量化训练（核心功能）
```bash
# 使用二进制量化（推荐设置）
python engine.py \
  --dataset_name gowalla \
  --num_clusters 500 \
  --num_composition_centroid 2 \
  --latent_dim 128 \
  --use_pretrain_init \
  --pretrain_state_dict checkpoints/bgr-gowalla-128-rank.pth.tar \
  --device_id 0
```

### 批量实验
使用提供的脚本进行大规模实验：
```bash
# 运行所有预配置实验
chmod +x run.sh
./run.sh
```

## ⚙️ 核心参数配置

### 高级参数
```python
# 量化相关
--assignment_update_frequency "every-epoch"  # 聚类分配更新频率
--init_anchor_weight 0.5-0.9                 # 锚点权重初始化

# 训练优化
--use_pretrain_init                          # 使用预训练初始化
--pretrain_state_dict "checkpoints/..."      # 预训练模型路径
--early_stop_patience 50                     # 早停耐心值
```

## 📁 项目结构

```
GNAQ-master/
├── data/                     # 数据集目录
│   ├── gowalla/             # Gowalla数据集
│   ├── yelp2020/            # Yelp2020数据集
│   └── amazon-book/         # Amazon-Book数据集
├── models/                  # 模型实现
│   ├── binary_quantization.py   # 二进制量化核心
│   ├── com_embedding.py         # 组合嵌入实现
│   ├── mat_approx.py           # 矩阵近似方法
│   ├── recsys.py               # 推荐系统框架
│   └── rerank.py               # 重排序模块
├── data_loading/            # 数据加载
│   └── data_loader.py       # 数据预处理与加载
├── utils/                   # 工具函数
│   └── util.py              # 评估指标与工具
├── engine.py                # 主训练引擎
├── funcs.py                 # 核心训练与评估函数
├── set_parse.py             # 参数配置
├── run.sh                   # 批量实验脚本
└── requirements.txt         # 依赖列表

