
# 基于神经网络与计算智能优化的金融风控模型
[![GitHub Repo Status](https://img.shields.io/github/status/your-username/your-repo.svg)](https://github.com/your-username/your-repo)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 目录
- [项目背景](#项目背景)
- [技术路线](#技术路线)
- [主要内容](#主要内容)
- [如何运行](#如何运行)
- [文件结构](#文件结构)
- [贡献说明](#贡献说明)
- [许可证](#许可证)
- [联系信息](#联系信息)


## 项目背景
在互联网金融快速发展的背景下，信用风险评估面临数据高维、非线性和不平衡等挑战。传统机器学习方法（如XGBoost）对复杂模式捕捉能力有限，且超参数调优依赖人工经验。本项目提出“**神经网络+计算智能优化**”混合框架，通过遗传算法（GA）、蚁群算法（ACO）和模拟退火（SA）优化模型，提升金融风控模型的准确性与鲁棒性。


## 技术路线
1. **数据预处理**  
   - 缺失值处理：删除缺失率>60%的列，40%-60%的列填充-1，其余用均值填充  
   - 特征工程：地理位置编码（经纬度）、时间离散化、独热编码  
   - 特征选择：基于XGBoost重要性排序和遗传算法降维  

2. **模型优化三阶段架构**  
   - **阶段1：遗传算法（GA）特征选择**：通过二进制染色体优化特征子集，以LightGBM交叉验证AUC为适应度函数  
   - **阶段2：蚁群算法（ACO）超参数优化**：搜索Transformer头数（2/4/8）和隐藏层数（1/2/3）的最优组合  
   - **阶段3：模拟退火（SA）权重微调**：通过温度控制扰动，避免模型陷入局部最优  

3. **神经网络模型**  
   - 混合结构：融合特征交叉层与Transformer分支  
   - 损失函数：Focal Loss缓解类别不平衡  
   - 正则化：Dropout + L2正则化 + 梯度裁剪  


## 主要内容
### 1. 数据处理流程
```python
# 缺失值处理示例
data_set = data_set.drop(columns_to_delete, axis=1)  # 删除缺失率>60%的列
data_set[col] = data_set[col].fillna(-1)  # 填充40%-60%缺失率的列
```

### 2. 特征工程
- **地理位置处理**：通过高德地图API获取城市经纬度，将城市名转换为经纬度特征  
- **时间离散化**：将成交时间按每10天划分区间  
- **独热编码**：对字符串类型特征进行One-Hot编码  

### 3. 模型优化
- **遗传算法特征选择**：通过3折交叉验证评估特征子集的AUC  
- **蚁群算法超参数搜索**：基于信息素浓度动态调整参数探索概率  
- **模拟退火权重微调**：通过Metropolis准则接受劣解，提升全局搜索能力  


## 如何运行
### 1. 环境依赖
```bash
# 安装依赖
pip install pandas numpy scikit-learn tensorflow lightgbm xgboost requests matplotlib seaborn
```

### 2. 数据获取
- 数据集来源：[和鲸社区金融风控比赛](https://www.heywhale.com/home/competition/56cd5f02b89b5bd026cb39c9/content/1)  
- 下载后解压到`data/`目录，文件结构如下：
  ```
  data/
  └─ dataset.csv
  ```

### 3. 代码运行步骤
```bash
# 1. 数据预处理与特征工程
python code/data_preprocessing.py

# 2. 特征选择（遗传算法）
python code/feature_selection_ga.py

# 3. 模型训练与优化（蚁群算法+模拟退火）
python code/model_training_aco_sa.py

# 4. 结果可视化
python code/visualization.py
```


## 文件结构
```
.
├─ code/                # 代码目录
│  ├─ data_preprocessing.py   # 数据预处理
│  ├─ feature_selection_ga.py # 遗传算法特征选择
│  ├─ model_training_aco_sa.py# 模型训练与优化
│  └─ visualization.py        # 结果可视化
├─ data/                # 数据集目录（需手动下载）
├─ docs/                # 文档目录
│  └─ 研究报告.pdf       # 项目详细说明
├─ models/              # 训练好的模型权重
├─ LICENSE             # 许可证文件
└─ README.md           # 项目说明
```


## 贡献说明
欢迎提交Issue或Pull Request改进模型：  
1. 提出新的特征工程方法  
2. 优化计算智能算法参数  
3. 改进模型架构设计  


## 许可证
本项目采用MIT许可证，详情见[LICENSE](LICENSE)。

如果需要帮助或合作，请随时联系！
