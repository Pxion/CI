
# ### 一、数据清洗

# #### 1. 缺失值处理

# (1) 导入需要用到的模块

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
import lightgbm as lgb 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# In[3]:


plt.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  #让负号正常显示


# (2) 加载数据

# In[4]:


#加载数据集
data_set=pd.read_csv('dataset.csv',encoding='gb18030')


# (3) 总体数据分析

# In[5]:


data_set.shape  #数据集的行数、列数


# In[6]:


data_set.info()  #获取并展示该数据集的详细信息


# In[7]:


list(data_set.columns)  #获取data_set所有的列名


# In[8]:


data_set.head()  #快速查看data_set中的前5行数据


# (4) 计算缺失比率并可视化

# In[9]:


data_set_null=data_set
# 按列统计缺失值个数
missing_values_count = data_set_null.isnull().sum()

# 计算各列缺失比率
total_rows = len(data_set_null)
missing_ratios = (missing_values_count / total_rows) * 100

# 筛选出有缺失值的列
columns_with_missing = missing_values_count[missing_values_count > 0].index
missing_ratios_selected = missing_ratios[missing_ratios.index.isin(columns_with_missing)]

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(missing_ratios_selected.index, missing_ratios_selected)
plt.xlabel('Columns')
plt.ylabel('Missing Ratio (%)')
plt.title('Missing Value Ratios of Columns with Missing Values')
plt.xticks(rotation=90)
plt.show()


# (5) 删除缺少比率大于60%的列

# In[10]:


# 筛选出缺失比率高于60%的列
columns_to_delete = missing_ratios[missing_ratios > 60].index

# 删除这些列
data_set=data_set.drop(columns_to_delete, axis=1)

# 查看处理后的数据信息
data_set.info()


# (6) 将缺少比率大于40%小于60%的列的值用 -1 填充

# In[11]:


# 计算每列的缺失值比率
missing_rates = data_set.isnull().mean()

# 筛选出符合缺失比率范围的列名
columns_to_fill = missing_rates[(missing_rates > 0.4) & (missing_rates <= 0.6)].index

# 对符合条件的列进行填充
for col in columns_to_fill:
    data_set[col] = data_set[col].fillna(-1)


# (7) 将缺失比率为0-20%的列用均值填充

# In[12]:


# 计算每列的缺失值比率
missing_rates = data_set.isnull().mean()

# 筛选出符合缺失比率范围的列名
columns_to_fill = missing_rates[(missing_rates >= 0) & (missing_rates <= 0.2)].index

# 对符合条件的列进行填充
for col in columns_to_fill:
    if not pd.api.types.is_string_dtype(data_set[col].dtype):
        mean_value = data_set[col].mean()
        data_set[col] = data_set[col].fillna(mean_value)


# #### 2. 文本处理

# (1) UserInfo_9 字段的取值包含了空格字符，如“中国移动”和“中国移动  ”， 它们是同⼀种取值，需要将空格符去除。

# In[13]:


data_set['UserInfo_9'] = data_set['UserInfo_9'].str.strip() #去掉这一列的空格


# (2) UserInfo_8、UserInfo_7、UserInfo_19、UserInfo_20列中包含有“重庆”、“重庆市”；“山东”、“山东省”等取值，它们实际上是同⼀个城市或省份，需要把 字符中的“市”、“省”全部去掉。

# In[14]:


# 将“省”、“市”去掉
data_set['UserInfo_8'] = data_set['UserInfo_8'].str.replace('市', '', regex=False)
data_set['UserInfo_7'] = data_set['UserInfo_7'].str.replace('市', '', regex=False)
data_set['UserInfo_19'] = data_set['UserInfo_19'].str.replace('市', '', regex=False)
data_set['UserInfo_20'] = data_set['UserInfo_20'].str.replace('市', '', regex=False)
data_set['UserInfo_8'] = data_set['UserInfo_8'].str.replace('省', '', regex=False)
data_set['UserInfo_7'] = data_set['UserInfo_7'].str.replace('省', '', regex=False)
data_set['UserInfo_19'] = data_set['UserInfo_19'].str.replace('省', '', regex=False)


# #### 3.剔除常变量

# In[15]:


# 提取数值型列
numeric_columns = data_set.select_dtypes(include=['int64', 'float64']).columns

# 计算每列的标准差
std_devs = data_set[numeric_columns].std()

# 定义阈值，接近 0 的标准差
threshold = 1e-1  # 根据需求调整阈值

# 找到标准差接近于 0 的特征
low_variance_features = std_devs[std_devs < threshold]

# 输出这些特征及其标准差
print("以下特征的标准差接近于 0，将被剔除：")
for feature, std in low_variance_features.items():
    print(f"{feature}: 标准差 = {std:.6f}")

# 剔除这些特征
data_set = data_set.drop(columns=low_variance_features.index)


# #### 4.删除索引列

# In[16]:


data_set=data_set.drop(columns="Idx") #删除Index的列


# ### 二、特征工程

# #### 1.地理位置的处理

# ##### （1）省份特征的处理，UserInfo_7列和 UserInfo_19列是省份信息

# In[17]:


# 选择省份列和目标列
province_columns = ['UserInfo_7', 'UserInfo_19']
target_column = 'target'

# 合并省份信息，创建一个新 data_setFrame
province_data_set = data_set[province_columns + [target_column]]

# 将 UserInfo_7 和 UserInfo_19 合并为一个省份列
melted_data_set = province_data_set.melt(id_vars=[target_column], value_vars=province_columns, var_name='Source', value_name='Province')

# 计算每个省份的违约率
default_rates = melted_data_set.groupby('Province')[target_column].mean().reset_index()

# 找到违约率最高的 6 个省份
top_default_rates = default_rates.nlargest(6, target_column)

# 输出违约率最高的 6 个省份
print("违约率最高的 6 个省份及其违约率：")
print(top_default_rates)

# 构建二值特征
for province in top_default_rates['Province']:
    data_set[f'Is_{province}'] = data_set['UserInfo_7'].apply(lambda x: 1 if x == province else 0) | data_set['UserInfo_19'].apply(lambda x: 1 if x == province else 0)

# 输出构建的特征
print("\n构建的二值特征：")
print(data_set[[f'Is_{province}' for province in top_default_rates['Province']]].head())

# 删除原来列
data_set = data_set.drop(columns=["UserInfo_7", "UserInfo_19"])


# ##### (2)城市特征的处理："UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"是城市列，经纬度特征的引入

# （a）提取全国城市的经纬度信息

# In[18]:


import requests

# 高德地图 API 信息
api_url = "https://restapi.amap.com/v3/config/district"
api_key = "5c9c9cbb82bf1caeab02c4261d318d04"

# 请求中国的地级市信息
params = {
    "keywords": "中国",
    "subdistrict": 2,  # 2 表示返回地级市和区县信息
    "key": api_key,   
    "extensions": "base"  #表示获取基本的行政区划数据格式，不包含一些额外的详细扩展信息
}
response = requests.get(api_url, params=params)
data = response.json()

# 提取城市名称和经纬度
city_list = []
for province in data['districts'][0]['districts']:
    for city in province['districts']:
        city_name = city['name']
        if 'center' in city:
            lng, lat = map(float, city['center'].split(','))
            city_list.append({"City": city_name, "Latitude": lat, "Longitude": lng})

# 转换为 DataFrame 并保存
city_df = pd.DataFrame(city_list)
city_df.to_csv("china_city_list.csv", index=False)
print("城市列表已保存到 china_city_list.csv")
city_df.head()


# （b）将城市名用经纬度替换，得到北纬和东经两个数值型特征

# In[19]:


# 读取包含城市和经纬度的CSV文件
city_coords_df = pd.read_csv("china_city_list.csv")  # 替换为你的经纬度文件路径

# 将城市的“市”删去
city_coords_df['City'] = city_coords_df['City'].str.replace('市', '', regex=False)

# 创建城市名到经纬度的映射字典
# city_coords_df有三列："City"（城市名），"Latitude"（纬度），"Longitude"（经度）
city_to_coords = {row['City']: (row['Latitude'], row['Longitude']) for index, row in city_coords_df.iterrows()}

# 定义一个替换函数，将城市名替换为经纬度元组
def replace_with_coords(city):
    return city_to_coords.get(city, (None, None))  # 如果城市名不在字典中，返回(None, None)

# 对需要替换的列进行处理
for col in ["UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"]:
    data_set[col] = data_set[col].apply(lambda x: replace_with_coords(x) if isinstance(x, str) else (None, None))

# 拆分经纬度成两个独立的列
for col in ["UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"]:
    data_set[f"{col}_Latitude"] = data_set[col].apply(lambda x: x[0] if x != (None, None) else None)
    data_set[f"{col}_Longitude"] = data_set[col].apply(lambda x: x[1] if x != (None, None) else None)

# 查看处理后的数据
columns_to_view = []
for col in ["UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"]:
    columns_to_view.extend([col, f"{col}_Latitude", f"{col}_Longitude"])
print(data_set[columns_to_view].head())


# ##### （3）城市特征的处理：构建地理位置差异特征
# 1,2,4,6 列都是城市。我们构建⼀个城市差异的特征，
# 
# diff_12：比较“UserInfo_2”和“UserInfo_4”列中的城市是否相同。
# 
# diff_14：比较“UserInfo_2”和“UserInfo_8”列中的城市是否相同。
# 
# diff_16：比较“UserInfo_2”和“UserInfo_20”列中的城市是否相同。
# 
# diff_24：比较“UserInfo_4”和“UserInfo_8”列中的城市是否相同。
# 
# diff_26：比较“UserInfo_4”和“UserInfo_20”列中的城市是否相同。
# 
# diff_46：比较“UserInfo_8”和“UserInfo_20”列中的城市是否相同。

# In[20]:


# 创建城市差异特征
data_set['diff_12'] = (data_set['UserInfo_2'] == data_set['UserInfo_4']).astype(int)
data_set['diff_14'] = (data_set['UserInfo_2'] == data_set['UserInfo_8']).astype(int)
data_set['diff_16'] = (data_set['UserInfo_2'] == data_set['UserInfo_20']).astype(int)
data_set['diff_24'] = (data_set['UserInfo_4'] == data_set['UserInfo_8']).astype(int)
data_set['diff_26'] = (data_set['UserInfo_4'] == data_set['UserInfo_20']).astype(int)
data_set['diff_46'] = (data_set['UserInfo_8'] == data_set['UserInfo_20']).astype(int)

# 查看新增的差异特征
print(data_set[['diff_12', 'diff_14', 'diff_16', 'diff_24', 'diff_26', 'diff_46']].head())

# 删除原有列
data_set = data_set.drop(columns=["UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"])


# ##### （4）成交时间特征的处理
# 将成交时间的字段 Listinginfo 做离散化处理
# 
# 每 10 天作 为⼀个区间，也就是将日期 0~10 离散化为1，日期 11~20 离散化为2

# In[21]:


"""将成交日期离散化处理"""
def date_discretize(date_str):
    """
    函数用于把日期字符串列表进行离散化处理。
    参数:
    date_str (list): 日期字符串列表，格式如 '年/月/日'。
    返回值:
    ans (list): 离散化后的日期代码列表。
    """
    ans = []
    for date in date_str:
        parts = date.split('/')  # 分割日期字符串为年、月、日
        day = int(parts[-1])
        month = int(parts[-2])
        year = int(parts[-3])
        date_code = 3 * month - 2  # 按规则计算初始日期代码

        if year == 2014 and month == 11:  # 特殊情况处理
            ans.append(37)
            continue

        if date_code > 30:  # 调整日期代码
            date_code -= 30
        else:
            date_code += 6

        if 0 <= day <= 10:  # 根据日的范围确定最终日期代码
            ans.append(date_code)
        elif 11 <= day <= 20:
            ans.append(date_code + 1)
        else:
            ans.append(date_code + 2)

    return ans

# 应用函数对数据集的 'ListingInfo' 列进行离散化处理
data_set["ListingInfo"] = date_discretize(list(data_set["ListingInfo"]))

# 用于统计目标值为0时不同离散化日期出现的次数
date_count_0 = {}
# 用于统计目标值为1时不同离散化日期出现的次数
date_count_1 = {}
for i in range(len(list(data_set["ListingInfo"]))):
    if data_set.loc[i, "target"] == 0:  # 当目标值为0时
        date_count_0[data_set.loc[i, "ListingInfo"]] = date_count_0.get(data_set.loc[i, "ListingInfo"], 0) + 1
    elif data_set.loc[i, "target"] == 1:  # 当目标值为1时
        date_count_1[data_set.loc[i, "ListingInfo"]] = date_count_1.get(data_set.loc[i, "ListingInfo"], 0) + 1

# 通过字典推导式构建新的有序字典
date_count_1 = {key: date_count_1[key] for key in sorted(date_count_1.keys())}
date_count_0 = {key: date_count_0[key] for key in sorted(date_count_0.keys())}

# 提取字典1的键和值
x1 = list(date_count_0.keys())
y1 = list(date_count_0.values())
# 提取字典2的键和值
x2 = list(date_count_1.keys())
y2 = list(date_count_1.values())

# 绘制第一条折线，设置线条颜色等样式
plt.plot(x1, y1, 'r-o', label='count_0')
# 绘制第二条折线
plt.plot(x2, y2, 'b-s', label='count_1')

# 设置图表标题、坐标轴标签以及添加图例
plt.title('Date(20131101-20141110)')
plt.xlabel('date')
plt.ylabel('count')
plt.legend()

# 显示网格线
plt.grid(True)
# 展示图表
plt.show()


# ##### (5) 独热编码
# 除掉上述特殊⽣成的特征，其余都做独热编码

# In[22]:


# 选择所有字符串类型的列（通常是 object 类型）
string_columns = data_set.select_dtypes(include=['object']).columns

# 对这些列进行 One-Hot Encoding
data_set = pd.get_dummies(data_set, columns=string_columns)

# 查看编码后的数据
print(data_set.head())


# ### 三、特征选择

# #### 基于学习模型的特征排序⽅法
# 这种⽅法有⼀个好处：模型学习的过程和特征选择的过程是同时进⾏的，因此采用这种⽅法，基于xgboost来做特征选择，xgboost模型训练完成后可以输出特征的重要性，据此可以保留 Top N 个特征，从⽽达到特征选择的目的。

# ##### （1）检查数据

# In[23]:


# 查看数据的前几行，了解数据结构
print(data_set.head())

# 检查是否有缺失值
print(data_set.isnull().sum())


# ##### （2）分离特征和标签
# 将特征（X）和标签（y）分开

# In[24]:


X = data_set.drop(columns=['target'])  # 特征
y = data_set['target']  # 标签


# ##### （3）划分训练集和测试集
# 接下来，将数据集划分为训练集和测试集，通常 80% 用于训练，20% 用于测试

# In[25]:


from sklearn.model_selection import train_test_split

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ##### （4）训练 XGBoost 模型
# 然后，使用 XGBoost 训练模型，并获取特征重要性

# In[26]:


import xgboost as xgb

# 创建并训练模型
model_1 = xgb.XGBClassifier()  # 分类问题，使用XGBClassifier；如果是回归问题，使用 XGBRegressor
model_1.fit(X_train, y_train)


# ##### （5）获取特征重要性
# 训练好模型后，查看每个特征的重要性。XGBoost 提供了多种方式来评估特征重要性：

# 对于金融风控，importance_type我们选择gain（特征在树中分裂时的平均增益），原因如下：
# 
# 1.与模型性能紧密相关：金融风控模型（如信用评分、贷款违约预测、欺诈检测）最重要的是提升预测性能和区分不同风险等级，而 gain 恰好衡量了特征对模型性能的增益。
# 
# 2.考虑到数据不平衡：在风控任务中，负样本（如违约者或欺诈者）和正样本（如守信者或正常交易者）比例差异大，使用 gain 使得模型能更多地关注那些能有效区分这两类样本的特征，而不仅仅是频繁出现的特征。
# 
# 3.提高模型准确性：金融风控中的特征重要性不仅仅是计算特征在树中出现的次数，而是需要计算特征对模型分类决策的影响大小，因此 gain 提供了一个更具指导意义的衡量标准。

# In[27]:


#使用 model.feature_importances_ 获取特征重要性：
# 获取特征重要性
importances = model_1.feature_importances_

# 将重要性分数和特征名称结合起来，方便查看
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

#使用 xgb.plot_importance() 绘制特征重要性图：
import matplotlib.pyplot as plt

# 绘制特征重要性
xgb.plot_importance(model_1, importance_type='gain', max_num_features=50)
plt.show()


# ##### （6）根据特征重要性进行特征选择
# 根据特征的重要性，选择选择前 50 个特征进行训练。

# In[28]:


# 选择前 50 个重要特征
top_features = feature_importance_df.head(50)['Feature'].values

# 创建新数据集变量
selected_features_dataset = {
    'X_train': X_train.loc[:, top_features],  # 使用 loc 保持 DataFrame 结构
    'X_test': X_test.loc[:, top_features],
    'y_train': y_train,
    'y_test': y_test
}

# 输出前 50 个重要特征
print("前 50 个重要特征：")
print(top_features)



# ### 使用遗传算法进行进一步特征选择

# In[29]:


# ------------------------- 4. 遗传算法特征选择 -------------------------

class GeneticFeatureSelector:
    def __init__(self, n_features, pop_size=30, generations=15):
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations

    def _initialize_population(self, total_features):
        return np.random.choice([0,1], (self.pop_size, total_features), p=[0.8,0.2])

    def _fitness(self, individual, X, y):
        selected = individual.astype(bool)
        if np.sum(selected) == 0: return 0.0
        # 关闭LightGBM的日志输出
        model = lgb.LGBMClassifier(verbosity=0)  # 添加参数关闭日志
        scores = []
        kf = KFold(n_splits=3)

        y = y.values if isinstance(y, pd.Series) else y

        for train_idx, val_idx in kf.split(X):
            X_train = X[train_idx][:, selected]
            X_val = X[val_idx][:, selected]
            model.fit(X_train, y[train_idx])
            preds = model.predict_proba(X_val)[:,1]
            scores.append(roc_auc_score(y[val_idx], preds))
        return np.mean(scores)

    def select_features(self, X, y):
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = []
        X = X.values if isinstance(X, pd.DataFrame) else X
        #feature_names = X.columns if hasattr(X, 'columns') else []
        pop = self._initialize_population(X.shape[1])
        ga_history = {'best_fitness': [], 'avg_fitness': []}
        selection_counts = np.zeros(X.shape[1])
        population_diversity = []
        best_score = -np.inf
        best_individual = None



        for generation in range(self.generations):
            scores = np.array([self._fitness(ind, X, y) for ind in pop])
            # 新增数据记录（在精英选择前添加）
            ga_history['best_fitness'].append(np.max(scores))
            ga_history['avg_fitness'].append(np.mean(scores))
            population_diversity.append(np.mean(pop, axis=0))  # 记录种群特征均值
            selection_counts += np.sum(pop, axis=0)  # 累计所有个体特征选择

            elite_idx = np.argsort(-scores)[:5]
            new_pop = pop[elite_idx]

             # 交叉与变异（添加特征数控制）
            while len(new_pop) < self.pop_size:
                parents = pop[np.random.choice(len(pop), 2, replace=False)]
                crossover_point = np.random.randint(1, X.shape[1]-1)
                child = np.hstack([parents[0][:crossover_point], parents[1][crossover_point:]])

                # 变异并控制特征数量
                mutation_mask = np.random.rand(X.shape[1]) < 0.05
                child = np.where(mutation_mask, 1-child, child)

                # 确保特征数量为10
                current_count = np.sum(child)
                if current_count != self.n_features:
                    if current_count < self.n_features:
                        # 随机补充特征
                        zero_indices = np.where(child == 0)[0]
                        add_indices = np.random.choice(zero_indices, self.n_features - current_count, replace=False)
                        child[add_indices] = 1
                    else:
                        # 随机删除多余特征
                        one_indices = np.where(child == 1)[0]
                        remove_indices = np.random.choice(one_indices, current_count - self.n_features, replace=False)
                        child[remove_indices] = 0

                new_pop = np.vstack([new_pop, child])

            pop = new_pop
            current_best = np.max(scores)
            if current_best > best_score:
                best_score = current_best
                best_individual = pop[np.argmax(scores)]

        selected = best_individual.astype(bool)
        return {
        'selected_mask': best_individual.astype(bool),
        'ga_history': ga_history,
        'selection_counts': selection_counts,
        'population_diversity': np.array(population_diversity),
        'feature_names': feature_names
    }

# 应用遗传算法选择特征
X = selected_features_dataset['X_train']
y = selected_features_dataset['y_train'].values
selector = GeneticFeatureSelector(n_features=10)
results = selector.select_features(X, y)
selected_mask = selector.select_features(X, y)
selected_cols = X.columns[results['selected_mask']]
print(f"选择的特征数：{len(selected_cols)}")
print("选择的特征：", selected_cols)
X_selected = X[selected_cols]

# 可视化部分应使用 results 且补充图表元素
plt.figure(figsize=(10, 4))
plt.plot(results['ga_history']['best_fitness'], 'r-', label='最佳适应度')
plt.plot(results['ga_history']['avg_fitness'], 'b--', label='平均适应度')
plt.title('适应度进化曲线')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(results['feature_names'], results['selection_counts'])
plt.xticks(rotation=45, ha='right')
plt.title('特征选择频次统计')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(results['population_diversity'].T, cmap='coolwarm')  # 转置矩阵
plt.xlabel('进化代数')
plt.ylabel('特征索引')
plt.title('种群特征分布热力图')
plt.show()


# In[40]:


# -------------------------  优化前神经网络模型构建 -------------------------
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 自定义Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def _focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow(1.0 - p_t, gamma)
        return -K.sum(alpha_factor * modulating_factor * K.log(p_t), axis=-1)
    return _focal_loss

# 构建混合神经网络
def build_hybrid_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))

    # 特征交叉层
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)

    # 时序特征提取分支（假设特征中包含时序信息）
    lstm_branch = layers.Reshape((input_dim, 1))(inputs)  # 转换为时序输入
    lstm_branch = layers.Conv1D(64, 3, activation='relu')(lstm_branch)
    lstm_branch = layers.LSTM(32, return_sequences=False)(lstm_branch)



    # 特征融合
    merged = layers.concatenate([x, lstm_branch])

    # 深度特征提取
    merged = layers.Dense(64, activation="relu", 
                        kernel_regularizer=regularizers.l2(0.01))(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.5)(merged)

    outputs = layers.Dense(1, activation="sigmoid")(merged)

    model = models.Model(inputs=inputs, outputs=outputs)

    # 配置优化器带梯度裁剪
    opt = Adam(learning_rate=0.001, clipvalue=0.5)
    model.compile(optimizer=opt,
                loss=focal_loss(gamma=2, alpha=0.25),
                metrics=['AUC'])
    return model

# 确保数据类型为数值型且转换为numpy数组
X_selected = X_selected.astype('float32') 
y = y.astype('float32')

# 检查数据有效性（新增）
assert not np.any(np.isnan(X_selected)), "存在NaN值，请先处理缺失值"
assert not np.any(np.isinf(X_selected)), "存在Infinity值，请检查数据"
# 模型训练
model = build_hybrid_model(len(selected_cols))
history = model.fit(
    X_selected, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
best_auc = max(history.history['val_AUC'])
best_epoch = np.argmax(history.history['val_AUC']) + 1
print('\n' + '='*60)
print(f'※ 最佳验证AUC值: {best_auc:.4f} (第 {best_epoch} 个epoch)')
print('='*60)
# ------------------------- 训练效果可视化 -------------------------
plt.figure(figsize=(10, 6))
plt.plot(history.history['AUC'], label='训练集AUC')
plt.plot(history.history['val_AUC'], label='验证集AUC', linestyle='--')
plt.title(f'AUC曲线 [最佳验证AUC: {best_auc:.4f}]')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# 单独绘制验证集AUC曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_AUC'], color='darkorange', lw=2)
plt.title('验证集AUC变化趋势')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()

# 绘制AUC与Loss的对比图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['AUC'], label='训练AUC')
plt.plot(history.history['val_AUC'], label='验证AUC')
plt.title('AUC对比')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失对比')
plt.legend()

plt.tight_layout()
plt.show()


# In[45]:


# ------------------------- 5.1 蚁群超参数优化 -------------------------
import itertools
import numpy as np
import tensorflow as tf

class AntColonyOptimizer:
    def __init__(self, n_ants=10, iterations=20, evaporation=0.5):
        # 定义离散参数空间
        self.param_grid = {
            'n_heads': [2, 4, 8],
            'hidden_layers': [1, 2, 3]
        }
        self.combinations = list(itertools.product(*self.param_grid.values()))
        self.pheromone = np.ones(len(self.combinations))  # 初始信息素
        self.n_ants = n_ants
        self.iterations = iterations
        self.evaporation = evaporation
        self.best_auc = -np.inf
        self.best_params = None
        self.best_auc_history = []  # 新增最佳AUC记录
        self.avg_auc_history = []   # 新增平均AUC记录
        self.pheromone_history = [] # 新增信息素记录

    def _select_params(self):
        # 带启发式的概率计算（添加空历史记录处理）
        if len(self.history) == 0:
            # 初始阶段使用均匀分布
            probabilities = self.pheromone.copy()
        else:
            # 创建参数组合的AUC累计数组
            history_auc = np.zeros(len(self.combinations)) + 1e-8  # 防止零值
            for params, auc in self.history:
                idx = self.combinations.index(params)
                history_auc[idx] += auc

            probabilities = self.pheromone ** 1.5 * history_auc

        probabilities /= probabilities.sum()
        return np.random.choice(range(len(self.combinations)), p=probabilities)
    def optimize(self, X, y):
        X = np.expand_dims(X, axis=1)

        self.history = []

        from concurrent.futures import ThreadPoolExecutor

        for iter in range(self.iterations):
            ants_indices = [self._select_params() for _ in range(self.n_ants)]
            ants_params = [self.combinations[i] for i in ants_indices]

            # 使用chunksize减少通信开销
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self._evaluate_single_model, params, X, y) 
                          for params in ants_params]
                results = [f.result() for f in futures]

            for params, auc in zip(ants_params, results):
                self.history.append( (params, auc) )  # 记录参数组合和对应AUC

            # 信息素更新（Δτ与AUC正相关）
            delta_pheromone = np.zeros_like(self.pheromone)
            for i, idx in enumerate(ants_indices):
                delta_pheromone[idx] += results[i] * 0.1

            self.pheromone = self.pheromone * self.evaporation + delta_pheromone

            # 记录最佳参数
            current_best_idx = np.argmax(results)
            if results[current_best_idx] > self.best_auc:
                self.best_auc = results[current_best_idx]
                self.best_params = ants_params[current_best_idx]
            # 新增数据记录（在信息素更新后添加）
            self.best_auc_history.append(self.best_auc)
            self.avg_auc_history.append(np.mean(results))
            self.pheromone_history.append(self.pheromone.copy())
        return self.best_params

    def _build_model_with_params(self, params, input_dim):
        """根据参数组合构建神经网络模型"""
        n_heads, hidden_layers = params
        from tensorflow.keras import layers, Model

        inputs = layers.Input(shape=(1, input_dim)) 

        # 多头注意力机制
        attn_output = layers.MultiHeadAttention(
            key_dim=4, 
            num_heads=n_heads
        )(inputs, inputs)

        # 隐藏层堆叠
        x = layers.Concatenate()([inputs, attn_output])
        x = layers.Flatten()(x) 

        for _ in range(hidden_layers):
            x = layers.Dense(64, activation='relu')(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def _evaluate_single_model(self, params, X, y):
        """单个模型评估函数"""
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        model = self._build_model_with_params(params, X.shape[2])
        early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        patience=3, 
        mode='max',
        restore_best_weights=True
    )
        history = model.fit(X, y, epochs=15,  # 减少epoch数
                           validation_split=0.2, 
                           verbose=0,
                           callbacks=[early_stop])
        return np.max(history.history['val_auc'])

    def _wrap_evaluate(self, params_args, X, y):
        """Windows多进程需要的包装函数"""
        params, input_dim = params_args
        return self._evaluate_single_model(params, X, y)

# 执行优化流程
aco = AntColonyOptimizer()
optimized_params = aco.optimize(X_selected, y)
best_auc = aco.best_auc
print(f"最优参数组合：头数={optimized_params[0]}, 隐藏层数={optimized_params[1]}")

# 在优化完成后添加可视化代码
plt.figure(figsize=(15, 8))


# 2. 信息素浓度变化（独立显示）
plt.figure(figsize=(8, 4))
for i in range(5):
    plt.plot([p[i] for p in aco.pheromone_history], label=f'组合{i+1}')
plt.xlabel('迭代次数')
plt.ylabel('信息素浓度')
plt.title('前5参数组合信息素变化')
plt.legend()
plt.tight_layout()
plt.show()

# 3. 最终信息素分布热力图（独立显示）
plt.figure(figsize=(6, 4))
sns.heatmap(aco.pheromone.reshape(len(aco.param_grid['n_heads']), 
                                len(aco.param_grid['hidden_layers'])),
          annot=True, cmap='YlGnBu')
plt.xlabel('隐藏层数')
plt.ylabel('注意力头数')
plt.title('最终信息素分布')
plt.tight_layout()
plt.show()

# 4. 参数组合探索分布（独立显示）
plt.figure(figsize=(10, 4))
param_counts = np.zeros(len(aco.combinations))
for params, _ in aco.history:
    param_counts[aco.combinations.index(params)] += 1
plt.bar(range(len(param_counts)), param_counts)
plt.xlabel('参数组合索引')
plt.ylabel('探索次数')
plt.title('参数组合探索分布')
plt.tight_layout()
plt.show()


# In[2]:


# ------------------------- 5.2 神经网络模型构建 -------------------------
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# 自定义Focal Loss
@tf.keras.utils.register_keras_serializable(package='Custom')
def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = K.pow(1.0 - p_t, gamma)
    return -K.sum(alpha_factor * modulating_factor * K.log(p_t), axis=-1)

# 构建混合神经网络
def build_hybrid_model(input_dim,n_heads, hidden_layers):
    inputs = layers.Input(shape=(input_dim,))

    # 特征交叉层
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)

    # 时序特征提取分支（假设特征中包含时序信息）
    #lstm_branch = layers.Reshape((input_dim, 1))(inputs)  # 转换为时序输入
    #lstm_branch = layers.Conv1D(64, 3, activation='relu')(lstm_branch)
    #lstm_branch = layers.LSTM(32, return_sequences=False)(lstm_branch)

     # Transformer结构
    transformer_input = layers.Reshape((input_dim, 1))(inputs)
    transformer_branch = layers.MultiHeadAttention(
        num_heads=n_heads,  # 使用优化后的头数
        key_dim=64  # 修正为合法参数值
    )(transformer_input, transformer_input)
    # 添加维度转换层
    transformer_branch = layers.GlobalAveragePooling1D()(transformer_branch)
    transformer_branch = layers.Dense(64)(transformer_branch)  # 与全连接分支对齐

    # 特征融合
    merged = layers.concatenate([x, transformer_branch])

    # 动态构建隐藏层（使用优化参数）
    for _ in range(hidden_layers-1):  # 保持至少一层
        merged = layers.Dense(64, activation='relu')(merged)

    # 深度特征提取
    merged = layers.Dense(64, activation="relu", 
                        kernel_regularizer=regularizers.l2(0.01))(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.5)(merged)

    outputs = layers.Dense(1, activation="sigmoid")(merged)

    model = models.Model(inputs=inputs, outputs=outputs)

    # 配置优化器带梯度裁剪
    opt = Adam(learning_rate=0.001, clipvalue=0.5)
    model.compile(optimizer=opt,
                loss=focal_loss,
                metrics=['AUC'])
    return model
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, 
    y, 
    test_size=0.2, 
    random_state=42
)

# 模型训练
model = build_hybrid_model(
    input_dim=len(selected_cols),
    n_heads=optimized_params[0],
    hidden_layers=optimized_params[1]
)
history = model.fit(
    X_train, y_train,  # 使用训练集
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),  # 使用验证集
    callbacks=[
        EarlyStopping(
            monitor='val_AUC',
            patience=5,
            mode='max',
            restore_best_weights=True),
        ModelCheckpoint(
            'best_model.h5', 
            monitor='val_AUC',
            mode='max',
            save_best_only=True)
    ]
)

# 加载最佳模型并评估
from tensorflow.keras.models import load_model

best_model = load_model('best_model.h5', custom_objects={'focal_loss': focal_loss})
val_auc = best_model.evaluate(X_val, y_val, verbose=0)[1]
print(f"\n最佳验证集AUC: {val_auc:.4f}")

# 可选：输出训练过程中的最高AUC
max_auc = max(history.history['val_AUC'])
print(f"训练过程中最高val_AUC: {max_auc:.4f}")


# In[51]:


# ------------ 6.1模拟退火微调 ------------
def enhanced_sa_tuning(initial_params, X, y, init_temp=100.0, cooling_rate=0.9):
    """改进的模拟退火微调算法"""
    # 参数边界约束
    param_bounds = {
        'n_heads': (2, 8),
        'hidden_layers': (1, 3)
    }

    # 初始化
    current_params = np.array(initial_params, dtype=float)
    best_params = current_params.copy()
    current_auc = evaluate_params(current_params, X, y)
    best_auc = current_auc
    T = init_temp
    no_improve_counter = 0
    history = []

    # 退火循环
    while T > 1.0 and no_improve_counter < 10:
        # 生成新参数（温度相关扰动）
        new_params = current_params + np.round(np.random.normal(0, T/20, size=2))  
        new_params = np.clip(new_params, [pb[0] for pb in param_bounds.values()], 
                            [pb[1] for pb in param_bounds.values()])
        new_params = new_params.astype(int)

        valid_heads = [2,4,8]  # 蚁群设定的可选值
        if new_params[0] not in valid_heads:
            # 寻找最近的合法值
            new_params[0] = min(valid_heads, key=lambda x: abs(x - new_params[0]))

        # 评估新参数
        new_auc = evaluate_params(new_params, X, y)
        delta = new_auc - current_auc

        # Metropolis接受准则
        if delta > 0 or np.random.rand() < np.exp(delta/T):
            current_params = new_params
            current_auc = new_auc
            if new_auc > best_auc:
                best_auc = new_auc
                best_params = new_params
                no_improve_counter = 0
            else:
                no_improve_counter += 1

        # 记录状态
        history.append({
            'temp': T,
            'current_auc': current_auc,
            'best_auc': best_auc,
            'params': current_params
        })

        # 降温
        T *= cooling_rate

    # 可视化优化过程
    plt.figure(figsize=(10, 6))
    plt.plot([h['best_auc'] for h in history], 'r-o', label='Best AUC')
    plt.plot([h['current_auc'] for h in history], 'b--', label='Current AUC')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.title('SA Optimization Process')
    plt.legend()
    plt.show()

    return best_params, best_auc, history

def evaluate_params(params, X, y):
    """快速评估参数性能"""
    try:
        model = build_hybrid_model(
            input_dim=X.shape[1],
            n_heads=int(params[0]),
            hidden_layers=int(params[1])
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train, epochs=15, verbose=0,
                 validation_data=(X_val, y_val))
        return model.evaluate(X_val, y_val, verbose=0)[1]
    except:
        return 0.0

# 使用方式（在蚁群优化后调用）
print("\n=== 模拟退火微调 ===")
final_params, final_auc, sa_history = enhanced_sa_tuning( 
    initial_params=optimized_params,
    X=X_selected,
    y=y
)

print(f"\n最终参数：头数={final_params[0]}, 隐藏层={final_params[1]}")

# 可视化优化过程（独立显示）


# 2. 参数变化轨迹
plt.figure(figsize=(10, 4))
params = np.array([h['params'] for h in sa_history])
plt.plot(params[:,0], 'g-s', label='注意力头数')
plt.plot(params[:,1], 'm-o', label='隐藏层数')
plt.xlabel('迭代次数')
plt.ylabel('参数值')
plt.title('参数优化轨迹')
plt.legend()
plt.show()

# 3. 温度衰减曲线
plt.figure(figsize=(10, 4))
temps = [h['temp'] for h in sa_history]
plt.semilogy(temps, 'k-', label='温度曲线')
plt.xlabel('迭代次数')
plt.ylabel('温度（对数刻度）')
plt.title('温度衰减过程')
plt.show()

# 4. 参数空间搜索路径
plt.figure(figsize=(8, 6))
sc = plt.scatter(params[:,0], params[:,1], c=[h['best_auc'] for h in sa_history], 
                cmap='viridis', s=50)
plt.colorbar(sc, label='AUC值')
plt.xlabel('注意力头数')
plt.ylabel('隐藏层数')
plt.title('参数空间搜索路径')
plt.show()


# In[1]:


# ------------------------- 6.2 最终模型训练 -------------------------
print("\n=== 最终模型训练（模拟退火优化版） ===")

# 使用模拟退火优化后的参数构建模型
final_model = build_hybrid_model(
    input_dim=len(selected_cols),
    n_heads=int(final_params[0]),
    hidden_layers=int(final_params[1])
)

# 训练配置（增加训练epoch）
final_history = final_model.fit(
    X_train, y_train,
    epochs=100,  # 增加训练轮次
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(
            monitor='val_AUC',
            patience=10,  # 增加耐心值
            mode='max',
            restore_best_weights=True),
        ModelCheckpoint(
            'final_model.h5',
            monitor='val_AUC',
            mode='max',
            save_best_only=True)
    ]
)

# 加载最佳模型并评估
final_best_model = load_model('final_model.h5', custom_objects={'focal_loss': focal_loss})
final_val_auc = final_best_model.evaluate(X_val, y_val, verbose=0)[1]

# 输出最佳验证集AUC
print('\n' + '='*60)
print(f'※ 最佳验证集 AUC: {final_val_auc:.4f}')
print('='*60)

