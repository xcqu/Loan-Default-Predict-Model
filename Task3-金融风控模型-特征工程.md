<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#什么是特征工程" data-toc-modified-id="什么是特征工程-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>什么是特征工程</a></span><ul class="toc-item"><li><span><a href="#特征工程目标" data-toc-modified-id="特征工程目标-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>特征工程目标</a></span></li><li><span><a href="#简要介绍" data-toc-modified-id="简要介绍-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>简要介绍</a></span></li></ul></li><li><span><a href="#基础包加载" data-toc-modified-id="基础包加载-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>基础包加载</a></span></li><li><span><a href="#基础函数" data-toc-modified-id="基础函数-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>基础函数</a></span></li><li><span><a href="#基础数据处理" data-toc-modified-id="基础数据处理-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>基础数据处理</a></span></li><li><span><a href="#空值处理" data-toc-modified-id="空值处理-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>空值处理</a></span></li><li><span><a href="#异常值处理" data-toc-modified-id="异常值处理-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>异常值处理</a></span><ul class="toc-item"><li><span><a href="#数值特征异常值处理" data-toc-modified-id="数值特征异常值处理-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>数值特征异常值处理</a></span><ul class="toc-item"><li><span><a href="#数字特征变量" data-toc-modified-id="数字特征变量-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>数字特征变量</a></span></li><li><span><a href="#通过Boxplot法识别异常值" data-toc-modified-id="通过Boxplot法识别异常值-6.1.2"><span class="toc-item-num">6.1.2&nbsp;&nbsp;</span>通过Boxplot法识别异常值</a></span></li><li><span><a href="#均值标准差方法（Mean-Standard-Deviation-Method）判别异常值" data-toc-modified-id="均值标准差方法（Mean-Standard-Deviation-Method）判别异常值-6.1.3"><span class="toc-item-num">6.1.3&nbsp;&nbsp;</span>均值标准差方法（Mean-Standard Deviation Method）判别异常值</a></span></li><li><span><a href="#绝对中位差（Median-Absolute-Deviation）判别异常值" data-toc-modified-id="绝对中位差（Median-Absolute-Deviation）判别异常值-6.1.4"><span class="toc-item-num">6.1.4&nbsp;&nbsp;</span>绝对中位差（Median Absolute Deviation）判别异常值</a></span></li><li><span><a href="#直接采用截断法" data-toc-modified-id="直接采用截断法-6.1.5"><span class="toc-item-num">6.1.5&nbsp;&nbsp;</span>直接采用截断法</a></span></li><li><span><a href="#loanAmnt-贷款金额-异常处理" data-toc-modified-id="loanAmnt-贷款金额-异常处理-6.1.6"><span class="toc-item-num">6.1.6&nbsp;&nbsp;</span>loanAmnt 贷款金额 异常处理</a></span></li><li><span><a href="#annualIncome-年收入-异常处理" data-toc-modified-id="annualIncome-年收入-异常处理-6.1.7"><span class="toc-item-num">6.1.7&nbsp;&nbsp;</span>annualIncome 年收入 异常处理</a></span></li><li><span><a href="#installment----分期付款金额" data-toc-modified-id="installment----分期付款金额-6.1.8"><span class="toc-item-num">6.1.8&nbsp;&nbsp;</span>installment    分期付款金额</a></span></li><li><span><a href="#dti----债务收入比" data-toc-modified-id="dti----债务收入比-6.1.9"><span class="toc-item-num">6.1.9&nbsp;&nbsp;</span>dti    债务收入比</a></span></li></ul></li><li><span><a href="#类别特征异常值处理" data-toc-modified-id="类别特征异常值处理-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>类别特征异常值处理</a></span><ul class="toc-item"><li><span><a href="#类别特征变量" data-toc-modified-id="类别特征变量-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>类别特征变量</a></span></li><li><span><a href="#有序变量处理" data-toc-modified-id="有序变量处理-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>有序变量处理</a></span></li><li><span><a href="#无序变量处理" data-toc-modified-id="无序变量处理-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>无序变量处理</a></span><ul class="toc-item"><li><span><a href="#无序变量OneHot处理" data-toc-modified-id="无序变量OneHot处理-6.2.3.1"><span class="toc-item-num">6.2.3.1&nbsp;&nbsp;</span>无序变量OneHot处理</a></span></li><li><span><a href="#无序变量进行LabelEncoder处理" data-toc-modified-id="无序变量进行LabelEncoder处理-6.2.3.2"><span class="toc-item-num">6.2.3.2&nbsp;&nbsp;</span>无序变量进行LabelEncoder处理</a></span></li></ul></li></ul></li><li><span><a href="#日期特征异常值处理" data-toc-modified-id="日期特征异常值处理-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>日期特征异常值处理</a></span><ul class="toc-item"><li><span><a href="#日期特征变量" data-toc-modified-id="日期特征变量-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>日期特征变量</a></span></li></ul></li></ul></li><li><span><a href="#特征构造" data-toc-modified-id="特征构造-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>特征构造</a></span><ul class="toc-item"><li><span><a href="#数值变量特征构造" data-toc-modified-id="数值变量特征构造-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>数值变量特征构造</a></span></li><li><span><a href="#类别变量特征构造" data-toc-modified-id="类别变量特征构造-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>类别变量特征构造</a></span></li><li><span><a href="#日期变量特征构造" data-toc-modified-id="日期变量特征构造-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>日期变量特征构造</a></span></li><li><span><a href="#数值+类别变量特征构造" data-toc-modified-id="数值+类别变量特征构造-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>数值+类别变量特征构造</a></span></li></ul></li><li><span><a href="#特征选择" data-toc-modified-id="特征选择-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>特征选择</a></span></li><li><span><a href="#特征工程整合处理过程" data-toc-modified-id="特征工程整合处理过程-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>特征工程整合处理过程</a></span></li><li><span><a href="#-----------分割线---------------------------------------------" data-toc-modified-id="-----------分割线----------------------------------------------10"><span class="toc-item-num">10&nbsp;&nbsp;</span>-----------分割线---------------------------------------------</a></span></li></ul></div>

<center><font size=6.5>Task3-金融风控模型-特征工程</font></center>

**基本情况介绍**

赛题以预测用户贷款是否违约为任务，数据集报名后可见并可下载，该数据来自某信贷平台的贷款记录，总数据量超过120w，包含47列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取80万条作为训练集，20万条作为测试集A，20万条作为测试集B，同时会对employmentTitle、purpose、postCode和title等信息进行脱敏。


|     **字段英文名**      |                       **字段中文名**                        | **字段个人理解**|
| :---------------- | :-------------------------------------------------------------|:-------------------------------------|
|id|为贷款清单分配的唯一信用证标识|用于识别每笔贷款|
|loanAmnt|贷款金额|贷款放款的金额或批放的金额|
|term|贷款期限（year）||
|interestRate|贷款利率|需要确认是年利率还是月利率|
|installment|分期付款金额||
|grade|贷款等级|数据中贷款等级为A、B、C、D、E、F、G|
|subGrade|贷款等级之子级|每个等级的子等级又分五档，比如：A的子级为A1~A5|
|employmentTitle|就业职称|职称一般与收入成正比|
|employmentLength|就业年限（年）|就业年收入与收入成正比|
|homeOwnership|借款人在登记时提供的房屋所有权状况|抵质押物的情况，有抵质押物增加放款的概率|
|annualIncome|年收入|收入情况|
|verificationStatus|验证状态|不知道是什么？|
|issueDate|贷款发放的月份|数据不是方法日期，而是YYYY-MM-DD类型|
|purpose|借款人在贷款申请时的贷款用途类别||
|postCode|借款人在贷款申请中提供的邮政编码的前3位数字|用于区别借款的人的地区|
|regionCode|地区编码||
|dti|债务收入比|用于识别贷款人的还款能力|
|delinquency_2years|借款人过去2年信用档案中逾期30天以上的违约事件数|识别还款意愿|
|ficoRangeLow|借款人在贷款发放时的fico所属的下限范围|fico类似于人行征信评分|
|ficoRangeHigh|借款人在贷款发放时的fico所属的上限范围|fico类似于人行征信评分|
|openAcc|借款人信用档案中未结信用额度的数量|？？|
|pubRec|贬损公共记录的数量|用户识别贷款客户的品行|
|pubRecBankruptcies|公开记录清除的数量||
|revolBal|信贷周转余额合计||
|revolUtil|循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额||
|totalAcc|借款人信用档案中当前的信用额度总数||
|initialListStatus|贷款的初始列表状态|代码值为0或1|
|applicationType|表明贷款是个人申请还是与两个共同借款人的联合申请|代码值为0或1|
|earliesCreditLine|借款人最早报告的信用额度开立的月份|样本数据为：'Aug-2001'类型|
|title|借款人提供的贷款名称||
|policyCode|公开可用的策略_代码=1新产品不公开可用的策略_代码=2||
|n系列匿名特征|匿名特征n0-n14，为一些贷款人行为计数特征的处理|匿名变量无法识别业务含义|


经过初步分析后，特征类型如下：
- **类别型变量**：


```python
  categorical_col = ['grade', 'subGrade', 'employmentLength', 'employmentTitle', 'initialListStatus', 'term', 'verificationStatus',
                   'homeOwnership', 'applicationType', 'title', 'purpose',
                   'regionCode', 'postCode']


```
- **数值型变量**：

```python
  numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 
             'delinquency_2years', 'ficoRangeLow','ficoRangeHigh','openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal', 
             'revolUtil', 'totalAcc','n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']

```
- **日期型变量**：

```python
  date_col = ['issueDate'、'earliesCreditLine']
```


# 什么是特征工程

## 特征工程目标
对于特征进行进一步分析，并对于数据进行处理

完成对于特征工程的分析，并对于数据进行一些图表或者文字总结并打卡。


## 简要介绍
常见的特征工程包括：
1. 异常处理：
    - 通过箱线图（或 3-Sigma）分析删除异常值；
    - BOX-COX 转换（处理有偏分布）；
    - 长尾截断；
2. 特征归一化/标准化：
    - 标准化（转换为标准正态分布）；
    - 归一化（抓换到 [0,1] 区间）；
    - 针对幂律分布，可以采用公式： $log(\frac{1+x}{1+median})$
3. 数据分桶：
    - 等频分桶；
    - 等距分桶；
    - Best-KS 分桶（类似利用基尼指数进行二分类）；
    - 卡方分桶；
4. 缺失值处理：
    - 不处理（针对类似 XGBoost 等树模型）；
    - 删除（缺失数据太多）；
    - 插值补全，包括均值/中位数/众数/建模预测/多重插补/压缩感知补全/矩阵补全等；
    - 分箱，缺失值一个箱；
5. 特征构造：
    - 构造统计量特征，报告计数、求和、比例、标准差等；
    - 时间特征，包括相对时间和绝对时间，节假日，双休日等；
    - 地理信息，包括分箱，分布编码等方法；
    - 非线性变换，包括 log/ 平方/ 根号等；
    - 特征组合，特征交叉；
    - 仁者见仁，智者见智。
6. 特征筛选
    - 过滤式（filter）：先对数据进行特征选择，然后在训练学习器，常见的方法有 Relief/方差选择发/相关系数法/卡方检验法/互信息法；
    - 包裹式（wrapper）：直接把最终将要使用的学习器的性能作为特征子集的评价准则，常见方法有 LVM（Las Vegas Wrapper） ；
    - 嵌入式（embedding）：结合过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有 lasso 回归；
7. 降维
    - PCA/ LDA/ ICA；
    - 特征选择也是一种降维。

# 基础包加载


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-white')           #风格设置近似R这种的ggplot库
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  #字体乱码
plt.rcParams['font.family']='sans-serif'               #字体乱码  
plt.rcParams['axes.unicode_minus'] = False            #数字为负
# 设置精度
pd.set_option('display.float_format', lambda x: '%.2f'%x)
pd.set_option('max_columns', 1000) #设置最大列数
pd.set_option('max_row', 300)      #设置最大行数

%matplotlib inline
```

# 基础函数


```python
def missing_values_table(df):
    """统计制定数据框中的缺失值大小

    :param df:
    :return:
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


```

# 基础数据处理


```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem/1024/1024))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem/1024/1024))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

## 数据加载
train_df = reduce_mem_usage(pd.read_csv('../data/train.csv'))
test_df = reduce_mem_usage(pd.read_csv('../data/testA.csv'))

## 删除唯一值变量=policyCode
# del train_df['policyCode']
# del test_df['policyCode']

drop_list = ['policyCode', 'delinquency_2years', 'pubRec', 'pubRecBankruptcies', 'n0', 'n12', 'n13']
train_df.drop(drop_list, axis=1, inplace = True)
test_df.drop(drop_list, axis=1, inplace = True)


## 分类变量转化
# category_col = ['grade', 'subGrade', 'employmentLength', 'employmentTitle', 'initialListStatus', 'term', 'verificationStatus',
#                 'homeOwnership', 'applicationType', 'title', 'purpose','regionCode', 'postCode', 'issueDate', 'earliesCreditLine']

# train_df[category_col] = train_df[category_col].astype('object')
# test_df[category_col]  = test_df[category_col].astype('object')

## 训练数据与测试数据进行合并，并与后续进行特征处理
train_df['isFlag'] = '1'
test_df['isFlag']  = '0'
## merge_df = pd.concat([train_df, test_df], ignore_index=True)

## 数据合并
merge_df = pd.concat([train_df, test_df], ignore_index=True)

print("处理完毕")
```

    Memory usage of dataframe is 286.87 MB
    Memory usage after optimization is: 69.46 MB
    Decreased by 75.8%
    Memory usage of dataframe is 70.19 MB
    Memory usage after optimization is: 17.20 MB
    Decreased by 75.5%
    处理完毕
    

# 空值处理


```python
# 这种方法并没有把 NaN的值处理掉
numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow','ficoRangeHigh','openAcc',  'revolBal', 
                'revolUtil', 'totalAcc', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']


category_col = ['grade', 'subGrade', 'employmentLength', 'employmentTitle', 'initialListStatus', 'term', 'verificationStatus',
                'homeOwnership', 'applicationType', 'title', 'purpose', 'regionCode', 'postCode', 'issueDate', 'earliesCreditLine']

merge_df[category_col] = merge_df[category_col].astype('object')

#按照中值填充数值特征
for col in numberic_col:
    print('{}列中值是:{}'.format(col,merge_df[col].median()))
    merge_df[col] = merge_df[col].fillna(merge_df[col].median())

    
for col in category_col:
    print('{} 列众数是:{}'.format(col,merge_df[col].mode().values[0]))
    merge_df[col] = merge_df[col].fillna(merge_df[col].mode().values[0])
    
# 查看空值率
missing_values_table(merge_df)
```

    loanAmnt列中值是:12000.0
    interestRate列中值是:12.7421875
    installment列中值是:375.5
    annualIncome列中值是:65000.0
    dti列中值是:17.625
    delinquency_2years列中值是:0.0
    ficoRangeLow列中值是:690.0
    ficoRangeHigh列中值是:694.0
    openAcc列中值是:11.0
    pubRec列中值是:0.0
    pubRecBankruptcies列中值是:0.0
    revolBal列中值是:11133.0
    revolUtil列中值是:52.1875
    totalAcc列中值是:23.0
    n0列中值是:0.0
    n1列中值是:3.0
    n2列中值是:5.0
    n3列中值是:5.0
    n4列中值是:4.0
    n5列中值是:7.0
    n6列中值是:7.0
    n7列中值是:7.0
    n8列中值是:13.0
    n9列中值是:5.0
    n10列中值是:11.0
    n11列中值是:0.0
    n12列中值是:0.0
    n13列中值是:0.0
    n14列中值是:2.0
    grade 列众数是:B
    subGrade 列众数是:C1
    employmentLength 列众数是:10+ years
    employmentTitle 列众数是:54.0
    initialListStatus 列众数是:0
    term 列众数是:3
    verificationStatus 列众数是:1
    homeOwnership 列众数是:0
    applicationType 列众数是:0
    title 列众数是:0.0
    purpose 列众数是:0
    regionCode 列众数是:8
    postCode 列众数是:134.0
    issueDate 列众数是:2016-03-01
    earliesCreditLine 列众数是:Aug-2001
    Your selected dataframe has 46 columns.
    There are 1 columns that have missing values.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>isDefault</th>
      <td>200000</td>
      <td>20.00</td>
    </tr>
  </tbody>
</table>
</div>



# 异常值处理

## 数值特征异常值处理


### 数字特征变量
```python
  numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow','ficoRangeHigh','openAcc',  'revolBal', 
                'revolUtil', 'totalAcc', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']


```


```python
numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow','ficoRangeHigh','openAcc',  'revolBal', 
                'revolUtil', 'totalAcc', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']

print("数值变量个数：{}".format(len(numberic_col)))

print("数值变量分布情况:\n", merge_df[numberic_col].describe().T)
```

### 通过Boxplot法识别异常值


```python
# 绘图行列情况
numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow','ficoRangeHigh','openAcc',  'revolBal', 
                'revolUtil', 'totalAcc', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']

ncols = 2
nrows = int(len(numberic_col)/ncols+0.5)

## 空值先直接fillna(-1)处理

# 绘制连续变量直方图
f, ax = plt.subplots(nrows, ncols, figsize=(14, nrows*3))
for i in range(nrows):
    j= i*2
    if j+1 <= len(numberic_col) -1:
        ax[i,0].boxplot(train_df[numberic_col[j]])
        ax[i,0].set_title("%s 箱线图分析" % str(numberic_col[j]),fontsize=20)
        
        ax[i,1].boxplot(train_df[numberic_col[j+1]])
        ax[i,1].set_title("%s 箱线图分析" % str(numberic_col[j+1]),fontsize=20)
    else:
        ax[i,0].boxplot(train_df[numberic_col[j]])
        ax[i,0].set_title("%s 箱线图分析" % str(numberic_col[j]),fontsize=20)
f.tight_layout()

plt.show()
```

通过如上箱线图分析法，可以确定的异常变量如下：
 - annualIncome  年收入字段  90000.00（75%） 10999200.00(max Value)
 - dti	债务收入比 24.06(75%)  999.00(max Value)
 - delinquency_2years 借款人过去2年信用档案中逾期30天以上的违约事件数  0.00(75%)  39.00(max Value)
 - 
 
 
 
 可疑变量如下：
 - 	openAcc	借款人信用档案中未结信用额度的数量
 - 
 


```python
# 这里我包装了一个异常值处理的代码，可以随便调用。
def outliers_proc_boxplot(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    ## 绘制图形
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n


merge_box_df = merge_df.copy()
#installment

#merge_box_df

outliers_proc_boxplot(merge_box_df, 'installment')
```

### 均值标准差方法（Mean-Standard Deviation Method）判别异常值


```python
# 这里我包装了一个异常值处理的代码，可以随便调用。
def outliers_proc_msdm(data, col_name, scale=5):
    """
    用于清洗异常值，默认用切比雪夫不等式进行数据清洗（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def msd_outliers(data_ser, scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        mean = data_ser.mean()
        std = data_ser.std()
        val_low = mean - scale*std
        val_up =  mean + scale*std
        
        rule_low = ( data_ser < val_low )
        rule_up  = ( data_ser > val_up)
        
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = msd_outliers(data_series, scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n
```

### 绝对中位差（Median Absolute Deviation）判别异常值


![image.png](attachment:image.png)


```python
# 这里我包装了一个异常值处理的代码，可以随便调用。
def outliers_proc_mad(data, col_name, scale=5):
    """
    用于清洗异常值，默认用切比雪夫不等式进行数据清洗（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def mad_outliers(data_ser, scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        md = data_ser.median()
        mad = abs(data_ser -md).median()*1.483
        
        
        val_low = md - scale*mad
        val_up =  md + scale*mad
        
        rule_low = ( data_ser < val_low )
        rule_up  = ( data_ser > val_up)
        
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = mad_outliers(data_series, scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n

```

### 直接采用截断法


```python
def mad_outliers(data_ser, scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    md = data_ser.median()
    mad = abs(data_ser -md).median()*1.483

    val_low = md - scale*mad
    val_up =  md + scale*mad

    return val_low, val_up

def box_plot_outliers(data_ser, box_scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    val_low = data_ser.quantile(0.25) - iqr
    val_up = data_ser.quantile(0.75) + iqr
    
    return val_low, val_up


def msd_outliers(data_ser, scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    mean = data_ser.mean()
    std = data_ser.std()
    val_low = mean - scale*std
    val_up =  mean + scale*std
    
    return  val_low, val_up

numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow','ficoRangeHigh','openAcc',  'revolBal', 
                'revolUtil', 'totalAcc', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']

for col in numberic_col:   
    mean_val   = merge_df[col].mean()
    median_val = merge_df[col].median()
    val_low, val_up = mad_outliers(merge_df[col], 3)
    merge_df[col+ str('_mad_mean')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    merge_df[col+ str('_mad_median')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)

    val_low, val_up = box_plot_outliers(merge_df[col], 3)
    merge_df[col+ str('_box_mean')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    merge_df[col+ str('_box_median')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    
    val_low, val_up = msd_outliers(merge_df[col], 3)
    merge_df[col+ str('_msd_mean')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    merge_df[col+ str('_msd_median')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)


print("截断法处理异值完成!!!")


```

### loanAmnt 贷款金额 异常处理


```python
merge_df.loanAmnt.hist()

```




    <matplotlib.axes._subplots.AxesSubplot at 0x16ef3550>




![png](output_24_1.png)


### annualIncome 年收入 异常处理


```python
merge_df['annualIncome_mean']   = merge_df['annualIncome']
merge_df['annualIncome_median'] = merge_df['annualIncome']
merge_df['annualIncome_mode']   = merge_df['annualIncome']

merge_df.loc[merge_df['annualIncome_mean']>100000, 'annualIncome_mean'] = merge_df.annualIncome_mean.mean()
merge_df.loc[merge_df['annualIncome_median']>100000, 'annualIncome_median'] = merge_df.annualIncome_median.median()
merge_df.loc[merge_df['annualIncome_mode']>100000,  'annualIncome_mode'] = merge_df.annualIncome_mode.mode()

```


```python
# 填充效果
f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(14, 7))
sns.boxplot(y=merge_df.annualIncome_mean, ax=ax1)
sns.boxplot(y=merge_df.annualIncome_median, ax=ax2)
sns.boxplot(y=merge_df.annualIncome_mode, ax=ax3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x150c4c88>




![png](output_27_1.png)


### installment	分期付款金额



```python
# merge_df.columns.to_list()
merge_df.installment.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15452e80>




![png](output_29_1.png)


### dti	债务收入比



```python
merge_df['dti_mean']   = merge_df['dti']
merge_df['dti_median'] = merge_df['dti']
merge_df['dti_mode']   = merge_df['dti']
merge_df['dti_new']   =  merge_df['dti']

merge_df.loc[merge_df['dti_mean']>40, 'dti_mean'] = merge_df.dti_mean.mean()
merge_df.loc[merge_df['dti_median']>40, 'dti_median'] = merge_df.dti_median.median()
merge_df.loc[merge_df['dti_mode']>40,  'dti_mode'] = merge_df.dti_mode.mode()
merge_df.loc[merge_df['dti_new']>40,  'dti_new'] = 

# 填充效果
f, ax = plt.subplots(1, 1, figsize=(12, 7))
sns.boxplot(y=merge_df.dti, ax=ax)


f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(14, 7))
sns.boxplot(y=merge_df.dti_mean, ax=ax1)
sns.boxplot(y=merge_df.dti_median, ax=ax2)
sns.boxplot(y=merge_df.dti_mode, ax=ax3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x67e98208>




![png](output_31_1.png)



![png](output_31_2.png)


## 类别特征异常值处理

**分类变量**

```python
categorical_col = ['grade', 'subGrade', 'employmentLength', 'employmentTitle', 'initialListStatus', 'term','verificationStatus','homeOwnership', 'applicationType', 'title', 'purpose','regionCode', 'postCode']

```

### 类别特征变量


```python
##分类变量中唯一变量个数
categorical_col = ['grade', 'subGrade', 'employmentLength', 'employmentTitle', 'initialListStatus', 'term', 'verificationStatus',
                   'homeOwnership', 'applicationType', 'title', 'purpose',
                   'regionCode', 'postCode']

print("类别变量个数{}\n".format(len(categorical_col)))

for col in categorical_col:
    print("{} 变量唯一值个数：{}".format(col, train_df[col].nunique()))
```

### 有序变量处理

- 有序变量为：'employmentLength','grade','subGrade' 三个变量


```python
mapping_dict = {  
    "employmentLength": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0
    },
    "grade":{
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7
    } ,
    "subGrade":{'A1':1.1, 'A2':1.2, 'A3':1.3, 'A4':1.4, 'A5':1.5, 'B1':2.1, 'B2':2.2, 'B3':2.3, 'B4':2.4, 'B5':2.5, 
                'C1':3.1, 'C2':3.2, 'C3':3.3, 'C4':3.4, 'C5':3.5, 'D1':4.1, 'D2':4.2, 'D3':4.3, 'D4':4.4, 'D5':4.5, 
                'E1':5.1, 'E2':5.2, 'E3':5.3, 'E4':5.4, 'E5':5.5, 'F1':6.1, 'F2':6.2, 'F3':6.3, 'F4':6.4, 'F5':6.5, 
                'G1':7.1, 'G2':7.2, 'G3':7.3, 'G4':7.4, 'G5':7.5}
}

# 进行有序变量替换
merge_df = merge_df.replace(mapping_dict) #变量映射

# 查看数据
merge_df[['grade','employmentLength','subGrade']].head()
```

### 无序变量处理

**无序变量**

| **字段英文名** | **字段中文名** | **变量唯一值** | **变量个数** | **计划处理方式** |  
| :---------------- | :--------------------|:-----------------|----------:|:--------|  
|initialListStatus|贷款的初始列表状态|变量唯一值个数|2|OneHot|
|term|贷款期限（year）|变量唯一值个数|2|OneHot|
|verificationStatus|验证状态|变量唯一值个数|3|OneHot|
|homeOwnership|借款人在登记时提供的房屋所有权状况|变量唯一值个数|6|OneHot|
|applicationType|表明贷款是个人申请还是与两个共同借款人的联合申请|变量唯一值个数|2|OneHot|
|employmentTitle|就业职称|变量唯一值个数|298102||
|title|借款人提供的贷款名称|变量唯一值个数|47904||
|purpose|借款人在贷款申请时的贷款用途类别|变量唯一值个数|14||
|regionCode|地区编码|变量唯一值个数|51||
|postCode|借款人在贷款申请中提供的邮政编码的前3位数字|变量唯一值个数|936||

####  无序变量OneHot处理


```python
categorical_onehot_col = ['initialListStatus', 'term', 'verificationStatus', 'homeOwnership', 'applicationType', 'regionCode', 'purpose']

## 对以上变量进行 OneHot 处理
dummy_df = pd.get_dummies(merge_df[categorical_onehot_col].astype('object'))
merge_df = pd.concat([merge_df, dummy_df], axis=1)

## 查看几列数据
merge_df.head()
```

####  无序变量进行LabelEncoder处理


```python
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

for col in tqdm(['employmentTitle', 'postCode', 'title', 'regionCode']):
    print(col)
    le = LabelEncoder()
    le.fit(list(merge_df[col].astype(str).values) + list(merge_df[col].astype(str).values))
    merge_df[col] = le.transform(list(merge_df[col].astype(str).values))

print('Label Encoding 完成')
```

      0%|                                                    | 0/4 [00:00<?, ?it/s]

    employmentTitle
    

     25%|███████████                                 | 1/4 [00:04<00:14,  4.83s/it]

    postCode
    

     50%|██████████████████████                      | 2/4 [00:08<00:08,  4.38s/it]

    title
    

     75%|█████████████████████████████████           | 3/4 [00:11<00:03,  3.94s/it]

    subGrade
    

    100%|████████████████████████████████████████████| 4/4 [00:12<00:00,  3.08s/it]

    Label Encoding 完成
    

    
    


```python
def class_anlysis(data, col):
    f, ax = plt.subplots(1, 1, figsize=(10,6))
    sum_df = data.groupby(['isDefault', col]).agg({'id':'count'}).reset_index().rename(columns={'id':'num'})
    sns.barplot(x=col, y='num', hue='isDefault', data=sum_df)
    
    ax.set_title("{} Frequency of each Class".format(col))
    ax.set_xticklabels(ax.get_xticks(), rotation=90)  #旋转坐标轴

    f.tight_layout( pad = 5)
    plt.show()

class_anlysis(merge_df, 'purpose')
    #print(col)
```


```python
def class_anlysis(data, col):
    f, ax = plt.subplots(1, 1, figsize=(10,6))
    sum_df = data.groupby(['isDefault', col]).agg({'id':'count'}).reset_index().rename(columns={'id':'num'})
    sns.barplot(x=col, y='num', hue='isDefault', data=sum_df)
    
    ax.set_title("{} Frequency of each Class".format(col))
    ax.set_xticklabels(ax.get_xticks(), rotation=90)  #旋转坐标轴

    f.tight_layout( pad = 5)
    plt.show()


class_anlysis(merge_df, 'title')
```

## 日期特征异常值处理

在所有特征中，日期变量特征为：
- issueDate 贷款月份变量，数据中其实为年月日类型，并不代表实际月份，后期可以进行衍生变量年、月两个变量
- earliesCreditLine 借款人最早报告的信用额度开立的月份,格式为：'Aug-2001'类型，在特征工程阶段可以衍生变量。      

### 日期特征变量


```python
# 衍生变量 年 月 变量
merge_df['issueDate_year'] = merge_df.issueDate.apply(lambda x:int(str(x)[0:4]))
merge_df['issueDate_month'] = merge_df.issueDate.apply(lambda x:int(str(x)[5:7]))

# 最早贷款日期变量
merge_df['earliesCreditLine_year'] = merge_df.earliesCreditLine.apply(lambda s: int(s[-4:]))

# 转化成时间格式
merge_df['issueDate'] = pd.to_datetime(merge_df['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    #构造时间特征
merge_df['issueDateDT'] = merge_df['issueDate'].apply(lambda x: x-startdate).dt.days
    
    
merge_df.drop(['issueDate','earliesCreditLine'], axis=1, inplace=True)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-093f77c83811> in <module>
          7 
          8 # 转化成时间格式
    ----> 9 for data in [data_train, data_test_a]:
         10     data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
         11     startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    

    NameError: name 'data_train' is not defined


# 特征构造

## 数值变量特征构造



```python
numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow','ficoRangeHigh','openAcc',  'revolBal', 
                'revolUtil', 'totalAcc', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']

print("数值变量分布情况:\n",merge_df[numberic_col].describe().T)
```

    数值变量分布情况:
                        count     mean      std    min      25%      50%      75%  \
    loanAmnt      1000000.00      nan      nan 500.00  8000.00 12000.00 20000.00   
    interestRate  1000000.00      nan     0.00   5.31     9.75    12.74    15.99   
    installment   1000000.00      nan      nan  14.01   248.50   375.50   580.50   
    annualIncome  1000000.00 76197.47 70776.45   0.00 45671.03 65000.00 90000.00   
    dti           1000000.00      nan      nan  -1.00    11.80    17.62    24.06   
    ficoRangeLow  1000000.00      nan      nan 625.00   670.00   690.00   710.00   
    ficoRangeHigh 1000000.00      nan      nan 629.00   674.00   694.00   714.00   
    openAcc       1000000.00      nan     0.00   0.00     8.00    11.00    14.00   
    revolBal      1000000.00 16234.13 22452.57   0.00  5943.00 11133.00 19743.00   
    revolUtil     1000000.00      nan      nan   0.00    33.50    52.19    70.69   
    totalAcc      1000000.00      nan     0.00   2.00    16.00    23.00    32.00   
    n1            1000000.00      nan     0.00   0.00     2.00     3.00     5.00   
    n2            1000000.00      nan     0.00   0.00     3.00     5.00     7.00   
    n3            1000000.00      nan     0.00   0.00     3.00     5.00     7.00   
    n4            1000000.00      nan     0.00   0.00     3.00     4.00     6.00   
    n5            1000000.00      nan     0.00   0.00     5.00     7.00    10.00   
    n6            1000000.00      nan     0.00   0.00     4.00     7.00    11.00   
    n7            1000000.00      nan     0.00   0.00     5.00     7.00    10.00   
    n8            1000000.00      nan     0.00   1.00     9.00    13.00    18.00   
    n9            1000000.00      nan     0.00   0.00     3.00     5.00     7.00   
    n10           1000000.00      nan     0.00   0.00     8.00    11.00    14.00   
    n11           1000000.00     0.00     0.00   0.00     0.00     0.00     0.00   
    n14           1000000.00      nan     0.00   0.00     1.00     2.00     3.00   
    
                          max  
    loanAmnt         40000.00  
    interestRate        30.98  
    installment       1715.00  
    annualIncome  10999200.00  
    dti                999.00  
    ficoRangeLow       845.00  
    ficoRangeHigh      850.00  
    openAcc             90.00  
    revolBal       2904836.00  
    revolUtil          892.50  
    totalAcc           162.00  
    n1                  33.00  
    n2                  63.00  
    n3                  63.00  
    n4                  63.00  
    n5                  70.00  
    n6                 132.00  
    n7                  83.00  
    n8                 128.00  
    n9                  45.00  
    n10                 90.00  
    n11                  4.00  
    n14                 30.00  
    

## 类别变量特征构造


```python

```


```python

```

## 日期变量特征构造


```python

```

## 数值+类别变量特征构造


```python

```


```python

```

# 特征选择

特征选择技术可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度。特征选择不是为了减少训练时间（实际上，一些技术会增加总体训练时间），而是为了减少模型评分时间。

特征选择的方法：

- 1 Filter
  - 方差选择法
  - 相关系数法（pearson 相关系数）
  - 卡方检验
  - 互信息法
- 2 Wrapper （RFE）
  - 递归特征消除法
- 3 Embedded
  - 基于惩罚项的特征选择法
  - 基于树模型的特征选择
  


```python

```

# 特征工程整合处理过程


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-white')           #风格设置近似R这种的ggplot库
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  #字体乱码
plt.rcParams['font.family']='sans-serif'               #字体乱码  
plt.rcParams['axes.unicode_minus'] = False            #数字为负
# 设置精度
pd.set_option('display.float_format', lambda x: '%.2f'%x)
pd.set_option('max_columns', 1000) #设置最大列数
pd.set_option('max_row', 300)      #设置最大行数

%matplotlib inline

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem/1024/1024))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem/1024/1024))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

## 1、数据加载
train_df = reduce_mem_usage(pd.read_csv('../data/train.csv'))
test_df  = reduce_mem_usage(pd.read_csv('../data/testA.csv'))

## 删除唯一值变量=policyCode
# del train_df['policyCode']
# del test_df['policyCode']

drop_list = ['policyCode', 'delinquency_2years', 'pubRec', 'pubRecBankruptcies', 'n0', 'n12', 'n13']
train_df.drop(drop_list, axis=1, inplace = True)
test_df.drop(drop_list, axis=1, inplace = True)

# 添加打印标识
train_df['isFlag'] = 1
test_df['isFlag']  = 0

merge_df = pd.concat([train_df, test_df], ignore_index=True)

## 2、数据空值处理
# 这种方法并没有把 NaN的值处理掉
numberic_col = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti', 'ficoRangeLow', 'ficoRangeHigh',
                'openAcc',  'revolBal', 'revolUtil', 'totalAcc', 
                'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n14']

# 数字变量填充中值
for col in numberic_col:
    # print('{}列中值是:{}'.format(col,merge_df[col].median()))
    merge_df[col] = merge_df[col].fillna(merge_df[col].median())



category_col = ['grade', 'subGrade', 'employmentLength', 'employmentTitle', 'initialListStatus', 'term', 'verificationStatus',
                'homeOwnership', 'applicationType', 'title', 'purpose', 'regionCode', 'postCode', 'issueDate', 'earliesCreditLine']

merge_df[category_col] = merge_df[category_col].astype('object') #category变量无法填充数字
# 类别变量填充众数
for col in category_col:
    # print('{} 列众数是:{}'.format(col,merge_df[col].mode().values[0]))
    merge_df[col] = merge_df[col].fillna(merge_df[col].mode().values[0])

    
    
    
## 3、类别变量处理 employmentLength、grade、subGrade
mapping_dict = {  
    "employmentLength": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0
    },
    "grade":{
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7
    } ,
    "subGrade":{'A1':1.1, 'A2':1.2, 'A3':1.3, 'A4':1.4, 'A5':1.5, 'B1':2.1, 'B2':2.2, 'B3':2.3, 'B4':2.4, 'B5':2.5, 
                'C1':3.1, 'C2':3.2, 'C3':3.3, 'C4':3.4, 'C5':3.5, 'D1':4.1, 'D2':4.2, 'D3':4.3, 'D4':4.4, 'D5':4.5, 
                'E1':5.1, 'E2':5.2, 'E3':5.3, 'E4':5.4, 'E5':5.5, 'F1':6.1, 'F2':6.2, 'F3':6.3, 'F4':6.4, 'F5':6.5, 
                'G1':7.1, 'G2':7.2, 'G3':7.3, 'G4':7.4, 'G5':7.5}
}

# 低维度有序变量处理
merge_df = merge_df.replace(mapping_dict) #变量映射

# 低维度有序变量处理 OneHot 变量
onehot_col = ['initialListStatus', 'term', 'verificationStatus', 
              'homeOwnership', 'applicationType', 'regionCode', 'purpose']

## 对以上变量进行 OneHot 处理
dummy_df = pd.get_dummies(merge_df[onehot_col].astype('object'))
merge_df = pd.concat([merge_df, dummy_df], axis=1)

del onehot_col
del dummy_df

# 高维无序变量处理
for col in tqdm(['employmentTitle', 'postCode', 'title']):
    print(col)
    le = LabelEncoder()
    le.fit(list(merge_df[col].astype(str).values) + list(merge_df[col].astype(str).values))
    merge_df[col] = le.transform(list(merge_df[col].astype(str).values))

print('Label Encoding 完成')

## 4、日期变量处理
### 4.1 衍生年月变量
merge_df['issueDate_year'] = merge_df.issueDate.apply(lambda x:int(str(x)[0:4]))
merge_df['issueDate_month'] = merge_df.issueDate.apply(lambda x:int(str(x)[5:7]))
merge_df['earliesCreditLine_year'] = merge_df.earliesCreditLine.apply(lambda s: int(s[-4:]))

### 4.2 截止到某一天的天数
merge_df['issueDate'] = pd.to_datetime(merge_df['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
#构造时间特征
merge_df['issueDateDT'] = merge_df['issueDate'].apply(lambda x:x-startdate).dt.days

# 删除这两个变量
del merge_df['issueDate_year']
del merge_df['earliesCreditLine_year']

## 5、数值变量异常值处理与特征衍生
def mad_outliers(data_ser, scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    md = data_ser.median()
    mad = abs(data_ser -md).median()*1.483

    val_low = md - scale*mad
    val_up =  md + scale*mad

    return val_low, val_up

def box_plot_outliers(data_ser, box_scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    val_low = data_ser.quantile(0.25) - iqr
    val_up = data_ser.quantile(0.75) + iqr
    
    return val_low, val_up

def msd_outliers(data_ser, scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，
    :return:
    """
    mean = data_ser.mean()
    std = data_ser.std()
    val_low = mean - scale*std
    val_up =  mean + scale*std
    
    return  val_low, val_up

# 首先要判断该列有异常值，否则衍生出异常值列就有问题
for col in numberic_col:   
    mean_val   = merge_df[col].mean()
    median_val = merge_df[col].median()
    val_low, val_up = mad_outliers(merge_df[col], 3)
    merge_df[col+ str('_mad_mean')]  = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    merge_df[col+ str('_mad_median')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)

    val_low, val_up = box_plot_outliers(merge_df[col], 3)
    merge_df[col+ str('_box_mean')]   = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    merge_df[col+ str('_box_median')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    
    val_low, val_up = msd_outliers(merge_df[col], 3)
    merge_df[col+ str('_msd_mean')]   = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)
    merge_df[col+ str('_msd_median')] = merge_df[col].apply(lambda x: mean_val if x<val_low or x>val_up else x)


## 数据保存
merge_df.loc[merge_df['isFlag']==1, merge_df.columns.difference(['isFlag'])].to_csv('../tmp/train_tmp.csv', index=False)
merge_df.loc[merge_df['isFlag']==1, merge_df.columns.difference(['isFlag'])].to_csv('../tmp/testA_tmp.csv', index=False)

print("数据保存完成!!!")t
```

    Memory usage of dataframe is 286.87 MB
    Memory usage after optimization is: 69.46 MB
    Decreased by 75.8%
    Memory usage of dataframe is 70.19 MB
    Memory usage after optimization is: 17.20 MB
    Decreased by 75.5%
    

      0%|                                                    | 0/3 [00:00<?, ?it/s]

    employmentTitle
    

     33%|██████████████▋                             | 1/3 [00:04<00:09,  4.82s/it]

    postCode
    

     67%|█████████████████████████████▎              | 2/3 [00:08<00:04,  4.44s/it]

    title
    

    100%|████████████████████████████████████████████| 3/3 [00:11<00:00,  3.81s/it]
    

    Label Encoding 完成
    数据保存完成!!!
    


```python
for col in numberic_col:   
    mean_val   = merge_df[col].mean()
    median_val = merge_df[col].median()
    val_low, val_up = mad_outliers(merge_df[col], 3)
    outlier_num =  merge_df.loc[(merge_df[col]<val_low) | (merge_df[col]>val_up)].shape[0]
    #print("{} 变量判断边界为低于：{} 或高于：{}".format(col, val_low, val_up))
    print("{} 列异常值数量为：{} 异常值占比:{}".format(col,  outlier_num, outlier_num/merge_df.shape[0]))

```

    loanAmnt 列异常值数量为：5201 异常值占比:0.005201
    interestRate 列异常值数量为：9027 异常值占比:0.009027
    installment 列异常值数量为：32593 异常值占比:0.032593
    annualIncome 列异常值数量为：48003 异常值占比:0.048003
    dti 列异常值数量为：3284 异常值占比:0.003284
    delinquency_2years 列异常值数量为：192816 异常值占比:0.192816
    ficoRangeLow 列异常值数量为：28527 异常值占比:0.028527
    ficoRangeHigh 列异常值数量为：28527 异常值占比:0.028527
    openAcc 列异常值数量为：27293 异常值占比:0.027293
    pubRec 列异常值数量为：169142 异常值占比:0.169142
    pubRecBankruptcies 列异常值数量为：124585 异常值占比:0.124585
    revolBal 列异常值数量为：66699 异常值占比:0.066699
    revolUtil 列异常值数量为：31 异常值占比:3.1e-05
    totalAcc 列异常值数量为：13257 异常值占比:0.013257
    n0 列异常值数量为：225212 异常值占比:0.225212
    n1 列异常值数量为：55919 异常值占比:0.055919
    n2 列异常值数量为：26875 异常值占比:0.026875
    n3 列异常值数量为：26875 异常值占比:0.026875
    n4 列异常值数量为：20183 异常值占比:0.020183
    n5 列异常值数量为：20200 异常值占比:0.0202
    n6 列异常值数量为：38939 异常值占比:0.038939
    n7 列异常值数量为：67350 异常值占比:0.06735
    n8 列异常值数量为：20475 异常值占比:0.020475
    n9 列异常值数量为：24348 异常值占比:0.024348
    n10 列异常值数量为：26606 异常值占比:0.026606
    n11 列异常值数量为：710 异常值占比:0.00071
    n12 列异常值数量为：3057 异常值占比:0.003057
    n13 列异常值数量为：54488 异常值占比:0.054488
    n14 列异常值数量为：25633 异常值占比:0.025633
    

**测试数据集与训练数据集分布情况**


```python
# 绘图行列情况
ncols = 2
nrows = int(len(numberic_col)/ncols+0.5)

# 绘制连续变量直方图
f, ax = plt.subplots(nrows, ncols, figsize=(14, nrows*3))
for i in range(nrows):
    j= i*2
    if j+1 <= len(numberic_col) -1:
        sns.distplot(merge_df.loc[merge_df.isFlag==0, numberic_col[j]], hist=False,label='测试集' , ax=ax[i,0])
        sns.distplot(merge_df.loc[merge_df.isFlag==1, numberic_col[j]], hist=False, label='训练集', ax=ax[i,0])
        ax[i,0].set_title("%s 变量分析" % str(numberic_col[j]),fontsize=20)
        
        sns.distplot(merge_df.loc[merge_df.isFlag==0, numberic_col[j+1]], hist=False, ax=ax[i,1], label='测试集')
        sns.distplot(merge_df.loc[merge_df.isFlag==1, numberic_col[j+1]], hist=False, ax=ax[i,1], label='训练集')
        ax[i,1].set_title("%s 变量分析" % str(numberic_col[j+1]),fontsize=20)
    else:
        sns.distplot(merge_df.loc[merge_df.isFlag==0, numberic_col[j]], hist=False, ax=ax[i,0], label='测试集')
        sns.distplot(merge_df.loc[merge_df.isFlag==1, numberic_col[j]], hist=False, ax=ax[i,0], label='训练集')
        ax[i,0].set_title("%s 变量分析" % str(numberic_col[j]),fontsize=20)
f.tight_layout()

plt.show()
```


![png](output_60_0.png)


# -----------分割线---------------------------------------------
