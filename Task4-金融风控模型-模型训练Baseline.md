<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"></ul></div>

<center><font size=6.5>Task4-金融风控模型-模型训练Baseline</font></center>



通过Baseline模型作为模型的基准



```python
# -*- coding: utf-8 -*-
## 加载软件包
import pandas   as pd
import lightgbm as lgb
import xgboost  as xgb
import pandas as pd
import numpy as np
import math
import os
import sys
import joblib
import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from  sklearn import metrics


pd.set_option('display.float_format',lambda x : '%.4f'%x)

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

## 减少内存函数
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df



## 定义lgb模型
def build_model_lgb(x_train, y_train):
    import lightgbm as lgb

#     lgm = lgb.LGBMClassifier(boosting_type='gbdt',
#                              objective= 'binary',
#                              learning_rate= 0.1,
#                              metric= 'auc',        
#                              min_child_weight= 1e-3,
#                              num_leaves= 31,
#                              max_depth= -1,
#                              reg_lambda= 0,
#                              reg_alpha= 0,
#                              feature_fraction= 1,
#                              bagging_fraction= 1,
#                              bagging_freq= 0,
#                              seed= 2020,
#                              nthread= 8,
#                              silent= True,
#                              verbose= -1
#                             )
    lgm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective= 'binary',
                             learning_rate= 0.05,
                             metric= 'auc',        
                             min_child_weight= 1.6,
                             num_leaves= 31,
                             max_depth= -1,
                             reg_lambda= 9,
                             reg_alpha= 7,
                             feature_fraction= 0.69,
                             bagging_fraction= 0.98,
                             bagging_freq= 96,
                             seed= 2020,
                             nthread= 8,
                             silent= True,
                             verbose= -1
                            )
    lgm.fit(x_train, y_train)
    return lgm

if __name__ == "__main__":
    
    # now_date = datetime.datetime.now().strftime('%Y-%m-%d')
    Train_data = reduce_mem_usage(pd.read_csv('../tmp/train_tmp.csv'))
    Test_data  = reduce_mem_usage(pd.read_csv('../tmp/testA_tmp.csv'))

#     Train_data = reduce_mem_usage(pd.read_csv('../data/train.csv'))
#     Test_data  = reduce_mem_usage(pd.read_csv('../data/testA.csv'))

    select_col = [col for col in Train_data.columns.tolist() if col not in ['id','issueDate','isDefault','earliesCreditLine']]


    Train_X = Train_data[select_col]
    Train_y = Train_data['isDefault'].astype(int)  
    Test_X  = Test_data[select_col]

    print('X train shape:', Train_X.shape)
    print('X test shape:',  Test_X.shape)
    print('Y train shape:', Train_y.shape)

    # 数据切分
    x_train, x_val, y_train, y_val = train_test_split(Train_X, Train_y, test_size=0.3, random_state=123)

    ## lgb模型训练结果
    print('Train Data lgb...')
    model_lgb = build_model_lgb(x_train, y_train)
    val_lgb   = model_lgb.predict(x_val)
    auc_lgb = roc_auc_score(y_val, val_lgb)
    print('MAE of val with lgb:',auc_lgb)

    
#     ## 保存joblib
#     joblib.dump(model_lgb_pre,'../tmp/model_lgb_pre_{}.pkl'.format(now_date))
#     ## 保存预测值
#     lgb_model = pd.DataFrame()
#     lgb_model['price']  = np.exp(subA_lgb)
#     lgb_model.to_csv('../tmp/lgb_model_used_car_submit_{}.csv'.format(now_date), index=False)
    
#     # 保存权重数据
#     with open('../tmp/MAE_lgb_{}.txt'.format(now_date), 'w+') as f:
#         f.write(str(MAE_lgb))
    
    
    print("LGB模型训练完成，数据保存完成!!!")

    """
    1、20200606：使用该模型训练集成绩为514分、测试集提交结果为499.8105,比上个模型提高了18分（518.7181）
    
    """
```

    Memory usage of dataframe is 1664000128.00 MB
    Memory usage after optimization is: 363232600.00 MB
    Decreased by 78.2%
    Memory usage of dataframe is 1664000128.00 MB
    Memory usage after optimization is: 363232600.00 MB
    Decreased by 78.2%
    X train shape: (800000, 256)
    X test shape: (800000, 256)
    Y train shape: (800000,)
    Train Data lgb...
    MAE of val with lgb: 0.5328576522318256
    LGB模型训练完成，数据保存完成!!!
    


```python
MAE of val with lgb: 0.5428968107544275
```


```python
import matplotlib.pyplot as plt
from  sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_val, val_lgb)
roc_auc = metrics.auc(fpr, tpr)
print('调参后lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
"""画出roc曲线图"""
plt.figure(figsize=(8, 8))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.4f' % roc_auc)
plt.ylim(0,1)
plt.xlim(0,1)
plt.legend(loc='best')
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# 画出对角线
plt.plot([0,1],[0,1],'r--')
plt.show()
```

    调参后lightgbm单模型在验证集上的AUC：0.5330402608396811
    


![png](output_4_1.png)



```python
# -*- coding: utf-8 -*-
## 加载软件包
import pandas   as pd
import lightgbm as lgb
import xgboost  as xgb
import pandas as pd
import numpy as np
import math
import os
import sys
import joblib
import datetime

pd.set_option('display.float_format',lambda x : '%.4f'%x)
pd.set_option('max_columns', 1000) #设置最大列数
pd.set_option('max_row', 300)      #设置最大行数


from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

## 减少内存函数
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
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
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# now_date = datetime.datetime.now().strftime('%Y-%m-%d')
# Train_data = reduce_mem_usage(pd.read_csv('../tmp/train_tmp.csv'))
# Test_data  = reduce_mem_usage(pd.read_csv('../tmp/testA_tmp.csv'))



select_col = [col for col in Train_data.columns.tolist() if col not in ['id','issueDate','isDefault','earliesCreditLine']]


Train_X = Train_data[select_col]
Train_y = Train_data['isDefault'].astype(int)  
Test_X  = Test_data[select_col]

print('X train shape:', Train_X.shape)
print('X test shape:',  Test_X.shape)
print('Y train shape:', Train_y.shape)

# 数据切分
x_train, x_val, y_train, y_val = train_test_split(Train_X, Train_y, test_size=0.3, random_state=123)

#isDefault

print("执行完成!!!")
```

    Memory usage of dataframe is 780800128.00 MB
    Memory usage after optimization is: 129632600.00 MB
    Decreased by 83.4%
    Memory usage of dataframe is 780800128.00 MB
    Memory usage after optimization is: 129632600.00 MB
    Decreased by 83.4%
    


```python
select_col = [col for col in Train_data.columns.tolist() if col not in ['id','issueDate','isDefault','earliesCreditLine']]


Train_X = Train_data[select_col]
Train_y = Train_data['isDefault'].astype(int)  
Test_X  = Test_data[select_col]

print('X train shape:', Train_X.shape)
print('X test shape:',  Test_X.shape)
print('Y train shape:', Train_y.shape)
```

    X train shape: (800000, 118)
    X test shape: (800000, 118)
    Y train shape: (800000,)
    
