import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from pyecharts.charts import Line,Bar,Tab
from pyecharts import options as opts
import matplotlib.pyplot as plt

data = pd.read_csv('/home/mw/input/data3431/brain_stroke.csv',encoding='gbk')
data.head()

data.info()
#无缺失值
data[data.duplicated()]
#无重复值

data['性别'] = data['性别'].apply(lambda x:1 if x=='男性' else 0)#将性别进行映射，男性为1
data['BMI_'] = data['BMI'].apply(lambda x:'体重过轻' if x<18.5 else '体重正常' if x<23.9 else '超重' if x<27.9 else '肥胖')
data['血糖水平_'] = data['血糖水平'].apply(lambda x:'低血糖' if x<70 else '正常' if x<100 else '糖尿病')
data['年龄_'] = pd.cut(data['年龄'],bins=5)
data.describe()


data[data['年龄'].apply(lambda x:str(x)[-2:]!='.0')].shape[0]/data.shape[0]
#占比较小，直接删除

data = data[data['年龄'].apply(lambda x:str(x)[-2:]=='.0')]
data.describe()


#中风患者EDA
data_zf = data[data['是否中风']==1]
#接下来的数据分析及可视化只对 中风患者 进行探索


def getBarLine(name):
    r = pd.DataFrame((data_zf[name].value_counts()/data_zf.shape[0]*100).round(2))
    c = pd.DataFrame(data_zf[name].value_counts())
    temp = pd.concat([c,r],axis=1)
    temp.columns = ['count','rate']
    bar = (
        Bar()
        .add_xaxis(temp.index.astype('str').tolist())
        .add_yaxis("人数", temp['count'].tolist(),z=1)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="不同{} 中风人数及占比图".format(name)),
            tooltip_opts=opts.TooltipOpts(
                is_show=True, trigger="axis", axis_pointer_type="cross"
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
            )
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="中风占比",type_="value",
            ),
        )
    )
    line = (
        Line()
        .add_xaxis(temp.index.astype('str').tolist())
        .add_yaxis(
            "中风占比", temp['rate'].tolist(),yaxis_index=1,z=2
        )
    )
    return bar.overlap(line)

#占比图输出
col_list = data_zf.columns.tolist()
col_list = set(col_list).difference(['年龄','BMI','血糖水平','是否中风'])#对没有映射的字段剔除
tab1 = Tab()
for col in col_list:
    tab1.add(getBarLine(col),col)
tab1.render_notebook()





'''
#特征工程
'''
X = data.iloc[:,[5,6,9,0,1,2,3,4,7,8]]#调整字段顺序
X.iloc[:,:3] = OrdinalEncoder().fit_transform(X.iloc[:,:3])
Y = data['是否中风']


#特征选择
from sklearn.feature_selection import chi2
chivalue, p = chi2(X,Y)
print('其中p值大于0.05的有：',X.columns[p>0.05].tolist())
[*zip(X.columns,p)]

del X['住宅类型']
del X['性别']
X.head()


#上采样
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=1) #实例化
X,Y = sm.fit_resample(X,Y)#返回已经上采样完毕过后的特征矩阵和标签
n_sample_ = X.shape[0]
n_1_sample = pd.Series(Y).value_counts()[1]
n_0_sample = pd.Series(Y).value_counts()[0]
print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample_,n_1_sample/n_sample_,n_0_sample/n_sample_))


#划分训练集
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3,random_state=1)


'''
模型预测
'''
#决策树
model_list = []
from sklearn.tree import DecisionTreeClassifier as DTC
tr = []
te = []
for i in range(20):
    clf = DTC(random_state=1,max_depth=i+1)
    clf = clf.fit(xtrain,ytrain)
    score_tr = clf.score(xtrain,ytrain)
    score_te = cross_val_score(clf,xtest,ytest,cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1,21),tr,color='red',label='train')
plt.plot(range(1,21),te,color='blue',label='test')
plt.xticks(range(1,21))
plt.legend()
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.show()
model_list.append(max(te))

#随机森林
from sklearn.ensemble import RandomForestClassifier as RFC
tr = []
te = []
for i in range(20):
    rfc = RFC(random_state=2,max_depth=i+1,n_estimators=100)
    rfc = rfc.fit(xtrain,ytrain)
    score_tr = rfc.score(xtrain,ytrain)
    score_te = cross_val_score(rfc,xtest,ytest,cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1,21),tr,color='red',label='train')
plt.plot(range(1,21),te,color='blue',label='test')
plt.xticks(range(1,21))
plt.legend()
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.show()
model_list.append(max(te))


#LGB
from lightgbm import LGBMClassifier as LGB
from sklearn.metrics import accuracy_score
te = []
for i in np.arange(10,210,10):
    lgb = LGB(random_state=1,n_estimators=i)
    lgb.fit(xtrain,ytrain)
    ypred = lgb.predict(xtest)
    te.append(accuracy_score(ytest, ypred))
print(max(te))
plt.plot(np.arange(10,210,10),te,color='blue')
plt.xticks(np.arange(10,210,10))
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.show()

lgb = LGB(random_state=1,n_estimators=137,learning_rate=0.35556)
lgb.fit(xtrain,ytrain)
ypred = lgb.predict(xtest)
print(accuracy_score(ytest, ypred))
model_list.append(accuracy_score(ytest, ypred))



'''
模型对比
'''
model_list = pd.DataFrame(model_list,['决策树','随机森林','LGB'])
model_list = model_list.reset_index()
model_list.columns = ['model','accuracy']
bar = (
    Bar()
    .add_xaxis(model_list['model'].tolist())
    .add_yaxis("准确度", (model_list['accuracy']*100).round(3).tolist(),z=1)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="不同模型准确度对比图"),
        tooltip_opts=opts.TooltipOpts(
            is_show=True, trigger="axis", axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
        )
        ,yaxis_opts=opts.AxisOpts(
            min_=90
        )
    )
)
bar.render_notebook()

#因素重要性展示
pd.DataFrame([*zip(X.columns,lgb.feature_importances_)]
,columns=['col','feature_importances_']).sort_values(
    'feature_importances_',ascending=False)


