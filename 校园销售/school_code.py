# 导入库

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')  # 更改绘图风格 R语言绘图库的风格
plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus']= False


#导入数据
df = pd.read_excel('/home/mw/input/userconsumption5607/order2021.xlsx')
# 数据预览
df.head()
#数据信息
df.info()
# 一共10万多条数据 显示渠道编号有缺失值
#描述统计分析
df.describe()
#1.显示付款金额的最小值为-12 有异常值
#2.订单金额 大多数订单消费金额在600-1300

#对数据进行去文本空格处理
df.columns = df.columns.str.strip()
df.columns

#查看数据集是否存在重复值
#两条记录中所有的数据都相等时duplicate才会判为重复值
df.duplicated().sum()

#缺失值处理
df.isnull().sum()
#显示渠道编号有8个缺失值，选择删除

#删除渠道编号缺失的数据
#检查是否删除成功
df.dropna(inplace=True)
df.isnull().sum()

#异常值处理
#查找付款金额小于0的数据
abnormal=df[np.where(df['付款金额'] < 0,True,False)]
abnormal


#提取异常值的索引并删除
abindex = abnormal.index
print(abindex)
#删除付款金额小于0的数据
df.drop(abindex,inplace=True)
#查看是否删除成功
print(df.shape)


#新增一列订单日期
df['订单日期'] = pd.to_datetime(df['付款时间'],format='%Y-%m-%d').dt.date
#将订单日期转化为精度为月份的数据列
df['月份'] = df['订单日期'].astype('datetime64[M]') #[M]控制后转换的精度  只能看到月份 看不到具体哪天
df.head()

#筛选没有退款的订单
df1 = df[np.where(df['是否退款']=='否',True,False)]
df1['订单日期'] = df1['订单日期'].astype('datetime64[D]')


'''
画出图像
'''
# 按月份统计产品购买数量，消费金额，消费次数，消费人数
plt.figure(figsize=(20,15),dpi=120)
# 每月产品购买数量
plt.subplot(221)  #两行两列 占据第一个位置
df1.groupby(by='月份')['商品编号'].count().plot()  # 默认折线图
plt.title('每月的产品购买数量')

# 每月产品消费金额
plt.subplot(222)  #两行两列 占据第二个位置
df1.groupby(by='月份')['付款金额'].sum().plot()  # 默认折线图
plt.title('每月的产品消费金额')

# 每月产品消费次数   这里由于每笔订单仅一个商品 故购买数量和消费次数是一样的结果
plt.subplot(223)  #两行两列 占据第三个位置
df1.groupby(by='月份')['用户名'].count().plot()  # 默认折线图
plt.title('每月的产品消费次数')

# 每月产品消费人数  （根据用户名去重）
plt.subplot(224)  #两行两列 占据第四个位置
df1.groupby(by='月份')['用户名'].apply(lambda x:len(x.drop_duplicates())).plot()  # 默认折线图
plt.title('每月的产品消费人数')


# 分析结果
# 前四个月销量较低，从第四个月开始销量迅速攀升，7-10月销量轻微下降，10月后继续上升，总体呈现上升趋势
# 四个图的比例均呈相同的变化趋势
# 可能是从4月开始 商家加大了促销力度  或是由于产品属性的原因 和季节相关




#用户个体的消费分析
user_grouped = df1.groupby(by='用户名').sum()
print(user_grouped.describe())
print('用户数量：',len(user_grouped))
# 总共71253名用户
# 每月用户平均消费金额为1487，中位数805，3/4分位数为1779，结合分位数和最大值、平均值来看，为典型的右偏分布，说明存在小部分用户（后25%）
# 的高额消费

# 绘制产品的购买量与消费金额的散点图
plt.figure(figsize=(8,4),dpi=120)



#用户消费分布图(输出对应图像)
#用直方图和核密度看下‘订单金额’的数据分布
plt.figure(figsize=(12,6),dpi=120)
plt.hist(df1['订单金额'],bins=60,
         color='g',label='订单金额',
         histtype='bar',density=True,
         edgecolor = 'white',alpha=0.4)
plt.title('订单金额分布',fontdict={'fontsize':16,'color':'r'})
plt.xlabel('订单金额')
plt.ylabel('密度')
plt.legend(loc = 'best')

data1=df1['订单金额']
data1.plot(kind='kde',color='b')

print('偏度',data1.skew())
print('峰度',data1.kurtosis())


#用直方图和核密度看下‘付款金额’的数据分布
plt.figure(figsize=(12,6),dpi=120)
plt.hist(df1['付款金额'],bins=60,
         color='b',label='付款金额',
         histtype='bar',density=True,
         edgecolor = 'white',alpha=0.4)
plt.title('付款金额分布',fontdict={'fontsize':16,'color':'r'})
plt.xlabel('付款金额')
plt.ylabel('密度')
plt.legend(loc = 'best')

data2=df1['付款金额']
data2.plot(kind='kde',color='g')

print('偏度',data2.skew())
print('峰度',data2.kurtosis())



#用户累计消费金额占比分析（用户的贡献度）
# 进行用户分组，取出消费金额，进行求和，排序，重置索引
user_cumsum = df1.groupby('用户名')['付款金额'].sum().sort_values().reset_index()
user_cumsum
# 每个用户消费金额累加 cumsum函数
user_cumsum['付款金额累加'] = user_cumsum['付款金额'].cumsum()
user_cumsum.tail()

amount_total = user_cumsum['付款金额累加'].max()  # 消费金额总值
user_cumsum['prop'] = user_cumsum.apply(lambda x:x['付款金额累加']/amount_total,axis=1) #前xx名用户的总贡献率
user_cumsum.tail()

plt.figure(figsize=(8,4),dpi=120)
user_cumsum['prop'].plot()

# 前6万名用户贡献了一半的金额 而后面一万多名贡献了剩下一半的金额  符合二八原则




#用户消费行为
# 用户分组，取最小值 即为首购时间
plt.figure(figsize=(8,4),dpi=120)
df1.groupby(by='用户名')['订单日期'].min().value_counts().plot()
# 首次购买的用户量子从3月开始呈上升趋势，在7月开始逐步回落  猜测：公司产品的推广力度或价格的改变

#最后一次购买时间
plt.figure(figsize=(8,4),dpi=120)
df1.groupby(by='用户名')['订单日期'].max().value_counts().plot()

# 随着时间的推移，用户的最后一次购买的数量逐渐增多，说明忠诚客户逐渐增多


#用户分层
# 构建数据透视表
# 透视表的使用(index：相当于groupby分组，values：取出的数据列，aggfunc：key值必须存在于values中 且必须跟随有效的聚合函数)
rfm = df1.pivot_table(index='用户名',
                    values=['订单日期','订单号','付款金额'],
                    aggfunc={
                        '订单日期':'max',  # 最后一次购买
                        '订单号':'count',  # 总订单数
                        '付款金额':'sum'   # 消费总金额
,                    })
rfm.head()


rfm['R'] = -(rfm['订单日期'] - rfm['订单日期'].max())/np.timedelta64(1,'D')  #取相差的天数  小数点为0 精确到天
rfm.rename(columns={'订单号':'F','付款金额':'M'},inplace=True)
rfm.head()


# RFM计算方式 ：每一列数据减去数据所在列的平均值（有正有负），根据结果值与1作比较，如果>=1，设置为1，否则为0
#rfm['R'] - rfm['R'].mean()
def rfm_func(x):  #x代表每一列数据
    level = x.apply(lambda x:'1' if x>=1 else '0')
    label = level['R'] + level['F'] + level['M']  # 举例：1 0 0    1 1 1
    d = {
        '111':'重要价值客户',
        '011':'重要保持客户',
        '101':'重要发展客户',
        '001':'重要挽留客户',
        '110':'一般价值客户',
        '010':'一般保持客户',
        '100':'一般发展客户',
        '000':'一般挽留客户'
    }
    result = d[label]
    return result
rfm['label'] = rfm[['R','F','M']].apply(lambda x:x-x.mean()).apply(rfm_func,axis=1) # axis=1 向右逐步取列
rfm.head()


# 客户分层可视化
plt.figure(figsize=(8,4),dpi=120)
for label,grouped in rfm.groupby(by='label'):
    x= grouped['F']
    y= grouped['M']
    plt.scatter(x,y,label=label)
plt.legend()
plt.xlabel('F')
plt.ylabel('M')


pivoted_counts = df1.pivot_table(
index='用户名',
columns='月份',
values='订单日期',
aggfunc='count').fillna(0)
pivoted_counts


# 浮点数转化为0  1  1代表有过消费记录 0代表没有
df_purchase = pivoted_counts.applymap(lambda x:1 if x>0 else 0)
# apply: 作用于dataframe数据中的一行或一列数据  默认为列
# applymap： 作用于dataframe数据中的每一个元素
# map： 作用于series中的每一个元素 df结构中无法使用
df_purchase.head()


# 判断是否为 新、活跃、不活跃、回流用户
def active_status(data):  # data整行数据 共12列 即一个用户的12个月的消费记录
    status = []  # 负责存储用户 12 个月的状态：unreg|new|active|unactive|return
    for i in range(12):
        # 本月没有消费
        if data[i] == 0:
            if len(status) == 0:  # 前面没有任何记录（21年1月份）
                status.append('unreg')
            else:  # 开始判断上一个月状态
                if status[i - 1] == 'unreg':  # 一直未消费过
                    status.append('unreg')
                else:  # 只要本月没有消费当前的为0且不是unreg 只能为unactive
                    status.append('unactive')
        # 本月有消费==1
        else:
            if len(status) == 0:  # 前面没有任何记录（21年1月份）
                status.append('new')
            else:  # 之前有过记录  开始判断上一个月状态
                if status[i - 1] == 'unactive':  # 上个月没有消费
                    status.append('return')
                elif status[i - 1] == 'unreg':  # 以前没有消费过
                    status.append('new')
                else:
                    status.append('active')
    return pd.Series(status, df_purchase.columns)  # 值为status 列名为df_purchase中的列名


purchase_states = df_purchase.apply(active_status, axis=1)  # axis=1 朝列的方向读取
purchase_states.head()


# 用Nan替换unreg
purchase_states_ct = purchase_states.replace('unreg',np.NaN).apply(lambda x:pd.value_counts(x))
purchase_states_ct.head()

purchase_states_ct.T.fillna(0).plot.area()

# 前三个月以新用户为主，从第四个月新用户开始上升，在7 8月经过小幅的下降后趋近于平稳
# 回流用户主要产生在3月份之后，一直呈现逐渐攀升的趋势 是商家的重要客户
# 活跃用户主要产生在2月份之后，一直呈现逐渐攀升的趋势  在9月经过短暂下降后回升



# 回流用户的占比
#画出图像
plt.figure(figsize=(8,4),dpi=120)
rate = purchase_states_ct.T.fillna(0).apply(lambda x:x/x.sum(),axis=1)
plt.plot(rate['return'],label='return')
plt.plot(rate['active'],label='active')
plt.legend()
#  回流用户整年持续上涨
#  活跃用户前2个月大涨 猜测刚开始搞活动吸引很多新的用户持续消费 3月份后开始下降 并维持在1%附近



#用户购买周期
data1 =pd.DataFrame({
    'a':[0,1,2,3,4],
    'b':[5,4,3,2,1]
})
data1.shift()
# shift函数：将数据移动到一定的位置（整体向下或向右,默认值axis=0向下）
# 计算购买周期
order_diff = df1.groupby(by='用户名').apply(lambda x:abs(x['订单日期']-x['订单日期'].shift())) #当前订单日期 — 上一次订单日期
order_diff.head()

order_diff.describe()
order_diff.head()
(order_diff/np.timedelta64(1,'D')).hist(bins=20)
# 平均购买周期为116天 绝大多数用户的消费周期低于150天 用户消费周期在200天以上（不积极用户）占少数
# 用户的购买周期的人数随着时间的增长而减少
# 对于不积极用户可以在用户消费后3天内通过短信回访会赠送优惠券的方式，增大消费频率


'''
#用户生命周期
'''
#用户生命周期
# 计算方式： 用户最后一次购买 — 第一次购买的日期  如果差值=0 说明用户只够买了一次
user_life = df1.groupby(by='用户名')['订单日期'].agg(['min','max'])
(user_life['max'] == user_life['min']).value_counts().plot.pie(autopct='%1.1f%%') # 判断只够买一次的用户占比  格式化一位小数
plt.legend(['仅消费一次','多次消费'])

# 有77.2% 的用户仅消费了一次，说明运营不利，留存率不高
# 用户平均生命周期为29天 但是中位数为以及75%分位数0  再次验证大多数用户消费了一次  低质量用户
# 说明75%之后有很多生命周期很长的用户，属于核心用户 需要着重维持

# 绘制所有用户生命周期直方图+多次消费
plt.figure(figsize=(8,4),dpi=120)
plt.subplot(121)
((user_life['max'] - user_life['min'])/np.timedelta64(1,'D')).hist(bins=30)
plt.title('所有用户生命周期直方图')
plt.xlabel('生命周期天数')
plt.ylabel('用户人数')

plt.subplot(122)
u_1 = (user_life['max'] - user_life['min']).reset_index()[0]/np.timedelta64(1,'D')
u_1[u_1>0].hist(bins=25)
plt.title('多次消费用户生命周期直方图')
plt.xlabel('生命周期天数')
plt.ylabel('用户人数')


u_1[u_1>0].mean()
# 对比可知 ，第二幅图过滤了用户生命周期为0的用户，用户人数随着生命周期的增加呈现递减趋势
#  虽然二图中还有一部分用户的生命周期为0天左右 但是人数比第一幅图少了很多 虽然进行了多次消费 但是不够长久 属于普通用户
# 可针对性进行营销活动
# 对于商家的忠诚顾客较少 可针对性维护此类用户


'''

'''
#复购率和回购率分析
#计算方式： 在自然月内，购买多次的用户在总消费人数中的占比（同一天消费多次也算）
# 消费者有三种：1.本月消费多次（复购） 2.消费一次 3.本月无消费
# 复购： 1  非复购且有消费：0  无消费：NAN(不参与count计数)
purchase_r = pivoted_counts.applymap(lambda x:1 if x>1 else np.NaN if x==0 else 0)
purchase_r.head()
# purchase_r.sum() 求出复购用户
# purchase_r.count() 求出所有参与消费的用户
(purchase_r.sum()/purchase_r.count()).plot(figsize=(12,6))

# 前五个月复购率逐渐上升  后续趋于平稳  但是总体复购率都不高


#回购率分析
# 计算方式：在一个时间窗口内进行了消费，在下一个窗口又进行了消费
# 1：回购用户：当前月消费了，下个月又消费了  0：当前月消费 下个月未消费   nan：当前月未消费
def purchase_back(data):
    status = [] #存储用户回购率状态
    for i in range(11):
        # 当前月消费了
        if data[i] == 1:
            if data[i+1] ==1:
                status.append(1) # 回购用户
            elif data[i+1] == 0:
                status.append(0) # 下个月未消费
        # 当当前月未消费
        else:
            status.append(np.NaN)
    status.append(np.NaN)  # 填充最后一列数据
    return pd.Series(status,df_purchase.columns)
purchase_b = df_purchase.apply(purchase_back,axis=1)
purchase_b.head()


# 回购率可视化
plt.figure(figsize=(20,15),dpi=200)
plt.subplot(211)
# 回购率
(purchase_b.sum()/ purchase_b.count()).plot(label='回购率')
#复购率
(purchase_r.sum()/purchase_r.count()).plot(label='复购率')
plt.legend()
plt.ylabel('百分比')
plt.title('用户复购率和回购率对比图')


# 回购人数与购物总人数
plt.subplot(212)
plt.plot(purchase_b.sum(),label='回购人数')
plt.plot(purchase_b.count(),label='购物总人数')
plt.xlabel('月份')
plt.ylabel('人数')

# 由回购率可知 前五个月逐渐上升  平稳后在0.05左右 具有一定的波动性
# 复购率一直低于回购率 平稳后在0.25左右
# 前五个月无论是回购还是复购 均呈现上升 说明新用户需要一定的时间来变成复购或者回购用户
# 结合新老用户分析，新客户忠诚度远低于老客户忠诚度


# 2-5月 购物总人数与回购人数逐渐拉大 说明多了很多新用户  可能是商家促销活动力度加大的原因
# 但是回购、复购率仍不足  需要营销策略积极引导其再次消费及持续消费
# 对于多次消费的顾客，也要适时推出反馈老客户的优惠活动，以加强老客的忠诚度




