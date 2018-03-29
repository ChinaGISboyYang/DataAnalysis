# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange #导入拉格朗日插值函数
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4, threshold=500)
pd.options.display.max_rows = 100


###缺失值处理——拉格朗日插值法
# 
inputfile = 'd:/data/catering_sale.xls' #销量数据路径
outputfile = 'd:/data/sales.xls' #输出数据路径

data = pd.read_excel(inputfile) #读入数据
data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None #过滤异常值，将其变为空值

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) #插值并返回插值结果

#逐个元素判断是否需要插值
for i in data.columns:
  for j in range(len(data)):
    if (data[i].isnull())[j]: #如果为空即插值。
      data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile) #输出结果，写入文件


###dataframe合并
#1
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'],
                 'data2': range(3)})
df1
df2
# 合并
pd.merge(df1, df2)
# on：指定key
pd.merge(df1, df2, on='key')

#2
df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
# left_on='lkey', right_on='rkey'指左边（右边）键的名称
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
# 连接方式(how)：外部链接
"""
left  : 参与合并的左侧DataFrame
right : 参与合并的右侧DataFrame
how   : "inner" "outer" "left" "right" 默认是“----”

on    : 用于连接的列明，必须存在左右两个DATaFrame对象中
left_on（right_on） : 左侧(右侧)DataFrame中用与连接键的列
left_index : 将左侧的索引应用做其连接键，right_index 同理

sort : 根据连接键对合并后的数据进行排序，默认为True
"""
pd.merge(df1, df2, how='outer')

#3
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})
df1
df2
# 左连接方式
pd.merge(df1, df2, on='key', how='left')
pd.merge(df1, df2, how='inner')

#4
left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')

#5
pd.merge(left, right, on='key1')

pd.merge(left, right, on='key1', suffixes=('_left', '_right'))


###索引上的合并
#
#1
left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
left1
right1

pd.merge(left1, right1, left_on='key', right_index=True)

pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

#2
lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
lefth
righth

pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)

pd.merge(lefth, righth, left_on=['key1', 'key2'],
         right_index=True, how='outer')

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],
                 columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                   index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
left2
right2
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)

#3
left2.join(right2, how='outer')

left1.join(right1, on='key')

#4
another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                    index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
left2.join([right2, another])

left2.join([right2, another], how='outer')


###轴向连接
#1
arr = np.arange(12).reshape((3, 4))
arr
 # axis： 平行连接
np.concatenate([arr, arr], axis=1)

#2
s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
#连接concat
"""
轴向连接
    Panads对象：
        objs   参与链接的pandas对象的列或者字典，唯一必须的参数
        axis   指明连接的轴向，默认为0
        join   "inner","outer"其中之一，默认为”outer“
              （指明其他轴向上的索引是按照交集（inner）还是并集（outer）进行合并）
        join_axes  指明用于其他n-1条轴的索引，不执行并集或者交集运算
        keys   于连接对象有关的值，用于连接轴向上的层次化索引
        levels 指定用作层次化索引个级别上的索引
        names  用于创建分层级别的名称
        verify_integrity 检查结果对象新轴向上的重复情况，
               如果发现则引起异常，默认（false）允许重复
        ignore_index    不保留连接轴上的索引。产生一组新索引range（total_length）
        
"""
pd.concat([s1, s2, s3])

pd.concat([s1, s2, s3], axis=1)

s4 = pd.concat([s1 * 5, s3])
pd.concat([s1, s4], axis=1)

pd.concat([s1, s4], axis=1, join='inner')

pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

#3
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result

result.unstack()

#4
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
                columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
                columns=['three', 'four'])

pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])

pd.concat({'level1': df1, 'level2': df2}, axis=1)

pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
          names=['upper', 'lower'])

#5
df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])

df1

df2

pd.concat([df1, df2], ignore_index=True)


###合并重叠数据
#1
a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64),
           index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan

a

b

np.where(pd.isnull(a), b, a)

#2
# combline_first 填充缺失值
b[:-2].combine_first(a[2:])

#3
df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                 'b': [np.nan, 2., np.nan, 6.],
                 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                 'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)


###重塑层次化索引
"""
Stack : 将数据的列“旋转”为行
Unstack :  将数据的行“旋转”为列
"""
#1
data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
data

result = data.stack()
result

result.unstack()

result.unstack(0) 
result.unstack('state')

#2
s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2.unstack()

data2.unstack().stack()

data2.unstack().stack(dropna=False)

#3
df = DataFrame({'left': result, 'right': result + 5},
               columns=pd.Index(['left', 'right'], name='side'))
df

df.unstack('state')

df.unstack('state').stack('side')


###长宽格式的转换
#1
data = pd.read_csv('d:data/macrodata.csv')
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
data = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end'))

ldata = data.stack().reset_index().rename(columns={0: 'value'})
wdata = ldata.pivot('date', 'item', 'value')

#2
ldata[:10]

pivoted = ldata.pivot('date', 'item', 'value')
pivoted.head()

ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]

pivoted = ldata.pivot('date', 'item')
pivoted[:5]

pivoted['value'][:5]

unstacked = ldata.set_index(['date', 'item']).unstack('item')
unstacked[:7]


###移除重复数据
data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data
# 检测数据（行）和前面是否有相同的数据行
data.duplicated()
# 删除（丢弃）重复行
data.drop_duplicates()

data['v1'] = range(7)
data.drop_duplicates(['k1'])

data.drop_duplicates(['k1', 'k2'], take_last=True)


###利用函数或映射进行数据转换
# 
#1
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                           'corned beef', 'Bacon', 'pastrami', 'honey ham',
                           'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
# .map(str, lower)将大写字符串转换成全部小写字符串
# .map(meat_to_animal)映射为字典，生成data数据

data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data 

data['food'].map(lambda x: meat_to_animal[x.lower()])

# 数据标准化
datafile = 'd:/data/normalization_data.xls' #参数初始化
data = pd.read_excel(datafile, header = None) #读取数据

(data - data.min())/(data.max() - data.min()) #最小-最大规范化
(data - data.mean())/data.std() #零-均值规范化
data/10**np.ceil(np.log10(data.abs().max())) #小数定标规范化


###替换值
data = Series([1., -999., 2., -999., -1000., 3.])
data
# 替换
data.replace(-999, np.nan)

data.replace([-999, -1000], np.nan)

data.replace([-999, -1000], [np.nan, 0])

data.replace({-999: np.nan, -1000: 0})


###重命名轴索引
data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
# 将字符串全部转换成大写格式
data.index.map(str.upper)

data.index = data.index.map(str.upper)
data
# 重命名（index指主列，columns指）
data.rename(index=str.title, columns=str.upper)

data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})

# 总是返回DataFrame的引用
_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data


###离散化与面元划分
#1
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

bins = [18, 25, 35, 60, 100]
# 分组（以bins为标准，拆分ages。）
cats = pd.cut(ages, bins)
cats

cats.labels

cats.levels
# 统计分组后，个组项的值（频数  ）
pd.value_counts(cats)
# 更改分组区间的左右闭合（是否取到），默认为"( ]"
# 如果改为“right”= False 则是“[  )”
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
# 更改分组的名称（18-26是青年组，如此类推哦）
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
# labels = 更改后的分组情况
pd.cut(ages, bins, labels=group_names)
# 随机生成长度为20的随机数组
data = np.random.rand(20)
# 再将这个数组等长 的分成4分
pd.cut(data, 4, precision=2)

#2
data = np.random.randn(1000) # Normally distributed
# qcut 是和cut擦汗不多，是根据数据的分位数划分数据
cats = pd.qcut(data, 4) # Cut into quartiles
cats

pd.value_counts(cats)

pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])


###检测和过滤异常值
#1随机生成符合正态分布的随机数组
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
# 查看数据的分布状况
data.describe()

col = data[3]
# 找出绝对值大于3 的值
col[np.abs(col) > 3]

data[(np.abs(data) > 3).any(1)]

#2 .sign 是符号判断函数
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()


###排列与随机采样
#1 随机产生一个dataFrame
df = DataFrame(np.arange(5 * 4).reshape((5, 4)))
# 对数据进行排序
sampler = np.random.permutation(5)
sampler
df
# 对数组进行随机乱排
df.take(sampler)

#2 将数组重排后在抽样
df.take(np.random.permutation(len(df))[:3])

#3
bag = np.array([5, 7, -1, 6, 4])
# 通过数组产生随机的整数
sampler = np.random.randint(0, len(bag), size=10)
sampler

draws = bag.take(sampler)
draws


###计算指标与哑变量
#1
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy

#2
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('d:/data/movies.dat', sep='::', header=None,
                        names=mnames)
movies[:10]

genre_iter = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iter))

dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)

for i, gen in enumerate(movies.genres):
    dummies.ix[i, gen.split('|')] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.ix[0]

#3
np.random.seed(12345)
values = np.random.rand(10)
values

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))

#线损率属性构造


#参数初始化
inputfile= 'd:/data/electricity_data.xls' #供入供出电量数据
outputfile = 'd:/data/electricity_data.xls' #属性构造后数据文件

data = pd.read_excel(inputfile) #读入数据
data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量'])/data[u'供入电量']

data.to_excel(outputfile, index = False) #保存结果
"""
字符串操作：
    字符串.split(’，‘)  以逗号分割，拆分成单个字符串
    [x.strip() for x in val.split(',')] 去除字符串中的空格
"""
"""
PYTHON 内置的字符串方法：
    count      ：返回子串在字符串中出现的次数
    endswith   ：如果字符串以某个后缀结尾，则返回Ture
    startswith ：如果字符串以某个前缀开头，则返回Ture
    join       ：将字符串用作连接其他字符串序列的分隔符
    index      ：如果在字符串中找到子串，则返回子串第一个字符所在的位置，
                 如果没有找到，就会引发ValueError错误
    find       ：如果在字符串中找到子串，则返回第一个发现子串的第一个字符所在的位置，
                 如果没有找到，则返回-1
    rfind      ：如果在字符串中找到子串，则返回最后一个发现子串的第一个字符所在的位置，
                 如果没有找到，则返回-1
    replace    ：用另一个字符串替换指定子串
    strip，rstrip,Istrip
               : 去除空白符（包括换行符），相当于对各个元素执行x.strip()
    split      : 通过指定的分隔符将字符串拆分成为一组子串
    lower,upper: 分别将字母字符转换为小写或者大写
    ljust，rjust：用空格（或者其他字符）填充字符串的空白侧以返回符合最低宽度的字符串
"""
###字符串对象方法
val = 'a,b,  guido'
val.split(',')

pieces = [x.strip() for x in val.split(',')]
pieces

first, second, third = pieces
first + '::' + second + '::' + third
# 通过“：：”连接符将字符串连接起来
'::'.join(pieces)
# 判断’字符串‘是否在字符串中
'guido' in val
# 判断’，‘是否在字符串中，如果是，那么返回的就是1；如果不是，那么就会报错
val.index(',')
# .find()函数也是判断是否在字符串中
val.find(':')

val.index(':')
# 统计字符串中的元素出现的次数
val.count('a')
# 替换
val.replace(',', '::')

val.replace(',', '')


###正则表达式
'''
Re模块（正则表达式模块）
（匹配模式；替换；拆分）
方法：
    findall，finditer   ：返回字符串中所有非重叠匹配模式。
            （findall返回的是由所有模式组成的列表；finditer则通过一个迭代器逐个返回）
    match               : 从字符串起始位置匹配模式，还可以对模式个部分进行分组。
                            如果匹配到模式，则返回匹配对象，否则就返回None
    search              : 扫描整个字符串以匹配模式，如果找到则返回一个匹配对象。和match
                            不同，其匹配项可以位于字符串的任意位置，不单单指在起始位置。
    split               ：根据模式将字符串拆分成数段
    sub, subn           : 将字符串中所有的（sub）或者前N个（subn）模式替换为指定表达式
                            在替换字符串中可以通过 \1, \2等符号表示各分组项
                            

'''
#1
import re
# \t 换行符； \n 空格符
text = "foo    bar\t baz  \tqux"
# \s+  指多个空格符 
re.split('\s+', text)
# 将字符串编辑下来，指去掉空格
regex = re.compile('\s+')
regex.split(text)

regex.findall(text)

#2
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
# 普通电子邮件的邮箱格式
# [A-Z0-9._%+-] ： 任意的数字字母或者下划线百分号等字符
# 下例中“\.” 指特定的“.”符号，是不能更改的 
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE 的作用是使正则表达式对大小写不敏感
regex = re.compile(pattern, flags=re.IGNORECASE)

regex.findall(text)

m = regex.search(text)
m

text[m.start():m.end()]

print(regex.match(text))

print(regex.sub('REDACTED', text))

#3
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)

m = regex.match('wesm@bright.net')
m.groups()

regex.findall(text)

print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

#4
regex = re.compile(r"""
    (?P<username>[A-Z0-9._%+-]+)
    @
    (?P<domain>[A-Z0-9.-]+)
    \.
    (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE|re.VERBOSE)

m = regex.match('wesm@bright.net')
m.groupdict()

 
###pandas中矢量化的字符串函数
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
# 将字符串形式的数据转成数组形式
data = Series(data)

data
# 判断数组内是否存在为空的数据
data.isnull()
# .str.contains() : 检查字符串中是否存在‘gnail’的特定子串，返回是Ture或False
data.str.contains('gmail')

pattern
# 通过正则表达式，找出各部分的组成
data.str.findall(pattern, flags=re.IGNORECASE)
#分割成不同的部分
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches

matches.str.get(1)

matches.str[0]

data.str[:5]



###示例：USDA食品数据库
'''
{
  "id": 21441,
  "description": "KENTUCKY FRIED CHICKEN, Fried Chicken, EXTRA CRISPY,
Wing, meat and skin with breading",
  "tags": ["KFC"],
  "manufacturer": "Kentucky Fried Chicken",
  "group": "Fast Foods",
  "portions": [
    {
      "amount": 1,
      "unit": "wing, with skin",
      "grams": 68.0
    },

    ...
  ],
  "nutrients": [
    {
      "value": 20.8,
      "units": "g",
      "description": "Protein",
      "group": "Composition"
    },

    ...
  ]
}
'''

import json
db = json.load(open('d:/data/data/foods-2011-10-03.json'))
len(db)

db[0].keys()

db[0]['nutrients'][0]

nutrients = DataFrame(db[0]['nutrients'])
nutrients[:7]

info_keys = ['description', 'group', 'id', 'manufacturer']
info = DataFrame(db, columns=info_keys)

info[:5]
info

pd.value_counts(info.group)[:10]

nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

nutrients
# 丢掉重复行数
nutrients.duplicated().sum()

nutrients = nutrients.drop_duplicates()

col_mapping = {'description' : 'food',
               'group'       : 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
info

col_mapping = {'description' : 'nutrient',
               'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
nutrients

ndata = pd.merge(nutrients, info, on='id', how='outer')

ndata

ndata.ix[30000]
# 绘制中位数的图
result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].order().plot(kind='barh')

by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])

get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())

max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]

# make the food a little smaller
max_foods.food = max_foods.food.str[:50]

max_foods.ix['Amino Acids']['food']