
# coding: utf-8
'''
聚类：也是将数据分成几个类别，但没有任何的参考信息
分类判别：虽也是分类，但它是由学习集，有参考信息的
样本点的关键度量指标：距离
欧氏距离(enclidean)：其实就是通常意义上的直线上的距离
马氏距离(Manhattan)：考虑到变量的相关性，并且与变量的单位无关
余弦距离：（cosine）衡量变量相似性
    
（凝聚的）层次聚类法：
    思想：
    1.开始时，每个样本各自作为一类
    2.规定某种度量作为样本之间的距离以及类类与类之间的距离，并计算。
        各类之间距离计算的方法：
            离差平方和--ward
            类平均法--average
            最大距离发--complete
    3.将距离最短的两个类合并成一个新类
    4.重复2-3，即不断的合并最近的两个类每次减少一个类，知道所有的样本被合并为一类
方法具体应用： 
AgglomerativeClustering(n_clusters = 2, 聚类的类别数目
                        affinity='euclidean',  点之间的距离计算方式
                        menpory=Memory(cachedir=None),  缓存系谱图的目录，默认是空的 
                        connectivity= None,  连接矩阵,就是距离矩阵
                        n_components=None,  不用考虑，
                        compute_full_tree='auto  计算完整树，通过数据的大小考虑是否计算全树
                        linkage='ward',  类和类之间的距离的计算方法（默认是离差平方和法）
                        pooling_func= <function mean>) 将输入的变量进行降维计算

'''
# In[1]:

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets


# In[2]:
# 使用模块内所自带的数据集
digits = datasets.load_digits(n_class=10)
# x是多维的数据集
X = digits.data
# 对原数据及进行标记，查看是否已经对其进行分类
# y是一维的数据即
y = digits.target
n_samples, n_features = X.shape
print( X[:5,:])
print (n_samples,n_features)


# In[3]:

# Visualize the clustering
# 定义一个画图该函数，将聚类结果不同的分类有不同的颜色
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()



# 2D embedding of the digits dataset，数字数据集的2D嵌入
print("Computing embedding")
# 对数据集进行计算
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering
# 主要是对比三个类（ward，average，complete）之间的差别
for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    # 对数据进行拟合 
    clustering.fit(X_red)
    # 在进行画图
    plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)
plt.show()

'''
动态聚类：K-means方法
    算法：
        1.选择K个点作为初始质心
        2.将每个点指派到最近的质心，形成K个簇（聚类）
        3.重新计算每个簇的质心
        4.重复2-3直到质心不发生变化
方法具体应用：
    KMeans(n_clusters=8,  聚类的类别数目
           init='k-means++',  聚类数目的计算方法，默认是智能的K-means++，
           n_init=10,   k-means算法运行的次数
           max_iter=300,  最大的迭代次数，
           tol=0.0001,  质心收敛的精度
           precompute_distances='auto', 默认是自动的，True是运算快，但是类群上很耗费
                                         自动样本数乘以聚类数大于12000万，不会计算。
           verbose=O,   是否进入冗长模式，默认是不进入
           random_state= None,  
           copy_x=True,  计算更均值的再去计算
           n_jobs= 1)    并行计算（-1为使用并行计算）

'''

get_ipython().magic(u'matplotlib inline')
X0 = np.array([7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3])
X1 = np.array([5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7])
plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0, X1, 'k.');


# In[4]:

C1 = [1, 4, 5, 9, 11]
C2 = list(set(range(12)) - set(C1))
X0C1, X1C1 = X0[C1], X1[C1]
X0C2, X1C2 = X0[C2], X1[C2]
plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0C1, X1C1, 'rx')
plt.plot(X0C2, X1C2, 'g.')
plt.plot(4,6,'rx',ms=12.0)
plt.plot(5,5,'g.',ms=12.0);


# In[5]:

C1 = [1, 2, 4, 8, 9, 11]
C2 = list(set(range(12)) - set(C1))
X0C1, X1C1 = X0[C1], X1[C1]
X0C2, X1C2 = X0[C2], X1[C2]
plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0C1, X1C1, 'rx')
plt.plot(X0C2, X1C2, 'g.')
plt.plot(3.8,6.4,'rx',ms=12.0)
plt.plot(4.57,4.14,'g.',ms=12.0);


# In[6]:

C1 = [0, 1, 2, 4, 8, 9, 10, 11]
C2 = list(set(range(12)) - set(C1))
X0C1, X1C1 = X0[C1], X1[C1]
X0C2, X1C2 = X0[C2], X1[C2]
plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(X0C1, X1C1, 'rx')
plt.plot(X0C2, X1C2, 'g.')
plt.plot(5.5,7.0,'rx',ms=12.0)
plt.plot(2.2,2.8,'g.',ms=12.0);


# In[7]:

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T
plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.');


# In[8]:

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('The average degree of distortion')
plt.title('Best k')


# In[9]:

import numpy as np
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
plt.figure()
plt.axis([0, 10, 0, 10])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.');


# In[10]:

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('The average degree of distortion')
plt.title('Best K')


# In[11]:


"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.
DBSAN 方法（基于密度的方法）：
    算法基本思想：
        1指定合适的r和M
        2 计算所有的样本点，如果点p的r邻域里有超过M个点，则创建一个以p为核心点的新簇
        3 反复寻找这些核心点直接密度可达(之后可能是密度可达)的点，将其加入到相应的簇，
            对于核心点发生“密度相连”状况的簇，给予合并
        4 当没有新的点可以被添加到任何簇时，算法结束
相关概念：
      r-邻域: 给定点半径r内的区域
      核心点: 如果一个点的r-邻域至少包含最少数目M个点，则称该点为核心点
      直接密度可达: 如果点p在核心点q的r-邻域内，则称p是从q出发可以直接密度可达
      如果存在点链p1P2...Pn,P1=q,Pn=p,P;+ 1是从p;关于r和M直接密度可达，则称点p
          是从q关于r和M 密度可达的
      如果样本集D中存在点o，使得点p、q是从o关于r和M密度可达的，那么点p、q是关于r
          和M密度相连的
DBSAN算法描述：
    输入: 包含n个对象的数据库，半径e，最少数目MinPts;
    输出:所有生成的簇，达到密度要求。
    (1)Repeat
    (2)从数据库中抽出一个未处理的点;
    (3)IF抽出的点是核心点TH EN 找出所有从该点密度可达的对象，形成一个簇;
    (4)ELSE 抽出的点是边缘点(非核心对象)，跳出本次循环，寻找下一个点;
    (5)UNTIL 所有的点都被处理。
    DBSCAN对用户定义的参数很敏感，细微的不同都可能导致差别很大的结果，而参数的选
        择无规律可循，只能靠经验确定。
DBSAN 方法
    DBSAN（eps = 0.5,  r的距离大小
            min_samples = 5,  邻域内的样本数目（M）
            metric = 'eucliean',  距离计算方式，欧氏距离
            algorithm = 'auto',  
            leaf_size =  30,  叶子的数目限制在30内
            p = None,  初始的核心点，默认是自动计算
            random_state = None）  随机因子



"""
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


##############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

##############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

##############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# In[ ]:



