import numpy as np
from numpy import mat
from numpy.linalg import *
from math import exp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import metrics

# 设置数据集
def datSet():
    path = "ex3/iris.data"
    data = pd.read_csv(path)
    # 打乱数据顺序
    data = data.sample(frac=1.0)
    data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'types']
    data.drop('sepal width',axis=1,inplace=True)
    data.drop('petal width',axis=1,inplace=True)

    # 将每一个含有类别变量的列都转化为含有整数值的列
    object_cols = ['types']
    label_encoder = LabelEncoder()
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # 删除重复的点
    data.drop_duplicates(inplace=True)

    # 将标签分离
    labels = data['types']
    data.drop('types',axis=1,inplace=True)

    # 重置索引
    data.index = range(len(data))
    labels.index = range(len(labels))

    data = np.mat(data)

    return data,labels

"""
多维高斯分布：
变量：
D：维度
E：协方差矩阵(2X2)
XX：数据(1X2)
mu：均值(1X2)
"""
def gaussian_Distribute(D, E, XX, mu):
    return 1/((2*np.pi)**(D/2)*(det(E))**0.5) * exp(-(1/2) * (XX-mu) * E.I * (XX - mu).T)

"""
alpha:每个多维高斯分布的权值（1X2）
theta:要求的参数（2个），包括E（协方差矩阵2X2，有K个）,mu（均值1X2，有K个）
XX:二维的观测数据（1X2）
D:指观测数据的维度
K:指的K个分类
注意：返回的gama是一个3X1的列向量
"""
def E_Step(alpha, theta, XX, D, K):
    tmp = 0
    for i in range(K):
        tmp += alpha[i]*gaussian_Distribute(D, theta['E'][i], XX, theta['mu'][i])
    gama = np.zeros((K,1))

    for i in range(K):
        gama[i] = (alpha[i]*gaussian_Distribute(D, theta['E'][i], XX, theta['mu'][i]))/tmp
    return gama     # 返回一个（KX1）的gama矩阵

"""
gama:隐含变量（KX1）一共N个
XX:观测数据（1X2）一共N个
返回一个mu(总共有K项，每一项都是1X2的矩阵)
"""
def M_Step_Mu(gama,XX):
    mu = []
    for i in range(len(gama[0])):
        tmp1=np.zeros((1,2))
        tmp2 =0
        for j in range(len(XX)):
            tmp1 += gama[j][i]*XX[j]
            tmp2 += gama[j][i]
        mu.append(mat(tmp1/tmp2))
    return mu

"""
gama:隐含变量
XX；观测数据
mu：均值（K个1X2）
"""
def M_Step_E(gama, XX, mu):
    E = []
    for i in range(len(gama[0])):
        tmp1 =np.zeros((2,2))
        tmp2 = 0
        for j in range(len(XX)):
            tmp3 = float(gama[j][i])
            tmp1 += tmp3*((XX[j]-mu[i]).T * (XX[j]-mu[i]))
            tmp2 += tmp3
        E.append(mat(tmp1/tmp2))
    return E

"""
gama:隐含变量
"""
def M_Step_alpha(gama):
    N = len(gama)
    alpha = []
    for i in range(len(gama[0])):
        tmp1=0
        for j in range(N):
            tmp1 +=gama[j][i]
        tmp1 /=N
        alpha.append(float(tmp1))
    return alpha

"""
data:观测数据
theta:要求的参数（2个），包括E（协方差矩阵2X2，有K个）,mu（均值1X2，有K个）
gama:隐含变量（KX1）一共N个
alpha:每个多维高斯分布的权值（1X2）
D：数据维度
K：类别的个数
"""
def GMM(data, theta, gama, alpha, D, K):
    N = len(data)   # 数据个数
    for times in range(100):
        # 计算新的gama
        for i in range(N):
            gama.append(E_Step(alpha,theta,data[i], D, K))

        # 计算新的Mu
        theta['mu'] = M_Step_Mu(gama,data)

        # 计算新的协方差矩阵E
        theta['E'] = M_Step_E(gama,data,theta['mu'])
        # 计算alpha
        alpha = M_Step_alpha(gama)

        gama = []           # 每次清空一次gama列表
    return theta, alpha

def painting(data, labels, gama, theta):
    data = np.array(data)

    M = theta['mu']

    yh = np.zeros(len(data))
    for i in range(len(data)):
        tmp = 0
        tmp_y = gama[i][0][0]
        for j in range(len(gama[0][0])):
            if tmp_y < gama[i][0][j]:
                tmp = j
                tmp_y = gama[i][0][j]
        yh[i] = tmp
    # 将真实值转成数组型
    labels = np.array(labels)

    # 画出高斯分布的中心点
    for i in range(len(M)):
        M[i] = np.array(M[i])
        plt.plot(M[i][0][0], M[i][0][1], '*g')

    # 画出所有点的分布
    for i in range(len(data)):
        if yh[i] == 0:
            plt.plot(data[i][0], data[i][1], '.y')
        if yh[i] == 1:
            plt.plot(data[i][0], data[i][1], '.b')
        elif yh[i] == 2:
            plt.plot(data[i][0],data[i][1],'.r')

    print("兰德指数为：",metrics.adjusted_rand_score(labels, yh))
    plt.show()

def GMM_test():
    data,labels = datSet()
    E = []
    E.append(mat(np.cov(data[0:50].T)))
    E.append(mat(np.cov(data[50:100].T)))
    E.append(mat(np.cov(data[100:150].T)))
    mu=[]
    mu.append(data[0:50].mean(axis=0))
    mu.append(data[50:100].mean(axis=0))
    mu.append(data[100:150].mean(axis=0))
    theta = {}
    theta['E'] = E
    theta['mu'] = mu
    alpha = [0.4,0.4,0.2]
    gama = []
    # 混合高斯
    theta,alpha = GMM(data, theta, gama, alpha, D=2, K=3)
    gama = []
    for i in range(len(data)):
        gama.append(E_Step(alpha,theta,data[i],D=2,K=3).T)

    painting(data,labels,gama,theta)

if __name__ == '__main__':
    GMM_test()