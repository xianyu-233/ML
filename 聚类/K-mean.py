import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# 设置数据集
def datSet():
    path = "ex3/iris.data"
    data = pd.read_csv(path)
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

    return data,labels

# 画图
def painting(data,labels,mu):
    tmp1 = []
    tmp2 = []
    for i in range(len(mu)):
        tmp1.append(mu[i][0])
        tmp2.append(mu[i][1])
    plt.figure()
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    for i in range(len(data)):
        if labels[i] == 0:
            plt.plot(data[i][0],data[i][1],'.g')
        if labels[i] == 1:
            plt.plot(data[i][0],data[i][1],'.b')
        elif labels[i]==2:
            plt.plot(data[i][0],data[i][1],'.c')
        elif labels[i]==3:
            plt.plot(data[i][0],data[i][1],'.k')
        elif labels[i]==4:
            plt.plot(data[i][0],data[i][1],'.m')
    for i in range(len(mu)):
        plt.plot(mu[i][0],mu[i][1],'r*')
    plt.show()

# 计算欧氏距离
def distance(mu,XX):
    dis = 0
    for i in range(len(XX)):
        dis += (mu[i]-XX[i])**2
    dis = np.sqrt(dis)
    return dis

# 找到最小值的标签
def find_Min(data):
    idx = 0
    for i in range(len(data)):
        if data[idx]>data[i]:
            idx = i
    return idx

# K-mean算法
def K_mean(data, K, loop_times):
    row = len(data)
    columns = len(data[0])
    mu = np.zeros((K,columns))
    # 到所有中心点的距离
    tmp_dis = np.zeros(K)
    # 预测标签
    labels = np.zeros(row)
    labels = labels-1
    # 记录每种类的所有下标
    idx = {}

    # 初始化中心点
    for i in range(K):
        mu[i] = data[i]
    # 循环
    for times in range(loop_times):
        # 初始化idx
        for i in range(K):
            idx[i] = []
        # 记录每一类的个数
        idx_num = np.zeros(K)

        # 计算所有点到各中心点的距离
        for i in range(row):
            for j in range(K):
                # 计算每个点到各中心点的距离
                tmp_dis[j] = distance(mu[j],data[i])
            # 决定样本点的标签（找到最近的中心点）
            labels[i] = find_Min(tmp_dis)
            idx_num[int(labels[i])] +=1
            # 记录每一类的下标
            idx[int(labels[i])].append(i)

        tmp = 0
        # painting(data,labels, mu)
        # 重新设置中心点
        # K个中心点
        for i in range(K):
            # 中心点的各个参数x
            for j in range(columns):
                # 属于该中心点的所有训练点
                for k in idx[i]:
                    tmp += data[k][j]

                mu[i][j] = tmp/idx_num[i]
                # tmp每次都要重置
                tmp=0
    # 画结果图
    painting(data,labels, mu)
    return labels

if __name__ == '__main__':
    data,labels = datSet()
    data = np.array(data)

    yh = np.zeros(len(labels))
    K = 3

    yh = K_mean(data, K, 200)

    print("兰德指数为：",metrics.adjusted_rand_score(labels, yh))
    painting(data,labels,[[4,2],[4,2],[4,2]])
    # a = {0:[1,2,3],1:[1,7]}
    # print(a)
    # for i in a[1]:
    #     print(i)