import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
import matplotlib.pyplot as plt


# 设置数据集
def dataSet():
    path = 'housing.csv'
    data = pd.read_csv(path)
    # 删除空数据
    data.dropna(axis=0,how='any',inplace=True)

    # 删除异常值
    # id=[]
    # for i in range(len(data)):
    #     if data['median_house_value'].values[i] > 495000:
    #         id.append(data.index[i])
    # data.drop(index=id,axis=0,inplace=True)


    # 对每一个含有类别变量的列都使用独热编码
    s = (data.dtypes == 'object')
    cols = list(s[s].index)
    # handle_unknown='ignore' 是为了避免验证集中出现了训练集中没有的新类别向量，sparse=False 是为了确保返回的经编码过的列，是以numpy的数组形式出现，而不是以稀疏矩阵的形式返回
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_data = pd.DataFrame(OH_encoder.fit_transform(data[cols]))
    # 独热编码会移除索引，因此要将它们取回
    OH_cols_data.index = data.index
    # 移除数据集含有类别变量的列（因为我们已经获得了含有独热编码的列）
    num_data = data.drop(cols, axis=1)
    # 将含有独热编码的列与经过处理的数据集（不包含类别变量）连接
    data = pd.concat([num_data, OH_cols_data], axis=1)


    data=data.sample(frac=1.0)      #打乱数据的顺序
    label = data['median_house_value']    #获取房价标签
    data = data.drop('median_house_value',axis=1)   #获取不含标签的数据集

    # 对数据集进行切割（训练70%，测试30%）
    training_data = data.head(int(0.7*len(data)))
    training_label = label.head(int(0.7*len(data)))
    test_data = data.tail(int(1 - 0.7*len(data))-1)
    test_label = label.tail(int(1 - 0.7*len(data))-1)

    return training_data,training_label,test_data,test_label

# 计算一个点的预测值h(x)
def h(XX,theta):
    return np.dot(XX,theta)

# 计算单个x_j的梯度
def Gradient(h_x,label,x):
    return (h_x - label) * x

# 梯度下降
def Gradient_Descent(training_data, training_labels, learning_rate, loop_times):
    row = len(training_data)            # 计算数据的行
    column = len(training_data[0])      # 计算数据的列
    length = 1000                       # 随机梯度选取长度

    theta = np.zeros(column)
    tmp = np.zeros(column)

    for times in range(loop_times):     # 循环次数
        g = np.zeros(column)
        h_y = np.zeros(row)  # 预测值

        # 批量梯度下降
        # 随机在训练集中抽取1000个数据的索引
        idx = []
        for i in range(length):
            idx.append(random.randint(0,len(training_data)-1))

        # 计算预测值
        for i in idx:
            h_y[i] = h(training_data[i], theta)

        # 计算梯度
        for i in range(column):
            for j in idx:
                tmp[i] = Gradient(h_y[j],training_labels[j],training_data[j][i])
                g[i] = g[i] + tmp[i]
            g[i] = g[i]/length

        # 计算每一次更新的theta
        for i in range(column):
            theta[i] = theta[i] - learning_rate * g[i]

        print("循环次数：",times)
    print("theta（GD）:",theta)

    return theta

# 解析式求解
def Analytic_Method(XX,target):
    theta = np.zeros(np.shape(XX[0]))
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(XX.T, XX)), XX.T),target)
    print('theta(AM):',theta)
    return theta

# 计算预测值与实际值的误差
def MSE(training_y, labels):
    sum = 0
    m = len(training_y)
    for i in range(m):
        sum = sum + (training_y[i]-labels[i])*(training_y[i]-labels[i])
    sum = sum/m
    return sum

def MAE(training_y,labels):
    sum=0
    m=len(training_y)
    for i in range(m):
        sum = sum + abs(training_y[i]-labels[i])
    sum = sum/m
    return sum

def R_2(training_y,labels):
    sum_mean=0
    sum_se=0
    m=len(labels)
    mean=sum(labels)/m

    for i in range(m):
        sum_mean = sum_mean+(mean-labels[i])*(mean-labels[i])
        sum_se=sum_se+(training_y[i]-labels[i])*(training_y[i]-labels[i])
    return 1-(sum_se/sum_mean)

def model_Evaluate(training_y,labels):
    R2 = R_2(training_y,labels)
    print("R2:",R2)

if __name__ == '__main__':
    training_data,training_label,test_data,test_label = dataSet()

    training_data2 = training_data

    # 将df转成np中的array
    training_data = np.array(training_data)
    training_labels = np.array(training_label)

    test_data = np.array(test_data)
    test_labels = np.array(test_label)

    # 设置学习率
    learning_rate = 0.0000001

    theta = np.zeros(len(training_data[0]))
    # 梯度下降
    theta = Gradient_Descent(training_data, training_labels, learning_rate, loop_times=10000)
    h_y = np.zeros(len(test_labels))

    for i in range(len(test_labels)):
        h_y[i] = h(test_data[i],theta)
    print("梯度下降：")
    model_Evaluate(h_y, test_labels)

    # 解析式求解
    theta = Analytic_Method(training_data, training_labels)

    h_y = np.zeros(len(test_labels))

    for i in range(len(test_labels)):
        h_y[i] = h(test_data[i],theta)

    print("解析式求解：")
    model_Evaluate(h_y,test_labels)
