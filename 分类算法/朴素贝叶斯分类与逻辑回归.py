import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import math

# 设置数据
def dataSet(path):
    # 读取数据
    data = pd.read_csv(path, header=None)
    # 给数据设置列名
    list_name = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    data.columns = list_name
    # 清洗数据
    # 将缺省值替换成nan
    data.replace({' ?': np.nan}, inplace=True)
    # 删除带有nan的数据
    data.dropna(axis=0, how='any', inplace=True)
    # 删除完全重复的一行
    data.drop_duplicates(inplace=True)

    # 将每一个含有类别变量的列都转化为含有整数值的列
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    label_encoder = LabelEncoder()
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # 将数据的标签值分离出来
    labels = data['salary']
    data.drop('salary', axis=1, inplace=True)

    # 将第三列的数据缩小，防止对判断造成过大的干扰
    data[list_name[2]] = data[list_name[2]]*0.0001

    # 重新设置索引
    data.index = range(len(data))
    labels.index = range(len(labels))

    return data, labels

# Sigmoid函数,返回一个预测值的计算
def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

# 梯度上升：
def gradient_ascend(labels, training_data, lr, threshold, loop_times):
    # 数据的总行数
    row = len(training_data)
    # 数据的总列数
    column = len(training_data[0])
    # 小批量梯度下降一次选用的数据量
    length = 2500
    theta = np.ones((column,1))
    theta = np.mat(theta)
    # 预测值
    hy = np.ones((length, 1))
    hy = np.mat(hy)
    # 真实值
    lb = np.ones((length, 1))
    lb = np.mat(lb)

    # 循环次数
    for times in range(loop_times):
        # 从0开始每次取length长的数据
        num = (times*length)%row
        # 记录lenght个样本点的下标
        idx = np.zeros(length)
        for i in range(length):
            idx[i] = int((num+i) % row)
        idx = idx.astype(int)       # 将idx的数据类型转成int型

        XX = np.zeros((length, column))
        XX = np.mat(XX)
        for i in range(length):
            XX[i] = training_data[idx[i]]       # 将2500个数据放入XX中(2000 X 14)
            lb[i] = labels[idx[i]]              # 将2500个标签放入lb中(2000 X 1)

        hy = Sigmoid(XX * theta)
        error = (lb - hy)
        # 梯度上升
        theta =theta + lr * (XX.T * error)

        error = error.getA()
        # 统计误差值
        for i in range(length):
            error[i] = abs(error[i])
            if error[i] < 0.5:      # 划分0和1
                error[i] = 0
            else:
                error[i] = 1

        # print("真实值：", sum(lb))
        # print("猜测的真实值：",sum(hy))
        # print("误差数", sum(error))
        # print("theta:",theta.T)
        print("正确率（训练集）：",1-(sum(error)/length))

        # 设置阈值，当训练集的学习率超过某一个值时直接返回
        if 1-(sum(error)/length) > threshold:
            theta = theta.getA()
            return theta
    theta = theta.getA()
    return theta

# 计算正态分布
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

# 计算似然概率
def L(data,labels):
    colunms = len(data.values[0])
    row = len(data.values)
    list_name = data.columns.tolist()

    # 将离散型变量和连续型变量的索引分开
    dv = []     # 离散型变量的索引
    cv = []     # 连续型变量
    for j in range(colunms):
        if (j !=0 and j!=2 and j!=4 and j!=10 and j!=11 and j!=12):
            dv.append(j)
        else:
            cv.append(j)

    L1 = {}          # 存放概率的字典，其中y=1
    L0 = {}          # 存放概率的字典，其中y=0
    tmp_n = []        # 暂时存放某些数的列名

    for j in dv:    # 将离散值的列表名存到tmp_n中
        tmp_n.append(list_name[j])

    # 计算所有似然函数(离散的)
    for i in range(len(dv)):
        total1 = sum(labels)
        total0 = row-total1
        l = len(data[tmp_n[i]].value_counts())
        tmp1 = np.zeros(l)
        tmp0 = np.zeros(l)

        tmp_dv = data[tmp_n[i]]

        # # 统计数量
        for j in range(l):
            for k in range(row):
                if tmp_dv[k]==j and labels[k]==1:
                    tmp1[j] +=1
                if tmp_dv[k]==j and labels[k]==0:
                    tmp0[j] +=1

        # 拉普拉斯平滑
        for j in range(l):
            if tmp1[j] ==0:
                tmp1[j] +=1
                total1 +=1
            if tmp0[j] ==0:
                tmp0[j] +=1
                total0 +=1

        tmp1 = tmp1/total1
        tmp0 = tmp0/total0

        # 计算每个特征x中每个特征值的条件概率
        L1[tmp_n[i]] = tmp1
        L0[tmp_n[i]] = tmp0

    L_mean_var_y1 = {}
    L_mean_var_y0 = {}
    tmp_n2 = []
    for i in cv:    # 将离散值的列表名存到tmp_n中
        tmp_n2.append(list_name[i])

    # 将数据分成正负两类
    dt_y1 = pd.DataFrame(columns=list_name)
    dt_y0 = pd.DataFrame(columns=list_name)
    for i in range(row):
        if labels[i] ==1:
            tmp = pd.DataFrame([data.values[i]],columns=list_name)
            dt_y1 = dt_y1.append(tmp)

        else:
            tmp = pd.DataFrame([data.values[i]],columns=list_name)
            dt_y0 = dt_y0.append([tmp])


    # # 计算所有似然函数的平均值与方差(连续的)
    for i in range(len(cv)):
        tmp = np.zeros(2)
        tmp[0] = np.mean(dt_y1[tmp_n2[i]])
        tmp[1] = np.std(dt_y1[tmp_n2[i]])
        L_mean_var_y1[tmp_n2[i]] = tmp

    for i in range(len(cv)):
        tmp = np.zeros(2)
        tmp[0] = np.mean(dt_y0[tmp_n2[i]])
        tmp[1] = np.std(dt_y0[tmp_n2[i]])
        L_mean_var_y0[tmp_n2[i]] = tmp

    return L1,L0,L_mean_var_y1,L_mean_var_y0


# 朴素贝叶斯
def naive_B(data, labels):
    # 读取测试集数据
    L1, L0, L_mean_var_y1,L_mean_var_y0 = L(data,labels)
    test_data,test_labels = dataSet("ex2/adult_test.csv")
    hy = np.zeros((len(test_labels),1))
    dv_name = L1.keys()
    cv_name = L_mean_var_y1.keys()
    list_name = data.columns.tolist()
    row = len(test_data)
    column = len(test_data.values[0])

    # 预测所有样本点的值
    for times in range(row):
        y0=1
        y1=1
        for i in range(column):
            if list_name[i] in dv_name:
                tmp = test_data[list_name[i]][times]
                y1 = y1*L1[list_name[i]][tmp]
                y0 = y0*L0[list_name[i]][tmp]
            elif list_name[i] in cv_name:
                y1 = y1*normal_distribution(test_data[list_name[i]][times], L_mean_var_y1[list_name[i]][0], L_mean_var_y1[list_name[i]][1])
                y0 = y0*normal_distribution(test_data[list_name[i]][times], L_mean_var_y0[list_name[i]][0], L_mean_var_y0[list_name[i]][1])
        # 计算测试集中正负例的概率
        y1 = y1*(sum(labels)/len(labels))
        y0 = y0*(1-(sum(labels)/len(labels)))

        if y1>y0:
            hy[times] = 1
        else:
            hy[times] = 0
    # 模型估计
    test_labels = np.array(test_labels)
    evaluate_model(hy,test_labels)

# 测试模型(梯度上升的)
def test_model(theta):
    # 对测试集进行预测
    test_data,test_labels = dataSet("ex2/adult_test.csv")
    row,columns = np.shape(test_data)
    hy = np.ones((row, 1))
    hy = np.mat(hy)
    theta = np.mat(theta)
    test_data = np.mat(test_data)
    hy = Sigmoid(test_data * theta)

    # 计算测试集的正确率、AUC并且画出ROC曲线
    evaluate_model(hy,test_labels)

# 画AUC图
def AUC_P(FPR,TPR,AUC):
    plt.figure()
    plt.title('ROC CURVE (AUC={:.2f})'.format(AUC))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot(FPR, TPR, color='g')
    plt.plot([0, 1], [0, 1], color='m', linestyle='--')
    plt.show()

# 评估模型函数
def evaluate_model(hy,target):
    l = len(target)
    FPR, TPR, threshold = roc_curve(target, hy, pos_label=1)
    AUC = auc(FPR,TPR)
    hy = np.mat(hy)
    target = np.mat(target)
    target = target.T

    error = target - hy

    # 将预测值规范化
    for i in range(l):
        if hy[i]<0.5:
            hy[i]=0
        else:
            hy[i]=1
    # 评估测试集的正确率
    for i in range(l):
        error[i] = abs(error[i])
        if error[i]<0.5:
            error[i] = 0
        else:
            error[i] = 1
    print("正确率(测试集)：",1-(sum(error)/l))

    # 画图
    AUC_P(FPR,TPR,AUC)

# 逻辑回归
def Logistic_regression(data,labels):
    # 学习率、循环次数、训练集中正确率的最大阈值
    lr = 0.000001
    loop_times = 100000
    threshold = 0.805
    theta = gradient_ascend(labels, data, lr,threshold, loop_times)

    # 模型评价
    test_model(theta)

if __name__ == '__main__':
    data, labels = dataSet("ex2/adult_data.csv")

    # 将数据类型转为array
    data_1 = np.array(data)
    labels_1 = np.array(labels)
    # 逻辑回归
    Logistic_regression(data_1,labels_1)
    # 朴素贝叶斯
    naive_B(data, labels)

    # 测试点
    # list_name = data.columns.tolist()
    # a = pd.DataFrame(columns=list_name)
    # c = []
    # c = data.values[0]
    # b = pd.DataFrame([c], columns=list_name)
    # # b = data.values[0]
    # print(b)
    # a = a.append(b)
    # a = a.append(b)
    # print(a)



