import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
import random

pd.set_option('display.max_columns',None)

"""
设置数据
path：文件路径
"""
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

    # 暂时删除耗时较长的维度
    # # data.drop(list_name[0], axis=1, inplace=True)
    # data.drop(list_name[1], axis=1, inplace=True)
    # data.drop(list_name[2], axis=1, inplace=True)
    # data.drop(list_name[3], axis=1, inplace=True)
    # # data.drop(list_name[4], axis=1, inplace=True)
    # # data.drop(list_name[5], axis=1, inplace=True)
    # # data.drop(list_name[6], axis=1, inplace=True)
    # # data.drop(list_name[7], axis=1, inplace=True)
    # data.drop(list_name[8], axis=1, inplace=True)
    # data.drop(list_name[9], axis=1, inplace=True)
    # data.drop(list_name[10], axis=1, inplace=True)
    # data.drop(list_name[11], axis=1, inplace=True)
    # data.drop(list_name[12], axis=1, inplace=True)
    # data.drop(list_name[13], axis=1, inplace=True)

    # 重新设置索引
    data.index = range(len(data))
    labels.index = range(len(labels))
    return data, labels

"""
将数据中的连续值转化成离散变量
data:含连续值的数据
labels：标签
feature：要离散化的特征
"""
def cv_to_dv(data,labels,feature):
    # 要离散化的数据
    # 将标签和特征值的一列合并在一起
    dt = pd.concat([data[feature],labels],axis=1)
    # 将刚刚合并的数据，按照特征那一列排序
    dt.sort_values(by=feature, inplace=True,ascending=True)
    # 将经过排序的标签切出来备用
    y = dt['salary']

    # 计算特征的分布情况
    num = data[feature].value_counts()
    # 计算各特征的所占概率（备用）
    p_num = num/len(labels)
    idx = sorted(num.index)
    # 定义插入点存放变量（划分点）
    insert_pot = np.zeros(len(idx)-1)
    # 计算插入点
    for i in range(len(idx)-1):
        insert_pot[i] = (idx[i]+idx[i+1])/2
    # 记录前x个数据
    head_num = 0

    # 计算原数据的信息熵
    H_p = Infor_Entropy(y,2)
    # 记录信息增益最大的特征值(第一个数为特征值，第二个为信息增益)
    max_sign = np.zeros(2)

    for i in idx:
        head_num +=num[i]
        # 分别计算切分两边数据的信息熵
        a = Infor_Entropy(y[:head_num],2)
        b = Infor_Entropy(y[head_num:],2)
        # 计算不同划分的信息熵
        H = sum(p_num[:i])*a + sum(p_num[i:])*b

        # print(H)

        # 查找最大值
        if max_sign[1] <(H_p-H):
            max_sign[0] = i
            max_sign[1] = (H_p-H)

    # 记录划分点的值
    pot = 0
    for i in insert_pot:
        if max_sign[0]<i:
            pot = i
            break

    # 将特征值离散化
    f = data[feature]
    data.loc[f < pot,feature] = 0
    data.loc[f > pot,feature] = 1
    print("分离点为：",pot)
    print("数据",feature,"离散化成功")

    return data,pot


"""
计算信息熵
labels：不同的分类的标签
K：分类的个数
"""
def Infor_Entropy(labels,K):
    N = len(labels)
    if N==0:
        return 0
    p = []
    # 计算所有类别的概率
    p.append(sum(labels) / N)
    p.append(1 - (sum(labels) / N))
    H = 0
    # 计算信息熵

    for i in range(K):
        if p[i]==0 or p[i]==1:
            H +=0
        else:
            H += -p[i] * math.log2(p[i])
    return H

"""
计算信息增益
data:数据
K：标签类别的个数
labels：标签
feature：选择的特征
"""
def Infor_Gain(data, K, labels, feature):
    # 总数据个数
    N = len(labels)
    # 统计该特征所有的可能取值
    num = data[feature].value_counts()
    # 记录特征值的所有可能取值数
    n = len(num)
    # 获取所有特征的取值
    xxi = num.index
    xxi = sorted(xxi)

    # 计算所有取值的概率p(X=xi)，注：p的索引是乱的
    p_i = num/N

    # 计算特征所有取值的条件概率p(Y=1|X=xi)
    # 将标签和特征值的一列合并在一起
    dt = pd.concat([data[feature],labels],axis=1)
    # 将刚刚合并的数据，按照特征那一列排序
    dt.sort_values(by=feature, inplace=True,ascending=True)
    # 将排好序的标签拿出来
    y = dt['salary']
    # 设置条件概率p(Y=1|X=xi)的记录变量
    P_y1_i = np.zeros(n)
    # 标记量，用于作为标签值的索引
    tmp = 0
    # 统计计算条件概率p(Y=1|X=xi)
    for i in range(n):
        P_y1_i[i] = sum(y[tmp:tmp+num[xxi[i]]])/num[xxi[i]]
        # 每一次都更改一下标签的索引
        tmp = num[xxi[i]]
    # 此时的P_y1_i就是p(Y=1|X=xi)的一个矩阵
    # 将条件概率及xi的取值一起打包
    P = {}
    P['xi'] = xxi
    P['p'] = P_y1_i

    # 计算条件信息熵
    H = 0
    # 计算条件信息熵
    for i in range(n):
        tmp_p = P['p'][i]
        # 将条件概率为0或1的p*log(p)设为0
        if tmp_p ==0 or tmp_p==1:
            h=0
        else:
            h = -p_i[P['xi'][i]] * (tmp_p * math.log2(tmp_p)-(1-tmp_p)*math.log2(1-tmp_p))

        H += h

    return Infor_Entropy(labels,K)-H

"""
计算信息增益比
data:数据
K：标签类别的个数
labels：标签
feature：选择的特征
"""
def Infor_Gain_Rate(data,K,labels,feature):
    # 总数据个数
    N = len(labels)
    # 统计该特征所有的可能取值
    num = data[feature].value_counts()
    # 记录特征值的所有可能取值数
    n = len(num)
    # 获取所有特征的取值
    xxi = num.index
    xxi = sorted(xxi)

    # 计算所有取值的概率p(X=xi)，注：p的索引是乱的
    p_i = num/N

    # 计算特征所有取值的条件概率p(Y=1|X=xi)
    # 将标签和特征值的一列合并在一起
    dt = pd.concat([data[feature],labels],axis=1)
    # 将刚刚合并的数据，按照特征那一列排序
    dt.sort_values(by=feature, inplace=True,ascending=True)
    # 将排好序的标签拿出来
    y = dt['salary']
    # 设置条件概率p(Y=1|X=xi)的记录变量
    P_y1_i = np.zeros(n)
    # 标记量，用于作为标签值的索引
    tmp = 0
    # 统计计算条件概率p(Y=1|X=xi)
    for i in range(n):
        P_y1_i[i] = sum(y[tmp:tmp+num[xxi[i]]])/num[xxi[i]]
        # 每一次都更改一下标签的索引
        tmp = num[xxi[i]]
    # 此时的P_y1_i就是p(Y=1|X=xi)的一个矩阵
    # 将条件概率及xi的取值一起打包
    P = {}
    P['xi'] = xxi
    P['p'] = P_y1_i

    # 计算条件信息熵
    H = 0
    # 计算条件信息熵
    for i in range(n):
        tmp_p = P['p'][i]
        # 将条件概率为0或1的p*log(p)设为0
        if tmp_p ==0 or tmp_p==1:
            h=0
        else:
            h = -p_i[P['xi'][i]] * (tmp_p * math.log2(tmp_p)-(1-tmp_p)*math.log2(1-tmp_p))

        H += h

    # 计算HA
    HA = 0
    for i in num.index:
        HA -= (num[i]/len(data))*math.log2(num[i]/len(data))

    if HA==0:
        HA = 1

    return (Infor_Entropy(labels,K)-H)/HA

"""
生成树结点：
Tree：决策树的存储结构（字典）
data：数据
labels：数据对应的标签
list_name：属性
"""
def createTree(data, labels, list_name):
    # 生成叶子结点
    # 当所有属性划分完时,选取多数进行分类
    if not list_name:
        # 小于0.5代表labels中多数为0
        if sum(labels)/len(labels) < 0.5:
            return 'no'
        # 大于0.5代表labels中多数为1
        else:
            return 'yes'

    # 记录属性名称的临时变量
    list_name_tmp = list_name.copy()

    # 记录信息增益
    HI = np.zeros(len(list_name_tmp))
    # 计算信息增益
    for i in range(len(list_name_tmp)):
        HI[i] = Infor_Gain(data, 2, labels, list_name_tmp[i])
    # 记录信息增益最大的属性的下标
    idx = 0
    # 记录信息增益的最大值
    tmp_max = 0
    for i in range(len(HI)):
        if HI[i] > tmp_max:
            tmp_max = HI[i]
            idx = i

    # 记录信息增益最大的属性
    tmp_feature = list_name_tmp.pop(idx)
    # 生成一颗树待用
    Tree = {tmp_feature:{}}
    # 将数据按照某一属性排序
    # 将标签整合在一起再排序
    data = pd.concat([data,labels],axis=1)

    data.sort_values(by=tmp_feature,inplace=True,ascending=True)

    labels = data['salary']
    # 再将标签删掉
    data.drop('salary', axis=1, inplace=True)
    # 统计数据中某一属性的分布情况
    num = data[tmp_feature].value_counts()
    # 将属性特征的下标进行排序
    tmp_idx = sorted(num.index)

    tmp_be = 0
    tmp_ed = 0
    # 生成叶子结点
    # 当全0或全1时直接返回结果
    if sum(labels)==0 or sum(labels)==len(labels):
        if sum(labels)==0:
            return 'no'
        elif sum(labels)==len(labels):
            return 'yes'

    # 根据属性的特征值生成结点
    for i in tmp_idx:
        tmp_ed +=num[i]
        # 循环递归生成子树
        Tree[tmp_feature][i]=createTree(data[tmp_be:tmp_ed],labels[tmp_be:tmp_ed], list_name_tmp)

        tmp_be +=num[i]

    # 返回最终结果
    return Tree

"""
生成树结点：
Tree：决策树的存储结构（字典）
data：数据
labels：数据对应的标签
list_name：属性
"""
def createTree_Infor_GainRate(data, labels, list_name):
    # 生成叶子结点
    # 当所有属性划分完时,选取多数进行分类
    if not list_name:
        # 小于0.5代表labels中多数为0
        if sum(labels)/len(labels) < 0.5:
            return 'no'
        # 大于0.5代表labels中多数为1
        else:
            return 'yes'

    # 记录属性名称的临时变量
    list_name_tmp = list_name.copy()

    # 记录信息增益
    HI_Rate = np.zeros(len(list_name_tmp))
    # 计算信息增益
    for i in range(len(list_name_tmp)):
        HI_Rate[i] = Infor_Gain_Rate(data, 2, labels, list_name_tmp[i])
    # 记录信息增益最大的属性的下标
    idx = 0
    # 记录信息增益的最大值
    tmp_max = 0
    for i in range(len(HI_Rate)):
        if HI_Rate[i] > tmp_max:
            tmp_max = HI_Rate[i]
            idx = i

    # 记录信息增益最大的属性
    tmp_feature = list_name_tmp.pop(idx)
    # 生成一颗树待用
    Tree = {tmp_feature:{}}
    # 将数据按照某一属性排序
    # 将标签整合在一起再排序
    data = pd.concat([data,labels],axis=1)

    data.sort_values(by=tmp_feature,inplace=True,ascending=True)

    labels = data['salary']
    # 再将标签删掉
    data.drop('salary', axis=1, inplace=True)
    # 统计数据中某一属性的分布情况
    num = data[tmp_feature].value_counts()
    # 将属性特征的下标进行排序
    tmp_idx = sorted(num.index)

    tmp_be = 0
    tmp_ed = 0
    # 生成叶子结点
    # 当全0或全1时直接返回结果
    if sum(labels)==0 or sum(labels)==len(labels):
        if sum(labels)==0:
            return 'no'
        elif sum(labels)==len(labels):
            return 'yes'

    # 根据属性的特征值生成结点
    for i in tmp_idx:
        tmp_ed +=num[i]
        # 循环递归生成子树
        Tree[tmp_feature][i]=createTree(data[tmp_be:tmp_ed],labels[tmp_be:tmp_ed], list_name_tmp)

        tmp_be +=num[i]

    # 返回最终结果
    return Tree

"""
*
使用决策树进行分类：
Tree：决策树
list_name：待划分的特征值
data：待分类的数据
"""
# 根据决策树对点进行分类
def classify(Tree, list_name, data):
   firstStr = next(iter(Tree))  # 获取根节点
   secondTree = Tree[firstStr]  # 获取根节点下的两颗树
   featIndex = list_name.index(firstStr)
   classLabel = -1
   for key in secondTree.keys():
       if data[featIndex] == key:
           if type(secondTree[key]).__name__ == 'dict':  # 如果子节点是一棵树，那么递归调用分类器
               classLabel = classify(secondTree[key], list_name, data)
           else:                                        # 如果子节点是一个叶子节点，那么直接返回
               if secondTree[key]=='yes':
                   classLabel=1
               else:
                   classLabel=0
   return classLabel

"""
将多个结果总结出一个预测值：
yh：一组二维的预测值
优先度为：1>0>-1
"""
def class_labels(yh):
    conlumn = len(yh)
    one = 0
    zero = 0
    miss = 0
    for j in range(conlumn):
        if yh[j] == 1:
            one +=1
        elif yh[j] == 0:
            zero +=1
        else:
            miss +=1
    if one >=2:
        return 1
    elif zero >=2:
        return 0
    else:
        return -1

"""
随机生成树
data：带标签的数据集
"""
def random_CreateTree(data):
    # 删除特征的数量
    N = random.randint(3,10)
    # 随机抽取6成数据进行决策树的构成
    tmp_data = data.sample(frac=0.6)
    tmp_labels = tmp_data['salary']
    tmp_data.drop('salary', axis=1, inplace=True)

    # 随机选取列数据删掉
    list_name = tmp_data.columns.tolist()
    del_num = random.sample(range(0, 13), N)
    # 记录要删的列名
    del_name = []
    for i in del_num:
        del_name.append(list_name[i])
    for i in del_name:
        list_name.remove(i)
        tmp_data.drop(i, axis=1, inplace=True)
    # 随机选择信息增益还是信息增益比来生成树
    num = random.randint(0,1)
    if num ==0:
        return createTree_Infor_GainRate(tmp_data,tmp_labels,list_name)
    else:
        return createTree(tmp_data,tmp_labels,list_name)


"""
随机森林
data：带标签的数据集
"""
def random_forest(data):
    # 树的棵数
    N = 5
    RF = {}
    for i in range(N):
        RF[i] = {}
        # 随机抽取数据进行构建决策树
        RF[i] = random_CreateTree(data)

    return RF

"""
测试函数
"""
def test(dt,cv_num,cv_values):
    N = 5       #树的棵数
    data,labels = dataSet("ex4/adult_test.csv")
    yh = np.zeros((len(labels),N))
    list_name = data.columns.tolist()
    RF = {}
    RF = random_forest(dt)

    # 将测试集的数据离散化
    for i in range(len(cv_num)):
        f = data[list_name[cv_num[i]]]
        data.loc[f < cv_values[i],list_name[cv_num[i]]] = 0
        data.loc[f > cv_values[i],list_name[cv_num[i]]] = 1

    print("开始分类：")
    yh = np.zeros((len(labels),N))
    print(yh[0])
    print(len(yh[0]))

    tt = np.zeros(N)
    for i in range(N):
        Tree = RF[i]
        tt[i] = classify(Tree,list_name,data.values[0])
        for j in range(len(labels)):
            yh[j][i]=classify(Tree,list_name,data.values[j])

    # 将多个预测值综合的出最终结果
    y_h = np.zeros(len(yh))
    errors = np.zeros(len(yh))

    for i in range(len(yh)):
        y_h[i] = class_labels(yh[i])
        if y_h[i]==-1:
            labels.drop(i,inplace=True)
        else:
            errors[i] = abs(labels[i] - y_h[i])
    print("测试集错误率为：",sum(errors)/len(labels))
    print("测试数据数为：",len(labels))




if __name__ == '__main__':
    # 决策树
    Tree = {}

    data,labels = dataSet("ex4/adult_data.csv")
    list_name = data.columns.tolist()
    # 连续变量的下标
    cv_num = [0,2,4,10,11,12]

    # 存放连续变量离散化的值
    cv_values = np.zeros(len(cv_num))
    # 将连续值转成离散值
    for i in range(len(cv_num)):
        data,cv_values[i] = cv_to_dv(data,labels,list_name[cv_num[i]])

    dt = pd.concat([data,labels],axis=1)

    test(dt,cv_num,cv_values)

