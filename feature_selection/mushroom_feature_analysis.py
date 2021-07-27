'''
参考：https://mp.weixin.qq.com/s/oqGpYOy-riqgvL5HFxnzPQ
'''
import pandas as pd
#from pygments.lexers import graphviz
from graphviz import Source
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import graphviz

mush_file = '../datas/mushrooms.csv' #8124 rows x 22 columns
mush_df = pd.read_csv(mush_file)
print(type(mush_df),mush_df)
X = mush_df.drop(['class'],axis = 1) #labels : single label or list-like Index or column labels to drop. axis : {0 or 'index', 1 or 'columns'}
Y = mush_df['class']
print('X',X)
print('Y',Y)
#编码方式 one-hot
Y = LabelEncoder().fit_transform(Y) #y should be a 1d array
print('\n---',Y) #[1 0 0 ... 0 1 0]

'''
get_dummies: #Convert categorical variable into dummy/indicator variables.
prefix_sep or prefix :在column_name后面追加设定的prefix 如cap-shape_b  cap-shape_c
'''
X = pd.get_dummies(X,prefix_sep='_')
print('get_dummies \n',X)
'''
StandardScaler:去均值和方差归一化
 z = (x - u) / s , Fit to data, then transform it
 
fit_transform,fit,transform区别和作用详解！！！！！！： https://blog.csdn.net/weixin_38278334/article/details/82971752
fit_transform：sklearn中抽象的特征预处理方法，包括fit和transform两个步骤
fit:一般来说就是根据数据求得训练集X的均值，方差，最大值最小值等这些训练集X的固有属性
transform:在fit的基础上，进行标准化，降维，归一化等操作
一般这样可以消除不同的特征上量纲的影响
不同的特征处理方式得到的结果不同，常用的有
StandardScaler：
CountVectorizer
TfidfTransformer

'''
X2 =X #StandardScaler().fit_transform(X)
#print('\n X2',type(X2),X2) # numpy.ndarray 每个数据表示一个样本及对应的特征

'''
Split arrays or matrices into random train and test subsets
'''
X_train,X_test,Y_train,Y_test = train_test_split(X2,Y,test_size=0.3,random_state=101)

'''
n_estimators:The number of trees in the forest.
'''
trained_forest = RandomForestClassifier(n_estimators=700).fit(X_train,Y_train)
predictionforest  = trained_forest.predict(X_test)
print(predictionforest)

print(confusion_matrix(Y_test,predictionforest))
'''
Build a text report showing the main classification metrics
precision/recall/f1-score
macro/micro/weighted avg
'''
print(classification_report(Y_test,predictionforest))
print('feature size:',len(trained_forest.feature_importances_),trained_forest.feature_importances_)
#查看随机森林中特征重要性
plt.figure(figsize=(20,22),dpi=80,facecolor='w',edgecolor='k')
feat_importances = pd.Series(trained_forest.feature_importances_,index=X.columns)
#查看特征最重要的7个（nlargest(7)）
feat_importances.nlargest(7).plot(kind='barh')
# plt.show()

# 尝试使用前三个特征训练模型
X_reduce = X[['odor_n','odor_f', 'gill-size_n','gill-size_b']]
X_reduce = StandardScaler().fit_transform(X_reduce)
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_reduce,Y,test_size=0.30,random_state=101)
trainedforest2 = RandomForestClassifier(n_estimators=700).fit(X_train2,Y_train2)
predictionforest2 = trainedforest2.predict(X_test2)
print(confusion_matrix(Y_test2,predictionforest2))
print(classification_report(Y_test2,predictionforest2))
'''
通过一颗决策树可视化特征重要性
export_graphviz:
Export a decision tree in DOT format

step1:pip install graphviz
step2:下载graphviz http://www.graphviz.org/download/
增加系统路径：D:\Program Files\Graphviz\bin
'''

trainedtree  = DecisionTreeClassifier().fit(X_train,Y_train)
predicetree = trainedtree.predict(X_test)

print(confusion_matrix(Y_test,predicetree))
print(classification_report(Y_test,predicetree))

# data = export_graphviz(trainedtree,out_file=None,feature_names=X.columns,
#                 class_names=['edible','poisonous'],
#                 filled=True,rounded=True,max_depth=2,special_characters=True)
# print('======',type(data),data)
# graph: Source = graphviz.Source(data)
# graph.view('tree_feature',directory='../datas')


