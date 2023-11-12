from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import bayesian

import data_loader, feature_extract, evaluation

classes = ['airplane', 'automobile','brid','cat','deer','dog','frog','horse','ship','truck']

# 从文件中读取数据
x_train, y_train, x_test, y_test = data_loader.load_data()

# 特征提取
x_train, x_test = feature_extract.get_feature(x_train, x_test)

# 标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# PCA
pca = PCA(n_components=0.8)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

print('--- start fitting ---')
Model_Bayesian_Diy = bayesian.Bayesian()
Model_Bayesian_Diy.fit(x_train,y_train)

Model_KNeighbors = KNeighborsClassifier(n_neighbors=8)
Model_KNeighbors = Model_KNeighbors.fit(x_train, y_train)

Model_Beyasian = GaussianNB()
Model_Beyasian.fit(x_train,y_train)

Model_Decisiontree = DecisionTreeClassifier()
Model_Decisiontree.fit(x_train,y_train)

Model_Randomforest = RandomForestClassifier()
Model_Randomforest.fit(x_train,y_train)

Model_SVC = SVC(kernel="rbf", decision_function_shape="ovo")
Model_SVC.fit(x_train,y_train)

print('--- fitting done ---')
print('--- Bayesian Diy Result ---')
y_pred = Model_Bayesian_Diy.predict(x_test)
evaluation.ModelEvaluation(y_true = y_test, y_pred = y_pred, ModelName = 'test')

print('--- Bayesian Result ---')
y_pred = Model_Beyasian.predict(x_test)
evaluation.ModelEvaluation(y_true = y_test, y_pred = y_pred, ModelName = 'test')

print('--- KNeighbors Result ---')
y_pred = Model_KNeighbors.predict(x_test)
evaluation.ModelEvaluation(y_true = y_test, y_pred = y_pred, ModelName = 'test')

print('--- Decision Tree Result ---')
y_pred = Model_Decisiontree.predict(x_test)
evaluation.ModelEvaluation(y_true = y_test, y_pred = y_pred, ModelName = 'test')

print('--- Decision Tree Diy Result ---')



print('--- Random Forest Result ---')
y_pred = Model_Randomforest.predict(x_test)
evaluation.ModelEvaluation(y_true = y_test, y_pred = y_pred, ModelName = 'test')

print('--- SVC Result ---')
y_pred = Model_SVC.predict(x_test)
evaluation.ModelEvaluation(y_true = y_test, y_pred = y_pred, ModelName = 'test')

