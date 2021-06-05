import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import train_test_split
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

df_train = pd.read_csv('C://Users//Root//PycharmProjects//train.csv')
df_test = pd.read_csv('C://Users//Root//PycharmProjects///test.csv')

df_head_train = df_train.head(n=5)
print(df_head_train)

df_unique_train = df_train.tare.unique()
print(df_unique_train)
print(len(df_unique_train))
print(len(df_train['name']))

sw = stopwords.words('english')
for i in sw:
    df_train['new_name'] = df_train['question_text'].replace(i, '')
pattern = r"\d+."
df_train['new_name'] = df_train['new_name'].str.replace(pattern, '')
df_head_train = df_train.head(n=5)
print(df_head_train)

#определяем тестовую и тренировочную выборки
features_train, features_test, labels_train, labels_test = train_test_split(df_train['new_name'], df_train['target'], test_size=0.3, random_state=42)
vectorizer = TfidfVectorizer(sublinear_tf=False)
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed = vectorizer.transform(features_test)

n_components = 100
pca = TruncatedSVD(n_components=n_components, random_state=42, n_iter=10).fit(features_train_transformed)

X_train_pca = pca.transform(features_train_transformed)
X_test_pca = pca.transform(features_test_transformed)

parameters = {'n_estimators':(10, 40, 100), 'criterion':('gini', 'entropy'), 'random_state':[42]}
first_tree = RandomForestClassifier()
clf = GridSearchCV(first_tree, parameters)
clf.fit(X_train_pca, labels_train)
best_par = clf.best_params_
print(best_par)

#Таким способом было выявлено, что при n_components=100 (для TruncatedSVD) оптимальные параметры
#случайного леса 'criterion': 'gini', 'n_estimators': 100, 'random_state': 42. При увеличении
#количества деревьев и/или компонент точность увеличивается на 1-2%, время тренировки в 1,25
#и более раз.

#RandomForest - наибольшее значение accuracy=0.750717507175, время обучения: 281,541s.
#n_components=250, estimators=200
#ExtraTreesClassifier - наибольшее значение accuracy=0.753341533415, время обучения:  225.115 s
#n_components = 250, estimators=150
#GaussianNB accuracy=0.376055760558, время обучения 5.6s. Параметры:default, n_components = 250
#RidgeClassifierCV - наилучшими параметрами оказались fit_intercept': False, 'normalize': True).
#С указанными параметрами accuracy=0.627306273063, training time: 12.34 s, n_components = 250

#Выше указаны только те параметры моделей, которые менялись/подбирались Gridsearch. Остальные значения
#были default.

#далее надо выбрать модель, обучить и посмотреть результаты

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=150, criterion='gini',random_state=42)
clf.fit(X_train_pca, labels_train)
predicted_values_0 = clf.predict(X_test_pca)
accuracy = accuracy_score(labels_test, predicted_values_0)
features_test_transformed_submission = vectorizer.transform(df_test.question_text)
X_test_pca_submission = pca.transform(features_test_transformed_submission)
predicted_values_for_submission = clf.predict(X_test_pca_submission)

print(accuracy)
#print(predicted_values)
submisiion = pd.DataFrame({'qid':df_test.qid, 'prediction': predicted_values_for_submission}, columns=['qid', 'prediction'])
submisiion.to_csv('submission.csv', index=False)
'''[5 rows x 4 columns]
0.94086061295896
[0 0 0 ... 0 0 0]
Process finished with exit code 0
'''
#далее в силу того что в начале были поделены тренировочные данные, часть операций надо повторить
#уже на полном наборе данных

#performing same thing for all dataset
training_features = df_train['new_name']
testing_labels = df_train['target']
sw = stopwords.words('russian')
for i in sw:
    df_test['new_name'] = df_test['name'].replace(i, '')
pattern = r"\d+."
df_test['new_name'] = df_test['new_name'].str.replace(pattern, '')
testing_features = df_test['new_name']
training_features_transformed = vectorizer.fit_transform(training_features)
testing_features_transformed = vectorizer.transform(testing_features)
n_components = 250 #10, 15, 25, 50, 100, 250. for LSA recommended 100 components 
pca = TruncatedSVD(n_components=n_components, random_state=42, n_iter=10).fit(training_features_transformed)
X_train_pca_1 = pca.transform(training_features_transformed)
X_test_pca_1 = pca.transform(testing_features_transformed)
t0 = time()
second_tree = ExtraTreesClassifier()
clf1 = ExtraTreesClassifier(n_estimators=150, criterion='gini',random_state=42)
clf1.fit(X_train_pca_1, testing_labels)
predicted_values_1 = clf1.predict(X_test_pca_1)
