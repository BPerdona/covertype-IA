### DESCRIÇÃO DE TODOS OS CAMPOS
#1-Elevation: Elevação em metros
#2-Aspect: Aspecto em graus azimute
#3-Slope: Inclinação em graus
#4-Horizontal_Distance_To_Hydrology: distância horizontal para as características de água de superfície mais próximas
#5-Vertical_Distance_To_Hydrology: distância vertical para as características de água de superfície mais próximas
#6-Horizontal_Distance_To_Roadways: distância horizontal para as estradas mais próximas
#7-Hillshade_9am: índice de sombra às 9h
#8-Hillshade_Noon: índice sombra ao meio-dia
#9-Hillshade_3pm: índice de sombra às 15h
#10-Horizontal_Distance_To_Fire_Points: distância horizontal para os pontos de ignição de incêndios florestais mais próximos
#11-Wilderness_Area: 4 colunas binárias. 0 indica ausente e 1 indica presente
#12-Soil_Type: 40 colunas binárias. 0 indica ausente e 1 indica presente
#13-Cover_Type: 7 tipos de cobertura florestal

## Part 1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


## Exploração de dados
ct = pd.read_csv('covtype.csv')

## Descrevendo
ct.describe()
ct.info()

## Verificando duplicados
duplicado = ct[ct.duplicated()]
duplicado

## Verificando valores faltantes
ct.isna().sum()

## Tabela com todas as classes e suas quantidades
sns.countplot(data=ct, x='Cover_Type')

## Aplicando o StandardScaler
cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

scaler = StandardScaler()

for col in cols:
    ct[[col]] = scaler.fit_transform(ct[[col]])


## Dividindo entre previsores e classes
x_ct = ct.iloc[:,0:54].values
y_ct = ct.iloc[:,54].values


## Dividindo entre base de treinamento e testes
x_train, x_test, y_train, y_test = train_test_split(x_ct, y_ct, test_size=0.2, random_state=0)


#Armazenando com pickle
with open("covertype.plk", mode='wb') as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)


###### Executando ######
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

## Abrindo pickle
with open("covertype.plk", 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

## Naive Bayes
naive_cover = GaussianNB()
naive_cover.fit(X_train, y_train)
previsoes_cover = naive_cover.predict(X_test)
accuracy_score(y_test, previsoes_cover)

cm = ConfusionMatrix(naive_cover)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


## DecisionTree
arvore_cover = DecisionTreeClassifier(criterion='entropy')
arvore_cover.fit(X_train, y_train)
previsoes = arvore_cover.predict(X_test)
accuracy_score(y_test, previsoes)

cm = ConfusionMatrix(arvore_cover)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


## Random Forrest
random_forest_cover = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=0)
random_forest_cover.fit(X_train, y_train)
previsoes = random_forest_cover.predict(X_test)
accuracy_score(y_test, previsoes)

cm = ConfusionMatrix(random_forest_cover)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


## KNN
knn_cover = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_cover.fit(X_train, y_train)
previsoes = knn_cover.predict(X_test)
accuracy_score(y_test, previsoes)

cm = ConfusionMatrix(knn_cover)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


#Diminuindo tamanho da lista pra melhorar o desempenho
X_train_less = X_train[0:100000]
y_train_less = y_train[0:100000]
X_test_less = X_test[0:20000]
y_test_less = y_test[0:20000]


## SVM
svm_cover = SVC(C=1, kernel='rbf')
svm_cover.fit(X_train_less, y_train_less)
previsoes = svm_cover.predict(X_test_less)
accuracy_score(y_test_less, previsoes)
cm = ConfusionMatrix(svm_cover)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


## Rede neural
rede_neural_cover = MLPClassifier(max_iter=100, verbose=True, random_state=0, n_iter_no_change=10)
rede_neural_cover.fit(X_train, y_train)
previsoes = rede_neural_cover.predict(X_test)
accuracy_score(y_test, previsoes)
cm = ConfusionMatrix(rede_neural_cover)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)


#Unindo base
X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

#Diminuindo tamanho para melhorar o desempenho
X = X_all[:10000]
y = y_all[:10000]


### GridSearch
## Arvore descião
params = {
    'criterion': ['gini','entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1,5]}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params)
grid_search.fit(X_all, y_all)
best_result = grid_search.best_score_
best_params = grid_search.best_params_


#Random Forest
params = {
    'criterion': ['gini','entropy'],
    'n_estimators': [10,40],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1,5]}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params)
grid_search.fit(X, y)
best_result = grid_search.best_score_
best_params = grid_search.best_params_

#KNN
params = {'n_neighbors': [3,5,10,20],
          'p':[1,2]}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params)
grid_search.fit(X, y)
best_result = grid_search.best_score_
best_params = grid_search.best_params_


##SVM
params = {'tol': [0.001,0.0001,0.00001],
          'C':[1,1.5,2],
          'kernel':['rbg','linear','poly','sigmoid']}
grid_search = GridSearchCV(estimator=SVC(), param_grid=params)
grid_search.fit(X, y)
best_result = grid_search.best_score_
best_params = grid_search.best_params_


##Redes Neurais
params = {'activation': ['relu','logistic','tahn'],
          'solver': ['adam','sgd'],
          'batch_size': [10,60]}
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=params)
grid_search.fit(X, y)
best_result = grid_search.best_score_
best_params = grid_search.best_params_



