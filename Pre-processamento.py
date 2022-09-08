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

##Verificando valores faltantes
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




