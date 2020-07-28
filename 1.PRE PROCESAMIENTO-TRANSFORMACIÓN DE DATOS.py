# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:01:41 2020

@author: bisma
"""
#PREPROCESAMIENTO MEDIANTE TRANSFORMACIÓN DE DATOS
import pandas as pd
import numpy as np
"""
dataset = pd.read_csv('Datos-354.csv')
"""
#LEENDO EL ARCHIVO CSV
dataset = pd.read_csv('DATOS COVID-BOLIVIA.csv')

datos=dataset['casos']
#datos=datos.astype(float).fillna(0.0)


#DESCRIBE LOS DATOS
print(datos.describe())
arr=np.array(datos).reshape((-1,1))
print(arr.shape)

#MOSTRANDO EL RANGO DE MUERTES
ranges = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
print (dataset['muertes'].groupby(pd.cut(dataset.muertes, ranges,right=False)).count())


#HACIENDO EL PREPROCESAMIENTO
#HACIENDO LA TRANSFORMACION DE DATOS LA CUAL ELIMINA LA MEDIANA Y ESCALA LOS DATOS SEGUN SU RANGO
from sklearn import preprocessing
scaler=preprocessing.RobustScaler()
col=scaler.fit_transform(arr)
dataset_resultado=pd.DataFrame(col)
dataset_resultado.columns=['Robust scaler']
print(dataset_resultado.describe())

#HACINEDO LA TRANSFORMACION DE DATOS LA CUAL ESTANDARIZA LOS DATOS ELIMINANDO LA MEDIA Y ESCALA LOS DATOS TAL QUE SU VARIANZA SEA IGUAL A 1
scaler1=preprocessing.StandardScaler()
col=scaler1.fit_transform(arr)
dataset_resultado2=pd.DataFrame(col)
dataset_resultado2.columns=['Standar Scaler']
print(dataset_resultado2.describe())
