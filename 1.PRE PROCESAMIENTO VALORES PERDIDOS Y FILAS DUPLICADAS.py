# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:41:41 2020

@author: bisma
"""

import pandas as pd

dataset=pd.read_csv('DATOS COVID-BOLIVIA.csv', engine= 'python')

#VER SI LAS VARIABLES CARGARON CORRECTAMENTE VER VALORES PERDIDOS
'''
dataset.info()
dataset.head()
dataset.isnull().sum()
'''
#BORRAR FILAS O CLUMNAS QUE CONTENGAN VALORES PERDIDOS
data1=dataset.dropna()
data1

data2=dataset.dropna(axis=1)
data2

#si queremos conservar aquellos que contengan solamente 11 valores conocidos
data3=dataset.dropna(axis=1, thresh=11)
data3

#VERIFICAR SI EXIXSTEN FILAS DUPLICADAS
dataset.duplicated()
dataset
#ELIMINAR FILAS DUPLICADAS
dataset.drop_duplicates()
dataset.dropna(thresh=11)
dataset

