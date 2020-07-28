# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:10:04 2020

@author: bisma
"""

#REGRESION LINEAL CON SCIKITLEARN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

regr=linear_model.LinearRegression()

dataset=pd.read_csv('DATOS COVID-BOLIVIA.csv')
df=pd.DataFrame(dataset)
x=df['muertes']
y=df['recuperados']

#CALCULANDO LA REGRESION LINEAL TOMANDO X COMO MUERTES Y Y COMO RECUPERADOS
X=x[:,np.newaxis]
print(X)
print(regr.fit(X,y))
print(regr.coef_)
m=regr.coef_[0]
b=regr.intercept_
print('y={0}*x+{1}'.format(m,b))

#PREDECIR LOS PRIMEROS 5 DATOS
print(regr.predict(X)[0:5])
y_p=m*X+b
print("EL VALOR DE r^2:",r2_score(y,y_p))

#GRAFICA
plt.scatter(x,y, color='blue')
plt.plot(x,y_p, color='red')
plt.title('Regresion lineal', fontsize=16)
plt.xlabel('Muertes', fontsize=13)
plt.xlabel('Recuperados', fontsize=13)


