
import pandas as pd
#REDES NEURONALES USANDO KERAS LA CUAL ES UNA BIBLIOTECA DE REDES NEURONALES


#LEENDO LOS DATOS CON TOMANDO ENCUENTA LAS 10 PRIMERAS FILAS
dataset=pd.read_csv('DATOS COVID-BOLIVIA.csv')
df = pd.DataFrame(dataset)
print(df.head(10))

#CONVIRTIENDO EL DATAFRAME EN UN ARREGLO
data = df.values
print(data.shape)

#EXTRAER LAS CARACTERISTICAS X Y LA VARIABLE A PREDECIR Y DE LA COLUMNA 3
x = data[:,3:10]
y = data[:,10]

#PREPROCESAMIENTO DE DATOS
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scale = min_max_scaler.fit_transform(x)
print (x_scale.shape)
print (x_scale[3,:])
print (x_scale[:,3])

 #GRAFICO
'''import plotly.express as px
import numpy as np
x=np.arange(len(x_scale))
y=x_scale[:,2]
fig=px.line(df, x=x, y=y, title='Distribucion')
fig.show()
'''
'''
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)
print('x_val',len(x_val))
print(x_val)
print('x_test',len(x_test))
print(x_test)
print('y_val',len(y_val))
print(y_val)
print('y_test',len(y_test))
print(y_test)
'''

#SEPARANDO LOS VALORES DEL DATASET
from sklearn.model_selection import train_test_split
x_train, x_val, x_test, y_train, y_val, y_test = train_test_split(x_scale, y, test_size=0,3)
x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0,3)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)

#ARQUITECTURA DE LA RED NEURONAL
from keras.models import Sequential
from keras.layers import Dense
nodel = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
        ])

#OPTIMIZACION DEL MODELO
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acuracy'] )

#ENTRENAMIENTO DE LA RED NEURONAL
hist = model.fit(x_train, y_train, batch_size=32, epochs=100, vaalidation_data=[x_val, y_val])

#EVALUANDO EL MODELO
model.evaluate(x_test, y_test)[0]


