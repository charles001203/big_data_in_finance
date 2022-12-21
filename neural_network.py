import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import r2_score

#df = pd.read_csv('joined_signals_currency.csv', parse_dates=['DATE'])
#df = pd.read_csv('signals1.csv', parse_dates=['DATE'])
df = pd.read_csv('smoothed_data_wNA.csv', parse_dates=['DATE'])
print(df)
df = df.dropna()
df['smooth_ret'] = df['smooth_ret']*100
df['year'] = pd.DatetimeIndex(df['DATE']).year

column_name_list = df.columns.tolist()
column_name_list.remove('mvlag')
column_name_list.remove('PERMNO')
column_name_list.remove('DATE')
column_name_list.remove('RET')
column_name_list.remove('smooth_ret')
column_name_list.remove('tic')
column_name_list.remove('Date_x')
column_name_list.remove('Open_x')
column_name_list.remove('High_x')
column_name_list.remove('Low_x')
column_name_list.remove('Close_x')
column_name_list.remove('Adj.Close_x')
column_name_list.remove('Volume_x')
column_name_list.remove('Date_y')
column_name_list.remove('Open_y')
column_name_list.remove('High_y')
column_name_list.remove('Low_y')
column_name_list.remove('Close_y')
column_name_list.remove('Adj.Close_y')
column_name_list.remove('Volume_y')
column_name_list.remove('Date_x.1')
column_name_list.remove('Open_x.1')
column_name_list.remove('High_x.1')
column_name_list.remove('Low_x.1')
column_name_list.remove('Close_x.1')
column_name_list.remove('Adj.Close_x.1')
column_name_list.remove('Volume_x.1')
column_name_list.remove('Date_y.1')
column_name_list.remove('Open_y.1')
column_name_list.remove('High_y.1')
column_name_list.remove('Low_y.1')
column_name_list.remove('Close_y.1')
column_name_list.remove('Adj.Close_y.1')
column_name_list.remove('Volume_y.1')
column_name_list.remove('Date')
column_name_list.remove('Open')
column_name_list.remove('High')
column_name_list.remove('Low')
column_name_list.remove('Close')
column_name_list.remove('Adj.Close')
column_name_list.remove('Volume')

print(column_name_list)

X_train = df[df['year']<2016][column_name_list]
y_train = df[df['year']<2016]['smooth_ret']
X_valid = df[(df['year']>=2016) & (df['year']<2020)][column_name_list]
y_valid = df[(df['year']>=2016) & (df['year']<2020)]['smooth_ret']
X_test = df[df['year']==2020][column_name_list]
y_test = df[df['year']==2020]['smooth_ret']

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

nnet = Sequential()
nnet.add(Dense(32, input_dim = X_train.shape[1],activation='relu'))
nnet.add(layers.Dropout(0.3))
nnet.add(layers.BatchNormalization())
nnet.add(Dense(64, input_dim = X_train.shape[1],activation='relu'))
nnet.add(layers.Dropout(0.3))
nnet.add(layers.BatchNormalization())
nnet.add(Dense(128, input_dim = X_train.shape[1],activation='relu'))
nnet.add(layers.Dropout(0.3))
nnet.add(layers.BatchNormalization())
nnet.add(layers.Dense(1, activation='linear'))


initial_learning_rate = 0.02
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.70,
    staircase=True)

opt = keras.optimizers.Adam(learning_rate= lr_schedule)
nnet.compile(optimizer=opt, loss='mse') 

stop = EarlyStopping(monitor='val_loss', patience = 10, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', 
                     save_best_only=True, verbose=1)

history = nnet.fit(X_train, y_train, validation_data = (X_valid, y_valid), 
                     epochs = 100, batch_size=1000,  callbacks = [stop, mc])
best_model=load_model('best_model.h5')

train_preds = best_model.predict(X_train)
valid_preds = best_model.predict(X_valid)
test_preds  = best_model.predict(X_test)
print(r2_score(y_train, train_preds))
print(r2_score(y_valid, valid_preds))
print(r2_score(y_test,  test_preds))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()