# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:00:53 2017

@author: Sushma
"""



import pandas as pd
import numpy as np
import Quandl, math, datetime
import pickle
from sklearn import preprocessing, corss_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.pypllot as plt
from matplotlib import style
style.use('ggplot')


df = Quandl.get('WIKI/GOOGL')
df = df[['Adj. Open' ,'Adj. High', 'Adj. Low','Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj.Open'])/df['Adj. Close'] * 100.0
df = df[['Adj. CLose', 'HL_PCT','PCT_CHANGE','Adj Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop['label'],1)
x = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace =True)
 y =np.array(df['label'])
 
 X_train, X_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
 
 clf = LinearRegression(n_jobs = -1)
 clf.fit(X_train,y_train)
 with open('linearregression.pickle','wb') as f:
     pickle.dump(clf,f)
     
     
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
 print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan #why do we need a forecast label
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i  in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print (df.tail())
    
     
     
  
    
    
    
    
    
    
    ast_date= df.iloc[-1]
 

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.col))]
 