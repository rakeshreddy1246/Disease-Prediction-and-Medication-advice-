import pandas as pd
import sc
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
filename='Drug.csv'
data= read_csv(filename)
data.head()
data.dtypes
data.replace({'Gender':{'Female':0,'Male':1}},inplace=True)
x=data[['Disease']]
x.Disease.unique()
data.replace({'Disease':{'Acne':0, 'Allergy':1, 'Diabetes':2, 'Fungal infection':3,
       'Urinary tract infection':4, 'Malaria':5, 'Migraine':6, 'Hepatitis B':7,
       'AIDS':8}},inplace=True)
       #feature selection
df_x=data[['Disease','Gender','Age']]
df_y=data[['Drug']]
#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_x,df_y, test_size=0.2,random_state=0 )
#fitting random forst model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf = rf.fit(df_x,np.ravel(df_y))
#model Accuracy
from sklearn.metrics import accuracy_score
y_pred = rf.predict(X_test)
print(sc.accuaracy_score)
#score
rf.score(X_test, y_test)
pickle.dump(rf,open('model1.pkl','wb'))
# pickle.dump(clr_rf,open('model.pkl','wb'))