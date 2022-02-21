import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv('Iris.csv')
df.info()
print(df.head(5))
print(df.describe())
print('Data loaded!!')

X=df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
y=df[['Species']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

trclf=DecisionTreeClassifier()
trclf.fit(X_train, y_train)
dataClass=trclf.predict(X_test)
print("Score : ")
print(trclf.score(X_test, y_test))

