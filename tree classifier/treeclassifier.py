#tree classifier

import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('iriscsv.csv')
df = df.apply(LabelEncoder().fit_transform)
print(df)
print(df.info())
print(df.describe())

x = df[['sepal.length','sepal.width','petal.length','petal.width']]
y = df[['variety']]

x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.33)

trclf = DecisionTreeClassifier()

trclf.fit(x_train,y_train)

y_pred = trclf.predict(x_test)

print('score :')
print(accuracy_score(y_test,y_pred))
print(trclf.score(x_test,y_test));


