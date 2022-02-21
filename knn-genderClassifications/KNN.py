#knn-genderClassifier

import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import neighbors

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
dataframe=pd.read_csv('GanderClassification.csv')

dataframe=dataframe.apply(LabelEncoder().fit_transform)

print(dataframe)
print(dataframe.info())
print(dataframe.describe())


df_x = dataframe[['Favorite Color','Favorite Music Genre','Favorite Beverage','Favorite Soft Drink']]
df_y = dataframe[['Gender']]
print('-------------------------------------')
print(df_x)
print(df_y)



print('-----------spliting---------------')
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.33,random_state=42)


model=sk.neighbors.KNeighborsClassifier(n_neighbors=2)

model.fit(x_train,y_train.values.ravel())


predict_y = model.predict(x_test)



print("Prediction : ",predict_y)
print("Accuracy Score : ",accuracy_score(y_test,predict_y))
print("Confusion matrix : \n",confusion_matrix(y_test,predict_y))
