#svm non-linear

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn import svm
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

df = pd.read_csv('avocado.csv')
#df = df.apply(LabelEncoder().fit_transform)
print(df.info())




x = df[["AveragePrice","Total Volume","Total Bags"]]
y = df[["type"]]

x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.33,random_state=42)


svm_model = svm.SVC(kernel='rbf',C=1,gamma='auto')

svm_model.fit(x_train,y_train.values.ravel())
y_pred = svm_model.predict(x_test)

print('accuracy score : ',accuracy_score(y_test,y_pred))
