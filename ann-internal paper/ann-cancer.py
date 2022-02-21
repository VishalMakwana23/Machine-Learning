import pandas as pd
import sklearn as sk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
df = pd.read_csv('BreastCancerDataset.csv')
df = df.apply(LabelEncoder().fit_transform)

print(df)
print(df.info())
print(df.describe())

x = df.iloc[:,2:31].values
y = df.iloc[:,1].values


x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.33,random_state=42)


clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(5,3) ,alpha=0.000001,max_iter=400)
clf.fit(x_train,y_train)

data_test = clf.predict(x_test)
data_train = clf.predict(x_train)

print('prediction: ',data_test)
print('test data : ',accuracy_score(y_test,data_test))
print('test train: ',accuracy_score(y_train,data_train))

print(confusion_matrix(y_test,data_test,labels=[0,1]))

