import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = datasets.load_iris()

print(type(dataset))
print(dataset)

x = dataset.data
y = dataset.target


x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.33,random_state=42)

#svm_model = svm.LinearSVC()
svm_model = svm.SVC(kernel='linear')


svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

print('the iris type :',dataset.target_names[y_pred])
print('score :',accuracy_score(y_test,y_pred))
print('matrix :',confusion_matrix(y_test,y_pred,labels=[0,1,2]))

