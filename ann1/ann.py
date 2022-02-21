import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
dataset = datasets.load_iris()
print(dataset)

x = dataset.data
y = dataset.target

print(x)
print(y)

x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.33,random_state=42)

clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,3), alpha=0.000001, max_iter=200)

clf.fit(x_train,y_train)

data_test = clf.predict(x_test)
data_train = clf.predict(x_train)


print(accuracy_score(y_test,data_test))
print(accuracy_score(y_train,data_train))

    
print(confusion_matrix(y_test,y_test,labels=[0,1,2]))

print(clf.coefs_[0])
print(clf.coefs_[1])

print(classification_report(y_test,data_test))
