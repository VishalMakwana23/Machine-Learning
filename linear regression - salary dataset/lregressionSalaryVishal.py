import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mtp
df =pd.read_csv('Salary_Data.csv')


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

mtp.scatter(x_test,y_test,color="green")
mtp.plot(x_train,x_pred, color="red")
mtp.title("salary vs experiance (test database)")
mtp.xlabel("year of experiance")
mtp.ylabel("salary in ruppe ")
mtp.show()
