import  sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mtp

data_set= pd.read_csv('Salary_Data.csv')  
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 1].values   
# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  
#Fitting the Simple Linear Regression model to the training dataset  
regressor= LinearRegression()  
regressor.fit(x_train, y_train) 
#Prediction of Test and Training set result  
y_pred= regressor.predict(x_test)  
x_pred= regressor.predict(x_train)

mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Training Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()  
#visualizing the Test set results  
mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Salary vs Experience (Test Dataset)")  
mtp.xlabel("Years of Experience")  
mtp.ylabel("Salary(In Rupees)")  
mtp.show()
