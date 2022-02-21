import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([25,35,45,55,65,75])
y = np.array([5,20,14,32,22,38])

x = x.reshape((-1,1))
print(x)

model = LinearRegression()
model.fit(x,y)

score = model.score(x,y)
print('coefficent of determine :',score)
print('intercept : ',model.intercept_)
print('slope : ',model.coef_)

y_pred = model.predict(x)
print('predicted response : ',y_pred)
