import pandas as pd
import statsmodels.api as sms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


#reading some dummy data
# replace this with reading your data
king = pd.read_csv("king_county_workfile.csv")

#feature engineering
X = king[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'age', 
        'base_dum', 'reno_dum', 'water_dum']] # here we have 2 variables for the multiple linear regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example
Y = king['price']

#splitting data
print("-----  Splitting the data in train and test ----")
X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.1, random_state=42)

#adding the constant

X_train = sms.add_constant(X_train) # adding a constant
X_test = sms.add_constant(X_test) # adding a constant

#training the model
print("-----  Training the model ----")
model = sms.OLS(y_train, X_train).fit()
print_model = model.summary()


#predictions to check the model
print("-----  Evaluating the model ----")
predictions = model.predict(X_train)
err_train = np.sqrt(mean_squared_error(y_train, predictions))
predictions_test = model.predict(X_test)
err_test = np.sqrt(mean_squared_error(y_test, predictions_test))

print(print_model)
print ("-------------")
print ("RMSE on train data: {}".format(err_train))
print ("RMSE on train data: {}".format(err_test))