#Multiple Linear Regression Assignment
"""
Created on Wed Aug  5 00:20:41 2020

@author: kamakshi Gupta
"""
#%% Linear Regression -1 Marketing Data - Sales - YT, FB, print
#libraries
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import linear_model #1st method
import statsmodels.api as sm  #2nd method
import matplotlib.pyplot as plt
import seaborn as sns

url ='https://raw.githubusercontent.com/DUanalytics/datasets/master/R/marketing.csv'
marketing = pd.read_csv(url)
marketing.head()

#describe data
marketing.describe()
marketing.shape
marketing.info()

#visualise few plots to check correlation
#using seaborn
sns.scatterplot(data=marketing, x='youtube', y='sales') 
sns.scatterplot(data=marketing, x='facebook', y='sales')
sns.scatterplot(data=marketing, x='newspaper', y='sales')
#no significant relationship between newspaper and sales
sns.heatmap(marketing.corr(), annot=True) #to check correlation btw variables
#using matplot
plt.scatter(marketing['youtube'], marketing['sales'], color='blue')
plt.title('Sales vs Youtube marketing', fontsize=10)
plt.grid(True)
plt.show();

#split data into train and test
X=marketing[['youtube', 'facebook', 'newspaper']]
Y=marketing[['sales']]

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, random_state=123)

#build the model
#%%
#1st Method using sklearn - with all 3 IVS
MarketingModel = linear_model.LinearRegression()
MarketingModel.fit(X_train, Y_train)  
#finding intercept & coefficients
MarketingModel.intercept_
MarketingModel.coef_
#predict on test values
Y_predicted = MarketingModel.predict(X_test)
Y_predicted
#find metrics - R2, Adjt R2, RMSE, MAPE etc
r2_score(Y_train, MarketingModel.predict(X_train))
mse = mean_squared_error(Y_test, Y_predicted) 
print("Mean Squared error is", mse)
RMSE = math.sqrt(mse)
print("Root Mean Squared error is", RMSE)
MAE = mean_absolute_error(Y_test, Y_predicted)
print("Mean Absolute error is", MAE)

#%%
#2nd method using statsmodel
X_train = sm.add_constant(X_train)
Mm2=sm.OLS(Y_train, X_train).fit()
print(Mm2.summary())

#predict on test values
X_test = sm.add_constant(X_test)
Y_pred2 = Mm2.predict(X_test)
Y_pred2
#find metrics - R2, Adjt R2, RMSE, MAPE etc
Mm2.rsquared 
Mm2.rsquared_adj 

#predict on new value
newdata = pd.DataFrame({'youtube':[50,60,70], 'facebook':[20, 30, 40], 'newspaper':[70,75,80]})
newdata
Y_new = MarketingModel.predict(newdata)
#your ans should be close to [ 9.51, 11.85, 14.18] 
Y_new




