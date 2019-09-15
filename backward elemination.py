This is a temporary script file.
"""

import numpy as np
import pandas as pd

#opening csv file
op_file = pd.read_csv(r'C:\Users\raghu\Desktop\Datascience\scripts\Multilinear Regression\50_Startups.csv')
X = op_file.iloc[:, :-1]
y = op_file.iloc[:, -1]


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer1 = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = transformer1.fit_transform(X)

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


import statsmodels.api as sm
X = sm.add_constant(X)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(y, X).fit()
regressor_OLS.summary()
