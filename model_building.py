# -*- coding: utf-8 -*-
"""
Created on Tuesday June 1, 2021

@author: Eduardo
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("eda_data.csv")

# choose relevant columns
df.columns

df_model = df[['avg salary','Rating', 'seniority', 'job_simp', 'Size', 'Type of nership', 'Industry','Sector','Revenue','num_comp','Employer Provided Salary',
            'job city location','same_city','age', 'python y/n', 'tableau y/n','excel y/n', 'aws y/n', 'spark y/n', 'hadoop y/n']]



# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg salary', axis=1)
y = df['avg salary'].values

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))

# lasso regression
lm_lasso = Lasso(alpha=0.35)
np.mean(cross_val_score(lm_lasso, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))
lm_lasso.fit(X_train, y_train)

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lm_lasso = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lm_lasso, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

    
# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 3))

# tune models GridsearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':range(10, 300, 10), 'criterion': ("mse", "mae"), 'max_features':("auto","sqrt", "log2")}

gs = GridSearchCV(rf, parameters, scoring = "neg_mean_absolute_error", cv=3)
gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_

#test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_lasso.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test,tpred_rf) # best performing model

mean_absolute_error(y_test, ((tpred_lm*.05)+(tpred_rf*.95)))
