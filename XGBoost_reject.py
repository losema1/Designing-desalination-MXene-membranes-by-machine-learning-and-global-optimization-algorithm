# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:22:03 2022

@author: ma
"""

# Importing the required libraries
import xgboost as xgb
import pandas as pd
import scipy.stats as stats
import pickle
# First XGBoost model for Pima Indians dataset
import numpy as np
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import GridSearchCV
from pdpbox import pdp, info_plots
import shap
# Reading the csv file and putting it into 'df' object
df = pd.read_csv('reject.csv')
df.head()
# Putting feature variable to X
X = df.drop('rejection',axis=1)
# Putting response variable to y
y = df['rejection']
r_score1,r_score2,r_score3=[],[],[]
random2r=[]
rmse_score=[]
rmse_score1,rmse_score2,rmse_score3=[],[],[]
result,result_all=[],[]
choice=[]
#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_train.shape, X_test.shape
data_dmatrix = xgb.DMatrix(data=X_train, label=y_test)
    

######################################################################################


model = xgb.XGBRegressor(n_estimators=100, 
                         learning_rate=0.1,
                         max_depth=5, 
                         silent=True, 
                         objective='reg:squarederror',
                         random_state=7,
                         gamma=0,
                         importance_type='total_gain') 
model.fit(X_train, y_train)
#################################################################################################
# save model to file
pickle.dump(model, open("reject.dat", "wb"))

XGBoost_Training_predict=model.predict(X_train)
XGBoost_Training_error=XGBoost_Training_predict-y_train
# Verify the accuracy
from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_train,XGBoost_Training_predict))
pearson_r1=stats.pearsonr(y_train,XGBoost_Training_predict)
R21=metrics.r2_score(y_train,XGBoost_Training_predict)
RMSE1=metrics.mean_squared_error(y_train,XGBoost_Training_predict)**0.5

#Draw test plot
font = {"color": "darkred",
        "size": 18,
        "family" : "times new roman"}
font1 = {"color": "black",
        "size": 12,
        "family" : "times new roman"}

Text='r='+str(round(pearson_r1[0],2))
plt.figure(3)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(y_train,XGBoost_Training_predict,color='red')
plt.xlabel('True Values',fontdict=font)
plt.ylabel('Predictions',fontdict=font)
Lims=[0,110]
plt.xlim(Lims)
plt.ylim(Lims)
plt.tick_params(labelsize=10)
plt.plot(Lims,Lims,color='black')
plt.grid(False)
plt.title('ion',fontdict=font)
plt.text(2,10,Text,fontdict=font1)
plt.savefig('figure3.png', dpi=100,bbox_inches='tight') 
###############################################################################################
"""
Apply grid search combined with five-fold cross-validation method to search for the best hyperparameters of the model
"""
# params = {'max_depth':[x for x in range(2,11)], 
#         'n_estimators':[y for y in range(10, 200, 10) ],
#         'learning_rate':[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1 ]}
# gs =GridSearchCV(model, params, cv=5,n_jobs=-1,refit=True,verbose=1,scoring='r2')

# grid_result=gs.fit(X_train, y_train) 
# process=grid_result.cv_results_
# #print(grid_result.cv_results_)
# means_cv=grid_result.cv_results_['mean_test_score']
# params_cv=grid_result.cv_results_['params']
# for mean,param in zip(means_cv,params_cv):
#     print("%f  with:     %r" %(mean,param))
# print("#"*100)
# print("Best: %f using %s" %(grid_result.best_score_,grid_result.best_params_)) 

#################################################################################################
# Make predictions on the test set
y_pred = model.predict(X_test)
random_forest_error=y_pred-y_test
# evaluate predictions
from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test,y_pred))
pearson_r=stats.pearsonr(y_test,y_pred)
R2=metrics.r2_score(y_test,y_pred)
RMSE=metrics.mean_squared_error(y_test,y_pred)**0.5
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
print ('r2_score: %.2f' %R2)
rmse_score.append(RMSE)    
#Draw test plot
font = {"color": "darkred",
        "size": 18,
        "family" : "times new roman"}
font1 = {"color": "black",
        "size": 12,
        "family" : "times new roman"}

Text='r='+str(round(pearson_r[0],2))
plt.figure(1)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(y_test,y_pred,color='red')
plt.xlabel('True Values',fontdict=font)
plt.ylabel('Predictions',fontdict=font)
Lims=[0,110]
plt.xlim(Lims)
plt.ylim(Lims)
plt.tick_params(labelsize=10)
plt.plot(Lims,Lims,color='black')
plt.grid(False)
plt.title('ion',fontdict=font)
plt.text(2,10,Text,fontdict=font1)
plt.savefig('figure1.png', dpi=100,bbox_inches='tight')   


plt.figure(2)
plt.clf()
plt.hist(random_forest_error,bins=30)
plt.xlabel('Prediction Error',fontdict=font)
plt.ylabel('Count',fontdict=font)
plt.grid(False)
plt.title('ion',fontdict=font)
plt.savefig('figure2.png', dpi=100,bbox_inches='tight')
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))

# Show important features
plot_importance(model,importance_type=('total_gain'))
pyplot.show()
feature_importance = model.feature_importances_.tolist()
#=========================================================================================
#shap
explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_train)
shap.summary_plot(shap_values,X_train,show=False)
plt.savefig('importance_reject_xgb.png', format='png', dpi=1200, bbox_inches='tight')
#=========================================================================================



