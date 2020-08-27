import pandas as pd 
import numpy as np 
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
#testing data without lines:
#that didnt work well, also turns out the data was a mess, so now I have what I think is clean
#data including the lines
test_season = pd.read_csv('data/test_season.csv')
train_season = pd.read_csv('data/train_season.csv')
X_train_s = train_season.drop('GAME_TOTAL', axis = 1).to_numpy()
y_train_s = train_season['GAME_TOTAL'].to_numpy()
X_test_s = test_season.drop('GAME_TOTAL', axis = 1).to_numpy()
y_test_s = test_season['GAME_TOTAL'].to_numpy()
Test_Vegas = test_season['TOTAL_CLOSE'].to_numpy()
Train_Vegas = train_season['TOTAL_CLOSE'].to_numpy()

#Vegas BASELINE = 17.650007402704748 
mean_squared_error(np.append(y_train_s,y_test_s), np.append(Train_Vegas,Test_Vegas), squared = False)

#DUMMY REGRESSOR:

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train_s, y_train_s)
#-0.7833193001644205
dummy_regr.score(X_test_s, y_test_s)
#27.845427872989156
mean_squared_error(y_test_s, dummy_regr.predict(X_test_s), squared = False)

#OLS
regressor = sm.OLS(y_train_s, X_train_s)
regressor = regressor.fit()
#evidently this returned a 0.991 R**2
#second run gave us 0.993
regressor.summary()
preds = regressor.predict(X_test_s)
#18.5802074596655
mean_squared_error(y_test_s, preds, squared = False)

#RANDOM FOREST
rf = RandomForestRegressor(oob_score=True)
rf.fit(X_train_s,y_train_s)
#0.23057109964613554
rf.oob_score_
#uhhh 0.163
# 0.17830897459807504
rf_0_score = rf.score(X_test_s, y_test_s)
rf_0_score
pred = rf.predict(X_test_s)
#18.90138431469537
mean_squared_error(y_test_s, pred, squared = False)

#GB
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, verbose=1)
gb.fit(X_train_s,y_train_s)
# 0.17123
#0.1998819243181481
gb.score(X_test_s, y_test_s)
pred = gb.predict(X_test_s)
pred2= gb.predict(X_train_s)
# traing error = 16.666236130230207
# test error = 18.65161239394806
mean_squared_error(y_test_s, pred, squared = False)


import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=26,
                            verbosity=2, n_estimators = 200,
                            eta=0.05, gamma=1) #, colsample_bytree, subsample 

xgb_model.fit(X_train_s, y_train_s)

y_pred = xgb_model.predict(X_test_s)

mse=mean_squared_error(y_test_s, y_pred, squared=False)
#20.369190191737676
mse
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
xgb_model = xgb.XGBRegressor()

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(150, 200), # default 100
    "subsample": uniform(0.6, 0.4)
}


# search.best_estimator_

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=26, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train_s, y_train_s)

xgb_pred= search.best_estimator_.predict(X_test_s)
mean_squared_error(y_test_s, xgb_pred, squared=False)

search.best_estimator_.save_model('001.model')

bst = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.9791636319740136,
             gamma=0.031653016218398666, gpu_id=-1, importance_type='gain',
             interaction_constraints='', learning_rate=0.032887910887540325,
             max_delta_step=0, max_depth=2, min_child_weight=1,
             monotone_constraints='()', n_estimators=169, n_jobs=0,
             num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, subsample=0.6058651279281114,
             tree_method='exact', validate_parameters=1, verbosity=None)
bst.fit(X_train_s,y_train_s)
bst_pred = bst.predict(X_test_s)
mean_squared_error(y_test_s, bst_pred, squared=False)

import matplotlib.pyplot as plt


#taking out TEAM categorical columns:
train_season.iloc[:,:86]

X_train_nc = train_season.iloc[:,:86].drop('GAME_TOTAL', axis = 1).to_numpy()
y_train_nc = train_season['GAME_TOTAL'].to_numpy()
X_test_nc = test_season.iloc[:,:86].drop('GAME_TOTAL', axis = 1).to_numpy()
y_test_nc = test_season['GAME_TOTAL'].to_numpy()

base_nc = xgb.XGBRegressor(objective="reg:linear", random_state=26,
                            verbosity=2, n_estimators = 200,
                            eta=0.05, gamma=1)
base_nc.fit(X_train_nc, y_train_nc)
base_pred = base_nc.predict(X_test_nc)
mean_squared_error(y_test_s, base_pred, squared=False)
#returns 20.199345503121535 so better than the base with all cat data, interesting





from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
xgb_model = xgb.XGBRegressor()
params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(150, 200), # default 100
    "subsample": uniform(0.6, 0.4)
}


# search.best_estimator_

search_nc = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=26, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search_nc.fit(X_train_nc, y_train_nc)

xgb_pred_nc= search_nc.best_estimator_.predict(X_test_nc)
mean_squared_error(y_test_s, xgb_pred_nc, squared=False)

search.best_estimator_.save_model('001.model')







