import pandas as pd 
import numpy as np 
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# test_season = pd.read_csv('data/test_season.csv')
# train_season = pd.read_csv('data/train_season.csv')
# X_train_s = train_season.drop('GAME_TOTAL', axis = 1).to_numpy()
# y_train_s = train_season['GAME_TOTAL'].to_numpy()
# X_test_s = test_season.drop('GAME_TOTAL', axis = 1).to_numpy()
# y_test_s = test_season['GAME_TOTAL'].to_numpy()
# Test_Vegas = test_season['TOTAL_CLOSE'].to_numpy()
# Train_Vegas = train_season['TOTAL_CLOSE'].to_numpy()

season = pd.read_csv('data/avg_season.csv')
# [22007, 22008, 22009, 22010, 22011, 22012, 22013, 22014, 22015,
#        22016, 22017, 22018, 22019]
X_train = season[(season.SEASON_ID==22007) | (season.SEASON_ID==22008)].iloc[:1616, 2:]
y_train = season[(season.SEASON_ID==22007) | (season.SEASON_ID==22008)].iloc[:1616, 0]

X_test = season[(season.SEASON_ID==22007) | (season.SEASON_ID==22008)].iloc[1616:, 2:]
y_test = season[(season.SEASON_ID==22007) | (season.SEASON_ID==22008)].iloc[1616:, 0]
train_vegas = season[(season.SEASON_ID==22007) | (season.SEASON_ID==22008)].iloc[:1616, 5]
test_vegas = season[(season.SEASON_ID==22007) | (season.SEASON_ID==22008)].iloc[1616:, 5]


#Vegas BASELINE = 17.92971357029641 
mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)

#DUMMY REGRESSOR:

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)

#21.982204049481744
mean_squared_error(y_test, dummy_regr.predict(X_test), squared = False)



import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=26,
                            verbosity=2, n_estimators = 200,
                            eta=0.05, gamma=1) #, colsample_bytree, subsample 

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
#20.369190191737676
mean_squared_error(y_test, y_pred, squared=False)


from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
xgb_model = xgb.XGBRegressor()

params = {
    "colsample_bytree": [0.3, 0.5, 0.7, 1],
    "gamma": [0, 0.5, 1],
    "learning_rate": [0.01], # default 0.1 
    "max_depth": [2, 3, 5, 6], # default 3
    "n_estimators": [100, 150, 200], # default 100
    "subsample": [0.3, 0.5, 0.6, 1]
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


search_nc.best_estimator_

# search_nc.best_estimator_
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.871050138793756,
#              gamma=0.12845693745543396, gpu_id=-1, importance_type='gain',
#              interaction_constraints='', learning_rate=0.036085170670551454,
#              max_delta_step=0, max_depth=2, min_child_weight=1, missing=nan,
#              monotone_constraints='()', n_estimators=161, n_jobs=0,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=0.6964021257211186,
#              tree_method='exact', validate_parameters=1, verbosity=None)

search_nc = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=26, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search_nc.fit(X_train_nc, y_train_nc)

xgb_pred_nc= search_nc.best_estimator_.predict(X_test_nc)
mean_squared_error(y_test_s, xgb_pred_nc, squared=False)

search.best_estimator_.save_model('001.model')


regressor = sm.OLS(y_train_nc, X_train_nc)
regressor = regressor.fit()
#evidently this returned a 0.991 R**2
#second run gave us 0.993
regressor.summary()
preds = regressor.predict(X_test_nc)
#18.5802074596655
mean_squared_error(y_test_nc, preds, squared = False)

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train_nc,y_train_nc) 
X_train_ss = ss.transform(X_train_nc)
X_test_ss = ss.transform(X_test_nc)

sgd = SGDRegressor()
sgd.fit(X_train_ss, y_train_s)
sgd.predict(X_test_ss)
mean_squared_error(y_test_s, sgd.predict(X_test_ss), squared=False)