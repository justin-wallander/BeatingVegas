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
test_10 = pd.read_csv('test_10.csv')
train_10 = pd.read_csv('train_10.csv')
X_train_10 = train_10.drop('GAME_TOTAL', axis = 1).to_numpy()
y_train_10 = train_10['GAME_TOTAL'].to_numpy()
X_test_10 = test_10.drop('GAME_TOTAL', axis = 1).to_numpy()
y_test_10 = test_10['GAME_TOTAL'].to_numpy()
Test_Vegas = test_10['TOTAL_CLOSE'].to_numpy()
Train_Vegas = train_10['TOTAL_CLOSE'].to_numpy()

#Vegas BASELINE = 17.565434708173875 
mean_squared_error(y_test_10, Test_Vegas, squared = False)

#DUMMY REGRESSOR:

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train_10, y_train_10)
#returns -0.0011412
#second run with new data =  -0.00201585
dummy_regr.score(X_test_10, y_test_10)
#returns 21.1452
#second run = 21.1599
mean_squared_error(y_test_10, dummy_regr.predict(X_test_10), squared = False)

#OLS
regressor = sm.OLS(y_train_10, X_train_10)
regressor = regressor.fit()
#evidently this returned a 0.991 R**2
#second run gave us 0.993
regressor.summary()
preds = regressor.predict(X_test_10)
#this returns a RMSE of 19.29939303517463
#second run gives 17.708329120934696, which is close to vegas without any tuning... 
mean_squared_error(y_test_10, preds, squared = False)

#RANDOM FOREST
rf = RandomForestRegressor(oob_score=True)
rf.fit(X_train_10,y_train_10)
#hmmm.....0.1927
#second run 0.29414648409477695
rf.oob_score_
#uhhh 0.163
# second run 0.294
rf_0_score = rf.score(X_test_10, y_test_10)
rf_0_score
pred = rf.predict(X_test_10)
#RMSE 19.32287
#second run 17.7614
mean_squared_error(y_test_10, pred, squared = False)

#GB
gb = GradientBoostingRegressor()
gb.fit(X_train_10,y_train_10)
# 0.17123
#0.30384 second run
gb.score(X_test_10, y_test_10)
pred = gb.predict(X_test_10)
pred2= gb.predict(X_train_10)
# RMSE 19.239
# second run 17.637
mean_squared_error(y_test_10, pred, squared = False)


import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=26)

xgb_model.fit(X_train_10, y_train_10)

y_pred = xgb_model.predict(X_test_10)

mse=mean_squared_error(y_test_10, y_pred, squared=False)
#first run 17.6229
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

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=26, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train_10, y_train_10)

xgb_pred= search.best_estimator_.predict(X_test_10)
mean_squared_error(y_test_10, xgb_pred, squared=False)



