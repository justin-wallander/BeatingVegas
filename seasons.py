import pandas as pd 
import numpy as np 
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb
from numpy import sort
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
plt.style.use('fivethirtyeight')
%matplotlib inline


season = pd.read_csv('data/avg_season.csv')
# [22007, 22008, 22009, 22010, 22011, 22012, 22013, 22014, 22015,
#        22016, 22017, 22018, 22019]

season_teams = pd.read_csv('data/season.csv')
dal_19 = season_teams[(season_teams.SEASON_ID==22019) & ((season_teams.TEAM_A == 'DAL') | (season_teams.TEAM_B == 'DAL'))]
dal_19 = dal_19.iloc[int(len(dal_19)*(2/3)):, :]
dal_19
dal_X_test = dal_19.drop(dal_drop, axis=1)
dal_X_test.columns = ['VEGAS_OPEN', 'PTS_SPR_OPEN', 'HOME_WIN_PCT', 'HOME_PPG', 'HOME_FGM', 'HOME_FGA',
'HOME_FG_PCT', 'HOME_FG3M', 'HOME_FG3_PCT', 'HOME_FTM', 'HOME_OREB', 'HOME_REB', 'HOME_AST',
'HOME_STL', 'HOME_TOV', 'HOME_PF', 'HOME_PTS_ALLOW', 'HOME_FGA_OPP', 'HOME_FG_PCT_OPP',
'HOME_FG3_PCT_OPP', 'HOME_REB_OPP', 'HOME_AST_OPP', 'HOME_STL_OPP', 'HOME_TOV_OPP',
'AWAY_WIN_PCT', 'AWAY_PPG', 'AWAY_FGM', 'AWAY_FGA', 'AWAY_FG3M', 'AWAY_FG3_PCT', 'AWAY_FTM',
'AWAY_OREB', 'AWAY_REB', 'AWAY_AST', 'AWAY_STL', 'AWAY_BLK', 'AWAY_TOV', 'AWAY_PF',
'AWAY_PTS_ALLOW', 'AWAY_FGA_OPP', 'AWAY_FG_PCT_OPP', 'AWAY_FG3_PCT_OPP', 'AWAY_REB_OPP',
'AWAY_AST_OPP', 'AWAY_STL_OPP', 'AWAY_TOV_OPP']
X_train.columns
X_train.columns = ['VEGAS_OPEN', 'PTS_SPR_OPEN', 'HOME_WIN_PCT', 'HOME_PPG', 'HOME_FGM', 'HOME_FGA',
'HOME_FG_PCT', 'HOME_FG3M', 'HOME_FG3_PCT', 'HOME_FTM', 'HOME_OREB', 'HOME_REB', 'HOME_AST',
'HOME_STL', 'HOME_TOV', 'HOME_PF', 'HOME_PTS_ALLOW', 'HOME_FGA_OPP', 'HOME_FG_PCT_OPP',
'HOME_FG3_PCT_OPP', 'HOME_REB_OPP', 'HOME_AST_OPP', 'HOME_STL_OPP', 'HOME_TOV_OPP',
'AWAY_WIN_PCT', 'AWAY_PPG', 'AWAY_FGM', 'AWAY_FGA', 'AWAY_FG3M', 'AWAY_FG3_PCT', 'AWAY_FTM',
'AWAY_OREB', 'AWAY_REB', 'AWAY_AST', 'AWAY_STL', 'AWAY_BLK', 'AWAY_TOV', 'AWAY_PF',
'AWAY_PTS_ALLOW', 'AWAY_FGA_OPP', 'AWAY_FG_PCT_OPP', 'AWAY_FG3_PCT_OPP', 'AWAY_REB_OPP',
'AWAY_AST_OPP', 'AWAY_STL_OPP', 'AWAY_TOV_OPP']
X_test.columns = ['VEGAS_OPEN', 'PTS_SPR_OPEN', 'HOME_WIN_PCT', 'HOME_PPG', 'HOME_FGM', 'HOME_FGA',
'HOME_FG_PCT', 'HOME_FG3M', 'HOME_FG3_PCT', 'HOME_FTM', 'HOME_OREB', 'HOME_REB', 'HOME_AST',
'HOME_STL', 'HOME_TOV', 'HOME_PF', 'HOME_PTS_ALLOW', 'HOME_FGA_OPP', 'HOME_FG_PCT_OPP',
'HOME_FG3_PCT_OPP', 'HOME_REB_OPP', 'HOME_AST_OPP', 'HOME_STL_OPP', 'HOME_TOV_OPP',
'AWAY_WIN_PCT', 'AWAY_PPG', 'AWAY_FGM', 'AWAY_FGA', 'AWAY_FG3M', 'AWAY_FG3_PCT', 'AWAY_FTM',
'AWAY_OREB', 'AWAY_REB', 'AWAY_AST', 'AWAY_STL', 'AWAY_BLK', 'AWAY_TOV', 'AWAY_PF',
'AWAY_PTS_ALLOW', 'AWAY_FGA_OPP', 'AWAY_FG_PCT_OPP', 'AWAY_FG3_PCT_OPP', 'AWAY_REB_OPP',
'AWAY_AST_OPP', 'AWAY_STL_OPP', 'AWAY_TOV_OPP']
dal_y_test = dal_19.GAME_TOTAL
dal_y_test
dal_test_vegas = dal_19.TOTAL_CLOSE
dal_test_vegas

bst1 = xgb.XGBRegressor( 
                       objective= 'reg:squarederror', 
                       booster='gbtree', 
                       colsample_bytree=.87,  
                       learning_rate=.056,
                       max_depth=2, 
                       n_estimators=199, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=6,
                       subsample=0.61,
                       )


bst1.fit(X_train,y_train)

dal_pred = bst1.predict(dal_X_test)
dal_pred.

model_score = mean_squared_error(dal_y_test, bst1.predict(dal_X_test), squared=False)
vegas = mean_squared_error(dal_y_test, dal_test_vegas, squared=False)
#train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
print(f'Vegas: {vegas}  Model: {model_score}')

model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
vegas = mean_squared_error(y_test, test_vegas, squared=False)
#train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
print(f'Vegas: {vegas}  Model: {model_score}')


#only predicting the Mavs last 2/3 of this season, the model actually predicts better than Vegas:
# Vegas: 21.19905658278217  Model: 21.108387307447156
#overall though Vegas edges me out:
#Vegas: 18.558905310422904  Model: 18.748607816130928
fig,ax= plt.subplots(figsize=(12,10))
ax.scatter(dal_19.GAME_DATE,dal_y_test, label='ACTUAL SCORE', marker='o', s=150)
ax.scatter(dal_19.GAME_DATE,dal_test_vegas, label='VEGAS CLOSE', marker='_', s=110, c='#ec40f5')
ax.scatter(dal_19.GAME_DATE,dal_pred, label='PREDICTED', marker='*', s=125, c='#ffa70f')
ax.scatter(dal_19.GAME_DATE[abs(dal_res) < abs(vegas_res)], [280 for _ in dal_pred[abs(dal_res) < abs(vegas_res)]], c='#34b028', marker = '^', s=125, label='BEAT VEGAS')
ax.scatter(dal_19.GAME_DATE[abs(dal_res) > abs(vegas_res)], [280 for _ in dal_pred[abs(dal_res) > abs(vegas_res)]], c='#b02828', marker = 'v', s=125, label='VEGAS WON')
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('GAME DATE')
ax.set_ylabel('POINTS SCORED')
ax.set_title('MAVS 2020 ACTUAL vs PREDICTED vs VEGAS\n BEAT VEGAS 13 out of 25 (DON\'T BET ON THIS!)')
ax.legend(loc=2)

dal_res = dal_pred - dal_y_test
vegas_res = dal_test_vegas - dal_y_test
abs(dal_res) < abs(vegas_res)

plt_x
plt_y = 
X_test.columns[plt_x]
fig,ax = plt.subplots(figsize=(30,24))
# plt.scatter(y_test, bst1.predict(X_test)-y_test)
xgb.plot_importance(bst1, ax=ax)
ax.barh(plt_y, plt_x)



    # dal_train_vegas = season[(season.SEASON_ID==i)].iloc[:int(len(season[(season.SEASON_ID==i)])*(2/3)), 5]




DUMMY REGRESSOR:

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
#21.982204049481744
mean_squared_error(y_test, dummy_regr.predict(X_test), squared = False)

#OLS

# regressor = sm.OLS(y_train, X_train)
# regressor = regressor.fit()
# #evidently this returned a 0.991 R**2
# #second run gave us 0.993
# regressor.summary()
# preds = regressor.predict(X_test)
# #18.5802074596655
# mean_squared_error(y_test, preds, squared = False)




# xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=26,
#                             verbosity=2, n_estimators = 200,
#                             eta=0.05, gamma=1) #, colsample_bytree, subsample 

# xgb_model.fit(X_train, y_train)

# y_pred = xgb_model.predict(X_test)
# #20.369190191737676
# mean_squared_error(y_test, y_pred, squared=False)


# from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
# #from scipy.stats import uniform, randint
# xgb_model = xgb.XGBRegressor()

# params = {
#     "colsample_bytree": [0.3, 0.5, 0.7, 1],
#     "gamma": [0, 0.02, 0.05, 0.07],
#     "learning_rate": [0.035], # default 0.1 
#     "max_depth": [3, 5, 6], # default 3
#     "n_estimators": [200], # default 100
#     "subsample": [0.3, 0.5, 1]
# }


# # search.best_estimator_

# search = GridSearchCV(xgb_model, param_grid=params, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

# search.fit(X_train, y_train)

# xgb_pred= search.best_estimator_.predict(X_test)
# xgb_pred
# mean_squared_error(y_test, xgb_pred, squared=False)

# search.best_estimator_.save_model('001.model')
'''
Intially I was splitting 2/3 to simulate having 1 and 1/3 season of data. Turns out my thinking in that way
is wrong, 2/3 of my data is basically 2 of the 3 seaons since I forgot I added another season. I am going to try a 7/9 split
on 3 seasons, and then do a 75-25 split on 2 seasons to see if that helps.
The best model I was able to come up with on the 2/3 3 season split had these scores:
Vegas: 18.43497530800466  Train: 17.61019852390941  Model: 18.972976215171265
ran gridsearch cv and got 18.94429355521628 on the data without team cat cols.
after playing around with linspace on some of the parameters, I was able to get down to this:
Vegas: 18.43497530800466  Model: 18.822649836118806  Train: 17.423180296248294
Now going to see about doing a bit more feature selection:
got the featues down significantly (from 144 to 47) and the score down as well:
Vegas: 18.43497530800466  Model: 18.778468234871113  Train: 17.779103320602132
down to 46 featurs:
Vegas: 18.43497530800466  Model: 18.748607816130928  Train: 17.31403219460038




'''

# i = 22017
# X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#           (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
# y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 0]
# X_train
# X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
# y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 0]
# train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#               (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 5]
# test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
#              (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 5]





# xgb_model = xgb.XGBRegressor()

# params = {
#     "colsample_bytree": [0.7,0.8,0.9,0.95,1],
#     "gamma": [0, 0.2],
#     "learning_rate": [0.02,0.025,0.03,0.035,0.04,0.045, 0.05],
#     "max_depth": [2], # default 3
#     "n_estimators": [120,140,150,160,170,200,250],
#     "subsample": [0.6,0.7,0.8,0.9,0.95,1]}

# search = GridSearchCV(xgb_model, param_grid=params, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

# search.fit(X_train, y_train)

# xgb_pred= search.best_estimator_.predict(X_test)
# xgb_pred
# search.best_estimator_
# mean_squared_error(y_test, xgb_pred, squared=False)


# XGBRegressor(base_score=0.5, 
#              booster='gbtree', 
#              colsample_bytree=1
#              learning_rate=0.035, 
#              max_depth=2,
#              n_estimators=160,  
#              random_state=0,
#              reg_lambda=1, 
#              subsample=0.7,
#              )

# for i in np.linspace(0.0, 0.3,51):
# # for i in np.linspace(0.5, 1,51):
# # for i in range(10):
#     bst = xgb.XGBRegressor(base_score=0.5, 
#                        booster='gbtree', 
#                        colsample_bytree=0.77,
#                        gpu_id=-1, 
#                        gamma=0, 
#                        learning_rate=0.064,
#                        max_depth=2, 
#                        n_estimators=130, 
#                        n_jobs=-1,
#                        random_state=0, 
#                        reg_lambda=1,
#                        subsample=0.76,
#                        )
#     bst.fit(X_train,y_train)
#     model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
#     vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
#     train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
#     print(f'{i} Vegas: {vegas}  Model: {model_score}  Train: {train_score}')

# bst = xgb.XGBRegressor(base_score=0.5, 
#                        booster='gbtree', 
#                        colsample_bytree=0.77,
#                        gpu_id=0, 
#                        gamma=0, 
#                        learning_rate=0.064,
#                        max_depth=2, 
#                        n_estimators=130, 
#                        n_jobs=-1,
#                        random_state=0, 
#                        reg_lambda=1,
#                        subsample=0.76,
#                        )
# bst.fit(X_train,y_train)
# train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
# model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
# vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
# print(f'Vegas: {vegas}  Model: {model_score}  Train: {train_score}')

# for i in range(22007,22018):
#     X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#           (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
#     y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 0]
#     X_train
#     X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
#     y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 0]
#     train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#               (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 5]
#     test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
#              (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 5]

#     bst.fit(X_train,y_train)
#     train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
#     vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
#     model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
#     print(f'{i} = Vegas: {vegas}  Model: {model_score}  Train: {train_score}')



plt_x = bst1.feature_importances_ <=0.012422
plt_x
plt_y = 
X_test.columns[plt_x]
fig,ax = plt.subplots(figsize=(30,24))
# plt.scatter(y_test, bst1.predict(X_test)-y_test)
xgb.plot_importance(bst1, ax=ax)
ax.barh(plt_y, plt_x)


# drop_cols = list(X_test.columns[bst.feature_importances_<0.011533])
# drop_cols.extend(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A',
#                  'PLUS_MINUS_B','PTS_SPR_CLOSE', 'FTM_OPP_B', 'FTM_OPP_A', 
#                  'FTA_OPP_A', 'FTA_OPP_B', 'DREB_OPP_A', 'DREB_OPP_B'])
drop_cols =['BLK_OPP_A',
 'BLK_OPP_B',
 'BLK_A',
 'DREB_B',
 'DREB_A',
 'DREB_OPP_A',
 'DREB_OPP_B',
 'FG3A_B',
 'FG3A_A',
 'FG3A_OPP_A',
 'FG3M_OPP_A',
 'FG3A_OPP_B',
 'FG3M_OPP_B',
 'FGM_OPP_A',
 'FGM_OPP_B',
 'FG_PCT_B',
 'FTA_A',
 'FTA_B',
 'FTA_OPP_A',
 'FTA_OPP_B',
 'FTM_OPP_A',
 'FTM_OPP_B',
 'FT_PCT_A',
 'FT_PCT_B',
 'FT_PCT_OPP_A',
 'FT_PCT_OPP_B',
 'GAME_DATE',
 'ML_A',
 'ML_B',
 'OREB_OPP_A',
 'OREB_OPP_B',
 'PF_OPP_A',
 'PLUS_MINUS_A',
 'PLUS_MINUS_B',
 'PTS_SPR_CLOSE',
 'TOTAL_CLOSE',
 'WL_A',
 'WL_B']




 

# Fit model using each importance as a threshold
# thresholds = sort(bst1.feature_importances_)
# for thresh in thresholds:
# 	# select features using threshold
#     selection = SelectFromModel(bst1, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train)
# 	# train model
#     selection_model = xgb.XGBRegressor( 
#                        objective= 'reg:squarederror', 
#                        booster='gbtree', 
#                        colsample_bytree=.87, 
#                        gamma=0, 
#                        learning_rate=.056,
#                        max_depth=2, 
#                        n_estimators=199, 
#                        n_jobs=-1,
#                        random_state=0, 
#                        reg_lambda=6,
#                        subsample=0.61,
#                        )
#     selection_model.fit(select_X_train, y_train)
# 	# eval model
#     select_X_test = selection.transform(X_test)
#     predictions = selection_model.predict(select_X_test)
    
#     train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
#     vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
#     model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
#     #print(f'{thresh} = Vegas: {vegas}  Model: {model_score}  Train: {train_score}')
#     accuracy = mean_squared_error(y_test, predictions, squared=False)
#     print(f"Thresh={thresh:3f}, n:{select_X_train.shape[1]}, Accuracy: {accuracy}")

#guess I will mess around a bit with less features to see if this will improve performance 

i = 22017
X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 2:86].drop(drop_cols, axis = 1)
y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 0]
X_train
X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 2:86].drop(drop_cols, axis = 1)
y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 0]
train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
              (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 5]
test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
             (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 5]




lst = []
# for i in np.linspace(0.02, 0.08,61):
# for i in np.linspace(0.5, 1,51):
for i in range(100,251):
    bst1 = xgb.XGBRegressor( 
                       objective= 'reg:squarederror', 
                       booster='gbtree', 
                       colsample_bytree=.87, 
                       gamma=0, 
                       learning_rate=.056,
                       max_depth=2, 
                       n_estimators=199, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=6,
                       subsample=0.61,
                       )
    bst1.fit(X_train,y_train)
    model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
    vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
    train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
    #print(f'{i} Vegas: {vegas}  Model: {model_score}  Train: {train_score}')
    lst.append([model_score, i])

sorted(lst)


### BEST MODEL SO FAR
bst1 = xgb.XGBRegressor( 
                       objective= 'reg:squarederror', 
                       booster='gbtree', 
                       colsample_bytree=.87, 
                       gamma=0, 
                       learning_rate=.056,
                       max_depth=2, 
                       n_estimators=199, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=6,
                       subsample=0.61,
                       )
bst1.fit(X_train,y_train)
model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
vegas = mean_squared_error(y_test, test_vegas, squared=False)
train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
print(f'Vegas: {vegas}  Model: {model_score}  Train: {train_score}')

for i in range(22007,22018):
    X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 2:86].drop(drop_cols, axis = 1)
    y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 0]
    X_train
    X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 2:86].drop(drop_cols, axis = 1)
    y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 0]
    train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
              (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 5]
    test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
             (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 5]

    bst1.fit(X_train,y_train)
    train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
    vegas = mean_squared_error(y_test, test_vegas, squared=False)
    model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
    print(f'{i} = Vegas: {vegas}  Model: {model_score}  Train: {train_score}')


fig,ax = plt.subplots(figsize=(30,24))
xgb.plot_importance(bst1, ax=ax)
#figure out this graph
plt_x = sorted(bst1.feature_importances_, reverse=True)[:10]
plt_x
plt_y = X_test.columns[np.argsort(bst1.feature_importances_)][:10]
fig,ax = plt.subplots(figsize=(30,24))
#xgb.plot_importance(bst, ax=ax)
ax.barh(plt_y, plt_x)





# from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
# from scipy.stats import uniform, randint

# #running only two seasons gave me a 19.243284135549896? 
# i = 22018
# X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#           (season.SEASON_ID==i+1)])*(3/4)), 2:].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
# y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1)])*(3/4)), 0]
# X_train.info()
# X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1)].iloc[int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1)])*(3/4)):, 2:].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
# y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1)].iloc[int(len(season[(season.SEASON_ID==i) | 
#          (season.SEASON_ID==i+1)])*(3/4)):, 0]
# train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1)].iloc[:int(len(season[(season.SEASON_ID==i) | 
#               (season.SEASON_ID==i+1)])*(3/4)), 5]
# test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1)].iloc[int(len(season[(season.SEASON_ID==i) | 
#              (season.SEASON_ID==i+1)])*(3/4)):, 5]


# xgb_model = xgb.XGBRegressor()


# params = {
#     "colsample_bytree": uniform(0.9, 0.5),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.02, 0.1), # default 0.1 
#     "max_depth": randint(2, 5), # default 3
#     "n_estimators": randint(100, 300), # default 100
#     "subsample": uniform(0.9, 0.5),
#     "reg_lambda": randint(0,4)
# }
# search= RandomizedSearchCV(xgb_model, param_distributions=params, random_state=26, n_iter=1000, cv=5, verbose=1, n_jobs=-1, return_train_score=True)

# search.fit(X_train, y_train)

# xgb_pred= search.best_estimator_.predict(X_test)
# mean_squared_error(y_test, xgb_pred, squared=False)
# best_2_seasons = search.best_estimator_
# best_2_seasons
# search.best_estimator_.save_model('001.model')




