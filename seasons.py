import pandas as pd 
import numpy as np 
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb


season = pd.read_csv('data/avg_season.csv')
# [22007, 22008, 22009, 22010, 22011, 22012, 22013, 22014, 22015,
#        22016, 22017, 22018, 22019]





    # X_train = season[(season.SEASON_ID==i)].iloc[:int(len(season[(season.SEASON_ID==i)])*(2/3)), 2:]
    # y_train = season[(season.SEASON_ID==i)].iloc[:int(len(season[(season.SEASON_ID==i)])*(2/3)), 0]

    # X_test = season[(season.SEASON_ID==i)].iloc[int(len(season[(season.SEASON_ID==i)])*(2/3)):, 2:]
    # y_test = season[(season.SEASON_ID==i)].iloc[int(len(season[(season.SEASON_ID==i)])*(2/3)):, 0]
    # train_vegas = season[(season.SEASON_ID==i)].iloc[:int(len(season[(season.SEASON_ID==i)])*(2/3)), 5]
    # test_vegas = season[(season.SEASON_ID==i)].iloc[int(len(season[(season.SEASON_ID==i)])*(2/3)):, 5]



#DUMMY REGRESSOR:

# dummy_regr = DummyRegressor(strategy="mean")
# dummy_regr.fit(X_train, y_train)
# #21.982204049481744
# mean_squared_error(y_test, dummy_regr.predict(X_test), squared = False)

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



'''

i = 22017
X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 0]
X_train
X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 0]
train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
              (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 5]
test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
             (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 5]





xgb_model = xgb.XGBRegressor()

params = {
    "colsample_bytree": [0.7,0.8,0.9,0.95,1],
    "gamma": [0, 0.2],
    "learning_rate": [0.02,0.025,0.03,0.035,0.04,0.045, 0.05],
    "max_depth": [2], # default 3
    "n_estimators": [120,140,150,160,170,200,250],
    "subsample": [0.6,0.7,0.8,0.9,0.95,1]}

search = GridSearchCV(xgb_model, param_grid=params, cv=3, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train, y_train)

xgb_pred= search.best_estimator_.predict(X_test)
xgb_pred
search.best_estimator_
mean_squared_error(y_test, xgb_pred, squared=False)


XGBRegressor(base_score=0.5, 
             booster='gbtree', 
             colsample_bytree=1
             learning_rate=0.035, 
             max_depth=2,
             n_estimators=160,  
             random_state=0,
             reg_lambda=1, 
             subsample=0.7,
             )

for i in np.linspace(0.0, 0.3,51):
# for i in np.linspace(0.5, 1,51):
# for i in range(10):
    bst = xgb.XGBRegressor(base_score=0.5, 
                       booster='gbtree', 
                       colsample_bytree=0.77,
                       gpu_id=-1, 
                       gamma=0, 
                       learning_rate=0.064,
                       max_depth=2, 
                       n_estimators=130, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=1,
                       subsample=0.76,
                       )
    bst.fit(X_train,y_train)
    model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
    vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
    train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
    print(f'{i} Vegas: {vegas}  Model: {model_score}  Train: {train_score}')

bst = xgb.XGBRegressor(base_score=0.5, 
                       booster='gbtree', 
                       colsample_bytree=0.77,
                       gpu_id=-1, 
                       gamma=0, 
                       learning_rate=0.064,
                       max_depth=2, 
                       n_estimators=130, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=1,
                       subsample=0.76,
                       )
bst.fit(X_train,y_train)
train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
print(f'Vegas: {vegas}  Model: {model_score}  Train: {train_score}')

for i in range(22007,22018):
    X_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
          (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
    y_train = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 0]
    X_train
    X_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 2:86].drop(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'], axis = 1)
    y_test = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
         (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 0]
    train_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[:int(len(season[(season.SEASON_ID==i) | 
              (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)), 5]
    test_vegas = season[(season.SEASON_ID==i) | (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)].iloc[int(len(season[(season.SEASON_ID==i) | 
             (season.SEASON_ID==i+1) | (season.SEASON_ID==i+2)])*(7/9)):, 5]

    bst.fit(X_train,y_train)
    train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
    vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
    model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
    print(f'{i} = Vegas: {vegas}  Model: {model_score}  Train: {train_score}')



plt_x = sorted(bst.feature_importances_[bst.feature_importances_>=0.013398], reverse=False)
plt_x
plt_y = X_test.columns[np.argsort(bst.feature_importances_[bst.feature_importances_>=0.013398])]
fig,ax = plt.subplots(figsize=(30,24))
#xgb.plot_importance(bst, ax=ax)
ax.barh(plt_y, plt_x)

drop_cols = list(X_test.columns[bst.feature_importances_<0.011533])
drop_cols.extend(['TOTAL_CLOSE','ML_A','ML_B','WL_A','WL_B','PLUS_MINUS_A','PLUS_MINUS_B'])
drop_cols 

from numpy import sort
from sklearn.feature_selection import SelectFromModel
 

# Fit model using each importance as a threshold
thresholds = sort(bst.feature_importances_)
for thresh in thresholds:
	# select features using threshold
    selection = SelectFromModel(bst, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
	# train model
    selection_model = xgb.XGBRegressor(base_score=0.5, 
                       booster='gbtree', 
                       colsample_bytree=0.77,
                       gpu_id=-1, 
                       gamma=0, 
                       learning_rate=0.064,
                       max_depth=2, 
                       n_estimators=130, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=1,
                       subsample=0.76,
                       )
    selection_model.fit(select_X_train, y_train)
	# eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    
    train_score = mean_squared_error(y_train, bst.predict(X_train), squared=False)
    vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
    model_score = mean_squared_error(y_test, bst.predict(X_test), squared=False)
    #print(f'{thresh} = Vegas: {vegas}  Model: {model_score}  Train: {train_score}')
    accuracy = mean_squared_error(y_test, predictions, squared=False)
    print(f"Thresh={thresh:3f}, n:{select_X_train.shape[1]}, Accuracy: {accuracy}")

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

for i in np.linspace(0.02, 0.08,61):
for i in np.linspace(0.5, 1,51):
for i in range(20):
    bst1 = xgb.XGBRegressor(base_score=0.5, 
                       booster='gbtree', 
                       colsample_bytree=0.96,
                       gpu_id=-1, 
                       gamma=0, 
                       learning_rate=0.063,
                       max_depth=2, 
                       n_estimators=109, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=5,
                       subsample=0.61,
                       )
    bst1.fit(X_train,y_train)
    model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
    vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
    train_score = mean_squared_error(y_train, bst1.predict(X_train), squared=False)
    print(f'{i} Vegas: {vegas}  Model: {model_score}  Train: {train_score}')


### BEST MODEL SO FAR
bst1 = xgb.XGBRegressor(base_score=0.5, 
                       booster='gbtree', 
                       colsample_bytree=0.96,
                       gpu_id=-1, 
                       gamma=0, 
                       learning_rate=0.063,
                       max_depth=2, 
                       n_estimators=109, 
                       n_jobs=-1,
                       random_state=0, 
                       reg_lambda=5,
                       subsample=0.61,
                       )
bst1.fit(X_train,y_train)
model_score = mean_squared_error(y_test, bst1.predict(X_test), squared=False)
vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
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
    vegas = mean_squared_error(np.append(y_train,y_test), np.append(train_vegas,test_vegas), squared = False)
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




