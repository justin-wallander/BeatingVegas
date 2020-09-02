# Beating Vegas



## NBA Game Total Prediction
### Goals: 
* Create a Regression Model using historical NBA data to predict NBA Game total points scored. 

* Using Vegas closing line for O/U, the goal is to build a model that predicts more accurately than Vegas’ lines.

* Given my intuition on what stats might impact game totals, see if there are any stats that are surprising

* Data Soureces: NBA API, NBA.COM, BASKETBALL-REFERENCE.COM, https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm

### Challenges:
* Given that I do not have a time machine, I have to be very cognizant of what information I will have available to me at any given time
* Figuring out the test-train-split in order to avoid any potential info leakage

### Data Pipeline/Feature Engineering:

* First I had to gather all the individual game stats from the API and get it into a dataframe

* Once the data was in the data frame, I needed to combine home team and away teams on a single row. Fortunately the API provided a function to do that for me

* Once I had all the data on a single row I had to figure out what I wanted to do with it. Do I want to have a rolling average dating back maybe 10 or 20 games, or do I want to just use the season averages?

* Ultimately I decdided to go with the season averages in the final model. Figuring out how to average the info provided not be very difficult, I was able to create this function for it. It was very slow to run though, I am not sure if there are better methods to accomplish this task but will be tinkering with it more to find out: 
```python
def running_col_avg(df, year, team):
    df1 = df[(df.SEASON_ID == year)&(df.TEAM_ABBREVIATION_A == team)]
    df1= df1.reset_index(drop = True)
    df1.columns = ['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME',
                      'GAME_ID','GAME_DATE','MATCHUP','WL','MIN','PTS','FGM',
                      'FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
                      'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV',
                      'PF','PLUS_MINUS','TEAM_ID_OPP','TEAM_ABBREVIATION_OPP',
                      'TEAM_NAME_OPP','MATCHUP_OPP','WL_OPP','MIN_OPP','PTS_OPP',
                      'FGM_OPP','FGA_OPP','FG_PCT_OPP','FG3M_OPP','FG3A_OPP',
                      'FG3_PCT_OPP','FTM_OPP','FTA_OPP','FT_PCT_OPP','OREB_OPP',
                      'DREB_OPP','REB_OPP','AST_OPP','STL_OPP','BLK_OPP','TOV_OPP',
                      'PF_OPP','PLUS_MINUS_OPP', 'TEAM_TEST', 'TEST_OPP','ML_A',
                      'ML_B', 'TOTAL_OPEN', 'TOTAL_CLOSE', 'PTS_SPR_OPEN', 'PTS_SPR_CLOSE']

    avg_cols = ['WL','PTS','FGM',
                'FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
                'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV',
                'PF','PLUS_MINUS','PTS_OPP', 'FGM_OPP','FGA_OPP','FG_PCT_OPP',
                'FG3M_OPP','FG3A_OPP','FG3_PCT_OPP','FTM_OPP','FTA_OPP',
                'FT_PCT_OPP','OREB_OPP','DREB_OPP','REB_OPP','AST_OPP',
                'STL_OPP','BLK_OPP','TOV_OPP','PF_OPP']

    df1['WL']= df1['WL'].map({'W': 1, 'L' :0})
    df1['WL_OPP']=df1['WL_OPP'].map({'W': 1, 'L' :0})
    df1['GP'] = df1.index
    df1['GAME_TOTAL'] = df1['PTS'] + df1['PTS_OPP']

    df1[avg_cols] = df1[avg_cols].astype(float)
    counter = 1
    for idx in range(len(df1)):
        if counter != len(df1):
            df1.loc[counter, avg_cols] += df1.loc[idx, avg_cols]
            counter+=1
        df1.loc[idx, avg_cols] /= (idx +1)
    return df1
```

* Once I had the averages for each team's offensive and defensive stats, I took the odds dataframe I created and combined it with the averages dataframe. Then I recombine the teams again to have a final row that included the home teams offensive and defensive stats and the away teams offensive and defensive stats, as well as the odds for the O/U open and close as well as the open and close of the points spread.

* The next order of business was to insure that I was not using any information I would not have at the time of the prediction. I took out the points spread close and the O/U close. I also had to figure out how to shift the averages down 1 row while keeping everything else the same. This was to avoid the fact that if I did not do it, I would be using stats that occurred in the game in the average in order to predict that game. Fortunately I was able to figure out a some code to do this fairyly quickly: 
```python
same_list=['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME','GAME_ID','GAME_DATE',
        'MATCHUP','TEAM_TEST','TEST_OPP','ML_A','ML_B','TOTAL_OPEN','TOTAL_CLOSE',
        'PTS_SPR_OPEN','PTS_SPR_CLOSE','GP','GAME_TOTAL']


change_list= ['WL','MIN','PTS','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
	        'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PLUS_MINUS',	
            'TEAM_ID_OPP','TEAM_ABBREVIATION_OPP','TEAM_NAME_OPP','MATCHUP_OPP', 'WL_OPP',
            'MIN_OPP','PTS_OPP','FGM_OPP','FGA_OPP','FG_PCT_OPP','FG3M_OPP','FG3A_OPP',	
            'FG3_PCT_OPP','FTM_OPP','FTA_OPP','FT_PCT_OPP','OREB_OPP','DREB_OPP','REB_OPP',
            'AST_OPP','STL_OPP','BLK_OPP','TOV_OPP','PF_OPP','PLUS_MINUS_OPP']

avg_season.sort_values('GAME_ID', inplace= True)
avg_10.sort_values('GAME_ID', inplace= True)

shift_season_df = pd.DataFrame(columns=avg_season.columns)
for year in season_id_list:
    for team in team_abbrev_list:
        t_df = pd.DataFrame(columns=avg_season.columns)
        t_df[same_list] = avg_season[(avg_season.SEASON_ID == year) & (
                                      avg_season.TEAM_ABBREVIATION == team)][same_list]
        t_df[change_list] = avg_season[(avg_season.SEASON_ID == year) & (
                            avg_season.TEAM_ABBREVIATION == team)][change_list].shift(periods=1)
        t_df.dropna(inplace=True)
        shift_season_df = pd.concat([shift_season_df, t_df], ignore_index=True)
```

### Data Exploration
* Correlation map for the Offensive stats
![Offensive Correlation Map](/images/OFF_CORR.png)

* NBA Total Pts Scored Season Averages and O/U CLosing line Season Averages 

![Averages](/images/avg_pts.png)

* As we can see from this graph, Vegas is pretty dang good. Also, because of the increase in the last 3 seasons I had to completely reevaluate my strategy for my train-test-split. On the intial models that I ran, I was splitting my data by season, keeping the first 9 seasons in the training set and testing on the last 3. Due to the rise in scoring over those seasons, the model was not preforming well.

* I decided to split the data in groups of 3 seasons, and then mainly focused on the last 3 seasons for the modeling. I split those last 3 seasons into 7/9 for the train and 2/9 for the test. So I would train on 2 and 1/3 seasons and predict on the final 2/3 of the season. This is to simulate having to wait a bit into the 3rd season in order to account for any potential massive shifts in team data. 


### THE MODEL: XGBoost for the win? 

* Baseline Dummy Regressor score on all test data = 21.1072 (not too shabby) 

* The first couple of runs of my XGBoost model did not beat the Dummy (disconcerting to say the least)

* After quite a bit of tinkering with my splits, feature selection, and SKLearn’s RandomizedCV and GridsearchCV I was able to accomplish some decent results:
	* Features were whittled from 144 to 46 (categorical team data proved not to matter much in this model)
	* I ultimately used Vegas' closing line as my true baseline. Vegas’ RMSE on all of the data in my test set was 18.56 
	* My Model’s RMSE on all the data in my test set= 18.75
	* HOWEVER.... When running the model specifically on 1 team (My Dallas Mavericks) I was actually able to slightly beat Vegas:
		* Vegas RMSE when predicting DAL = 21.199
		* Model RMSE when predicting DAL= 21.108 
* The Final model hyperparameters:
```python
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
```

* A graph of the Mavs Actual Score, vs Vegas Prediction, vs Model Prediction: 

![Predicted](/images/Vegasvspredvsreal.png)

* Feature Importance- Given I would have the Vegas Opening Line, I kept that in the model and it comes as no surprise that it was the number one feature. Will be worth exploring what would happen if I take that out. Also of note, Home and Away Pts per game did not make it in the top 25 features to split on for this model, which certainly goes against what my intuition would expect. 

![F Importance](/images/F_importance.png)

### Next Steps:
* Connect to a live odds api to get current lines
* Get more granular data- advanced stats, player stats, etc
* Build a simulator to see how certain gambling strategies could play out
* ULTIMATE STEP: Add other leagues/sports into the mix








