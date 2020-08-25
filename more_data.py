from nba_api.stats.static import teams
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
from nba_api.stats.library.parameters import MeasureTypeDetailed
from nba_api.stats.library.parameters import MeasureTypeDetailedDefense
from nba_api.stats.endpoints import teamdashboardbyteamperformance
from nba_api.stats.endpoints import boxscoreadvancedv2
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime


nba_teams = teams.get_teams()
#Sameple code:
# Select the dictionary for the Celtics, which contains their team ID
# celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]

team_abbrev_list = sorted([team['abbreviation'] for team in nba_teams])
#SEA, NOH, NJN, NOK are odd team names I deal with later by mapping over them
team_abbrev_list


team_id_list = [team['id'] for team in nba_teams]



from nba_api.stats.endpoints import leaguegamefinder


game_id = pd.read_csv('shifted.csv')
game_id = game_id.GAME_ID
game_id[-1:]
df = pd.DataFrame()#columns=cols)
box_adv = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id = 21801211)
games = gamefinder.get_data_frames()[0]
print(games)

#creating database of games going back to 2007 - as far back as my odds database goes
for ele in game_id:
    gamefinder = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id= ele)
    games = gamefinder.get_data_frames()[0]
    print(games)
    #games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    #games = games[games.SEASON_ID.str[1:].astype(int) >= 2007]
    #df = pd.concat([df, games], ignore_index=True)
    break
# having an issue with IND and BOS only having 81 games, in 1 season that is the case but not all- resolved
df.head()


def checking_season_count(df, year, team):
    if 'TEAM_ABBREVIATION_A' in df.columns:
        return ((df[(df.SEASON_ID == year) & (df['TEAM_ABBREVIATION_A'] == team) & 
        (df['GAME_DATE'] > datetime.datetime(int(year[1:]), 8, 9))]
        [['GAME_ID', 'TEAM_ABBREVIATION_A', 'GAME_DATE', 'MATCHUP_A']]),
        (df[(df.SEASON_ID == year) & (df['TEAM_ABBREVIATION_B'] == team) & 
        (df['GAME_DATE'] > datetime.datetime(int(year[1:]), 8, 9))]
        [['GAME_ID', 'TEAM_ABBREVIATION_B', 'GAME_DATE', 'MATCHUP_B']]))
    return df[(df.SEASON_ID == year) & (df['TEAM_ABBREVIATION'] == team) & (df['GAME_DATE'] > datetime.datetime(int(year[1:]), 8, 9))][['GAME_ID', 'TEAM_ABBREVIATION','GAME_DATE', 'MATCHUP']]



#this was defined on the NBA_API github
def combine_team_games(df, keep_method='home'):
    '''Combine a TEAM_ID-GAME_ID unique table into rows by game. Slow.

        Parameters
        ----------
        df : Input DataFrame.
        keep_method : {'home', 'away', 'winner', 'loser', ``None``}, default 'home'
            - 'home' : Keep rows where TEAM_A is the home team.
            - 'away' : Keep rows where TEAM_A is the away team.
            - 'winner' : Keep rows where TEAM_A is the losing team.
            - 'loser' : Keep rows where TEAM_A is the winning team.
            - ``None`` : Keep all rows. Will result in an output DataFrame the same
                length as the input DataFrame.
                
        Returns
        -------
        result : DataFrame
    '''
    # Join every row to all others with the same game ID.
    joined = pd.merge(df, df, suffixes=['_A', '_B'],
                      on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
    # Filter out any row that is joined to itself.
    result = joined[joined.TEAM_ID_A != joined.TEAM_ID_B]
    # Take action based on the keep_method flag.
    if keep_method is None:
        # Return all the rows.
        pass
    elif keep_method.lower() == 'home':
        # Keep rows where TEAM_A is the home team.
        result = result[result.MATCHUP_A.str.contains(' vs. ')]
    elif keep_method.lower() == 'away':
        # Keep rows where TEAM_A is the away team.
        result = result[result.MATCHUP_A.str.contains(' @ ')]
    elif keep_method.lower() == 'winner':
        result = result[result.WL_A == 'W']
    elif keep_method.lower() == 'loser':
        result = result[result.WL_A == 'L']
    else:
        raise ValueError(f'Invalid keep_method: {keep_method}')
    return result




combined_df = combine_team_games(df, keep_method=None)
#Mapping incorrect teams DO for A and B
combined_df['TEAM_ABBREVIATION_A'] = combined_df['TEAM_ABBREVIATION_A'].map(
    {
        'SEA': 'OKC', 'NOH': 'NOP', 'NOK': 'NOP', 'NJN': 'BKN',
        'ATL': 'ATL', 'NYK': 'NYK', 'CHA': 'CHA', 'MEM': 'MEM', 
        'WAS': 'WAS', 'POR': 'POR', 'BKN': 'BKN', 'ORL': 'ORL', 
        'PHI': 'PHI', 'DAL': 'DAL', 'MIA': 'MIA', 'CLE': 'CLE', 
        'BOS': 'BOS', 'MIN': 'MIN', 'TOR': 'TOR', 'OKC': 'OKC', 
        'LAC': 'LAC', 'DET': 'DET', 'SAS': 'SAS', 'PHX': 'PHX', 
        'HOU': 'HOU', 'DEN': 'DEN', 'IND': 'IND', 'CHI': 'CHI', 
        'MIL': 'MIL', 'UTA': 'UTA', 'LAL': 'LAL', 'GSW': 'GSW', 
        'SAC': 'SAC', 'NOP': 'NOP'
    }
)

combined_df['TEAM_ABBREVIATION_B'] = combined_df['TEAM_ABBREVIATION_B'].map(
    {
        'SEA': 'OKC', 'NOH': 'NOP', 'NOK': 'NOP', 'NJN': 'BKN',
        'ATL': 'ATL', 'NYK': 'NYK', 'CHA': 'CHA', 'MEM': 'MEM', 
        'WAS': 'WAS', 'POR': 'POR', 'BKN': 'BKN', 'ORL': 'ORL', 
        'PHI': 'PHI', 'DAL': 'DAL', 'MIA': 'MIA', 'CLE': 'CLE', 
        'BOS': 'BOS', 'MIN': 'MIN', 'TOR': 'TOR', 'OKC': 'OKC', 
        'LAC': 'LAC', 'DET': 'DET', 'SAS': 'SAS', 'PHX': 'PHX', 
        'HOU': 'HOU', 'DEN': 'DEN', 'IND': 'IND', 'CHI': 'CHI', 
        'MIL': 'MIL', 'UTA': 'UTA', 'LAL': 'LAL', 'GSW': 'GSW', 
        'SAC': 'SAC', 'NOP': 'NOP'
    }
)



#I need to figure out what extra games are happening in each season and figure a way to delete them

season_id_list =sorted(combined_df['SEASON_ID'].unique())
season_id_list[:-1]
team_abbrev_list
#I figured out a good cutoff point to get rid of the extra preseaon games
clean_df = pd.DataFrame(columns=combined_df.columns)
for year in season_id_list:
    clean = combined_df[(combined_df['SEASON_ID'] == year) & (combined_df['GAME_DATE'] > datetime.datetime(int(year[1:]), 8, 9))].sort_values('GAME_DATE')
    clean_df = pd.concat([clean_df, clean], ignore_index=True)

#taking out 2019 for now, can add back later or use as testing
df_2019 = clean_df[clean_df.SEASON_ID == '22019']
clean_df = clean_df[clean_df.SEASON_ID != '22019']
clean_df

#I think this is a good spot to insert the Odds data
#well turns out I probably will be needing the vegas lines so here we go

'''couple of things to keep in mind here, what information am i going to actually have available to me. 
We arranged the averages in order to predict the next games totals. What about in terms of gambliing information. 
I will have the open line. Line referring to O/U.  I will have the current line. For these purposes, I am going to 
train the model on open and closing line. If training vastly outperfoms testing, I will take out the closing line, 
and use current line as open line when making predictions in real time, potentially. I believe the way I have my data
consturcted, team A is the home team. In this case, I will also add the points line, whether positive for favored or 
neg for not. I have an open and close situation here as well that I will need to test like above.
'''
odds_19 = pd.read_excel('data/nba_odds_2019.xlsx')
odds_19.Date.value_counts()


odds_df = pd.DataFrame(columns=['SEASON_ID','GAME_ID','MATCHUP','GAME_DATE','TEAM_ID','Date','VH', 'Team', 'Final','Open', 'Close', 'ML'])
for i in [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]:
    odds_i = pd.read_excel(f'data/nba_odds_{i}.xlsx')
    odds_i.Date = odds_i.Date.apply(lambda x: str(x)[:2] + '-' + str(x)[2:] + '-' + f'{i}' if x>999 else str(x)[:1] + '-' + str(x)[1:] + '-' + f'{i+1}')
    odds_i.Date = pd.to_datetime(odds_i.Date)
    odds_i.Team = odds_i.Team.map({'Atlanta':'ATL','Brooklyn':'BKN','NewJersey' :'BKN','Boston':'BOS',
                                    'Charlotte':'CHA','Chicago':'CHI','Cleveland':'CLE','Dallas':'DAL',
                                    'Denver':'DEN','Detroit':'DET','GoldenState':'GSW','Houston':'HOU',
                                    'Indiana':'IND','LAClippers':'LAC','LA Clippers' :'LAC','LALakers':'LAL',
                                    'LA Lakers': 'LAL','Memphis':'MEM','Miami':'MIA','Milwaukee':'MIL',
                                    'Minnesota':'MIN','NewOrleans':'NOP','New Orleans' :'NOP','NewYork':'NYK',
                                    'OklahomaCity':'OKC','Oklahoma City':'OKC','Seattle':'OKC','Orlando':'ORL',
                                    'Philadelphia':'PHI','Phoenix':'PHX','Portland':'POR','Sacramento':'SAC',
                                    'SanAntonio':'SAS','San Antonio':'SAS','Toronto':'TOR','Utah':'UTA',
                                    'Washington':'WAS'})
    odds_i['SEASON_ID']= np.nan
    odds_i['GAME_ID']= np.nan
    odds_i['MATCHUP'] = np.nan
    odds_i['TEAM_ID']= np.nan
    odds_i['GAME_DATE']= np.nan
    odds_i=odds_i[['SEASON_ID','GAME_ID','MATCHUP','GAME_DATE','TEAM_ID','Date','VH', 'Team', 'Final','Open', 'Close', 'ML']]
    odds_df = pd.concat([odds_df, odds_i], ignore_index=True)

odds_df['Date'].value_counts()[:100]



# very bad code that literally takes forever (1 hour only 19 unique Game Ids)
# for idx1 in range(len(clean_df)):
#     for idx2 in range(len(odds_df)):
#         if (clean_df.GAME_DATE[idx1] == odds_df.Date[idx2]) and (clean_df.MATCHUP_A.str.contains(odds_df.Team[idx2])[idx1]):
#             odds_df.loc[idx2,'SEASON_ID'] = clean_df.loc[idx1,'SEASON_ID']
#             odds_df.loc[idx2, 'MATCHUP'] = clean_df.loc[idx1, 'MATCHUP_A']
#             odds_df.loc[idx2, 'TEAM_ID'] = clean_df.loc[idx1, 'TEAM_ID_A']
#             odds_df.loc[idx2, 'GAME_ID'] = clean_df.loc[idx1, 'GAME_ID']
#             odds_df.loc[idx2, 'GAME_DATE'] = clean_df.loc[idx1, 'GAME_DATE']

clean_df2 = clean_df.copy()
odds_df2 = odds_df.copy()
odds_df2= odds_df2.drop('GAME_DATE', axis = 1)
odds_df2.columns = ['SEASON_ID', 'GAME_ID', 'MATCHUP', 'TEAM_ID', 'GAME_DATE', 'VH', 'Team',
                    'Final', 'Open', 'Close', 'ML']


clean_df2_dict = {}
for idx in range(len(clean_df2)):
    clean_df2_dict[str(clean_df2.loc[idx,'GAME_DATE']) +'_'+ str(clean_df2.loc[idx,'TEAM_ABBREVIATION_A'])]= clean_df2.loc[idx, ['SEASON_ID','MATCHUP_A','TEAM_ID_A','GAME_ID']]
    

for idx in range(len(odds_df2)):
    if str(odds_df2.loc[idx,'GAME_DATE']) +'_'+ str(odds_df2.loc[idx,'Team']) in clean_df2_dict:
        odds_df2.loc[idx,'GAME_ID'] = clean_df2_dict[str(odds_df2.loc[idx,'GAME_DATE']) +'_'+ str(odds_df2.loc[idx,'Team'])]['GAME_ID']
        odds_df2.loc[idx,'MATCHUP'] = clean_df2_dict[str(odds_df2.loc[idx,'GAME_DATE']) +'_'+ str(odds_df2.loc[idx,'Team'])]['MATCHUP_A']
        odds_df2.loc[idx,'TEAM_ID'] = clean_df2_dict[str(odds_df2.loc[idx,'GAME_DATE']) +'_'+ str(odds_df2.loc[idx,'Team'])]['TEAM_ID_A']
        odds_df2.loc[idx,'SEASON_ID'] = clean_df2_dict[str(odds_df2.loc[idx,'GAME_DATE']) +'_'+ str(odds_df2.loc[idx,'Team'])]['SEASON_ID']

#find missing ids + 1 id from 2011 that had the wrong date
# for i in clean_df2.GAME_ID.unique():
#     if i not in odds_df2.GAME_ID.unique():
#         print(i)

missing_id = ['0020700405','0020700406','0020700407','0021000436','0021000437',
            '0021000433','0021000435','0021000434', '0021000700']

odds_idx = [808,809,810,811,812,813,8748,8749,8750,8751,8752,8753,8754,8755,8756,8757,9281]
clean2_idx = [808,813,809,812,811,810,8249,8246,8250,8251,8248,8247,8242,8243,8244,8245,8773]
for idx1, idx2 in zip(odds_idx, clean2_idx):
    odds_df2.loc[idx1,'GAME_ID'] = clean_df2.loc[idx2,'GAME_ID']
    odds_df2.loc[idx1,'MATCHUP'] = clean_df2.loc[idx2,'MATCHUP_A'] 
    odds_df2.loc[idx1,'TEAM_ID'] = clean_df2.loc[idx2,'TEAM_ID_A'] 
    odds_df2.loc[idx1,'SEASON_ID'] = clean_df2.loc[idx2,'SEASON_ID']
    odds_df2.loc[idx1,'GAME_DATE'] = clean_df2.loc[idx2,'GAME_DATE']


### finally just took it into google sheets to clean it up
odds_df2.to_csv('odds_df2.csv')

odds_clean = pd.read_csv('odds_clean.csv')
odds_clean.info()
     

clean_df3 = clean_df.copy()
clean_df3 = clean_df3.sort_values('GAME_ID')
clean_df3.reset_index(drop = True, inplace=True)
# clean_df3['ML_A'] = np.nan 
# clean_df3['ML_B'] = np.nan 
# clean_df3['TOTAL_OPEN'] = np.nan 
# clean_df3['TOTAL_CLOSE'] = np.nan 
# clean_df3['PTS_SPR_OPEN'] = np.nan 
# clean_df3['PTS_SPR_CLOSE'] = np.nan 
clean_df3.info()
odds_clean['GAME_ID'] = '00' + odds_clean['GAME_ID'].astype(str)
odds_merge = odds_clean[['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'TEAM_A','TEAM_B', 'ML_A', 'ML_B', 'TOTAL_OPEN',
                        'TOTAL_CLOSE', 'PTS_SPR_OPEN', 'PTS_SPR_CLOSE']]

odds_merge.GAME_DATE = pd.to_datetime(odds_merge.GAME_DATE)
odds_merge['SEASON_ID'] = odds_merge['SEASON_ID'].astype(str)

odds_merge['GAME_DATE'][0] == clean_df3['GAME_DATE'][0]

clean_merge = pd.merge(clean_df3, odds_merge, on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
clean_merge[['MATCHUP_A', 'TEAM_A', 'TEAM_B', 'ML_A','TOTAL_OPEN', 'PTS_SPR_CLOSE']].tail()
clean_merge.info()

#i think this clean merge is the winner


#now I need to clean data types, remove columns and create the running average. I will try to do it with 1
#and then turn it into a function to loop through each team and season... function turned out ok, but
#takes for ever in the for loop 
def running_col_avg(df, year, team):
    df1 = df[(df.SEASON_ID == year)&(df.TEAM_ABBREVIATION_A == team)]
    #df1 = df1.sort_values('GAME_DATE', ascending = False)
    df1= df1.reset_index()
    df1= df1.drop('index', axis=1)
    df1.columns = ['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME',
                      'GAME_ID','GAME_DATE','MATCHUP','WL','MIN','PTS','FGM',
                      'FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
                      'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV',
                      'PF','PLUS_MINUS','TEAM_ID_OPP','TEAM_ABBREVIATION_OPP',
                      'TEAM_NAME_OPP','MATCHUP_OPP','WL_OPP','MIN_OPP','PTS_OPP',
                      'FGM_OPP','FGA_OPP','FG_PCT_OPP','FG3M_OPP','FG3A_OPP',
                      'FG3_PCT_OPP','FTM_OPP','FTA_OPP','FT_PCT_OPP','OREB_OPP',
                      'DREB_OPP','REB_OPP','AST_OPP','STL_OPP','BLK_OPP','TOV_OPP',
                      'PF_OPP','PLUS_MINUS_OPP']

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
    df1['GAME_TOTAL'] = df1['PTS'].astype(int) + df1['PTS_OPP'].astype(int)
    df1[avg_cols] = df1[avg_cols].astype(float)
    counter = 1
    for idx in range(len(df1)):
        # for col in avg_cols:
        #     df1[col]= df1[col].astype(float)
        #     if df1.iloc[idx]['SEASON_ID'] == '22012' and (df1.iloc[idx]['TEAM_ABBREVIATION'] == 'BOS'
        #                                                     or 
        #     if idx != 81 and df1.iloc[idx]['SEASON_ID'] != '22011': 
        #         df1.loc[idx + 1, col] += df1.loc[idx , col]
        #     elif df1.iloc[idx]['SEASON_ID'] == '22011' and idx != 65:
        #         df1.loc[idx + 1, col] += df1.loc[idx , col]


        #     df1.loc[idx , col] /= (idx+1)
        if counter != len(df1):
            df1.loc[counter, avg_cols] += df1.loc[idx, avg_cols]
            counter+=1
        df1.loc[idx, avg_cols] /= (idx +1)


    return df1 

#creating a second to create the totals on the line prior in order to avoid potential info leakage
#since the line I currently have the totals on incorporate the actual score of the game
def running_col_avg2(df, year, team):
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
    # for idx  in range(len(df1)):
    #     if idx + 1 != len(df1):
    df1['GAME_TOTAL'] = df1['PTS'] + df1['PTS_OPP']

    df1[avg_cols] = df1[avg_cols].astype(float)
    counter = 1
    for idx in range(len(df1)):
        if counter != len(df1):
            df1.loc[counter, avg_cols] += df1.loc[idx, avg_cols]
            counter+=1
        df1.loc[idx, avg_cols] /= (idx +1)


    return df1


#This takes forver, not sure if there is a way to rework this code to figure out a quicker process
# avg_df = pd.DataFrame(columns=['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME',
#                       'GAME_ID','GAME_DATE','MATCHUP','WL','MIN','PTS','FGM',
#                       'FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
#                       'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV',
#                       'PF','PLUS_MINUS','TEAM_ID_OPP','TEAM_ABBREVIATION_OPP',
#                       'TEAM_NAME_OPP','MATCHUP_OPP','WL_OPP','MIN_OPP','PTS_OPP',
#                       'FGM_OPP','FGA_OPP','FG_PCT_OPP','FG3M_OPP','FG3A_OPP',
#                       'FG3_PCT_OPP','FTM_OPP','FTA_OPP','FT_PCT_OPP','OREB_OPP',
#                       'DREB_OPP','REB_OPP','AST_OPP','STL_OPP','BLK_OPP','TOV_OPP',
#                       'PF_OPP','PLUS_MINUS_OPP'])

# for year in season_id_list[:-1]:
#     for team in team_abbrev_list:
#         avg_df = pd.concat([avg_df, running_col_avg(clean_df, year, team)], ignore_index=True)


#checking_season_count(avg_df, '22014', 'DAL')
#Messed around with some plots, hist, and kde of the PTS for a season
# avg_df[(avg_df.SEASON_ID > '22014') & (avg_df.TEAM_ABBREVIATION == 'GSW')]['PTS'].plot.kde()
# avg_df[(avg_df.SEASON_ID > '22004')]['PTS'].plot.kde()

#combine the data base again, this time keeping only home teams
# avg_combined_df = combine_team_games(avg_df)
# avg_comb_cols= avg_combined_df.columns
# avg_combined_df = avg_combined_df.sort_values('GAME_ID')
# avg_combined_df.reset_index(drop = True, inplace=True)
#need to get rid of columns, going to cheat a little and use google sheets to helo works through some
# avg_combined_df.head().to_csv('avg_comb_df_cols.csv',index=False)

# avg_combined_cols = ['GAME_TOTAL_A','SEASON_ID','GAME_DATE','TEAM_ABBREVIATION_A','GP_A','WL_A','PTS_A','FGM_A',
#                 'FGA_A','FG_PCT_A','FG3M_A','FG3A_A','FG3_PCT_A','FTM_A','FTA_A','FT_PCT_A','OREB_A','DREB_A',
#                 'REB_A','AST_A','STL_A','BLK_A','TOV_A','PF_A','PLUS_MINUS_A','PTS_OPP_A','FGM_OPP_A','FGA_OPP_A',
#                 'FG_PCT_OPP_A','FG3M_OPP_A','FG3A_OPP_A','FG3_PCT_OPP_A','FTM_OPP_A','FTA_OPP_A',
#                 'FT_PCT_OPP_A','OREB_OPP_A','DREB_OPP_A','REB_OPP_A','AST_OPP_A','STL_OPP_A','BLK_OPP_A',	
#                 'TOV_OPP_A','PF_OPP_A','TEAM_ABBREVIATION_B','GP_B','WL_B','PTS_B','FGM_B','FGA_B',	
#                 'FG_PCT_B','FG3M_B','FG3A_B','FG3_PCT_B','FTM_B','FTA_B','FT_PCT_B','OREB_B','DREB_B',	
#                 'REB_B','AST_B','STL_B','BLK_B','TOV_B','PF_B','PLUS_MINUS_B','PTS_OPP_B','FGM_OPP_B',	
#                 'FGA_OPP_B','FG_PCT_OPP_B','FG3M_OPP_B','FG3A_OPP_B','FG3_PCT_OPP_B','FTM_OPP_B',
#                 'FTA_OPP_B','FT_PCT_OPP_B','OREB_OPP_B','DREB_OPP_B','REB_OPP_B','AST_OPP_B','STL_OPP_B',
#                 'BLK_OPP_B','TOV_OPP_B','PF_OPP_B']

# avg_combined_df = avg_combined_df[avg_combined_cols]




#creating the second database and probably the one I will actually end up using
avg2_df = pd.DataFrame(columns= ['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME',
                      'GAME_ID','GAME_DATE','MATCHUP','WL','MIN','PTS','FGM',
                      'FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
                      'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV',
                      'PF','PLUS_MINUS','TEAM_ID_OPP','TEAM_ABBREVIATION_OPP',
                      'TEAM_NAME_OPP','MATCHUP_OPP','WL_OPP','MIN_OPP','PTS_OPP',
                      'FGM_OPP','FGA_OPP','FG_PCT_OPP','FG3M_OPP','FG3A_OPP',
                      'FG3_PCT_OPP','FTM_OPP','FTA_OPP','FT_PCT_OPP','OREB_OPP',
                      'DREB_OPP','REB_OPP','AST_OPP','STL_OPP','BLK_OPP','TOV_OPP',
                      'PF_OPP','PLUS_MINUS_OPP', 'TEAM_TEST', 'TEST_OPP','ML_A',
                      'ML_B', 'TOTAL_OPEN', 'TOTAL_CLOSE', 'PTS_SPR_OPEN', 'PTS_SPR_CLOSE'])


for year in season_id_list[:-1]:
    for team in team_abbrev_list:
        avg2_df = pd.concat([avg2_df, running_col_avg2(clean_merge, year, team)], ignore_index=True)

avg2_df[avg2_df.SEASON_ID=='22018'][['GAME_DATE','GAME_TOTAL', 'MATCHUP', 'TOTAL_CLOSE', 'PTS_SPR_OPEN']].head()
#checking_season_count(avg_df, '22014', 'DAL')
#Messed around with some plots, hist, and kde of the PTS for a season
avg2_df[(avg2_df.SEASON_ID > '22014') & (avg2_df.TEAM_ABBREVIATION == 'GSW')]['PTS'].plot.kde()
avg2_df[(avg2_df.SEASON_ID > '22004')]['PTS'].plot.kde()

plt.scatter(avg2_df[avg2_df.SEASON_ID=='22018'].GAME_DATE, avg2_df[avg2_df.SEASON_ID=='22018'].GAME_TOTAL)
plt.scatter(avg2_df[avg2_df.SEASON_ID=='22018'].GAME_DATE, avg2_df[avg2_df.SEASON_ID=='22018'].TOTAL_CLOSE)
plt.show()

avg2_df.to_csv('avg2_df.csv', index= False)
# i need to shift the df in order for there not to be info leakage, shift the stats down to predict on the next game
#given the most recent averages. Lines, totals, game dates, etc need to remain in the same spot though
same_list=['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME','GAME_ID','GAME_DATE',
        'MATCHUP','TEAM_TEST','TEST_OPP','ML_A','ML_B','TOTAL_OPEN','TOTAL_CLOSE',
        'PTS_SPR_OPEN','PTS_SPR_CLOSE','GP','GAME_TOTAL']





change_list= ['WL','MIN','PTS','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
	        'FT_PCT','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PLUS_MINUS',	
            'TEAM_ID_OPP','TEAM_ABBREVIATION_OPP','TEAM_NAME_OPP','MATCHUP_OPP', 'WL_OPP',
            'MIN_OPP','PTS_OPP','FGM_OPP','FGA_OPP','FG_PCT_OPP','FG3M_OPP','FG3A_OPP',	
            'FG3_PCT_OPP','FTM_OPP','FTA_OPP','FT_PCT_OPP','OREB_OPP','DREB_OPP','REB_OPP',
            'AST_OPP','STL_OPP','BLK_OPP','TOV_OPP','PF_OPP','PLUS_MINUS_OPP']

avg2_df.sort_values('GAME_ID', inplace= True)
avg2_df

shifted_df = pd.DataFrame(columns=avg2_df.columns)
for year in season_id_list[:-1]:
    for team in team_abbrev_list:
        t_df = pd.DataFrame(columns=avg2_df.columns)
        t_df[same_list] = avg2_df[(avg2_df.SEASON_ID == year) & (avg2_df.TEAM_ABBREVIATION == team)][same_list]
        t_df[change_list] = avg2_df[(avg2_df.SEASON_ID == year) & (avg2_df.TEAM_ABBREVIATION == team)][change_list].shift(periods=1)
        t_df.dropna(inplace=True)
        shifted_df = pd.concat([shifted_df, t_df], ignore_index=True)
        
        
shifted_df.to_csv('shifted.csv', index= False)







#combine the data base again, this time keeping only home teams
avg2_combined_df = combine_team_games(shifted_df)
# avg2_comb_cols= avg2_combined_df.columns
avg2_combined_df = avg2_combined_df.sort_values('GAME_ID')
avg2_combined_df.reset_index(drop = True, inplace=True)
#need to get rid of columns, going to cheat a little and use google sheets to helo works through some
# avg_combined_df.head().to_csv('avg_comb_df_cols.csv',index=False)
    


avg2_combined_cols = ['GAME_TOTAL_A','SEASON_ID','GAME_DATE','TEAM_ABBREVIATION_A','ML_A_A',
                'TOTAL_OPEN_A','TOTAL_CLOSE_A','PTS_SPR_OPEN_A','PTS_SPR_CLOSE_A','GP_A','WL_A','PTS_A','FGM_A',
                'FGA_A','FG_PCT_A','FG3M_A','FG3A_A','FG3_PCT_A','FTM_A','FTA_A','FT_PCT_A','OREB_A','DREB_A',
                'REB_A','AST_A','STL_A','BLK_A','TOV_A','PF_A','PLUS_MINUS_A','PTS_OPP_A','FGM_OPP_A','FGA_OPP_A',
                'FG_PCT_OPP_A','FG3M_OPP_A','FG3A_OPP_A','FG3_PCT_OPP_A','FTM_OPP_A','FTA_OPP_A',
                'FT_PCT_OPP_A','OREB_OPP_A','DREB_OPP_A','REB_OPP_A','AST_OPP_A','STL_OPP_A','BLK_OPP_A',	
                'TOV_OPP_A','PF_OPP_A','TEAM_ABBREVIATION_B','ML_B_A','GP_B','WL_B','PTS_B','FGM_B','FGA_B',	
                'FG_PCT_B','FG3M_B','FG3A_B','FG3_PCT_B','FTM_B','FTA_B','FT_PCT_B','OREB_B','DREB_B',	
                'REB_B','AST_B','STL_B','BLK_B','TOV_B','PF_B','PLUS_MINUS_B','PTS_OPP_B','FGM_OPP_B',	
                'FGA_OPP_B','FG_PCT_OPP_B','FG3M_OPP_B','FG3A_OPP_B','FG3_PCT_OPP_B','FTM_OPP_B',
                'FTA_OPP_B','FT_PCT_OPP_B','OREB_OPP_B','DREB_OPP_B','REB_OPP_B','AST_OPP_B','STL_OPP_B',
                'BLK_OPP_B','TOV_OPP_B','PF_OPP_B']

avg2_combined_df = avg2_combined_df[avg2_combined_cols]
avg2_combined_df.columns = ['GAME_TOTAL','SEASON_ID','GAME_DATE','TEAM_A', 'ML_A',
                'TOTAL_OPEN','TOTAL_CLOSE','PTS_SPR_OPEN','PTS_SPR_CLOSE','GP_A','WL_A','PTS_A','FGM_A',
                'FGA_A','FG_PCT_A','FG3M_A','FG3A_A','FG3_PCT_A','FTM_A','FTA_A','FT_PCT_A','OREB_A','DREB_A',
                'REB_A','AST_A','STL_A','BLK_A','TOV_A','PF_A','PLUS_MINUS_A','PTS_OPP_A','FGM_OPP_A','FGA_OPP_A',
                'FG_PCT_OPP_A','FG3M_OPP_A','FG3A_OPP_A','FG3_PCT_OPP_A','FTM_OPP_A','FTA_OPP_A',
                'FT_PCT_OPP_A','OREB_OPP_A','DREB_OPP_A','REB_OPP_A','AST_OPP_A','STL_OPP_A','BLK_OPP_A',	
                'TOV_OPP_A','PF_OPP_A','TEAM_B','ML_B','GP_B','WL_B','PTS_B','FGM_B','FGA_B',	
                'FG_PCT_B','FG3M_B','FG3A_B','FG3_PCT_B','FTM_B','FTA_B','FT_PCT_B','OREB_B','DREB_B',	
                'REB_B','AST_B','STL_B','BLK_B','TOV_B','PF_B','PLUS_MINUS_B','PTS_OPP_B','FGM_OPP_B',	
                'FGA_OPP_B','FG_PCT_OPP_B','FG3M_OPP_B','FG3A_OPP_B','FG3_PCT_OPP_B','FTM_OPP_B',
                'FTA_OPP_B','FT_PCT_OPP_B','OREB_OPP_B','DREB_OPP_B','REB_OPP_B','AST_OPP_B','STL_OPP_B',
                'BLK_OPP_B','TOV_OPP_B','PF_OPP_B']
avg2_combined_df.info()
avg2_combined_df.GAME_TOTAL = avg2_combined_df.GAME_TOTAL.astype(int)
#now i need to split this up the dummy vaiables, to ordinal, and then create train and test csvs
avg2_combined_df['GAME_DATE'] = avg2_combined_df['GAME_DATE'].apply(lambda x: x.toordinal())

avg2_combined_df= pd.concat([avg2_combined_df, pd.get_dummies(avg2_combined_df['TEAM_A'],prefix='TEAM_A', drop_first=True)], axis = 1)
avg2_combined_df= pd.concat([avg2_combined_df, pd.get_dummies(avg2_combined_df['TEAM_B'],prefix='TEAM_B', drop_first=True)], axis = 1)
avg2_combined_df= avg2_combined_df.drop(['TEAM_A', 'TEAM_B'], axis = 1)
avg2_combined_df.info()


test = avg2_combined_df[(avg2_combined_df['SEASON_ID'] == '22008') | (avg2_combined_df['SEASON_ID'] == '22012') | (avg2_combined_df['SEASON_ID'] == '22017')]
train = avg2_combined_df[(avg2_combined_df['SEASON_ID'] != '22008') & (avg2_combined_df['SEASON_ID'] != '22012') & (avg2_combined_df['SEASON_ID'] != '22017')]
train = train.drop('SEASON_ID', axis = 1)
test = test.drop('SEASON_ID', axis = 1)




train.to_csv('train1.csv',index=False)
test.to_csv('test1.csv',index=False)