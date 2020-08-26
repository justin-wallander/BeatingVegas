from nba_api.stats.static import teams
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime

#getting NBA teams
nba_teams = teams.get_teams()

#getting Team Abberviations
team_abbrev_list = sorted([team['abbreviation'] for team in nba_teams])
#SEA, NOH, NJN, NOK are odd team names I deal with later by mapping over them

#getting Team IDs
team_id_list = [team['id'] for team in nba_teams]


# importing the leaguegamefinder in order to get all the games for all the teams/seasons
from nba_api.stats.endpoints import leaguegamefinder

#snagged these columns from an original run, so wouldnt necessarrily have these prior creating a games df to begin with
cols = ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

df = pd.DataFrame(columns=cols)

#creating database of games going back to 2007 - as far back as my odds database goes
for ele in team_id_list:
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=ele, season_type_nullable=SeasonType.regular)
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games[games.SEASON_ID.str[1:].astype(int) >= 2007]
    df = pd.concat([df, games], ignore_index=True)
# having an issue with IND and BOS only having 81 games, in 1 season that is the case but not all- resolved


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



#combining the data frame to get both teams game info on the same row
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

season_id_list =sorted(clean_df['SEASON_ID'].unique())
season_id_list
team_abbrev_list
#I figured out a good cutoff point to get rid of the extra preseaon games
clean_df = pd.DataFrame(columns=combined_df.columns)
for year in season_id_list:
    clean = combined_df[(combined_df['SEASON_ID'] == year) & (combined_df['GAME_DATE'] > datetime.datetime(int(year[1:]), 8, 9))].sort_values('GAME_DATE')
    clean_df = pd.concat([clean_df, clean], ignore_index=True)

#clean_df.to_csv('clean_df.csv',index=False)
#clean_df = pd.read_csv('clean_df.csv')
clean_df.info()

#taking out 2019 for now, can add back later or use as testing
# df_2019 = clean_df[clean_df.SEASON_ID == '22019']
# clean_df = clean_df[clean_df.SEASON_ID != '22019']
# clean_df

#I think this is a good spot to insert the Odds data
#well turns out I probably will be needing the vegas lines so here we go
# took this to another file in order to make this code more readable

'''couple of things to keep in mind here, what information am i going to actually have available to me. 
I arranged the averages in order to predict the next games totals. What about in terms of gambliing information. 
I will have the open line. Line referring to O/U.  I will have the current line. For these purposes, I am going to 
train the model on open and closing line. If training vastly outperfoms testing, I will take out the closing line, 
and use current line as open line when making predictions in real time, potentially. I believe the way I have my data
consturcted, team A is the home team. In this case, I will also add the points line, whether positive for favored or 
neg for not. I have an open and close situation here as well that I will need to test like above.
'''

#read this in, this will only be necessary for the historical data, moving forward I am hoping to implement Odds API
odds_clean = pd.read_csv('odds_cleanw19.csv')
odds_clean.info()

     
#merging the game data and odds data
clean_df3 = clean_df.copy()
clean_df3 = clean_df3.sort_values('GAME_ID')
clean_df3.reset_index(drop = True, inplace=True)

odds_clean['GAME_ID'] = '00' + odds_clean['GAME_ID'].astype(str)
odds_merge = odds_clean[['SEASON_ID', 'GAME_ID', 'GAME_DATE', 'TEAM_A','TEAM_B', 'ML_A', 'ML_B', 'TOTAL_OPEN',
                        'TOTAL_CLOSE', 'PTS_SPR_OPEN', 'PTS_SPR_CLOSE']]

odds_merge.GAME_DATE = pd.to_datetime(odds_merge.GAME_DATE)
clean_df3.GAME_DATE = pd.to_datetime(clean_df3.GAME_DATE)
odds_merge['GAME_ID'] = odds_merge['GAME_ID'].astype(int)
odds_merge.info()
clean_df3.info()
clean_merge = pd.merge(clean_df3, odds_merge, on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
clean_merge.tail()

#i think this clean merge is the winner


#now I need to clean data types, remove columns and create the running average. I will try to do it with 1
#and then turn it into a function to loop through each team and season... function turned out ok, but
#takes for ever in the for loop 
#creating a second to create the totals on the line prior in order to avoid potential info leakage
#since the line I currently have the totals on incorporate the actual score of the game
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

def moving_col_avg(df, year, team, window = 10):
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

    df1[avg_cols] = df1[avg_cols].rolling(window=window).mean()


    return df1



#creating the second database and probably the one I will actually end up using
avg_season = pd.DataFrame(columns= ['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME',
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


for year in season_id_list:
    for team in team_abbrev_list:
        avg_season = pd.concat([avg_season, running_col_avg(clean_merge, year, team)], ignore_index=True)

#creating rolling average w/ window of 10
avg_10 = pd.DataFrame(columns= ['SEASON_ID','TEAM_ID','TEAM_ABBREVIATION','TEAM_NAME',
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

for year in season_id_list:
    for team in team_abbrev_list:
        avg_10 = pd.concat([avg_10, moving_col_avg(clean_merge, year, team)], ignore_index=True)


avg_10 = avg_10.dropna()
avg_10.info()


avg_season[avg_season.SEASON_ID=='22019'][['GAME_DATE','GAME_TOTAL', 'MATCHUP', 'TOTAL_CLOSE', 'PTS_SPR_OPEN', 'PTS']].tail()
#checking_season_count(avg_df, '22014', 'DAL')
#Messed around with some plots, hist, and kde of the PTS for a season
avg_season[(avg_season.SEASON_ID > '22014') & (avg_season.TEAM_ABBREVIATION == 'GSW')]['PTS'].hist()
avg_season[(avg_season.SEASON_ID > '22004')]['PTS'].plot.kde()

plt.scatter(avg_season[avg_season.SEASON_ID=='22018'].GAME_DATE, avg_season[avg_season.SEASON_ID=='22018'].GAME_TOTAL)
plt.scatter(avg_season[avg_season.SEASON_ID=='22018'].GAME_DATE, avg_season[avg_season.SEASON_ID=='22018'].TOTAL_CLOSE)
plt.show()

#avg_season.to_csv('avg_season.csv', index= False)
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

avg_season.sort_values('GAME_ID', inplace= True)
avg_10.sort_values('GAME_ID', inplace= True)


shift_season_df = pd.DataFrame(columns=avg_season.columns)
for year in season_id_list:
    for team in team_abbrev_list:
        t_df = pd.DataFrame(columns=avg_season.columns)
        t_df[same_list] = avg_season[(avg_season.SEASON_ID == year) & (avg_season.TEAM_ABBREVIATION == team)][same_list]
        t_df[change_list] = avg_season[(avg_season.SEASON_ID == year) & (avg_season.TEAM_ABBREVIATION == team)][change_list].shift(periods=1)
        t_df.dropna(inplace=True)
        shift_season_df = pd.concat([shift_season_df, t_df], ignore_index=True)


shift_10_df = pd.DataFrame(columns=avg_10.columns)
for year in season_id_list:
    for team in team_abbrev_list:
        t_df = pd.DataFrame(columns=avg_10.columns)
        t_df[same_list] = avg_10[(avg_10.SEASON_ID == year) & (avg_10.TEAM_ABBREVIATION == team)][same_list]
        t_df[change_list] = avg_10[(avg_10.SEASON_ID == year) & (avg_10.TEAM_ABBREVIATION == team)][change_list].shift(periods=1)
        t_df.dropna(inplace=True)
        shift_10_df = pd.concat([shift_10_df, t_df], ignore_index=True)
        
        
shift_season_df.to_csv('shift_season.csv', index= False)
shift_10_df.to_csv('shift_10.csv', index= False)








#combine the data base again, this time keeping only home teams
avg_season_comb_df = combine_team_games(shift_season_df)
# avg2_comb_cols= avg_season_comb_df.columns
avg_season_comb_df = avg_season_comb_df.sort_values('GAME_ID')
avg_season_comb_df.reset_index(drop = True, inplace=True)
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

avg_season_comb_df = avg_season_comb_df[avg2_combined_cols]
avg_season_comb_df.columns = ['GAME_TOTAL','SEASON_ID','GAME_DATE','TEAM_A', 'ML_A',
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
avg_season_comb_df.info()
avg_season_comb_df.GAME_TOTAL = avg_season_comb_df.GAME_TOTAL.astype(int)
#now i need to split this up the dummy vaiables, to ordinal, and then create train and test csvs
avg_season_comb_df['GAME_DATE'] = avg_season_comb_df['GAME_DATE'].apply(lambda x: x.toordinal())

avg_season_comb_df= pd.concat([avg_season_comb_df, pd.get_dummies(avg_season_comb_df['TEAM_A'],prefix='TEAM_A', drop_first=True)], axis = 1)
avg_season_comb_df= pd.concat([avg_season_comb_df, pd.get_dummies(avg_season_comb_df['TEAM_B'],prefix='TEAM_B', drop_first=True)], axis = 1)
avg_season_comb_df= avg_season_comb_df.drop(['TEAM_A', 'TEAM_B'], axis = 1)



test_season = avg_season_comb_df[(avg_season_comb_df['SEASON_ID'] == 22017) | (avg_season_comb_df['SEASON_ID'] == 22018) | (avg_season_comb_df['SEASON_ID'] == 22019)]
train_season = avg_season_comb_df[(avg_season_comb_df['SEASON_ID'] != 22017) & (avg_season_comb_df['SEASON_ID'] != 22018) & (avg_season_comb_df['SEASON_ID'] != 22019)]
train_season = train_season.drop('SEASON_ID', axis = 1)
test_season = test_season.drop('SEASON_ID', axis = 1)

train_season.info()
test_season.info()




train_season.to_csv('train_season.csv',index=False)
test_season.to_csv('test_season.csv',index=False)

 #combine the data base again, this time keeping only home teams
avg_10_comb_df = combine_team_games(shift_10_df)
# avg2_comb_cols= avg_10_comb_df.columns
avg_10_comb_df = avg_10_comb_df.sort_values('GAME_ID')
avg_10_comb_df.reset_index(drop = True, inplace=True)
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

avg_10_comb_df = avg_10_comb_df[avg2_combined_cols]
avg_10_comb_df.columns = ['GAME_TOTAL','SEASON_ID','GAME_DATE','TEAM_A', 'ML_A',
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

avg_10_comb_df.GAME_TOTAL = avg_10_comb_df.GAME_TOTAL.astype(int)
#now i need to split this up the dummy vaiables, to ordinal, and then create train and test csvs
avg_10_comb_df['GAME_DATE'] = avg_10_comb_df['GAME_DATE'].apply(lambda x: x.toordinal())

avg_10_comb_df= pd.concat([avg_10_comb_df, pd.get_dummies(avg_10_comb_df['TEAM_A'],prefix='TEAM_A', drop_first=True)], axis = 1)
avg_10_comb_df= pd.concat([avg_10_comb_df, pd.get_dummies(avg_10_comb_df['TEAM_B'],prefix='TEAM_B', drop_first=True)], axis = 1)
avg_10_comb_df= avg_10_comb_df.drop(['TEAM_A', 'TEAM_B'], axis = 1)



test_10 = avg_10_comb_df[(avg_10_comb_df['SEASON_ID'] == 22017) | (avg_10_comb_df['SEASON_ID'] == 22018) | (avg_10_comb_df['SEASON_ID'] == 22019)]
train_10 = avg_10_comb_df[(avg_10_comb_df['SEASON_ID'] != 22017) & (avg_10_comb_df['SEASON_ID'] != 22018) & (avg_10_comb_df['SEASON_ID'] != 22019)]
train_10 = train_10.drop('SEASON_ID', axis = 1)
test_10 = test_10.drop('SEASON_ID', axis = 1)

train_10.info()
test_10.info()




train_10.to_csv('train_10.csv',index=False)
test_10.to_csv('test_10.csv',index=False)

























