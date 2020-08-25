clean_df = pd.read_csv('clean_df.csv')
odds_19 = pd.read_excel('data/nba_odds_2019.xlsx')
odds_19.Date.value_counts()


odds_df = pd.DataFrame(columns=['SEASON_ID','GAME_ID','MATCHUP','GAME_DATE','TEAM_ID','Date','VH', 'Team', 'Final','Open', 'Close', 'ML'])
for i in [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
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
odds_df2.to_csv('odds_df2w19.csv')