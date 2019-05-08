import requests
import re
import yaml
import pandas as pd
import numpy as np
import math

#------------------------------------------------------------------------------------
# get imputed votes 
def get_imputed_votes(source, state_id):                                                       
     return source[source['state_id']==state_id]['votes'].values[0] 

#------------------------------------------------------------------------------------
# get number of seats in state 
def get_state_seat_count(df, state):                                                       
     return df[(df['state_id']==state) & (df['winner']==True)].winner.size

#------------------------------------------------------------------------------------
# get district total vote count 
def get_district_vote_count(df, state_id_seat):
     state_id = state_id_seat[0]
     seat = state_id_seat[1]
     return df[(df['state_id']==state_id) & (df['seat']==seat)]['votes'].sum() 

#------------------------------------------------------------------------------------
# get state total vote count 
def get_state_vote_count(df, state_id):
     return df[df['state_id']==state_id]['votes'].sum() 

#------------------------------------------------------------------------------------
# get district total vote count 
def get_district_candidate_count(df, state_id_seat):
     state_id = state_id_seat[0]
     seat = state_id_seat[1]
     return df[(df['state_id']==state_id) & (df['seat']==seat)]['votes'].size

#------------------------------------------------------------------------------------
# clean candidate results 
# in: candidate results dataframe
# out: original dataframe with the following changes
#      incumbent = NaN changed to False
#      winner = NaN changed to False
#      percent = 0 changed to 100.0
#      percent_display =  0 changed to 100.0
#      votes = 0 changed to imputed value:average votes for districts in state 
# Note: percent_display = votes = 0 means election was uncontested OR unopposed
#
def clean_candidate_results(df):

    df.loc[df['incumbent'].isnull(), 'incumbent'] = False
    df.loc[df['winner'].isnull(), 'winner'] = False
    df.loc[df['percent']==0.0, 'percent'] = 100.0 
    df.loc[df['percent_display']==0.0, 'percent_display'] = 100.0 

    # create fields to facilitate counts
    df['winner2'] = np.where(df['winner']==True,1,0)
    df['uncontested'] = np.where(df['percent']==100,1,0)

    # create field to facilitate grouping of votes into three parties only
    df['party_id2'] = np.where((df['party_id']=='democrat') | (df['party_id']=='republican'),\
                                 df['party_id'],'other')

    # identify missing parties for all seats 
    # we want every state/seat combination to have a row for a republican, a democrat and an "other"
    missing = list()                                                     
    for index, row in df[['state_id','seat']].iterrows():
       state_id = row['state_id']
       seat = row['seat']
       rs = df[(df['state_id']==state_id) & (df['seat']==seat) & (df['party_id2']=='republican')].size
       ds = df[(df['state_id']==state_id) & (df['seat']==seat) & (df['party_id2']=='democrat')].size
       os = df[(df['state_id']==state_id) & (df['seat']==seat) & (df['party_id2']=='other')].size
       if rs == 0: missing.append((state_id, seat, 'republican'))
       if ds == 0: missing.append((state_id, seat, 'democrat'))
       if os == 0: missing.append((state_id, seat, 'other'))
    # create dataset to concat to df with all rows missing from df 
    # fields in dataset
    # 'candidate_id', 'candidate_key', 'first_name', 'incumbent', 'last_name',
    # 'name_display', 'order', 'party_id', 'percent', 'percent_display',
    # 'seat', 'state_id', 'votes', 'winner', 'winner2', 'uncontested',
    # 'party_id2'
    missing_to_add_to_df = list()
    for state_id, seat, party_id2 in missing:
        missing_to_add_to_df.append(('dummy', 'dummy', 'dummy', False, 'dummy',\
                                     'dummy', 0, 'dummy', 0, 0,\
                                     seat, state_id, 0, False, 0, 0, party_id2))
    df_missing_to_add_to_df = pd.DataFrame(missing_to_add_to_df)
    df_missing_to_add_to_df.columns = df.columns
    df = pd.concat([df, df_missing_to_add_to_df], axis=0).copy()
    
    # will impute votes for uncontested OR unopposed winners
    # partition dataframeand then contactenate after imputing values
    # partition 1 = contested districts
    df_contested_districts = df[~((df['votes']==0) & (df['winner']==True))].copy()
    # partition 2 = uncontested OR unopposed districts
    df_uncontested_districts = df[((df['votes']==0) & (df['winner']==True))].copy()

    # calculate avg district votes for contested districts for each state
    df_avg_state_seat_votes = df_contested_districts.groupby(['state_id','seat']).agg({'votes': np.sum})
    df_avg_state_seat_votes.reset_index(inplace=True)
    df_avg_state_seat_votes = df_avg_state_seat_votes.groupby('state_id').agg({'votes': np.mean})
    df_avg_state_seat_votes['votes'] = df_avg_state_seat_votes['votes'].astype(int)
    df_avg_state_seat_votes.reset_index(inplace=True)

    # update zero votes in partition 2 dataframe
    df_uncontested_districts['votes'] = df_uncontested_districts['state_id'].\
                       apply(lambda x: get_imputed_votes(df_avg_state_seat_votes,x))     

    # concatenate back the two partitions
    df2 = pd.concat([df_contested_districts, df_uncontested_districts], axis=0) 
    df2.sort_values(['state_id','seat'], inplace=True)

    # create field to track number of seat count for state 
    df2['state_seat_count'] = df2['state_id'].\
                       apply(lambda x: get_state_seat_count(df2, x))     

    # create field to track total number of votes for district 
    df2['district_vote_count'] = df2[['state_id','seat']].\
                       apply(lambda x: get_district_vote_count(df2, x), axis=1)     

    # create field to track total number of votes for state 
    df2['state_vote_count'] = df2['state_id'].\
                       apply(lambda x: get_state_vote_count(df2, x))

    # create field to track total number of candidates in the district 
    df2['district_candidate_count'] = df2[['state_id','seat']].\
                       apply(lambda x: get_district_candidate_count(df2, x), axis=1)     

    # calculate wasted votes
    # for losers, it's everything 
    # for winners, is number of votes above 50% + 1 of total votes
    df2['wasted_votes'] = (np.where(df2['winner']!=True, df2['votes'],\
         df2['votes'] - (df2['district_vote_count']//2 + 1))).astype(int)

    return df2

#------------------------------------------------------------------------------------
# get_overall_results
# summarizes results at national level 
# returns:
# Party, Votes, Seats_Won, %_Votes, %_Seats_Won, Votes_to_Elect_Candidate, %_Total_Votes, sort_order 
# 
def get_overall_results(df):
    overall = df.groupby('party_id2').agg({'votes': np.sum, 'winner2': np.sum}).reset_index()
    overall['%votes'] = overall['votes'] / overall.votes.sum() * 100
    overall['%winner2'] = overall['winner2'] / overall.winner2.sum() * 100
    overall['votesperseat'] = overall['votes'] / overall['winner2']
    overall['%totalvotesperseat'] = overall['votesperseat'] / overall.votes.sum() * 100
    col_names = ['Party','Votes','Seats_Won','%_Votes','%_Seats_Won',\
                 'Votes_to_Elect_Candidate','%_Total_Votes']
    overall.columns = col_names
    overall['sort_order'] = np.where(overall['Party']=='republican', 1,\
                            np.where(overall['Party']=='democrat', 2, 3))
    overall = overall.sort_values('sort_order', ascending=True)
    return overall 

#------------------------------------------------------------------------------------
# get_results_bystate
# summarizes results at the state level 
# returns:
# 'State', 'Party', 'Votes', 'Seats_Won','State_Seat_Count','State_Vote_Count','%_Votes',
# '%_Winners','Votes_to_Elect_Candidate','%_of_Total'
# 
def get_results_bystate(df):
   bystate = df.groupby(['state_id','party_id2']).agg({'votes': np.sum, 'winner2': np.sum,\
                        'state_seat_count': np.max, 'state_vote_count': np.max}).\
                        reset_index()
   bystate['%votes'] = bystate['votes'] / bystate['state_vote_count'] * 100
   bystate['%winner2'] = bystate['winner2'] / bystate['state_seat_count'] * 100
   bystate['votesperseat'] = bystate['votes'] / bystate['winner2']
   bystate['%statevotesperseat'] = bystate['votesperseat'] / bystate.votes.sum() * 100
   col_names2 = ['State','Party','Votes','Seats_Won','State_Seat_Count','State_Vote_Count',\
                 '%_Votes','%_Winners','Votes_to_Elect_Candidate','%_of_Total']
   bystate.columns = col_names2
   bystate['sort_order'] = np.where(bystate['Party']=='republican', 1, \
                           np.where(bystate['Party']=='democrat', 2, 3))
   bystate = bystate.sort_values(['State','sort_order'], ascending=True)

   return bystate
    
if __name__ == "__main__":

    candidate_results_raw = pd.read_csv('data/candidate_results_raw.csv')
    
    candidate_results_clean = clean_candidate_results(candidate_results_raw)

    candidate_results_clean.to_csv('data/candidate_results_clean.csv', index=False)

