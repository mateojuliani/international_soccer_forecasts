from datetime import datetime
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
import math
import joblib

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#TODO: Document these functions


def convert_to_date(date_str): #converts date string to datetime format
    try: date_obj = datetime.strptime(date_str, "%B %d %Y")
    except: 
        try: date_obj = datetime.strptime(date_str, "%b %d %Y")
        except: 
            try: date_obj = datetime.strptime(date_str, "%b %Y").replace(day=1)
            except: date_obj = datetime(1990, 1, 1) #fin error catch


    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date

def convert_current_date(date_str):
    
    try: return datetime.strptime(date_str, "%a %b %d").replace(year=datetime.now().year).strftime("%Y-%m-%d") 
    except: return convert_to_date(date_str)
    

def split_space(str): #splits string based on ' ' character
    num1, num2 = str.split(" ")[0].strip(), str.split(" ")[1].strip()
    return pd.Series([num1, num2])

def convert_string_to_int(str): #the scraper picks up non-ASCII characters, so we replace them
    return int(str.replace('−', '-')) 

def result_list(score_1, score_2):
    
    if score_1 > score_2:
        return [1,0,0]
    elif score_1 == score_2:
        return [0,1,0]
    elif score_1 < score_2:
        return [0,0,1]
    else:
        return "ERROR"

def result_classification(score_1, score_2):
    
    if score_1 > score_2:
        return "T1"
    elif score_1 == score_2:
        return "D"
    elif score_1 < score_2:
        return "T2"
    else:
        return "ERROR"


def clean_raw_scraped_df(df_raw): #cleans the base df that we scraped from elorankings.com

    df_clean = pd.DataFrame()
    df_clean["date"] = df_raw["date"].apply(convert_to_date)
    df_clean[["team_1", "team_2"]] = df_raw[["team_1", "team_2"]]
    df_clean[["score_1", "score_2"]] = df_raw["score"].apply(lambda x: split_space(x))
    df_clean["location"] = df_raw["location"]
    df_clean[["elo_change_1", "elo_change_2"]] = df_raw["change_1"].apply(lambda x: split_space(x))
    df_clean[["new_elo_1", "new_elo_2"]] = df_raw["score_1_points"].apply(lambda x: split_space(x))
    df_clean[["rank_change_1", "rank_change_2"]] = df_raw["change_2"].apply(lambda x: split_space(x))
    df_clean[["new_rank_1", "new_rank_2"]] = df_raw["score_2_points"].apply(lambda x: split_space(x))

    df_clean["old_elo_1"] = df_clean["new_elo_1"].apply(int) - (df_clean["elo_change_1"].apply(convert_string_to_int))
    df_clean["old_elo_2"] = df_clean["new_elo_2"].apply(int) - (df_clean["elo_change_2"].apply(convert_string_to_int))

    df_clean["old_rank_1"] = df_clean["new_rank_1"].apply(int) - (df_clean["rank_change_1"].apply(convert_string_to_int))
    df_clean["old_rank_2"] = df_clean["new_rank_2"].apply(int) - (df_clean["rank_change_2"].apply(convert_string_to_int))

    df_clean["elo_diff_1"] = df_clean["old_elo_1"] - df_clean["old_elo_2"]
    df_clean["elo_diff_2"] = -1*df_clean["elo_diff_1"]

    df_clean["result"] = df_clean.apply(lambda row: result_list(row['score_1'], row['score_2']), axis=1)
    df_clean["result_class"] = df_clean.apply(lambda row: result_classification(row['score_1'], row['score_2']), axis=1)

    df_clean["game_id"] = df_clean["date"].apply(str) + df_clean["team_1"] + df_clean["team_2"]

    return df_clean


def create_one_col_df(df): #transform df to have one team per column

    l_1 = []
    l_2 = []

    for x in df.columns:

        if ("2" in x) & ("team" not in x):
            l_2.append(x)
        elif ("1" in x) & ("team" not in x):
            l_1.append(x)
        else:
            l_1.append(x)
            l_2.append(x)


    df_1 = df[l_1]
    df_2 = df[l_2]

    for y in df_2:
        if "_2" in y:
            df_2 = df_2.rename(columns={y: y.replace("_2", "")})

    df_2 = df_2.rename(columns={"team_1": "opp"})


    for z in df_1:
        if "_1" in z:
            df_1 = df_1.rename(columns={z: z.replace("_1", "")})

    df_1 = df_1.rename(columns={"team_2": "opp"})

    df_fin = pd.concat([df_1, df_2]).sort_values(by = "game_id").reset_index()[
        ["game_id", "date", "location", "team", "opp", "score", "old_elo", "elo_change", "new_elo", "old_rank", "rank_change", "new_rank", "elo_diff"]
    ]

    return df_fin


def get_opp_stats(df): #Used to get data on the opponent

    df_opp = df[["game_id", "team", "score", "old_elo", "old_rank"]].rename(columns = {

        "team":"opp",
        "score":"opp_score",
        "old_elo":"opp_elo",
        "old_rank":"opp_rank",
    }).set_index(['game_id', 'opp'])


    df_joined = df.join(df_opp, on = ["game_id", "opp"], how = "left")

    df_joined["result_class"] = df_joined.apply(lambda row: result_classification(row['score'], row['opp_score']), axis=1)

    df_joined = df_joined[
                        ["game_id", 
                         "date", 
                         "location", 
                         "team", 
                         "opp", 
                         "score", 
                         "opp_score", 
                         "old_elo", 
                         "opp_elo", 
                         "elo_diff", 
                         "result_class"]].rename(columns = {

                            "old_elo":"elo"
                        })
    
    return df_joined


def apply_elo_50_rounding(elo):
    return elo - (elo % 50)



def get_historical_elo(elo_diff, joined):

    elo_rounded = apply_elo_50_rounding(int(elo_diff))
    return joined[joined["rounded_elo"] == elo_rounded].sort_values("result_class")["pct"].to_list()



#Given two xgs, simulates scores of each side using a poisson distribution and then sees which side won / loss
def possion_sim(xg, opp_xg, num):
    
    win  = 0
    draw = 0
    loss = 0
    
    if xg <= 0:
        xg = 0.1
    
    if opp_xg <= 0:
        opp_xg = 0.1
    
    for x in range(0, num):
        team_goals = np.random.poisson(xg)
        opp_goals = np.random.poisson(opp_xg)
        
        if team_goals > opp_goals:
            win += 1
        elif team_goals == opp_goals:
            draw+=1
        else:
            loss+=1
    
    return [win / num, draw / num, loss / num]

#ensemble functions
def calculate_average(l1, l2, weights = [0.1, 0.9] ):

    arr = np.array([l1, l2])
    return  np.average(arr, axis=0, weights=weights).tolist()


def rebase_estimate(arr):
    val = 0
    for x in arr:
        val += float(x)

    return [float(x) / val for x in arr]

def odds_round(odds):
    return round(odds, 3)


#https://en.wikipedia.org/wiki/Brier_score
def single_brier_score(list_pred, list_outcome):
    
    cum = 0
    for x in range(0, len(list_pred)):
        cum+= (list_pred[x] - list_outcome[x])**2
    
    return cum
