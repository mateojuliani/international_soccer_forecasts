
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import joblib
import math

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from pathlib import Path


from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pprint import pprint
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go

import json

import joblib

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def model_estimate_odds(elo_diff, model, home=None, away=None, odds = False):


    if home is not None:
        preds = model.predict_proba(np.array([elo_diff, home, away]).reshape(1, -1))
    else:
        preds = model.predict_proba(np.array([elo_diff]).reshape(1, -1))


    #in format D / L / W
    if odds:
        return [1/preds[0][0], 1/preds[0][1], 1/preds[0][2]]
    else:
        return [preds[0][0], preds[0][1], preds[0][2]]
    

def single_brier_score(list_pred, list_outcome):
    """
    Calculates brier score from a list of predictions
    """
    
    cum_score = 0
    for x in range(0, len(list_pred)):
        cum_score+= (list_pred[x] - list_outcome[x])**2
    
    return cum_score 

def result_list(score_1, score_2):
    """
    Given two scores, returns classification of result in array format - used for brier score calcs
    Note list format is Draw - Away - Home
    """
    
    if score_1 > score_2:
        return [0,0,1]
    elif score_1 == score_2:
        return [1,0,0]
    elif score_1 < score_2:
        return [0,1,0]
    else:
        return "ERROR"
    
def result_classification(score_1, score_2):
    """
    Returns win, draw, loss based on scores
    Used for data cleaning
    """
    
    if score_1 > score_2:
        return "W"
    elif score_1 == score_2:
        return "D"
    elif score_1 < score_2:
        return "L"
    else:
        return "ERROR"


def to_list_preds(pred1, predtie, pred2):
    return [predtie, pred2, pred1]



#get nate silver preds
df_nate_silver = pd.read_csv("Data/spi_matches_intl.csv")
df_nate_clean = df_nate_silver[df_nate_silver["score1"].notna()]
df_nate_clean["probs"] = df_nate_clean.apply(lambda row: to_list_preds(row['prob1'], row['probtie'], row['prob2']), axis = 1)
df_nate_clean["result"] = df_nate_clean.apply(lambda row: result_list(row['score1'], row['score2']), axis = 1)
df_nate_clean["brier_single"] = df_nate_clean.apply(lambda row: single_brier_score(row['probs'], row['result']), axis=1)
#print(df_nate_clean)

#Our preds

df_base = pd.read_csv(f"../international_soccer_forecasts/Data/cleaned_matches_stacked.csv")
df = df_base.drop_duplicates(subset=['game_id'])
df["game_results"] = df.apply(lambda row: result_classification(row['score'], row['opp_score']), axis = 1)


df_train = df.loc[df.index < len(df) - 1501]
df_test = df.loc[df.index > len(df) - 1501]

print(df_test)

df_home = df_train[["elo_diff", "team_home", "opp_home", "game_results"]]
model_ft = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_ft.fit(df_home[["elo_diff", "team_home", "opp_home"]], df_home["game_results"])

model_ex_home_ft = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_ex_home_ft.fit(df_home[["elo_diff"]], df_home["game_results"])

df_test["final_results"] = df_test.apply(lambda row: result_list(row['score'], row['opp_score']), axis = 1)
df_test["model_probs"] = df_test.apply(lambda row: model_estimate_odds(row['elo_diff'], model_ft, row["team_home"], row["opp_home"]), axis = 1)
df_test["model_brier"] = df_test.apply(lambda row: single_brier_score(row['model_probs'], row['final_results']), axis=1)

df_test["model_ex_home_probs"] = df_test.apply(lambda row: model_estimate_odds(row['elo_diff'], model_ex_home_ft), axis = 1)
df_test["model_ex_home_brier"] = df_test.apply(lambda row: single_brier_score(row['model_ex_home_probs'], row['final_results']), axis=1)

df_test[['D_pred', 'L_pred', 'W_pred']] = pd.DataFrame(df_test['model_probs'].tolist(), index=df_test.index)
df_test[['D_act', 'L_act', 'W_act']] = pd.DataFrame(df_test['final_results'].tolist(), index=df_test.index)


w_true, w_pred = calibration_curve(np.array(df_test["W_act"].to_list()), np.array(df_test["W_pred"].to_list()), n_bins=20)
d_true, d_pred = calibration_curve(np.array(df_test["D_act"].to_list()), np.array(df_test["D_pred"].to_list()), n_bins=20)
l_true, l_pred = calibration_curve(np.array(df_test["L_act"].to_list()), np.array(df_test["L_pred"].to_list()), n_bins=20)

print(f"Model Brier {df_test["model_brier"].mean()}")
print(f"Model Ex Home Adv Brier {df_test["model_ex_home_brier"].mean()}")

# Create the plot
merged_df = df_nate_clean.merge(df_test, how='inner', left_on=['date', "team1", "team2"], right_on=['date', 'team', "opp"])
print(f"Model Brier {merged_df["model_brier"].mean()} vs Nate Brier {merged_df["brier_single"].mean()}")
merged_df.to_csv("Data/nate_silver_join_test.csv")

probs = True
if probs:

    fig = go.Figure()

    # Add the perfectly calibrated line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='Ideally Calibrated'
        )
    )

    # Add the model's calibration curve
    fig.add_trace(
        go.Scatter(
            x=w_pred,
            y=w_true,
            mode='markers+lines',
            marker=dict(size=8),
            name='Win'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d_pred,
            y=d_true,
            mode='markers+lines',
            marker=dict(size=8),
            name='Draw'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=l_pred,
            y=l_true,
            mode='markers+lines',
            marker=dict(size=8),
            name='Loss'
        )
    )

        # Update layout
    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Average Predicted Probability in each bin',
        yaxis_title='Ratio of positives',
        legend=dict(x=0.01, y=0.99)
    )

    # Show the plot
    fig.show()

