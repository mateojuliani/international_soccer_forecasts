import soccer_functions as sf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import math

def get_df_stacked():
    #simple function to return stacked df 
    return pd.read_csv(f"../international_soccer_forecasts/Data/cleaned_matches_stacked.csv")

def train_sk_models():

    """
    Train_sk_models: Create Linear and Gradient Boosted Regressions
    The goal of these two models will be to estimate the number of goals a team will score using the following variables:
    * The team's ELO rating
    * The opponent's ELO rating
    * The ELO difference between the two
    These models will then be used to forecast the number of goals two teams will score against each other. Then, we will run a poisson simulation to estimate the final score of X games, and calculate a teams win / draw / loss record based on those calculations

    At the end, we save down the models in the Model/ folder
    """


    df_stacked = get_df_stacked()


    #here we create a train test split based on game id, so the model does not have the same game in training and validation 
    splitter = GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 4231)

    split = splitter.split(df_stacked, groups=df_stacked['game_id'])
    train_id, test_id = next(split)

    train = df_stacked.iloc[train_id]
    test = df_stacked.iloc[test_id]

    #Currently model uses a teams elo, their opposition elo, and the elo difference 
    x_train = train[["elo", "opp_elo", "elo_diff"]]
    y_train = (train["score"])

    x_test = test[[ "elo", "opp_elo", "elo_diff"]]
    y_test = (test["score"])


    #initialize and fit the models
    gb = GradientBoostingRegressor(max_depth = 10, n_estimators = 1000, random_state=0).fit(x_train, y_train)
    linreg = LinearRegression().fit(x_train, y_train)


    #Get our validation set predictions
    y_pred_gb = gb.predict(x_test)
    y_pred_linreg = linreg.predict(x_test)

    #Calculated RMSE
    gb_rmse = math.sqrt(mean_squared_error(y_test, y_pred_gb))
    linreg_rmse = math.sqrt(mean_squared_error(y_test, y_pred_linreg))

    # print(
    # f"""
    # Gradient Boosted Regression RMSE:{gb_rmse}
    # Linear Regression RMSE:{linreg_rmse}
    # """)

    #save down models
    joblib.dump(gb, '../international_soccer_forecasts/Models/sklearn_gradient_boosted_model.pkl')
    joblib.dump(linreg, '../international_soccer_forecasts/Models/linreg_model.pkl')


def create_base_elo_dataframe(start: int = -1250, end: int = 1250, interval: int = 50):

    """
    create cross joined data frame of the all elo differences -1200 to -1200 in intervals of 50
    and for each class (T1, D, T2). Note: T1 is a team_1 win.
    Return: cross joined dataset

    """

    if end < start:
        raise Exception("end must be <= than start")

    elo = []
    result = []
    for x in range(-1200, 1250, 50):
        for y in ["T1", "D", "T2"]:
            elo.append(x)
            result.append(y)

    df_base = pd.DataFrame()
    df_base["rounded_elo"] = elo
    df_base["result_class"] = result

    return df_base

def calculate_historical_elo_win_rate_model():


    """
    Create ELO Average Win Rate Model
    The goal of this model is to look at the historical win rate of a team given its elo difference vs its opponent. We will then use the historical win rate for each ELO difference as our forecast

    Saved down the historical elo win rates in the Models folder. 
    """
    df_base = create_base_elo_dataframe()
    df_stacked = get_df_stacked()

    df_stacked["rounded_elo"] = df_stacked["elo_diff"].apply(sf.apply_elo_50_rounding)
    df_stacked["count"] = 1

    counts_by_elo = df_stacked.groupby(by = ["rounded_elo"])["count"].sum().reset_index()
    counts_by_elo.columns = ["rounded_elo", "total_c"]
    results_elo = df_stacked.groupby(by = ["rounded_elo", "result_class"])["count"].sum().reset_index()

    joined = df_base.merge(results_elo, on = ["rounded_elo", "result_class"], how = "left").merge(counts_by_elo, on = ["rounded_elo"], how = "left").fillna(0)
    joined["pct"] = joined["count"] / joined["total_c"]
    joined["result_class"] = pd.Categorical(joined["result_class"], categories = ["T1", "D", "T2"], ordered = True)
    joined.to_csv('../international_soccer_forecasts/Models/elo_win_rates.csv', index=False)