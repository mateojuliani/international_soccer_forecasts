import soccer_functions as sf
import pandas as pd
import joblib

def win_probability_inference():

    #read scraped incoming matches
    current_fixtures = pd.read_csv("../international_soccer_forecasts/Data/upcoming_matches.csv")

    #Clean up the raw scraped data
    current_fixtures['date'] = current_fixtures["date"].apply(sf.convert_current_date)
    current_fixtures[["elo_1", "elo_2"]] = current_fixtures["elo"].apply(lambda x: sf.split_space(x))
    current_fixtures["elo_diff_1"] = current_fixtures["elo_1"].apply(int) - current_fixtures["elo_2"].apply(int)
    current_fixtures["elo_diff_2"] = -current_fixtures["elo_diff_1"]
    current_fixtures = current_fixtures.drop(["rank", "elo", "win_pct"], axis = 1)

    #Import the two sklearn models
    gb_model = joblib.load('../soccer_forecasts/Models/sklearn_gradient_boosted_model.pkl')
    lin_reg_model = joblib.load('../soccer_forecasts/Models/linreg_model.pkl')


    #get the correct inputs for the xG Models
    team_1_xg_df = current_fixtures[["elo_1", "elo_2", "elo_diff_1"]].rename(columns = {

            "elo_1":"elo",
            "elo_2":"opp_elo",
            "elo_diff_1":"elo_diff",
        })


    team_2_xg_df = current_fixtures[["elo_2", "elo_1", "elo_diff_2"]].rename(columns = {

            "elo_2":"elo",
            "elo_1":"opp_elo",
            "elo_diff_2":"elo_diff",
        })


    #forecast the xgs for each team based on the two different models
    current_fixtures["gb_xg_1"] = gb_model.predict(team_1_xg_df)
    current_fixtures["gb_xg_2"] = gb_model.predict(team_2_xg_df)

    current_fixtures["lr_xg_1"] = lin_reg_model.predict(team_1_xg_df)
    current_fixtures["lr_xg_2"] = lin_reg_model.predict(team_2_xg_df)


    #get ELO probabilities
    elo_historical_average = pd.read_csv('../international_soccer_forecasts/Models/elo_win_rates.csv')
    current_fixtures["elo_probs"] = current_fixtures.apply(lambda row: sf.get_historical_elo(row['elo_diff_1'], elo_historical_average), axis = 1)

    #Get implied odds by using either the lr or gb model to forecast xG, and then simulate 500 games using poisson simulations
    current_fixtures['lr_probs'] = current_fixtures.apply(lambda row: sf.possion_sim(row['lr_xg_1'], row['lr_xg_2'], 500), axis=1)
    current_fixtures['gb_probs'] = current_fixtures.apply(lambda row: sf.possion_sim(row['gb_xg_1'], row['gb_xg_2'], 500), axis=1)

    #Since the Gradient boosted regression preforms worse than the linear regression, the ensemble predictions will be just based on a 50 / 50 split of the elo forecast and linear regression based poisson simulation

    #In this function we take the average of the T1 / D / T2 probabilities from the linear regression model and the ELO model
    current_fixtures["ensemble_preds"] = current_fixtures.apply(lambda row: sf.calculate_average(row['lr_probs'], row['elo_probs']), axis=1)

    #we rebased the aveage so that all probabilities sum to 1
    current_fixtures["ensemble_preds"] = current_fixtures["ensemble_preds"].apply(sf.rebase_estimate)


    #Parse out the probabilities into their own seperate columns
    current_fixtures[['team_1_w', 'draw', 'team_2_w']] = pd.DataFrame(current_fixtures['ensemble_preds'].tolist(), index=current_fixtures.index)

    #clean up table to get final results
    to_publish = current_fixtures[["date", "location", "team_1", "team_2", 'team_1_w', 'draw', 'team_2_w']]
    to_publish["game"] = current_fixtures["team_1"] + " vs " + current_fixtures["team_2"]
    to_publish["team_1_prob"] = to_publish["team_1_w"].apply(sf.odds_round)
    to_publish["draw_prob"] = to_publish["draw"].apply(sf.odds_round)
    to_publish["team_2_prob"] = to_publish["team_2_w"].apply(sf.odds_round)
    to_publish = to_publish[["date", "location", "game", "team_1_prob", "draw_prob", "team_2_prob"]]

    to_publish.to_csv('../international_soccer_forecasts/Data/current_odds.csv', index=False) #save current version



