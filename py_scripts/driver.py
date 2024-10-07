import scrape_games 
import clean_data
import training
import inference
import pandas as pd
import os 



# directory_path = '../international_soccer_forecasts/Data/'

# if os.path.isdir(directory_path):
#     print("The directory exists.")
# else:
#     print("The directory does not exist.")


# #scrape any historical data we are missing
# scrape_games.scrape_historical_games(2024, 2024)

# #clean and save down historical data we have scraped
# clean_data.clean_scraped_historical_data()

# #create / train / save down models that will be used for inference
# training.train_sk_models()
# training.calculate_historical_elo_win_rate_model()

# #scrape upcoming games that we will do inference on 
# scrape_games.scrape_latest_games()

#get win probabilities for all upcoming games and save down
inference.win_probability_inference()

