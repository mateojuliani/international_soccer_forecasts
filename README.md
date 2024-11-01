# international_soccer_forecasts
WIP - Workflow for scraping data and making forecasts for future international soccer games


## Folders

### py_script
To get forecast, run driver.py. In order, the following functions will run

1. scrape_historical_games - scrape historical games from elorating. We will use this data to create and train our models
2. clean_scraped_historical_data - cleans scraped historical data, adds extra columns, and transforms format to make it friendly for training
3. train_sk_models - Trains regression / tree based models to predict the number of goals each team will score.
4. calculate_historical_elo_win_rate_model - Calculates average win / draw / loss rate based on ELO difference between teams. Inspired by [this video](https://www.youtube.com/watch?v=KjISuZ5o06Q)
5. scrape_latest_games - Scrape all the upcoming games
6. win_probability_inference - Use upcoming games and trained models to create forecasts and ensemble them. Final output is current_odds.csv

### Data
Contains scraped data from elorating, old nate silver forecasts, and other misc data sources pulled from different websites

### Models
Folder containing models created in 03_create_models

If you've made it here, would love to chat! Feel free to reach out on any of my socials!
