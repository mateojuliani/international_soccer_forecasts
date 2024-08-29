# international_soccer_forecasts
WIP - Workflow for scraping data and making forecasts for future international soccer games


## Folders

### py_script
#### Python scripts for running scraper, training model, and performing inference on upcoming games
1. driver - run entire workflow - scrape historical games, clean data, create models, scrape upcoming games, and forcast win probabilities 
2. scrape_games - functions to scrape historical and upcoming games from elorating. TWe will use this data to create and train our models
3. clean_data - cleans historical scraped data, adds extra columns, and transforms format to make it friendly for training
5. training - creates/trains 3 models based on historical data
6. inference - uses models from 03_create_models and data scraped from 01_scrape_future_games to forecast win / draw / loss probabilities of upcoming international soccer games

### Data
Folder to save scraped / cleaned data. Contains scraped data from elorating

### Models
Folder containing models created from functions in training




