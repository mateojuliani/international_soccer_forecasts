import soccer_functions as sf
import pandas as pd

def clean_scraped_historical_data(start_year: int = 2016, end_year: int = 2024):

    """
    In this function, we ingest in the scraped data and create two dataframes:

    1) cleaned_matches.csv: The start_year - end_year matches cleaned and joined together in one dataframe
    2) cleaned_matches_stacked.csv: The cleaned_matches.csv dataframe with two rows per match with the format team and opponent. This table is created to simplify the training process. 

    At the end of the cleaning, we save down the csv files in the Data folder
    """

    if end_year < start_year:
        raise Exception("start_year must be <= than end_year")

    #ingest raw data
    years = [x for x in range(start_year, end_year + 1)]

    new_table = True
    df_joined_fin = None
    df_clean_fin = None

    #Look through each of the years
    for year in years:
        print(year)

        #Get data from year
        df_raw = pd.read_csv(f"../international_soccer_forecasts/Data/{year}_matches.csv")

        #clean the scraped data using the clean_raw_scraped_df function
        df_clean = sf.clean_raw_scraped_df(df_raw)

        #transform data into each team having a single row. This will make it easier to keep track of an individuals team performance
        #by being able to filter on df.team == "Afghanistan" for example, which will return all of Afghanistan's (home and away) matches
        df_merged = sf.create_one_col_df(df_clean)
        df_stacked = sf.get_opp_stats(df_merged)

        #Concat the different years of each table
        if new_table:
            
            df_clean_fin = df_clean
            df_stacked_fin = df_stacked
            new_table = False
        else:
            
            df_clean_fin = pd.concat([df_clean_fin, df_clean])
            df_stacked_fin = pd.concat([df_stacked_fin, df_stacked])

    #Save down the two tables
    df_clean_fin.to_csv(f'../international_soccer_forecasts/Data/cleaned_matches.csv', index=False)
    df_stacked_fin.to_csv(f'../international_soccer_forecasts/Data/cleaned_matches_stacked.csv', index=False)