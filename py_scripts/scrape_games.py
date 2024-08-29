# import libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime
import random


def scrape_historical_games(start_year: int, end_year: int):
    """
    Loops through and scrapes all the international games from the years inputed in the model 
    Saves down each year as its own file in the the Data/ folder
    """

    if end_year < start_year:
        raise Exception("start_year must be <= than end_year")


    years = [x for x in range(start_year, end_year+1)]


    for year in years:

        print(year)
        driver = webdriver.Chrome()
        #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        #driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
        url = f"https://www.eloratings.net/{year}_results"
        driver.get(url)
        driver.implicitly_wait(10)
        
        rows = driver.find_elements(By.CSS_SELECTOR, 'div.ui-widget-content.slick-row')
        
        # Initialize a list to hold the extracted data
        data = []
        
        # Loop through each row and extract the data
        for row in rows:
            cells = row.find_elements(By.CSS_SELECTOR, 'div.slick-cell')
            date = cells[0].get_attribute('innerHTML').replace('<br>', ' ').strip()
        
            teams_html = cells[1].get_attribute('innerHTML')
            teams_soup = BeautifulSoup(teams_html, 'html.parser')
            teams = teams_soup.find_all('a')
            teams_list = [team.get_text() for team in teams]
        
            score = cells[2].get_attribute('innerHTML').replace('<br>', ' ').strip()
        
            location_html = cells[3].get_attribute('innerHTML')
            location_soup = BeautifulSoup(location_html, 'html.parser')
            location = location_soup.get_text(separator=' ').strip()  # Use get_text to extract all text content
        
            change_1 = cells[4].get_attribute('innerHTML').replace('<br>', ' ').strip()
            score_1 = cells[5].get_attribute('innerHTML').replace('<br>', ' ').strip()
            change_2 = cells[6].get_attribute('innerHTML').replace('<br>', ' ').strip()
            score_2 = cells[7].get_attribute('innerHTML').replace('<br>', ' ').strip()
        
            # Create a dictionary for the current row
            match_data = {
                'date': date,
                'team_1': teams_list[0],
                'team_2':  teams_list[1],
                'score': score,
                'location': location,
                'change_1': change_1,
                'score_1_points': score_1,
                'change_2': change_2,
                'score_2_points': score_2
            }
        
            # Append the dictionary to the list
            data.append(match_data)
        
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(f'../international_soccer_forecasts/Data/{year}_matches.csv', index=False)

        #wait inbetween rescrapes 
        time.sleep(30 + 10 * random.random())
        
        
        # Close the WebDriver
        driver.quit()


def scrape_latest_games():
    """
    Function to go and scrape all upcoming games from eloratings.net
    Saves down csv with latest game in Data Folder 
    """


    today = datetime.today().strftime('%Y-%m-%d')

    #url we want to scrape
    url = "https://www.eloratings.net/fixtures"

    driver = webdriver.Chrome()
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    driver.implicitly_wait(10)

    rows = driver.find_elements(By.CSS_SELECTOR, 'div.ui-widget-content.slick-row')
    data = []

    # Loop through each row and extract the data
    for row in rows:
        cells = row.find_elements(By.CSS_SELECTOR, 'div.slick-cell')

        date = cells[0].get_attribute('innerHTML').replace('<br>', ' ').strip()

        teams_html = cells[1].get_attribute('innerHTML')
        teams_soup = BeautifulSoup(teams_html, 'html.parser')
        teams = teams_soup.find_all('a')
        teams_list = [team.get_text() for team in teams]

        location_html = cells[2].get_attribute('innerHTML')
        location_soup = BeautifulSoup(location_html, 'html.parser')
        location = location_soup.get_text(separator=' ').strip()  # Use get_text to extract all text content


        rank = cells[3].get_attribute('innerHTML').replace('<br>', ' ').strip()
        elo = cells[4].get_attribute('innerHTML').replace('<br>', ' ').strip()
        win_pct = cells[5].get_attribute('innerHTML').replace('<br>', ' ').strip()


        match_data = {
            'date': date,
            'team_1': teams_list[0],
            'team_2': teams_list[1],
            'location': location,
            'rank': rank,
            'elo': elo,
            'win_pct': win_pct,
            "scrape_date":today
        }

    # Append the dictionary to the list
        data.append(match_data)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    today = today.replace("-", "_")
    df.to_csv(f'../international_soccer_forecasts/Data/historical_data/upcoming_matches_{today}.csv', index=False) #save historical data
    df.to_csv(f'../international_soccer_forecasts/Data/upcoming_matches.csv', index=False) #save current version

    # Close the WebDriver
    driver.quit()

