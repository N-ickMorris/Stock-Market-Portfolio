# This file is a simple wrapper to execute code provided entirely by Brian Dew
# the original code is here: https://github.com/bdecon/econ_data/blob/master/APIs/FRED.ipynb

# data management modules
import requests
import pandas as pd
import numpy as np
import re

# a function for collecting average hourly earnings data from the Economic Research Division of the Federal Reserve Bank of St. Louis
def run(fred_key, start_date = "1980-01-01"):
    
    # set the base of the api url
    base = "https://api.stlouisfed.org/fred/series/observations?series_id="
    
    # List of FRED series IDs and their description
    s_dict = {"CES3000000008": "Manufacturing AHE, SA", 
              "CES1000000008": "Mining and Logging AHE, SA",
              "CES4000000008": "Trade, Transportation, and Utilities AHE, SA",
              "CES2000000008": "Construction AHE, SA",
              "CES5000000008": "Information AHE, SA",
              "CES5500000008": "Financial Activities AHE, SA",
              "CES6000000008": "Professional and Business Services AHE, SA",
              "CES6500000008": "Education and Health Services AHE, SA",
              "CES7000000008": "Leisure and Hospitality AHE, SA",
              "AHETPI": "Total Private AHE, SA",
              }
    
    # initilize the start date and api key
    dates = "&observation_start={}".format(start_date)
    api_key = "&api_key={}".format(fred_key)
    ftype = "&file_type=json"
    
    # set up an object to store results
    df = pd.DataFrame()
    
    # collect the data
    for code, name in s_dict.items():
        url = "{}{}{}{}{}".format(base, code, dates, api_key, ftype)
        r = requests.get(url).json()["observations"]
        df[name] = [i["value"] for i in r]
    df.index = pd.to_datetime([i["date"] for i in r])
    df = df.reset_index()
    
    # update column names
    gsub = np.vectorize(re.sub)
    df.columns = gsub(" ", "_", gsub(",", "", gsub(", SA", "", df.columns)))
    df = df.rename(columns = {"index": "datetime"})
    
    # export results
    return df