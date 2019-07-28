# This file is a simple wrapper to execute code provided entirely by Brian Dew
# the original code is here: https://github.com/bdecon/econ_data/blob/master/APIs/BLS.ipynb

# data management modules
import pandas as pd
import requests
import json

# a function for collecting unemployement data from the Bureau of Labor Statistics
def run(bls_key, start_year = 2000, end_year = 2019):
    
    # set the base of the api url
    api_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    # set the key
    key = "?registrationkey={}".format(bls_key)
    
    # Series stored as a dictionary
    series_dict = {
        "LNS14000003": "White", 
        "LNS14000006": "Black", 
        "LNS14000009": "Hispanic"}
    
    # Start year and end year
    date_r = (start_year, end_year)
    
    # Handle dates
    dates = [(str(date_r[0]), str(date_r[1]))]
    while int(dates[-1][1]) - int(dates[-1][0]) > 10:
        dates = [(str(date_r[0]), str(date_r[0]+9))]
        d1 = int(dates[-1][0])
        while int(dates[-1][1]) < date_r[1]:
            d1 = d1 + 10
            d2 = min([date_r[1], d1+9])
            dates.append((str(d1),(d2)))
    
    # set up an object to store results
    df = pd.DataFrame()
    
    # collect data
    for start, end in dates:
        # Submit the list of series as data
        data = json.dumps({
            "seriesid": list(series_dict.keys()),
            "startyear": start, "endyear": end})
    
        # Post request for the data
        p = requests.post(
            "{}{}".format(api_url, key), 
            headers={"Content-type": "application/json"}, 
            data=data).json()
        for s in p["Results"]["series"]:
            col = series_dict[s["seriesID"]]
            for r in s["data"]:
                date = pd.to_datetime("{} {}".format(
                    r["periodName"], r["year"]))
                df.at[date, col] = float(r["value"])
    
    # sort the data by time
    df = df.sort_index()
    df = df.reset_index()
    
    # update column names
    df.columns = ["datetime", "White_Unemployement", "Black_Unemployement", "Hispanic_Unemployement"]
    
    # export results
    return df