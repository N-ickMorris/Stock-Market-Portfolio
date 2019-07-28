# set the path of where the input files are
mywd = "D:\\Data Science\\Stocks"

# data management modules
import os
import pandas as pd
import numpy as np
import gc

# set the work directory
os.chdir(mywd)

# time study modules
from timeit import default_timer
from datetime import datetime
from datetime import timedelta

# finance modules
from yahoofinancials import YahooFinancials
import cpi
cpi.update()
import api

# get your FRED API key here: https://research.stlouisfed.org/useraccount/apikeys
fred_key = api.fred_key.run()

# get your BLS API key here: https://data.bls.gov/registrationEngine/
bls_key = api.bls_key.run()

# miscellaneous modules
import warnings

# should we hide warning messages?
hide_warnings = True

# handle warnings
if hide_warnings:
    warnings.filterwarnings("ignore")
else:
    warnings.filterwarnings("default")

# import the stock rank data
stock_rank = pd.read_csv("stock rank.csv")

# get the symbols as a list
symbols = list(stock_rank["symbol"])

start_time = default_timer()
# get stock financials for each symbol
stock_financials = YahooFinancials(symbols)

# get the daily stock history for each symbol
stock_prices = stock_financials.get_historical_price_data("1900-01-01", "2019-07-26", "daily")

# report progress
duration = default_timer() - start_time
print("Extracted daily stock history for " + str(len(symbols)) + " symbols after " + str(np.round(duration, 2)) + " seconds")

# set up an object to store results
stock_data = pd.DataFrame(columns = ["adjclose", "close", "date", "formatted_date", "high", "low", "open", "volume", "price_range", "diff_log_adjclose", "diff_log_volume", "diff_log_price_range", "symbol", "name", "mad_rank"])

# collect stock data on each symbol
for i in range(len(symbols)):
    try:
        start_time = default_timer()
        # get the symbol
        s = symbols[i]
        
        # get the prices as a data frame
        prices = pd.DataFrame(stock_prices[s]["prices"])
        
        # fill in missing values
        for j in ["adjclose", "close", "high", "low", "open", "volume"]:
            prices[j] = prices[j].fillna(method = "pad").fillna(method = "backfill")
        
        # compute the price range
        prices["price_range"] = prices["high"] - prices["low"]
        
        # compute the diff(log(adjclose)) for calculating percent change and volatility
        prices["diff_log_adjclose"] = np.append(np.nan, np.diff(np.log(prices["adjclose"])))
        
        # compute the diff(log(volume)) for calculating percent change and volatility
        prices["diff_log_volume"] = np.append(np.nan, np.diff(np.log(prices["volume"])))
        
        # compute the diff(log(price_range)) for calculating percent change and volatility
        prices["diff_log_price_range"] = np.append(np.nan, np.diff(np.log(prices["price_range"])))
        
        # remove the first observation
        prices = prices[1:]
        
        # add symbol, name, and mad (mean absolute deviation) rank to prices
        prices["symbol"] = s
        prices["name"] = stock_rank["name"][i]
        prices["mad_rank"] = stock_rank["rank"][i]
        
        # replace all inf values with 0
        prices = prices.replace(np.inf, 0).replace(-np.inf, 0)
        
        # store results
        stock_data = pd.concat([stock_data, prices], axis = 0, sort = False).reset_index(drop = True)
        
        # clean out garbage in RAM
        del s, prices
        gc.collect()
        
        # report progress
        duration = default_timer() - start_time
        print("Collected daily stock history for symbol " + str(i + 1) + " of " + str(len(symbols)) + " after " + str(np.round(duration, 2)) + " seconds")
        
    except:
        # report progress
        gc.collect()
        duration = default_timer() - start_time
        print("Failed to collect daily stock history for symbol " + str(i + 1) + " of " + str(len(symbols)) + " after " + str(np.round(duration, 2)) + " seconds")
        continue

start_time = default_timer()
# define a function for creating date objects from strings
def getdatetime(string, frmt):
    try:
        result = datetime.strptime(string, frmt).date()
    except:
        result = np.nan
    return result
np_getdatetime = np.vectorize(getdatetime)

# define a function for getting the year from date objects
def getyear(datetime_date):
    try:
        result = datetime_date.year
    except:
        result = np.nan
    return result
np_getyear = np.vectorize(getyear)

# define a function for geting the month from date objects
def getmonth(datetime_date):
    try:
        result = datetime_date.month
    except:
        result = np.nan
    return result
np_getmonth = np.vectorize(getmonth)

# define a function for geting the week from date objects
def getweek(datetime_date):
    try:
        result = int(datetime_date.strftime("%V"))
    except:
        result = np.nan
    return result
np_getweek = np.vectorize(getweek)

# define a function for geting the day from date objects
def getday(datetime_date):
    try:
        result = datetime_date.day
    except:
        result = np.nan
    return result
np_getday = np.vectorize(getday)

# define a function for geting the weekday from date objects
def getweekday(datetime_date):
    try:
        result = datetime_date.strftime("%A")
    except:
        result = np.nan
    return result
np_getweekday = np.vectorize(getweekday)

# convert formatted_date into a datetime object
stock_data["datetime"] = np_getdatetime(stock_data["formatted_date"], "%Y-%m-%d")

# get the year, month, quarter, week, day, and weekday from datetime
stock_data["year"] = np_getyear(stock_data["datetime"])
stock_data["month"] = np_getmonth(stock_data["datetime"])
stock_data["year_month"] = stock_data["year"].astype(str) + "_" + stock_data["month"].astype(str)
stock_data["quarter"] = np.ceil(stock_data["month"] / 3).astype(int)
stock_data["year_quarter"] = stock_data["year"].astype(str) + "_" + stock_data["quarter"].astype(str)
stock_data["week"] = np_getweek(stock_data["datetime"])
stock_data["year_week"] = stock_data["year"].astype(str) + "_" + stock_data["week"].astype(str)
stock_data["day"] = np_getday(stock_data["datetime"])
stock_data["weekday"] = np_getweekday(stock_data["datetime"])

# report progress
duration = default_timer() - start_time
print("Created time intervals after " + str(np.round(duration, 2)) + " seconds")

start_time = default_timer()
# compute annual percent change in adjclose, volume, and price_range for each symbol
stock_year_data = stock_data.groupby(["year", "symbol"]).sum()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_year_data["adjclose_annual_change"] = np.exp(stock_year_data["diff_log_adjclose"]) - 1
stock_year_data["volume_annual_change"] = np.exp(stock_year_data["diff_log_volume"]) - 1
stock_year_data["price_range_annual_change"] = np.exp(stock_year_data["diff_log_price_range"]) - 1

# compute annual volatility in adjclose, volume, and price_range for each symbol
stock_year_volatility = stock_data.groupby(["year", "symbol"]).std()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_year_data["adjclose_annual_volatility"] = stock_year_volatility["diff_log_adjclose"] * np.sqrt(255)
stock_year_data["volume_annual_volatility"] = stock_year_volatility["diff_log_volume"] * np.sqrt(255)
stock_year_data["price_range_annual_volatility"] = stock_year_volatility["diff_log_price_range"] * np.sqrt(255)

# compute quarterly percent change in adjclose, volume, and price_range for each symbol
stock_quarter_data = stock_data.groupby(["year_quarter", "symbol"]).sum()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_quarter_data["adjclose_quarterly_change"] = np.exp(stock_quarter_data["diff_log_adjclose"]) - 1
stock_quarter_data["volume_quarterly_change"] = np.exp(stock_quarter_data["diff_log_volume"]) - 1
stock_quarter_data["price_range_quarterly_change"] = np.exp(stock_quarter_data["diff_log_price_range"]) - 1

# compute quarterly volatility in adjclose, volume, and price_range for each symbol
stock_quarter_volatility = stock_data.groupby(["year_quarter", "symbol"]).std()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_quarter_data["adjclose_quarterly_volatility"] = stock_quarter_volatility["diff_log_adjclose"] * np.sqrt(255)
stock_quarter_data["volume_quarterly_volatility"] = stock_quarter_volatility["diff_log_volume"] * np.sqrt(255)
stock_quarter_data["price_range_quarterly_volatility"] = stock_quarter_volatility["diff_log_price_range"] * np.sqrt(255)

# compute monthly percent change in adjclose, volume, and price_range for each symbol
stock_month_data = stock_data.groupby(["year_month", "symbol"]).sum()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_month_data["adjclose_monthly_change"] = np.exp(stock_month_data["diff_log_adjclose"]) - 1
stock_month_data["volume_monthly_change"] = np.exp(stock_month_data["diff_log_volume"]) - 1
stock_month_data["price_range_monthly_change"] = np.exp(stock_month_data["diff_log_price_range"]) - 1

# compute monthly volatility in adjclose, volume, and price_range for each symbol
stock_month_volatility = stock_data.groupby(["year_month", "symbol"]).std()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_month_data["adjclose_monthly_volatility"] = stock_month_volatility["diff_log_adjclose"] * np.sqrt(255)
stock_month_data["volume_monthly_volatility"] = stock_month_volatility["diff_log_volume"] * np.sqrt(255)
stock_month_data["price_range_monthly_volatility"] = stock_month_volatility["diff_log_price_range"] * np.sqrt(255)

# compute weekly percent change in adjclose, volume, and price_range for each symbol
stock_week_data = stock_data.groupby(["year_week", "symbol"]).sum()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_week_data["adjclose_weekly_change"] = np.exp(stock_week_data["diff_log_adjclose"]) - 1
stock_week_data["volume_weekly_change"] = np.exp(stock_week_data["diff_log_volume"]) - 1
stock_week_data["price_range_weekly_change"] = np.exp(stock_week_data["diff_log_price_range"]) - 1

# compute weekly volatility in adjclose, volume, and price_range for each symbol
stock_week_volatility = stock_data.groupby(["year_week", "symbol"]).std()[["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"]].reset_index().fillna(0)
stock_week_data["adjclose_weekly_volatility"] = stock_week_volatility["diff_log_adjclose"] * np.sqrt(255)
stock_week_data["volume_weekly_volatility"] = stock_week_volatility["diff_log_volume"] * np.sqrt(255)
stock_week_data["price_range_weekly_volatility"] = stock_week_volatility["diff_log_price_range"] * np.sqrt(255)

# clean out garbage in RAM
del stock_year_volatility, stock_quarter_volatility, stock_month_volatility, stock_week_volatility
gc.collect()

# report progress
duration = default_timer() - start_time
print("Computed [annual, quarterly, monthly, & weekly] [percent change & volatility] in [price, volume, & price range] after " + str(np.round(duration, 2)) + " seconds")

start_time = default_timer()
# compute the start and end date of each year
year_start = stock_data.groupby(["year"]).min()[["datetime"]].reset_index()
year_end = stock_data.groupby(["year"]).max()[["datetime"]].reset_index()

# compute the start and end date of each quarter
quarter_start = stock_data.groupby(["year_quarter"]).min()[["datetime"]].reset_index()
quarter_end = stock_data.groupby(["year_quarter"]).max()[["datetime"]].reset_index()

# compute the start and end date of each month
month_start = stock_data.groupby(["year_month"]).min()[["datetime"]].reset_index()
month_end = stock_data.groupby(["year_month"]).max()[["datetime"]].reset_index()

# shift the end dates by 2 weeks (to prepare for the effect of monthly inflation)
year_end["datetime"] = year_end["datetime"] + timedelta(days = 14)
quarter_end["datetime"] = quarter_end["datetime"] + timedelta(days = 14)
month_end["datetime"] = month_end["datetime"] + timedelta(days = 14)

# update the column names of the start and end dates
year_start.columns = ["year", "start_datetime"]
year_end.columns = ["year", "end_datetime"]
quarter_start.columns = ["year_quarter", "start_datetime"]
quarter_end.columns = ["year_quarter", "end_datetime"]
month_start.columns = ["year_month", "start_datetime"]
month_end.columns = ["year_month", "end_datetime"]

# join the start and end date of each year onto annual stock data
stock_year_data = pd.merge(year_start, stock_year_data, left_on = "year", right_on = "year", how = "right")
stock_year_data = pd.merge(year_end, stock_year_data, left_on = "year", right_on = "year", how = "right")

# join the start and end date of each quarter onto quarterly stock data
stock_quarter_data = pd.merge(quarter_start, stock_quarter_data, left_on = "year_quarter", right_on = "year_quarter", how = "right")
stock_quarter_data = pd.merge(quarter_end, stock_quarter_data, left_on = "year_quarter", right_on = "year_quarter", how = "right")

# join the start and end date of each month onto monthly stock data
stock_month_data = pd.merge(month_start, stock_month_data, left_on = "year_month", right_on = "year_month", how = "right")
stock_month_data = pd.merge(month_end, stock_month_data, left_on = "year_month", right_on = "year_month", how = "right")

# define a function for computing inflation rate
def inflation(start_date, end_date):
    try:
        result = (cpi.inflate(1000, start_date, to = end_date) / 1000) - 1
    except:
        result = 0
    return result
np_inflation = np.vectorize(inflation)

# compute the annual, quarterly, and monthly inflation rates
stock_year_data["inflation_annual"] = np_inflation(stock_year_data["start_datetime"], stock_year_data["end_datetime"])
stock_quarter_data["inflation_quarterly"] = np_inflation(stock_quarter_data["start_datetime"], stock_quarter_data["end_datetime"])
stock_month_data["inflation_monthly"] = np_inflation(stock_month_data["start_datetime"], stock_month_data["end_datetime"])

# adjust annual percent change in adjclose and price_range for inflation
stock_year_data["adjclose_annual_change"] = ((1 + stock_year_data["adjclose_annual_change"]) / (1 + stock_year_data["inflation_annual"])) - 1
stock_year_data["price_range_annual_change"] = ((1 + stock_year_data["price_range_annual_change"]) / (1 + stock_year_data["inflation_annual"])) - 1

# adjust quarterly percent change in adjclose and price_range for inflation
stock_quarter_data["adjclose_quarterly_change"] = ((1 + stock_quarter_data["adjclose_quarterly_change"]) / (1 + stock_quarter_data["inflation_quarterly"])) - 1
stock_quarter_data["price_range_quarterly_change"] = ((1 + stock_quarter_data["price_range_quarterly_change"]) / (1 + stock_quarter_data["inflation_quarterly"])) - 1

# adjust monthly percent change in adjclose and price_range for inflation
stock_month_data["adjclose_monthly_change"] = ((1 + stock_month_data["adjclose_monthly_change"]) / (1 + stock_month_data["inflation_monthly"])) - 1
stock_month_data["price_range_monthly_change"] = ((1 + stock_month_data["price_range_monthly_change"]) / (1 + stock_month_data["inflation_monthly"])) - 1

# drop unnecessary columns from stock_year_data, stock_quarter_data, stock_month_data, and  stock_week_data
stock_year_data = stock_year_data.drop(columns = ["end_datetime", "start_datetime", "diff_log_adjclose", "diff_log_volume", "diff_log_price_range"])
stock_quarter_data = stock_quarter_data.drop(columns = ["end_datetime", "start_datetime", "diff_log_adjclose", "diff_log_volume", "diff_log_price_range"])
stock_month_data = stock_month_data.drop(columns = ["end_datetime", "start_datetime", "diff_log_adjclose", "diff_log_volume", "diff_log_price_range"])
stock_week_data = stock_week_data.drop(columns = ["diff_log_adjclose", "diff_log_volume", "diff_log_price_range"])

# create and index to remember the row order of stock_data
stock_data = stock_data.reset_index()

# join annual, quarterly, monthly, and weekly stock data onto stock_data
stock_data = pd.merge(stock_year_data, stock_data, left_on = ["year", "symbol"], right_on = ["year", "symbol"], how = "right")
stock_data = pd.merge(stock_quarter_data, stock_data, left_on = ["year_quarter", "symbol"], right_on = ["year_quarter", "symbol"], how = "right")
stock_data = pd.merge(stock_month_data, stock_data, left_on = ["year_month", "symbol"], right_on = ["year_month", "symbol"], how = "right")
stock_data = pd.merge(stock_week_data, stock_data, left_on = ["year_week", "symbol"], right_on = ["year_week", "symbol"], how = "right")

# sort stock_data by index
stock_data = stock_data.sort_values(by = "index", ascending = True).reset_index(drop = True)

# remove the index column
stock_data = stock_data.drop("index", axis = 1)

# clean out garbage in RAM
del stock_year_data, stock_quarter_data, stock_month_data, stock_week_data, year_start, year_end, quarter_start, quarter_end, month_start, month_end
gc.collect()

# report progress
duration = default_timer() - start_time
print("Adjusted percent changes in price & price range for inflation after " + str(np.round(duration, 2)) + " seconds")

# remove unnessary columns in stock_data
stock_data = stock_data.drop(columns = ["year_quarter", "year_month", "year_week", "diff_log_adjclose", "diff_log_volume", "diff_log_price_range", "formatted_date", "date"])

start_time = default_timer()
# get unemployement data
unemployement_data = api.bls_data.run(bls_key = bls_key, 
                                      start_year = min(stock_data["year"]), 
                                      end_year = max(stock_data["year"]))

# report progress
duration = default_timer() - start_time
print("Collected unemployement history after " + str(np.round(duration, 2)) + " seconds")

start_time = default_timer()
# get average hourly earnings data
earnings_data = api.fred_data.run(fred_key = fred_key, 
                                  start_date = str(min(stock_data["datetime"])))

# report progress
duration = default_timer() - start_time
print("Collected average hourly earnings history after " + str(np.round(duration, 2)) + " seconds")

start_time = default_timer()
# convert datetime to datetime64
stock_data["datetime"] = np.array(stock_data["datetime"], dtype = "datetime64")

# get the unique datetimes in stock_data 
econ_data = pd.DataFrame({"datetime": np.sort(np.unique(stock_data["datetime"]))})

# create and index to remember the row order of econ_data
econ_data = econ_data.reset_index()

# join unemployement and average hourly earnings data onto stock_data
econ_data = pd.merge(unemployement_data, econ_data, left_on = "datetime", right_on = "datetime", how = "right")
econ_data = pd.merge(earnings_data, econ_data, left_on = "datetime", right_on = "datetime", how = "right")

# sort stock_data by index
econ_data = econ_data.sort_values(by = "index", ascending = True).reset_index(drop = True)

# remove the index column
econ_data = econ_data.drop("index", axis = 1)

# fill in missing values
for j in range(1, econ_data.shape[1]):
    econ_data.iloc[:,j] = econ_data.iloc[:,j].fillna(method = "pad").fillna(method = "backfill")

# create and index to remember the row order of stock_data
stock_data = stock_data.reset_index()

# join unemployement and average hourly earnings data onto stock_data
stock_data = pd.merge(econ_data, stock_data, left_on = "datetime", right_on = "datetime", how = "right")

# sort stock_data by index
stock_data = stock_data.sort_values(by = "index", ascending = True).reset_index(drop = True)

# remove the index column
stock_data = stock_data.drop("index", axis = 1)

# clean out garbage in RAM
del unemployement_data, earnings_data, econ_data
gc.collect()

# report progress
duration = default_timer() - start_time
print("Combined stock and economic history after " + str(np.round(duration, 2)) + " seconds")

# export stock_data
stock_data.to_csv("stock data.csv", index = False)

# should we delete everything?
clean_work_space = False

if clean_work_space:
    
    # reset the work enviornment
    gc.collect()
    %reset -f