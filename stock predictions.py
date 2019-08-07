# set the path of where the input files are
mywd = "D:\\Data Science\\Stocks"

# data management modules
import os
import pandas as pd
import numpy as np
import gc
import re

# set the work directory
os.chdir(mywd)

# data science modules
import data_cleansing
import feature_engineering
import machine_learning
import data_visualization

# miscellaneous modules
import warnings

# should we hide warning messages?
hide_warnings = True

# handle warnings
if hide_warnings:
    warnings.filterwarnings("ignore")
else:
    warnings.filterwarnings("default")

# import the data
dat = pd.read_csv("stock data.csv")

# identify the key (the feature that identifies each row of the data frame)
key = "datetime"

# get timestamp data
timestamp = dat[["year", "quarter", "month", "week", "day", "weekday"]].copy()

# convert timestamp into binaries
timestamp = feature_engineering.dummies.run(data_frame = timestamp, 
                                            ignore_features = None, 
                                            display_progress = True)

# reduce the dimensionality of timestamp into 10 numerical features
timestamp_arch_features, timestamp_glrm_mod, timestamp_glrm_error, timestamp_wape, timestamp_arch_samples = machine_learning.unsupervised.projections.low_rank_model.grid.run(data_frame = timestamp, 
                                                                                                                                                                              rank = [10], 
                                                                                                                                                                              left_regularization = ["L2"], 
                                                                                                                                                                              right_regularization = ["L2"], 
                                                                                                                                                                              left_regularization_strength = [1e-3], 
                                                                                                                                                                              right_regularization_strength = [1e-3], 
                                                                                                                                                                              data_transformation = ["None"], 
                                                                                                                                                                              max_iterations = [1000], 
                                                                                                                                                                              min_step_size = [1e-4], 
                                                                                                                                                                              ignore_features = None, 
                                                                                                                                                                              cpu_fraction = 1, 
                                                                                                                                                                              grid_run_time = 2, 
                                                                                                                                                                              grid_size = 10, 
                                                                                                                                                                              seed = 3, 
                                                                                                                                                                              initialize_h2o = True, 
                                                                                                                                                                              shutdown_h2o = False, 
                                                                                                                                                                              display_progress = True)
del timestamp_glrm_mod, timestamp_glrm_error, timestamp_wape, timestamp_arch_samples

# update the column names in timestamp_arch_features
timestamp_arch_features.columns = np.core.defchararray.add("Timestamp_", timestamp_arch_features.columns.astype(str))

# combine timestamp_arch_features and dat
dat = pd.concat([dat, timestamp_arch_features], axis = 1)

# look at just symbols and year
check = dat[["year", "symbol"]].copy()

# drop all duplicates from check
check = check.drop_duplicates().reset_index(drop = True)

# count how many symbols exists per year
year_counts = pd.DataFrame()
for y in np.unique(check["year"]):
    year_counts = pd.concat([year_counts, pd.DataFrame({"year": [y], "count": [check.loc[check["year"] == y].shape[0]]})], axis = 0, sort = False).reset_index(drop = True)

# count how many years each symbols existed
symbol_counts = pd.DataFrame()
for s in np.unique(check["symbol"]):
    symbol_counts = pd.concat([symbol_counts, pd.DataFrame({"symbol": [s], "count": [check.loc[check["symbol"] == s].shape[0]]})], axis = 0, sort = False).reset_index(drop = True)

# plot the number of stocks per year
year_count_plot = data_visualization.line_plot.run(data_frame = year_counts, 
                                                   x_name = "year", 
                                                   y_name = "count", 
                                                   ribbon = False, 
                                                   y_max_name = None, 
                                                   y_min_name = None, 
                                                   smooth = False, 
                                                   smooth_span = 2/3, 
                                                   points = False, 
                                                   line_size = 1.5, 
                                                   point_size = 1.5, 
                                                   color_name = None, 
                                                   color = "cornflowerblue", 
                                                   group_color = False, 
                                                   color_palette = None, 
                                                   distill_palette = False, 
                                                   reverse_palette = False, 
                                                   transparency = 1, 
                                                   text_name = None, 
                                                   text_size = 11, 
                                                   text_color = "black", 
                                                   text_x_position = "center", 
                                                   text_y_position = "center", 
                                                   text_x_nudge = 0, 
                                                   text_y_nudge = 0, 
                                                   text_angle = 0, 
                                                   title = "Stocks in the Market", 
                                                   x_label = "Year", 
                                                   y_label = "Frequency", 
                                                   color_label = "", 
                                                   y_percent = False, 
                                                   y_comma = False, 
                                                   y_round = False, 
                                                   y_precision = 4, 
                                                   x_percent = False, 
                                                   x_comma = False, 
                                                   x_round = False, 
                                                   x_precision = 4, 
                                                   x_angle = 0, 
                                                   panel_formula = None, 
                                                   panel_scales = "fixed", 
                                                   panel_rows = None, 
                                                   panel_columns = None, 
                                                   legend_position = "right", 
                                                   legend_title_position = "top", 
                                                   legend_rows = None, 
                                                   legend_columns = None, 
                                                   font_size = 20, 
                                                   plot_theme = "light", 
                                                   grid_lines = False, 
                                                   display_progress = True)
year_count_plot

symbol_count_plot = data_visualization.histogram.run(data_frame = symbol_counts, 
                                                     x_name = "count", 
                                                     bin_width = 3, 
                                                     color_name = None, 
                                                     color = "cornflowerblue", 
                                                     group_color = False, 
                                                     color_palette = None, 
                                                     distill_palette = False, 
                                                     reverse_palette = False, 
                                                     transparency = 1, 
                                                     title = "Years in Market", 
                                                     x_label = "Years", 
                                                     y_label = "Frequency", 
                                                     color_label = "", 
                                                     y_comma = False, 
                                                     y_round = False, 
                                                     y_precision = 4, 
                                                     x_percent = False, 
                                                     x_comma = False, 
                                                     x_round = False, 
                                                     x_precision = 4, 
                                                     x_angle = 0, 
                                                     panel_formula = None, 
                                                     panel_scales = "fixed", 
                                                     panel_rows = None, 
                                                     panel_columns = None, 
                                                     legend_position = "right", 
                                                     legend_title_position = "top", 
                                                     legend_rows = None, 
                                                     legend_columns = None, 
                                                     font_size = 20, 
                                                     plot_theme = "light", 
                                                     grid_lines = False, 
                                                     display_progress = True)
symbol_count_plot

# only keep stocks for the last 20 years
dat = dat.loc[dat["year"] >= max(dat["year"]) - 20].reset_index(drop = True)

#####################

# collect all the non-stock specific data
other_data = dat[["datetime", "year", "quarter", "month", "week", 
                  "Timestamp_Arch1", "Timestamp_Arch2", "Timestamp_Arch3",
                  "Timestamp_Arch4", "Timestamp_Arch5", "Timestamp_Arch6",
                  "Timestamp_Arch7", "Timestamp_Arch8", "Timestamp_Arch9",
                  "Timestamp_Arch10", "inflation_annual", "inflation_quarterly", 
                  "inflation_monthly", "Manufacturing_AHE", "Mining_and_Logging_AHE",
                  "Trade_Transportation_and_Utilities_AHE", "Construction_AHE",
                  "Information_AHE", "Financial_Activities_AHE", "Professional_and_Business_Services_AHE",
                  "Education_and_Health_Services_AHE", "Leisure_and_Hospitality_AHE",
                  "Total_Private_AHE", "White_Unemployement", "Black_Unemployement", "Hispanic_Unemployement"]].copy()

# collect the stock features
stock_features = list(dat.drop(columns = np.append(other_data.columns, ["symbol", "name", "mad_rank", "day", "weekday", "close", "high", "low", "open", "adjclose", "volume", "price_range"])).columns)

#####################

# get weekly closing price change
weekly_price_change = dat[["datetime", "year", "quarter", "month", "week", "symbol", "adjclose_weekly_change"]].copy()

# drop all duplicates from weekly_price_change
weekly_price_change = weekly_price_change.drop_duplicates(subset = ["year", "quarter", "month", "week", "symbol"]).reset_index(drop = True)

# create a timestamp column
weekly_price_change["timestamp"] = "Y" + weekly_price_change["year"].astype(str) + "_Q" + weekly_price_change["quarter"].astype(str) + "_M" + weekly_price_change["month"].astype(str) + "_W" + weekly_price_change["week"].astype(str)

# convert weekly_price_change into wide format
weekly_price_change = weekly_price_change.pivot_table(index = "timestamp", columns = "symbol")["adjclose_weekly_change"].dropna(axis = 1).reset_index()

# reduce the dimensionality of weekly_price_change into 10 numerical features
weekly_price_change_arch_features, weekly_price_change_glrm_mod, weekly_price_change_glrm_error, weekly_price_change_wape, weekly_price_change_arch_samples = machine_learning.unsupervised.projections.low_rank_model.grid.run(data_frame = weekly_price_change.copy(), 
                                                                                                                                                                                                                                rank = [10], 
                                                                                                                                                                                                                                left_regularization = ["L2"], 
                                                                                                                                                                                                                                right_regularization = ["L2"], 
                                                                                                                                                                                                                                left_regularization_strength = [1e-3], 
                                                                                                                                                                                                                                right_regularization_strength = [1e-3], 
                                                                                                                                                                                                                                data_transformation = ["None"], 
                                                                                                                                                                                                                                max_iterations = [1000], 
                                                                                                                                                                                                                                min_step_size = [1e-4], 
                                                                                                                                                                                                                                ignore_features = ["timestamp"], 
                                                                                                                                                                                                                                cpu_fraction = 1, 
                                                                                                                                                                                                                                ram_fraction = 1,
                                                                                                                                                                                                                                grid_run_time = 2, 
                                                                                                                                                                                                                                grid_size = 10, 
                                                                                                                                                                                                                                seed = 3, 
                                                                                                                                                                                                                                initialize_h2o = False, 
                                                                                                                                                                                                                                shutdown_h2o = False, 
                                                                                                                                                                                                                                display_progress = True)
del weekly_price_change_arch_features, weekly_price_change_glrm_mod, weekly_price_change_glrm_error, weekly_price_change_wape

# remove anomalies from weekly_price_change_arch_samples
weekly_price_change_arch_samples, irf, irf_pred = data_cleansing.anomalies.isolation_forest.single.run(data_frame = weekly_price_change_arch_samples, 
                                                                                                       anomaly_fraction = 0.05, 
                                                                                                       trees = 50, 
                                                                                                       sample_fraction = 0.6320000291, 
                                                                                                       min_samples = 1, 
                                                                                                       max_depth = 14, 
                                                                                                       feature_fraction_tree = 1, 
                                                                                                       feature_fraction_split = None, 
                                                                                                       ignore_features = ["Feature"], 
                                                                                                       cpu_fraction = 1, 
                                                                                                       ram_fraction = 1, 
                                                                                                       model_run_time = 2, 
                                                                                                       seed = 62, 
                                                                                                       initialize_h2o = False, 
                                                                                                       shutdown_h2o = False, 
                                                                                                       display_progress = True)
del irf, irf_pred

# standardize weekly_price_change_arch_samples
weekly_price_change_arch_samples_std = feature_engineering.standardize.run(data_frame = weekly_price_change_arch_samples, 
                                                                           ignore_features = ["Feature"], 
                                                                           prefix = "std_", 
                                                                           display_progress = True)

# cluster weekly_price_change_arch_samples_std with kmeans
weekly_price_change_kmeans, km_mod, km_error, silhouette_kmeans = machine_learning.unsupervised.clustering.kmeans.grid.run(data_frame = weekly_price_change_arch_samples_std, 
                                                                                                                           clusters = list(np.arange(4, 31, 1)), 
                                                                                                                           standardize = [False], 
                                                                                                                           max_iterations = [50], 
                                                                                                                           ignore_features = ["Feature"], 
                                                                                                                           cpu_fraction = 1, 
                                                                                                                           ram_fraction = 1, 
                                                                                                                           grid_run_time = 2, 
                                                                                                                           grid_size = 50, 
                                                                                                                           seed = 71, 
                                                                                                                           initialize_h2o = False, 
                                                                                                                           shutdown_h2o = False, 
                                                                                                                           display_progress = True)
del km_mod, km_error

# cluster weekly_price_change_arch_samples_std with bayesian gaussian mixture
weekly_price_change_gmix, mod, gmix_score, silhouette_gmix = machine_learning.unsupervised.clustering.gaussian_mixture.grid.run(data_frame = weekly_price_change_arch_samples_std, 
                                                                                                                                grid_searches = 2, 
                                                                                                                                grid_size = 50, 
                                                                                                                                mutation_size = 7, 
                                                                                                                                mutation = 0.2, 
                                                                                                                                clusters = [7, 15, 20], 
                                                                                                                                covariance_type = ["full", "tied", "spherical"], 
                                                                                                                                weight_concentration_prior = [1/2, 1, 3/2],
                                                                                                                                mean_precision_prior = [1/2, 1, 3/2], 
                                                                                                                                iterations = 100, 
                                                                                                                                initializations = 5, 
                                                                                                                                cpu_fraction = 1, 
                                                                                                                                ram_fraction = 1, 
                                                                                                                                ignore_features = ["Feature"], 
                                                                                                                                seed = 48, 
                                                                                                                                display_progress = True)
del mod

# cluster weekly_price_change_arch_samples_std with random forest
weekly_price_change_rf, mod, weekly_price_change_importance = machine_learning.unsupervised.clustering.random_forest.single.run(data_frame = weekly_price_change_arch_samples_std, 
                                                                                                                                trees = 50, 
                                                                                                                                min_samples = 5, 
                                                                                                                                max_depth = 14, 
                                                                                                                                min_split_loss = 0, 
                                                                                                                                cpu_fraction = 1, 
                                                                                                                                ignore_features = ["Feature"], 
                                                                                                                                seed = 18, 
                                                                                                                                display_progress = True)
del mod




weekly_price_change_optics



# should we delete everything?
clean_work_space = False

if clean_work_space:
    
    # clean out the garbage in RAM
    h2o.remove_all()
    h2o.api("POST /3/GarbageCollect")
    gc.collect()
    
    # shutdown the h2o instance
    h2o.cluster().shutdown()

    # reset the work enviornment
    %reset -f