# set the path of where the input files are
mywd = "D:\\Data Science\\Stocks"

# time management modules
from timeit import default_timer

# start a timer
start_time = default_timer()

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

# univariate data modeling modules
import h2o
from machine_learning.supervised.prepare import run as prepare_data
from feature_engineering.selection.recursive_elimination.regression.univariate.xgboost.tree.single import run as rfe
from machine_learning.supervised.regression.univariate.linear_model.grid import run as linear
from machine_learning.supervised.regression.univariate.xgboost.tree.grid import run as xgbt
from machine_learning.supervised.regression.univariate.neural_network.dense.grid import run as nnet
from machine_learning.supervised.regression.univariate.evaluate import run as evaluate

# multivariate data modeling modules
from feature_engineering.selection.recursive_elimination.regression.multivariate.random_forest.single import run as rfe_m
from machine_learning.supervised.regression.multivariate.pls.grid import run as pls_m
from machine_learning.supervised.regression.multivariate.random_forest.grid import run as forest_m
from machine_learning.supervised.regression.multivariate.neural_network.dense.grid import run as nnet_m
from machine_learning.supervised.regression.multivariate.evaluate import run as evaluate_m

# unsupervised data modeling modules
from sklearn.metrics import silhouette_samples
from data_cleansing.anomalies.isolation_forest.single import run as isolate
from machine_learning.unsupervised.clustering.kmeans.grid import run as kmeans
from machine_learning.unsupervised.clustering.gaussian_mixture.grid import run as gmix
from machine_learning.unsupervised.clustering.optics.grid import run as optics
from machine_learning.unsupervised.clustering.random_forest.single import run as leafs
from machine_learning.unsupervised.clustering.connected_components.single import run as components
from machine_learning.unsupervised.clustering.colors.single import run as colors
from machine_learning.unsupervised.clustering.kcliques.single import run as kcliques
from machine_learning.unsupervised.projections.diffusion_maps.single import run as dmap
from machine_learning.unsupervised.projections.isomap.grid import run as imap
from machine_learning.unsupervised.projections.low_rank_model.grid import run as low_rank
from machine_learning.unsupervised.projections.pca.single import run as pca

# graphics modules
import networkx as nx
import matplotlib.pyplot as plt

# miscellaneous modules
import warnings

# should we hide warning messages?
hide_warnings = True

# should we delete everything at the end of the script?
clean_work_space = False

# handle warnings
if hide_warnings:
    warnings.filterwarnings("ignore")
else:
    warnings.filterwarnings("default")

# report progress
run_time1 = default_timer() - start_time
print("- Libraries loaded in " + str(np.round(run_time1, 3)) + " seconds")

# import the data
dat = pd.read_csv("stock data.csv")
         
# report progress
run_time2 = default_timer() - start_time - run_time1
print("- Data loaded in " + str(np.round(run_time2, 3)) + " seconds")

# identify the key (the feature that identifies each row of the data frame)
key = "datetime"

# get timestamp data
timestamp = dat[["year", "quarter", "month", "week", "day", "weekday"]].copy()

# convert timestamp into binaries
timestamp = feature_engineering.dummies.run(data_frame = timestamp, 
                                            ignore_features = None, 
                                            display_progress = True)

# report progress
run_time3 = default_timer() - start_time - run_time1 - run_time2
print("- Time stamps collected in " + str(np.round(run_time3, 3)) + " seconds")

# reduce the dimensionality of timestamp into 10 numerical features
timestamp_arch_features, timestamp_glrm_mod, timestamp_glrm_error, timestamp_wape, timestamp_arch_samples = low_rank(data_frame = timestamp, 
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
                                                                                                                     ram_fraction = 1,
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

# count how many symbols exist per year
symbol_counts = pd.DataFrame()
for y in np.unique(check["year"]):
    symbol_counts = pd.concat([symbol_counts, pd.DataFrame({"year": [y], "count": [check.loc[check["year"] == y].shape[0]]})], axis = 0, sort = False).reset_index(drop = True)

# count how many years each symbols existed
year_counts = pd.DataFrame()
for s in np.unique(check["symbol"]):
    year_counts = pd.concat([year_counts, pd.DataFrame({"symbol": [s], "count": [check.loc[check["symbol"] == s].shape[0]]})], axis = 0, sort = False).reset_index(drop = True)

# plot the number of stocks per year
symbol_count_plot = data_visualization.line_plot.run(data_frame = symbol_counts, 
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
symbol_count_plot

# plot the number of years stocks have been in the market
year_count_plot = data_visualization.histogram.run(data_frame = year_counts, 
                                                   x_name = "count", 
                                                   bin_width = 3, 
                                                   color_name = None, 
                                                   color = "forestgreen", 
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
year_count_plot

# only keep stocks for the last 20 years
dat = dat.loc[dat["year"] >= max(dat["year"]) - 20].reset_index(drop = True)

# start a timer
start_time = default_timer()

# collect weekly stock performance
weekly_indicators = ["adjclose_weekly_change", "adjclose_weekly_volatility", "price_range_weekly_change", "price_range_weekly_volatility", "volume_weekly_change", "volume_weekly_volatility"]

# get weekly closing price change
stock_performance = dat[np.append(["datetime", "year", "quarter", "month", "week", "symbol"], weekly_indicators)].copy()

# drop all duplicates from stock_performance
stock_performance = stock_performance.drop_duplicates(subset = ["year", "quarter", "month", "week", "symbol"]).reset_index(drop = True)

# create a timestamp column
stock_performance["timestamp"] = "Y" + stock_performance["year"].astype(str) + "_Q" + stock_performance["quarter"].astype(str) + "_M" + stock_performance["month"].astype(str) + "_W" + stock_performance["week"].astype(str)

# convert stock_performance into wide format for each indicator
stock_performance_weekly_list = [stock_performance.pivot_table(index = "timestamp", columns = "symbol")[j].dropna(axis = 1).reset_index() for j in weekly_indicators]

# report progress
run_time1 = default_timer() - start_time
print("- Weekly stock indicators collected in " + str(np.round(run_time1, 3)) + " seconds")

# collect monthly stock performance
monthly_indicators = ["adjclose_monthly_change", "adjclose_monthly_volatility", "price_range_monthly_change", "price_range_monthly_volatility", "volume_monthly_change", "volume_monthly_volatility"]

# get monthly closing price change
stock_performance = dat[np.append(["datetime", "year", "quarter", "month", "symbol"], monthly_indicators)].copy()

# drop all duplicates from stock_performance
stock_performance = stock_performance.drop_duplicates(subset = ["year", "quarter", "month", "symbol"]).reset_index(drop = True)

# only keep stocks that appeared in stock_performance_weekly_list
stock_performance = stock_performance.iloc[np.where(stock_performance["symbol"].isin(stock_performance_weekly_list[0].iloc[:,1:].columns))].reset_index(drop = True)

# create a timestamp column
stock_performance["timestamp"] = "Y" + stock_performance["year"].astype(str) + "_Q" + stock_performance["quarter"].astype(str) + "_M" + stock_performance["month"].astype(str)

# convert stock_performance into wide format for each indicator
stock_performance_monthly_list = [stock_performance.pivot_table(index = "timestamp", columns = "symbol")[j].dropna(axis = 1).reset_index() for j in monthly_indicators]

# report progress
run_time2 = default_timer() - start_time - run_time1
print("- Monthly stock indicators collected in " + str(np.round(run_time2, 3)) + " seconds")

# collect quarterly stock performance
quarterly_indicators = ["adjclose_quarterly_change", "adjclose_quarterly_volatility", "price_range_quarterly_change", "price_range_quarterly_volatility", "volume_quarterly_change", "volume_quarterly_volatility"]

# get quarterly closing price change
stock_performance = dat[np.append(["datetime", "year", "quarter", "symbol"], quarterly_indicators)].copy()

# drop all duplicates from stock_performance
stock_performance = stock_performance.drop_duplicates(subset = ["year", "quarter", "symbol"]).reset_index(drop = True)

# only keep stocks that appeared in stock_performance_weekly_list
stock_performance = stock_performance.iloc[np.where(stock_performance["symbol"].isin(stock_performance_weekly_list[0].iloc[:,1:].columns))].reset_index(drop = True)

# create a timestamp column
stock_performance["timestamp"] = "Y" + stock_performance["year"].astype(str) + "_Q" + stock_performance["quarter"].astype(str)

# convert stock_performance into wide format for each indicator
stock_performance_quarterly_list = [stock_performance.pivot_table(index = "timestamp", columns = "symbol")[j].dropna(axis = 1).reset_index() for j in quarterly_indicators]

# report progress
run_time3 = default_timer() - start_time - run_time1 - run_time2
print("- Quarterly stock indicators collected in " + str(np.round(run_time3, 3)) + " seconds")

# collect annual stock performance
annual_indicators = ["adjclose_annual_change", "adjclose_annual_volatility", "price_range_annual_change", "price_range_annual_volatility", "volume_annual_change", "volume_annual_volatility"]

# get annual closing price change
stock_performance = dat[np.append(["datetime", "year", "symbol"], annual_indicators)].copy()

# drop all duplicates from stock_performance
stock_performance = stock_performance.drop_duplicates(subset = ["year", "symbol"]).reset_index(drop = True)

# only keep stocks that appeared in stock_performance_weekly_list
stock_performance = stock_performance.iloc[np.where(stock_performance["symbol"].isin(stock_performance_weekly_list[0].iloc[:,1:].columns))].reset_index(drop = True)

# create a timestamp column
stock_performance["timestamp"] = "Y" + stock_performance["year"].astype(str)

# convert stock_performance into wide format for each indicator
stock_performance_annual_list = [stock_performance.pivot_table(index = "timestamp", columns = "symbol")[j].dropna(axis = 1).reset_index() for j in annual_indicators]

# report progress
run_time4 = default_timer() - start_time - run_time1 - run_time2 - run_time3
print("- Annual stock indicators collected in " + str(np.round(run_time4, 3)) + " seconds")

# combine the stock performance lists
stock_performance_list = stock_performance_annual_list + stock_performance_quarterly_list + stock_performance_monthly_list + stock_performance_weekly_list

# combine the indicator lists
indicators = annual_indicators + quarterly_indicators + monthly_indicators + weekly_indicators

# clean out garbage in RAM
del annual_indicators, quarterly_indicators, monthly_indicators, weekly_indicators, timestamp, timestamp_arch_features, check, stock_performance, stock_performance_annual_list, stock_performance_quarterly_list, stock_performance_monthly_list, stock_performance_weekly_list
h2o.remove_all()
h2o.api("POST /3/GarbageCollect")
gc.collect()

# reduce the dimensionality of stock_performance_list
stock_performance_arch_samples_list = []
for df in stock_performance_list:
    stock_performance_arch_features, stock_performance_glrm_mod, stock_performance_glrm_error, stock_performance_wape, stock_performance_arch_samples = low_rank(data_frame = df, 
                                                                                                                                                                 rank = [2], 
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
    stock_performance_arch_samples_list.append(stock_performance_arch_samples)
    del stock_performance_arch_features, stock_performance_glrm_mod, stock_performance_glrm_error, stock_performance_wape, stock_performance_arch_samples

# combine stock_performance_arch_samples_list into one data frame
# get the first indicator, and update the column names
stock_performance_arch_samples = stock_performance_arch_samples_list[0].copy()
stock_performance_arch_samples.columns = np.append(np.core.defchararray.add(indicators[0] + "_", stock_performance_arch_samples.columns[:-1]), ["Feature"])

# get the remaining indicators, and update the column names
for j in range(1, len(indicators)):
    df = stock_performance_arch_samples_list[j].drop(columns = ["Feature"]).copy()
    df.columns = np.core.defchararray.add(indicators[j] + "_", df.columns)
    stock_performance_arch_samples = pd.concat([df, stock_performance_arch_samples], axis = 1)

# clean out the garbage in RAM
del df, stock_performance_arch_samples_list
gc.collect()

# should we remove anomalies?
remove_symbols = False

# remove anomalies from stock_performance_arch_samples
if remove_symbols:
    stock_performance_arch_samples, irf, irf_pred = isolate(data_frame = stock_performance_arch_samples, 
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

# standardize stock_performance_arch_samples
stock_performance_arch_samples_std = feature_engineering.standardize.run(data_frame = stock_performance_arch_samples, 
                                                                         ignore_features = ["Feature"], 
                                                                         prefix = "std_", 
                                                                         display_progress = True)

# cluster stock_performance_arch_samples_std with kmeans
stock_performance_kmeans, km_mod, km_score, silhouette_kmeans, kmeans_distance = kmeans(data_frame = stock_performance_arch_samples_std, 
                                                                                        clusters = list(np.arange(10, 26, 1)), 
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
del km_mod

# check out the number of clusters & the mean silhouette coefficient
print("K-Means Clusters: " + str(len(set(stock_performance_kmeans["Cluster"]))) + "\n" +
      "K-Means Silhouette Coefficient: " + str(np.round(np.mean(silhouette_kmeans["Silhouette_Coefficient"]), 3)) + "\n")

# cluster stock_performance_arch_samples_std with bayesian gaussian mixture
    # distance_metric: "euclidean", "cosine", "sqeuclidean", "manhattan", "chebyshev", "correlation"
stock_performance_gmix, gmix_mod, gmix_score, silhouette_gmix, gmix_distance = gmix(data_frame = stock_performance_arch_samples_std, 
                                                                                    grid_searches = 2, 
                                                                                    grid_size = 50, 
                                                                                    mutation_size = 7, 
                                                                                    mutation = 0.2, 
                                                                                    clusters = [10, 15, 20], 
                                                                                    covariance_type = ["full", "tied"], 
                                                                                    weight_concentration_prior = [0.05, 0.1, 0.2], 
                                                                                    mean_precision_prior = [0.5, 1, 1.5, 2], 
                                                                                    iterations = 100, 
                                                                                    initializations = 5, 
                                                                                    distance_metric = "sqeuclidean",
                                                                                    cpu_fraction = 1, 
                                                                                    ram_fraction = 1, 
                                                                                    ignore_features = ["Feature"], 
                                                                                    seed = 48, 
                                                                                    display_progress = True)
del gmix_mod

# check out the number of clusters & the mean silhouette coefficient
print("Gaussian Mixture Clusters: " + str(len(set(stock_performance_gmix["Cluster"]))) + "\n" +
      "Gaussian Mixture Silhouette Coefficient: " + str(np.round(np.mean(silhouette_gmix["Silhouette_Coefficient"]), 3)) + "\n")

# euclidean:    10, 0.327
# cosine:       5, 0.43
# sqeuclidean:  10, 0.436
# manhattan:    5, 0.382
# chebyshev:    10, 0.212
# correlation:  5, 0.431

# cluster stock_performance with optics
    # distance_metric: "euclidean", "cosine", "sqeuclidean", "manhattan", "chebyshev", "correlation"
stock_performance_optics, optics_mod, optics_score, silhouette_optics, optics_distance = optics(data_frame = stock_performance_arch_samples_std, 
                                                                                                grid_searches = 2, 
                                                                                                grid_size = 50, 
                                                                                                mutation_size = 7, 
                                                                                                mutation = 0.2, 
                                                                                                sample_fraction = [0.01, 0.02, 0.03, 0.04], 
                                                                                                min_steepness = [0.0002, 0.0004, 0.0006, 0.0008, 0.002, 0.004, 0.006, 0.008], 
                                                                                                distance_metric = "cosine",
                                                                                                cpu_fraction = 1, 
                                                                                                ram_fraction = 1, 
                                                                                                ignore_features = ["Feature"], 
                                                                                                seed = 48, 
                                                                                                display_progress = True)
del optics_mod

# check out the number of clusters & the mean silhouette coefficient
print("Optics Clusters: " + str(len(set(stock_performance_optics["Cluster"]))) + "\n" +
      "Optics Silhouette Coefficient: " + str(np.round(np.mean(silhouette_optics["Silhouette_Coefficient"]), 3)) + "\n")

# euclidean:    4, -0.023
# cosine:       3, 0.31
# sqeuclidean:  22, -0.104
# manhattan:    5, -0.012
# chebyshev:    6, -0.048
# correlation:  3, 0.38

# cluster stock_performance_arch_samples_std with random forest
stock_performance_rf, rf_mod, stock_performance_importance = leafs(data_frame = stock_performance_arch_samples_std, 
                                                                   trees = 50, 
                                                                   min_samples = 15, 
                                                                   max_depth = 25, 
                                                                   min_split_loss = 0, 
                                                                   cpu_fraction = 1, 
                                                                   ignore_features = ["Feature"], 
                                                                   seed = 18, 
                                                                   display_progress = True)
del rf_mod

# update the column names of clusterings
stock_performance_kmeans.columns = ["symbol", "KM_Cluster"]
stock_performance_gmix.columns = np.append(["symbol", "GM_Cluster"], stock_performance_gmix.columns[2:])
stock_performance_optics.columns = ["symbol", "O_Cluster"]
stock_performance_rf.columns = np.append(["symbol"], stock_performance_rf.columns[1:])

# combine clusterings
stock_performance_clusters = pd.concat([stock_performance_kmeans, 
                                        stock_performance_optics[["O_Cluster"]],
                                        stock_performance_gmix[["GM_Cluster"]]], axis = 1)

# convert stock_performance_clusters into binaries
stock_performance_clusters = feature_engineering.dummies.run(data_frame = stock_performance_clusters, 
                                                             ignore_features = ["symbol"], 
                                                             display_progress = True)

# should we add leafs to the clusters?
include_leafs = False

# add random forest leafs to stock_performance_clusters
if include_leafs:
    stock_performance_clusters = pd.concat([stock_performance_clusters, 
                                            stock_performance_rf.drop(columns = ["symbol"])], axis = 1)

# drop the outlier cluster
stock_performance_clusters = stock_performance_clusters.drop(columns = ["O_Cluster_-1"])

# compute the cluster sizes
cluster_sizes = pd.DataFrame(stock_performance_clusters.iloc[:,1:].sum(axis = 0))

# compute max cluster sizes
max_size = np.round(stock_performance_clusters.shape[0] * 0.2, 0).astype(int)

# determine which clusters meet the cluster size requirement
cluster_names = cluster_sizes.loc[cluster_sizes.iloc[:,0] <= max_size].index.values

# transpose stock_performance_clusters to create an incidence matrix
stock_performance_incidence = stock_performance_clusters[cluster_names].T
stock_performance_incidence.columns = stock_performance_clusters["symbol"]

# cluster the incidence matrix using connected components
stock_performance_cc, stock_performance_graph, stock_performance_adjacency = components(incidence_data_frame = stock_performance_incidence, 
                                                                                        ignore_features = None, 
                                                                                        display_progress = True)

# check out the number of clusters
print("Connected Component Clusters: " + str(len(set(stock_performance_cc["Cluster"]))) + "\n")

# cluster the incidence matrix using coloring
    # strategy options: "largest_first", "random_sequential", "smallest_last", "independent_set", "connected_sequential_bfs", "connected_sequential_dfs", "DSATUR"
stock_performance_col, stock_performance_graph, stock_performance_adjacency = colors(incidence_data_frame = stock_performance_incidence, 
                                                                                     strategy = "largest_first",
                                                                                     interchange = False,
                                                                                     ignore_features = None, 
                                                                                     display_progress = True)

# check out the number of clusters
print("Colors: " + str(len(set(stock_performance_col["Color"]))) + "\n")

# cluster the incidence matrix using kcliques
stock_performance_kc, stock_performance_graph, stock_performance_adjacency = kcliques(incidence_data_frame = stock_performance_incidence,
                                                                                      size = 6,
                                                                                      ignore_features = None, 
                                                                                      display_progress = True)

# check out the number of clusters
print("K-Cliques Clusters: " + str(len(set(stock_performance_kc["Cluster"]))) + "\n")

# compute the silhouette coefficients for connected components
silhouette_cc_kmeans = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = kmeans_distance, labels = np.array(stock_performance_cc["Cluster"]), metric = "precomputed")})
silhouette_cc_gmix = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = gmix_distance, labels = np.array(stock_performance_cc["Cluster"]), metric = "precomputed")})
silhouette_cc_optics = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = optics_distance, labels = np.array(stock_performance_cc["Cluster"]), metric = "precomputed")})
print("Connected Component Silhouette Coefficient: " + str(np.round(np.mean([np.mean(silhouette_cc_kmeans["Silhouette_Coefficient"]), np.mean(silhouette_cc_gmix["Silhouette_Coefficient"]), np.mean(silhouette_cc_optics["Silhouette_Coefficient"])]), 3)))

# compute the silhouette coefficients for connected components
silhouette_col_kmeans = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = kmeans_distance, labels = np.array(stock_performance_col["Color"]), metric = "precomputed")})
silhouette_col_gmix = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = gmix_distance, labels = np.array(stock_performance_col["Color"]), metric = "precomputed")})
silhouette_col_optics = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = optics_distance, labels = np.array(stock_performance_col["Color"]), metric = "precomputed")})
print("Colors Silhouette Coefficient: " + str(np.round(np.mean([np.mean(silhouette_col_kmeans["Silhouette_Coefficient"]), np.mean(silhouette_col_gmix["Silhouette_Coefficient"]), np.mean(silhouette_col_optics["Silhouette_Coefficient"])]), 3)))

# compute the silhouette coefficients for kcliques
silhouette_kc_kmeans = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = kmeans_distance, labels = np.array(stock_performance_kc["Cluster"]), metric = "precomputed")})
silhouette_kc_gmix = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = gmix_distance, labels = np.array(stock_performance_kc["Cluster"]), metric = "precomputed")})
silhouette_kc_optics = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = optics_distance, labels = np.array(stock_performance_kc["Cluster"]), metric = "precomputed")})
print("K-Cliques Silhouette Coefficient: " + str(np.round(np.mean([np.mean(silhouette_kc_kmeans["Silhouette_Coefficient"]), np.mean(silhouette_kc_gmix["Silhouette_Coefficient"]), np.mean(silhouette_kc_optics["Silhouette_Coefficient"])]), 3)))

# rename the columns in stock_performance_cc and stock_performance_kc
stock_performance_cc.columns = np.append(["CC_Cluster"], stock_performance_cc.columns[1:])
stock_performance_kc.columns = np.append(["KC_Cluster"], stock_performance_kc.columns[1:])

# create a list of colors
color_list = ["#0099ff",
              "#4639d4",
              "#4daf4a",
              "#66923d",
              "#71cfca",
              "#87edc3",
              "#97b9d2",
              "#984ea3",
              "#a65628",
              "#ff4545",
              "#ff7f00",
              "#ff96ca",
              "#ffc100",
              "#fff654",
              "#f2d8ae",
              "#c09ecc",
              "#df5e3d",
              "#fa7e7e",
              "#eab16a",
              "#ffa23d",
              "#ff1493",
              "#b0b0b8",
              "#dddddd",
              "#be7759",
              "#6a6c60"]

# get the labels
labels = {i : stock_performance_cc.index.values[i] for i in stock_performance_graph.nodes()}

# get the positions
kk_positions = {i : np.array(stock_performance_cc[["KK_x_coordinate", "KK_y_coordinate"]].iloc[i]) for i in stock_performance_graph.nodes()}
fr_positions = {i : np.array(stock_performance_cc[["FR_x_coordinate", "FR_y_coordinate"]].iloc[i]) for i in stock_performance_graph.nodes()}
s_positions = {i : np.array(stock_performance_cc[["S_x_coordinate", "S_y_coordinate"]].iloc[i]) for i in stock_performance_graph.nodes()}

# get the node colors
np.random.shuffle(color_list)
km_colors = [color_list[i] for i in stock_performance_kmeans["KM_Cluster"]]
gm_colors = [color_list[i] for i in stock_performance_gmix["GM_Cluster"]]
o_colors = [color_list[i] for i in stock_performance_optics["O_Cluster"]]
cc_colors = [color_list[i] for i in stock_performance_cc["CC_Cluster"]]
col_colors = [color_list[i] for i in stock_performance_col["Color"]]
kc_colors = [color_list[i] for i in stock_performance_kc["KC_Cluster"]]

# plot the graph
plt.figure()
nx.draw_networkx(G = stock_performance_graph, 
                 pos = fr_positions,
                 labels = labels, 
                 node_size = 1200, 
                 node_color = kc_colors, 
                 width = 1)

# group stocks based on guassian mixture clustering
stock_performance_gmix

# get the order of clusters by size
gmix_labels = list(stock_performance_gmix["GM_Cluster"].value_counts().index.values)

# get symbols from cluster k
k = 0
gmix_symbols = list(stock_performance_gmix.loc[stock_performance_gmix["GM_Cluster"] == gmix_labels[k]]["symbol"])

# 



if clean_work_space:
    
    # clean out the garbage in RAM
    h2o.remove_all()
    h2o.api("POST /3/GarbageCollect")
    gc.collect()
    
    # shutdown the h2o instance
    h2o.cluster().shutdown()

    # reset the work enviornment
    %reset -f