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

# start a timer
start_time = default_timer()

# get weekly closing price change
price_change = dat[["datetime", "year", "quarter", "month", "week", "symbol", "adjclose_weekly_change"]].copy()

# drop all duplicates from price_change
price_change = price_change.drop_duplicates(subset = ["year", "quarter", "month", "week", "symbol"]).reset_index(drop = True)

# create a timestamp column
price_change["timestamp"] = "Y" + price_change["year"].astype(str) + "_Q" + price_change["quarter"].astype(str) + "_M" + price_change["month"].astype(str) + "_W" + price_change["week"].astype(str)

# convert price_change into wide format
price_change = price_change.pivot_table(index = "timestamp", columns = "symbol")["adjclose_weekly_change"].dropna(axis = 1).reset_index()

# report progress
run_time1 = default_timer() - start_time
print("- Weekly price changes collected in " + str(np.round(run_time1, 3)) + " seconds")

# reduce the dimensionality of price_change into 10 numerical features
price_change_arch_features, price_change_glrm_mod, price_change_glrm_error, price_change_wape, price_change_arch_samples = low_rank(data_frame = price_change, 
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
del price_change_arch_features, price_change_glrm_mod, price_change_glrm_error, price_change_wape

# remove anomalies from price_change_arch_samples
price_change_arch_samples, irf, irf_pred = isolate(data_frame = price_change_arch_samples, 
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

# standardize price_change_arch_samples
price_change_arch_samples_std = feature_engineering.standardize.run(data_frame = price_change_arch_samples, 
                                                                    ignore_features = ["Feature"], 
                                                                    prefix = "std_", 
                                                                    display_progress = True)

# cluster price_change_arch_samples_std with kmeans
price_change_kmeans, km_mod, km_score, silhouette_kmeans, price_change_distance = kmeans(data_frame = price_change_arch_samples_std, 
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
del km_mod

# cluster price_change_arch_samples_std with bayesian gaussian mixture
price_change_gmix, gmix_mod, gmix_score, silhouette_gmix, distance = gmix(data_frame = price_change_arch_samples_std, 
                                                                          grid_searches = 2, 
                                                                          grid_size = 50, 
                                                                          mutation_size = 7, 
                                                                          mutation = 0.2, 
                                                                          clusters = [5, 10, 15], 
                                                                          covariance_type = ["full", "tied", "diag", "spherical"], 
                                                                          weight_concentration_prior = [0.05, 0.1, 0.2],
                                                                          mean_precision_prior = [0.5, 1, 1.5], 
                                                                          iterations = 100, 
                                                                          initializations = 5, 
                                                                          cpu_fraction = 1, 
                                                                          ram_fraction = 1, 
                                                                          ignore_features = ["Feature"], 
                                                                          seed = 48, 
                                                                          display_progress = True)
del gmix_mod, distance

# cluster price_change with optics
price_change_optics, optics_mod, optics_score, silhouette_optics, distance = optics(data_frame = price_change_arch_samples_std, 
                                                                                    grid_searches = 2, 
                                                                                    grid_size = 50, 
                                                                                    mutation_size = 5, 
                                                                                    mutation = 0.2, 
                                                                                    sample_fraction = [0.001, 0.02, 0.05, 0.1], 
                                                                                    min_steepness = [0.001, 0.002, 0.003], 
                                                                                    cpu_fraction = 1, 
                                                                                    ram_fraction = 1, 
                                                                                    ignore_features = ["Feature"], 
                                                                                    seed = 48, 
                                                                                    display_progress = True)
del optics_mod, distance

# cluster price_change_arch_samples_std with random forest
price_change_rf, rf_mod, price_change_importance = leafs(data_frame = price_change_arch_samples_std, 
                                                         trees = 50, 
                                                         min_samples = 5, 
                                                         max_depth = 14, 
                                                         min_split_loss = 0, 
                                                         cpu_fraction = 1, 
                                                         ignore_features = ["Feature"], 
                                                         seed = 18, 
                                                         display_progress = True)
del rf_mod

# update the column names of clusterings
price_change_kmeans.columns = ["symbol", "KM_Cluster"]
price_change_gmix.columns = np.append(["symbol", "GM_Cluster"], price_change_gmix.columns[2:])
price_change_optics.columns = ["symbol", "O_Cluster"]
price_change_rf.columns = np.append(["symbol"], price_change_rf.columns[1:])

# check out the mean silhouette coefficients
np.mean(silhouette_kmeans["Silhouette_Coefficient"])
np.mean(silhouette_gmix["Silhouette_Coefficient"])
np.mean(silhouette_optics["Silhouette_Coefficient"])

# combine clusterings (except for optics)
price_change_clusters = pd.concat([price_change_kmeans, 
                                   price_change_gmix[["GM_Cluster"]]], axis = 1)

# convert price_change_clusters into binaries
price_change_clusters = feature_engineering.dummies.run(data_frame = price_change_clusters, 
                                                        ignore_features = ["symbol"], 
                                                        display_progress = True)

# transpose price_change_clusters to create an incidence matrix
price_change_incidence = price_change_clusters.drop(columns = ["symbol"]).T
price_change_incidence.columns = price_change_clusters["symbol"]

# cluster the incidence matrix using connected components
price_change_cc, price_change_graph, price_change_adjacency = components(incidence_data_frame = price_change_incidence, 
                                                                         ignore_features = None, 
                                                                         display_progress = True)

# compute the silhouette coefficients for connected components
silhouette_cc = pd.DataFrame({"Silhouette_Coefficient": silhouette_samples(X = price_change_distance, labels = np.array(price_change_cc["Cluster"]), metric = "precomputed")})
np.mean(silhouette_cc["Silhouette_Coefficient"])

# rename the columns in price_change_cc
price_change_cc.columns = np.append(["CC_Cluster"], price_change_cc.columns[1:])

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
labels = {i : price_change_cc.index.values[i] for i in price_change_graph.nodes()}

# get the positions
kk_positions = {i : np.array(price_change_cc[["KK_x_coordinate", "KK_y_coordinate"]].iloc[i]) for i in price_change_graph.nodes()}
fr_positions = {i : np.array(price_change_cc[["FR_x_coordinate", "FR_y_coordinate"]].iloc[i]) for i in price_change_graph.nodes()}
s_positions = {i : np.array(price_change_cc[["S_x_coordinate", "S_y_coordinate"]].iloc[i]) for i in price_change_graph.nodes()}

# get the node colors
np.random.shuffle(color_list)
km_colors = [color_list[i] for i in price_change_kmeans["KM_Cluster"]]
gm_colors = [color_list[i] for i in price_change_gmix["GM_Cluster"]]
cc_colors = [color_list[i] for i in price_change_cc["CC_Cluster"]]

# plot the Kamada-Kawai graph
plt.figure()
nx.draw_networkx(G = price_change_graph, 
                 pos = kk_positions,
                 labels = labels, 
                 node_size = 1200, 
                 node_color = gm_colors, 
                 width = 1)

# plot the Fruchterman-Reingold graph
plt.figure()
nx.draw_networkx(G = price_change_graph, 
                 pos = fr_positions,
                 labels = labels, 
                 node_size = 1200, 
                 node_color = gm_colors, 
                 width = 1)

# plot the Spectral graph
plt.figure()
nx.draw_networkx(G = price_change_graph, 
                 pos = s_positions,
                 labels = labels, 
                 node_size = 1200, 
                 node_color = gm_colors, 
                 width = 1)

# compute the communities of graph
communities = nx.algorithms.community.k_clique_communities(G = price_change_graph, k = 5)
communities = nx.algorithms.community.label_propagation_communities(G = price_change_graph)
communities = nx.algorithms.community.asyn_lpa_communities(G = price_change_graph, seed = 42)
communities = nx.algorithms.community.greedy_modularity_communities(G = price_change_graph)
communities = [list(c) for c in communities]

# create clusters from communities
community_clusters = pd.DataFrame({"Cluster": np.repeat(0, price_change_incidence.shape[1])})

# update Cluster
if len(communities) > 1:
    for c in range(1, len(communities)):
        for k in communities[c]:
            community_clusters.at[k, "Cluster"] = c

# compute the colors of graph
colors = nx.coloring.greedy_color(G = price_change_graph, strategy = "largest_first")
colors = pd.DataFrame(list(colors.items()), columns = ["Node", "Color"])

# sort colors by Node and then remove it
colors = colors.sort_values(by = "Node", ascending = True).reset_index(drop = True)
colors = colors.drop(columns = ["Node"])



if clean_work_space:
    
    # clean out the garbage in RAM
    h2o.remove_all()
    h2o.api("POST /3/GarbageCollect")
    gc.collect()
    
    # shutdown the h2o instance
    h2o.cluster().shutdown()

    # reset the work enviornment
    %reset -f