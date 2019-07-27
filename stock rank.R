# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path of where the input files are
mywd = list("D:/Data Science/Stocks")

# -----------------------------------------------------------------------------------
# ---- Packages ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# data handling
require(data.table)
require(tm)

# plotting
require(ggplot2)
require(gridExtra)
require(GGally)
require(scales)

# stock data
require(TTR)
require(quantmod)
require(FinancialInstrument)

# parallel computing
require(foreach)
require(parallel)
require(doSNOW)

}

# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# these are functions i like to use

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  # make dat into a data.frame
  dat = data.frame(dat)
  
  # get the column names
  column = colnames(dat)
  
  # get the class of the columns
  data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))
  
  # compute the number of levels for each column
  levels = sapply(1:ncol(dat), function(i) ifelse(data.type[i] == "factor", length(levels(droplevels(dat[,i]))), 0))
  
  return(data.frame(column, data.type, levels))
}

# ---- converts all columns to a character data type --------------------------------

tochar = function(dat)
{
  # make dat into a data.frame
  dat = data.frame(dat)
  
  # get the column names
  column = colnames(dat)
  
  # get the values in the columns and convert them to character data types
  values = lapply(1:ncol(dat), function(i) as.character(dat[,i]))
  
  # combine the values back into a data.frame
  dat = data.frame(do.call("cbind", values), stringsAsFactors = FALSE)
  
  # give dat its column names
  colnames(dat) = column
  
  return(dat)
}

# ---- gets the name of a stock based on its symbol ---------------------------------

getName = function(s)
{
  require(FinancialInstrument)
  
  return(tryCatch(
    {
      stock(s, currency("USD"))
      update_instruments.TTR(s)
      return(as.character(getInstrument(s)$Name))
      
    }, error = function(e) s))
}

# ---- prints out a dat file object in ampl syntax ----------------------------------

ampl = function(dat, object = "param", name = "c")
{
  # converts all columns to a character data type 
  tochar = function(dat)
  {
    # make dat into a data.frame
    dat = data.frame(dat)
    
    # get the column names
    column = colnames(dat)
    
    # get the values in the columns and convert them to character data types
    values = lapply(1:ncol(dat), function(i) as.character(dat[,i]))
    
    # combine the values back into a data.frame
    dat = data.frame(do.call("cbind", values), stringsAsFactors = FALSE)
    
    # give dat its column names
    colnames(dat) = column
    
    return(dat)
  }
  
  # make sure the data is a data frame object
  dat = tochar(dat)
  
  # every parameter/set object in an ampl dat file must end with a semicolon
  # so set up 1 semicolon to give to dat
  semicolon = c(";", rep(" ", ncol(dat) - 1))
  
  # add this semicolon as the last row of the data frame
  result = data.frame(rbind(dat, semicolon))
  
  # every parameter/set object in an ample dat file must begin with the name of the object and what it equals
  # for example: param c := 
  # so set up a header to give to dat
  header = c(paste(object, name, ":="), rep(" ", ncol(dat) - 1))
  
  # update the column names of dat to be the header we created
  colnames(result) = header
  
  # print out the result without any row names
  # print out the result left adjusted
  # print(result, right = FALSE, row.names = FALSE)
  
  return(result)	
}

}

# -----------------------------------------------------------------------------------
# ---- Prepare Yahoo Data -----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# set the work directory
setwd(mywd[[1]])

# do we need to clean up the symbols?
clean.symbols = FALSE

if(clean.symbols)
{
  # extract all of the known symbols from stockSymbols
  symbols = as.character(sort(unique(stockSymbols()$Symbol)))
  
  # lets find out which symbols throw errors
  count = 0
  pass = FALSE
  symbol_found = TRUE
  while(pass == FALSE)
  {
    # remove failing symbol
    if(symbol_found == FALSE)
    {
      symbols = symbols[-count]
      count = count - 1
    }
    
    # increment count
    count = count + 1
    
    # report progress
    print(paste("Evaluating symbol", count, "of", length(symbols)))
    
    # get symbol s
    s = symbols[count]
    
    # see if symbol s can be found
    symbol_found = tryCatch({is.numeric(Cl(getSymbols(s, src = "yahoo", auto.assign = FALSE)))}, 
                            error = function(e) FALSE)
    
    # if count reached its max then all passing symbols have been found
    if(count == length(symbols))
    {
      pass = TRUE
    }
  }
  
  # save this vector of symbols
  write.csv(data.table(symbols), "getSymbols.csv", row.names = FALSE)
  
} else
{
  # import symbols
  symbols = as.character(unlist(data.table(read.csv("getSymbols.csv"))))
}

# do we need to get the company prices?
get.prices = TRUE

if(get.prices)
{
  # create a name for a .txt file to log progress information while parallel processing
  myfile = "log.txt"
  file.create(myfile)
  
  # choose the number of workers and tasks for parallel processing
  workers = max(1, floor((3/4) * detectCores()))
  tasks = length(symbols)
  
  # set up a cluster if workers > 1, otherwise don't set up a cluster
  if(workers > 1)
  {
    # setup parallel processing
    cl = makeCluster(workers, type = "SOCK", outfile = "")
    registerDoSNOW(cl)
    
    # define %dopar%
    `%fun%` = `%dopar%`
    
    # write out start time to log file
    sink(myfile, append = TRUE)
    cat("\n------------------------------------------------\n")
    cat("preparing stock data\n")
    cat(paste(workers, "workers started at", Sys.time(), "\n"))
    sink()
    
  } else
  {
    # define %do%
    `%fun%` = `%do%`
    
    # write out start time to log file
    sink(myfile, append = TRUE)
    cat("\n------------------------------------------------\n")
    cat("preparing stock data\n")
    cat(paste("task 1 started at", Sys.time(), "\n"))
    sink()
  }
  
  # use symbols to get intrday prices
  prices = foreach(i = 1:tasks) %fun%
  {
    # load packages
    require(data.table)
    require(quantmod)
    
    # get symbol s
    s = symbols[i]
    
    # extract the closing price for symbol s
    close = data.frame(na.omit(Cl(getSymbols(s, src = "yahoo", auto.assign = FALSE))))
    
    # if close was not produced properly then return a dummy table
    if(is.null(dim(na.omit(close))))
    {
      close = data.table(close = NA, date = NA, symbol = s)
      
    } else
    {
      # get the dates
      dates = as.Date(row.names(close))
      
      # convert close to a data table
      close = data.table(close)
      
      # add date and symbol as columns
      close[, date := dates]
      close[, symbol := s]
      setnames(close, c("close", "date", "symbol"))
    }
    
    # free memory
    gc()
    
    # export progress information
    sink(myfile, append = TRUE)
    cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
    sink()
    
    return(close)
  }
  
  # write out end time to log file
  sink(myfile, append = TRUE)
  cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
  sink()
  
  # end the cluster if it was set up
  if(workers > 1)
  {
    stopCluster(cl)
  }
  
  # convert prices into one table
  prices = rbindlist(prices)
  
  # save this vector of symbols
  write.csv(prices, "yahoo.csv", row.names = FALSE)
  
  # choose the number of workers and tasks for parallel processing
  workers = max(1, floor((3/4) * detectCores()))
  tasks = length(symbols)
  
  # set up a cluster if workers > 1, otherwise don't set up a cluster
  if(workers > 1)
  {
    # setup parallel processing
    cl = makeCluster(workers, type = "SOCK", outfile = "")
    registerDoSNOW(cl)
    
    # define %dopar%
    `%fun%` = `%dopar%`
    
    # write out start time to log file
    sink(myfile, append = TRUE)
    cat("\n------------------------------------------------\n")
    cat("getting stock names\n")
    cat(paste(workers, "workers started at", Sys.time(), "\n"))
    sink()
    
  } else
  {
    # define %do%
    `%fun%` = `%do%`
    
    # write out start time to log file
    sink(myfile, append = TRUE)
    cat("\n------------------------------------------------\n")
    cat("getting stock names\n")
    cat(paste("task 1 started at", Sys.time(), "\n"))
    sink()
  }
  
  # use symbols to get names
  names = foreach(i = 1:tasks, .combine = "c") %fun%
  {
    # load packages
    require(data.table)
    require(FinancialInstrument)
    
    # get symbol s
    s = symbols[i]
    
    # lets find the full name of symbol s
    x = getName(s)
    
    # export progress information
    sink(myfile, append = TRUE)
    cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
    sink()
    
    return(x)
  }
  
  # write out end time to log file
  sink(myfile, append = TRUE)
  cat(paste(tasks, "tasks finished at", Sys.time(), "\n"))
  sink()
  
  # end the cluster if it was set up
  if(workers > 1)
  {
    stopCluster(cl)
  }
  
  # make names into a data table
  names = data.table(symbol = symbols, name = names)
  
  # save this table
  write.csv(names, "getName.csv", row.names = FALSE)
  
  # add an id column to preserve row order
  prices[, id := 1:nrow(prices)]
  
  # join names onto prices
  setkey(prices, symbol)
  setkey(names, symbol)
  prices = names[prices]
  
  # order prices by id
  prices = prices[order(id)]
  
  # remove id as a column
  prices[, id := NULL]
  
  # save this table
  write.csv(prices, "yahoo.csv", row.names = FALSE)
  
} else
{
  prices = data.table(read.csv("yahoo.csv", stringsAsFactors = FALSE))
}

# remove year, logDiff, and meanLogDiff
prices[, year := NULL]
prices[, logDiff := NULL]
prices[, meanLogDiff := NULL]

# update date
prices[, DATE := as.Date(paste(substr(date, 1, 4), substr(date, 5, 6), substr(date, 7, 8), sep = "-"))]

# only keep closing prices every 2 weeks
keep.dates = seq.Date(from = min(prices$DATE), to = max(prices$DATE), by = "2 weeks")

# remove the last date in keep.dates
keep.dates = head(keep.dates, length(keep.dates) - 1)

# only keep the dates of interest
prices = prices[DATE %in% keep.dates]

# split up prices into: all-year, 8-year, and 5-year data sets
date.8 = max(prices$DATE) - (365 * 8)
prices.8 = data.table(prices[DATE >= date.8])

date.5 = max(prices$DATE) - (365 * 5)
prices.5 = data.table(prices[DATE >= date.5])

# remove DATE from price tables
prices[, DATE := NULL]
prices.8[, DATE := NULL]
prices.5[, DATE := NULL]

# compute log differences for prices
prices.update = lapply(unique(prices$name), function(i)
{
  # get the data for company i
  DT = data.table(prices[name == i])
  
  # compute logDiff
  DT[, logDiff := c(NA, diff(log(close)))]
  
  # compute meanLogDiff
  DT[, meanLogDiff := mean(logDiff, na.rm = TRUE)]
  
  return(na.omit(DT))
})

# combine the list of tables back into one table
prices = rbindlist(prices.update)

# compute log differences for prices.8
prices.8.update = lapply(unique(prices.8$name), function(i)
{
  # get the data for company i
  DT = data.table(prices.8[name == i])
  
  # compute logDiff
  DT[, logDiff := c(NA, diff(log(close)))]
  
  # compute meanLogDiff
  DT[, meanLogDiff := mean(logDiff, na.rm = TRUE)]
  
  return(na.omit(DT))
})

# combine the list of tables back into one table
prices.8 = rbindlist(prices.8.update)

# compute log differences for prices.5
prices.5.update = lapply(unique(prices.5$name), function(i)
{
  # get the data for company i
  DT = data.table(prices.5[name == i])
  
  # compute logDiff
  DT[, logDiff := c(NA, diff(log(close)))]
  
  # compute meanLogDiff
  DT[, meanLogDiff := mean(logDiff, na.rm = TRUE)]
  
  return(na.omit(DT))
})

# combine the list of tables back into one table
prices.5 = rbindlist(prices.5.update)

}

# -----------------------------------------------------------------------------------
# ---- Export Yahoo Data for MAD LP -------------------------------------------------
# -----------------------------------------------------------------------------------

{

# all
margins = c(400, 800, 1400, 2000) / 100

# 8
margins.8 = c(200, 400, 700, 1000) / 100

# 5
margins.5 = c(100, 200, 350, 500) / 100

# should we export the yahoo data for the mad lp?
export.yahoo = FALSE

if(export.yahoo)
{
  # set the work directory
  setwd(mywd[[1]])
  
  # write out an ampl data file for each year in prices
  lapply(margins, function(m)
  {
    # extract year y from prices
    dat = data.table(prices)
    
    # create a subset of dat with just name and meanLogDiff
    dat.mean = data.table(dat[,.(name, meanLogDiff)])
    
    # remove all duplicates in dat.mean
    dat.mean = dat.mean[!duplicated(dat.mean)]
    
    # write out the set of days for margin m
    write.table(ampl(unique(dat$date), object = "set", name = "time"), 
                file = paste0("mad-yahoo-all-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE)
    
    # write out the set of names for margin m
    write.table(ampl(unique(dat$name), object = "set", name = "stocks"), 
                file = paste0("mad-yahoo-all-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of diff(log(price)) for margin m
    write.table(ampl(dat[,.(name, date, logDiff)], object = "param", name = "diffLog"), 
                file = paste0("mad-yahoo-all-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of mean(diff(log(price))) for margin m
    write.table(ampl(dat.mean, object = "param", name = "meanDiffLog"), 
                file = paste0("mad-yahoo-all-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of profit margin for margin m
    write.table(ampl(log(m + 1), object = "param", name = "logMarginPlusOne"), 
                file = paste0("mad-yahoo-all-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
  })
  
  # write out an ampl data file for each year in prices.8
  lapply(margins.8, function(m)
  {
    # extract year y from prices
    dat = data.table(prices.8)
    
    # create a subset of dat with just name and meanLogDiff
    dat.mean = data.table(dat[,.(name, meanLogDiff)])
    
    # remove all duplicates in dat.mean
    dat.mean = dat.mean[!duplicated(dat.mean)]
    
    # write out the set of days for margin m
    write.table(ampl(unique(dat$date), object = "set", name = "time"), 
                file = paste0("mad-yahoo-8-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE)
    
    # write out the set of names for margin m
    write.table(ampl(unique(dat$name), object = "set", name = "stocks"), 
                file = paste0("mad-yahoo-8-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of diff(log(price)) for margin m
    write.table(ampl(dat[,.(name, date, logDiff)], object = "param", name = "diffLog"), 
                file = paste0("mad-yahoo-8-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of mean(diff(log(price))) for margin m
    write.table(ampl(dat.mean, object = "param", name = "meanDiffLog"), 
                file = paste0("mad-yahoo-8-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of profit margin for margin m
    write.table(ampl(log(m + 1), object = "param", name = "logMarginPlusOne"), 
                file = paste0("mad-yahoo-8-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
  })
  
  # write out an ampl data file for each year in prices.5
  lapply(margins.5, function(m)
  {
    # extract year y from prices
    dat = data.table(prices.5)
    
    # create a subset of dat with just name and meanLogDiff
    dat.mean = data.table(dat[,.(name, meanLogDiff)])
    
    # remove all duplicates in dat.mean
    dat.mean = dat.mean[!duplicated(dat.mean)]
    
    # write out the set of days for margin m
    write.table(ampl(unique(dat$date), object = "set", name = "time"), 
                file = paste0("mad-yahoo-5-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE)
    
    # write out the set of names for margin m
    write.table(ampl(unique(dat$name), object = "set", name = "stocks"), 
                file = paste0("mad-yahoo-5-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of diff(log(price)) for margin m
    write.table(ampl(dat[,.(name, date, logDiff)], object = "param", name = "diffLog"), 
                file = paste0("mad-yahoo-5-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of mean(diff(log(price))) for margin m
    write.table(ampl(dat.mean, object = "param", name = "meanDiffLog"), 
                file = paste0("mad-yahoo-5-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of profit margin for margin m
    write.table(ampl(log(m + 1), object = "param", name = "logMarginPlusOne"), 
                file = paste0("mad-yahoo-5-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
  })
  
}

}

# -----------------------------------------------------------------------------------
# ---- Analyze MAD LP Results -------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# set up the names of the result files
results.files = c("mad-yahoo-5-100-results.txt", "mad-yahoo-5-200-results.txt", "mad-yahoo-5-350-results.txt", "mad-yahoo-5-500-results.txt",
                  "mad-yahoo-8-200-results.txt", "mad-yahoo-8-400-results.txt", "mad-yahoo-8-700-results.txt", "mad-yahoo-8-1000-results.txt", 
                  "mad-yahoo-all-400-results.txt", "mad-yahoo-all-800-results.txt", "mad-yahoo-all-1400-results.txt", "mad-yahoo-all-2000-results.txt")

# set up the year spans of the result files
yrs = c(rep(5, 4), rep(8, 4), rep(11, 4))

# set up the margins of the result files
mars = c(100, 200, 350, 500, 200, 400, 700, 1000, 400, 800, 1400, 2000)

# import results
results = lapply(1:length(results.files), function(r)
{
  # read in result file r
  DT = data.table(read.delim(results.files[r], sep = ",", stringsAsFactors = FALSE, row.names = NULL))
  
  # rename the columns of DT
  setnames(DT, c("variable", "name", "value"))
  
  # add a yearSpan column
  DT[, yearSpan := yrs[r]]
  
  # add a margin column
  DT[, margin := mars[r]]
  
  return(DT)
})

# combine the list of tables into one table
results = rbindlist(results)
  
# only keep portion > 0
results = results[variable == "portion" & value > 0]

# order results by yearSpan, margin, and value
results = results[order(-yearSpan, -margin, -value)]

# give results an id column
results[, id := 1:nrow(results)]

# remove variable
results[, variable := NULL]

# get the symbol and names from prices
symbol.mapping = data.table(prices[,.(name, symbol)])

# remove duplicates from symbol.mapping
symbol.mapping = symbol.mapping[!duplicated(symbol.mapping)]

# remove the space before every name in results
results[, name := substr(results$name, 2, 1000)]

# join symbol.mapping onto results
setkey(results, name)
setkey(symbol.mapping, name)
results = symbol.mapping[results]

# compute the variance of the logDiff for each company in prices
log.var = data.table(prices[,.(name, logDiff)])
log.var = log.var[, .(varLogDiff = var(logDiff)), by = .(name)]

# join log.var onto results
setkey(results, name)
setkey(log.var, name)
results = log.var[results]

# order results by id
results = results[order(id)]

# compute the reciprical of varLogDiff
results[, recipVarLogDiff := 1 / varLogDiff]

# rescale value and recipVarLogDiff to have the same bounds
results[, scaleRVLD := rescale(recipVarLogDiff, to = c(1, 99))]
results[, scaleV := rescale(value, to = c(1, 99))]

# multiple scaleRVLD and scaleV to create an index for finding companies with low variances and high values
results[, index := scaleV * scaleRVLD]

# sum up the index by name to rank comapanies
index = data.table(results[,.(name, index, symbol)])
index = index[,.(rank = sum(index)), by = .(name, symbol)]

# order index by rank
index = index[order(-rank)]

# rescale rank
index[, rank := rescale(rank, to = c(1, 99))]

# get the last known price of each company from prices
last.price = data.table(prices)
last.price = last.price[, .(close = tail(close, 1), date = tail(date, 1)), by = .(name)]
last.price[, date := as.Date(paste(substr(date, 1, 4), substr(date, 5, 6), substr(date, 7, 8), sep = "-"))]

# join last.price onto index
setkey(last.price, name)
setkey(index, name)
index = last.price[index]

# order index by rank
index = index[order(-rank)]

# write out index
fwrite(index, "stock rank.csv")

# check out the topp 100 companies that cost less than $125 per share
head(index[close < 125], 100)

}

