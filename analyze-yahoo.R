# -----------------------------------------------------------------------------------
# ---- Input Information ------------------------------------------------------------
# -----------------------------------------------------------------------------------

# set the path for the yahoo prices to be placed
prices.wd = "C:/Users/Nick Morris/Downloads/Data-Science/Projects/Stock-Market-Portfolio-Optimization/Models/Portfolio/prices"

# set the path for the yahoo.csv to be placed
yahoo.wd = "C:/Users/Nick Morris/Downloads/Data-Science/Projects/Stock-Market-Portfolio-Optimization/Models/Portfolio"

# set the path for the mad.csv to be placed
mad.wd = "C:/Users/Nick Morris/Downloads/Data-Science/Projects/Stock-Market-Portfolio-Optimization/Models/Portfolio/mad-results"

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

# disable the yahoo warning message
options("getSymbols.yahoo.warning" = FALSE)
options("getSymbols.warning4.0" = FALSE)

# set the work directory
setwd(prices.wd)

# extract all of the known symbols from stockSymbols
symbols = as.character(sort(unique(stockSymbols()$Symbol)))

# do we need to get the company names?
get.names = FALSE

if(get.names)
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
    
    # disable the yahoo warning message
    options("getSymbols.yahoo.warning" = FALSE)
    
    # get symbol s
    s = symbols[i]
    
    # extract the closing price for symbol s
    close = data.frame(na.omit(Cl(getSymbols(s, src = "yahoo", auto.assign = FALSE))))
    
    # if close was not produced properly then return a dummy table
    if(is.null(dim(na.omit(close))))
    {
      close = data.table(close = NA, date = NA, symbol = s)
      fwrite(x = close, file = paste0(s, ".csv"))
      
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
      
      fwrite(x = close, file = paste0(s, ".csv"))
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
  
  # get all price file names (while ignoring myfile)
  price.files = list.files()
  price.files = price.files[-which(price.files == myfile)]
  
  # read in the files
  prices = lapply(price.files, function(p) fread(p))
  
  # convert prices into one table
  prices = rbindlist(prices)
  
  # set the work directory for yahoo.csv
  setwd(yahoo.wd)
  
  # save this vector of symbols
  fwrite(prices, "yahoo.csv")
  
  # set the work directory
  setwd(prices.wd)
  
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
  
  # set the work directory for yahoo.csv
  setwd(yahoo.wd)
  
  # save this table
  fwrite(names, "getName.csv")
  
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
  
  # compute percent change in prices day to day for each symbol
  prices = prices[, .(date = date,
                      close = close,
                      change = abs((close / shift(close, 1)) - 1)),
                  by = .(symbol, name)]
  
  # remove any days that had a change larger than 50%
  prices = prices[change <= 0.50]
  
  # compute percent change in prices day to day for each symbol
  prices = prices[, .(date = date,
                      close = close,
                      change = abs((close / shift(close, 1)) - 1)),
                  by = .(symbol, name)]
  
  # remove any days that had a change larger than 50%
  prices = prices[change <= 0.50]
  
  # compute percent change in prices day to day for each symbol
  prices = prices[, .(date = date,
                      close = close,
                      change = abs((close / shift(close, 1)) - 1)),
                  by = .(symbol, name)]
  
  # remove any days that had a change larger than 50%
  prices = prices[change <= 0.50]
  
  # compute percent change in prices day to day for each symbol
  prices = prices[, .(date = date,
                      close = close,
                      change = abs((close / shift(close, 1)) - 1)),
                  by = .(symbol, name)]
  
  # remove any days that had a change larger than 50%
  prices = prices[change <= 0.50]
  
  # compute percent change in prices day to day for each symbol
  prices = prices[, .(date = date,
                      close = close,
                      change = abs((close / shift(close, 1)) - 1)),
                  by = .(symbol, name)]
  
  # remove any days that had a change larger than 50%
  prices = prices[change <= 0.50]
  
  # remove change
  prices = prices[, change := NULL]
  
  # save this table
  fwrite(prices, "yahoo.csv")
  
} else
{
  prices = fread("yahoo.csv")
}

# so we need to compute log differences?
compute.logs = FALSE

if(compute.logs)
{
  # get the year from date
  prices[, year := as.numeric(substr(date, 1, 4))]
  
  # lets loop through each company and compute the logDiff and meanLogDiff
  # create a name for a .txt file to log progress information while parallel processing
  myfile = "log.txt"
  file.create(myfile)
  
  # choose the number of workers and tasks for parallel processing
  workers = max(1, floor((3/4) * detectCores()))
  tasks = length(unique(prices$name))
  
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
    cat("computing log return\n")
    cat(paste(workers, "workers started at", Sys.time(), "\n"))
    sink()
    
  } else
  {
    # define %do%
    `%fun%` = `%do%`
    
    # write out start time to log file
    sink(myfile, append = TRUE)
    cat("\n------------------------------------------------\n")
    cat("computing log return\n")
    cat(paste("task 1 started at", Sys.time(), "\n"))
    sink()
  }
  
  # compute logDiff and meanLogDiff
  prices.update = foreach(i = 1:tasks) %fun%
  {
    # load packages
    require(data.table)
    
    # get the company name
    company = sort(unique(prices$name))[i]
    
    # get the data for company
    DT = data.table(prices[name == company])
    
    # compute logDiff
    DT[, logDiff := c(NA, diff(log(close)))]
    
    # compute meanLogDiff
    DT.avg = data.table(DT[, .(meanLogDiff = mean(logDiff, na.rm = TRUE)), by = .(year)])
    
    # give DT an id column to maintain its original order
    DT[, id := 1:nrow(DT)]
    
    # join DT.avg onto DT
    setkey(DT.avg, year)
    setkey(DT, year)
    DT = DT.avg[DT]
    
    # order DT by id
    DT = DT[order(id)]
    
    # remove id
    DT[, id := NULL]
    
    # free memory
    gc()
    
    # export progress information
    sink(myfile, append = TRUE)
    cat(paste("task", i, "of", tasks, "finished at", Sys.time(), "\n"))
    sink()
    
    return(DT)
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
  
  # convert prices.update into one table
  prices = na.omit(rbindlist(prices.update))
  
  # write out yahoo
  fwrite(prices, "yahoo.csv")
}

# do we need to reduce the number of symbols?
reduce.symbols = FALSE

if(reduce.symbols)
{
  # convert date into a date type
  prices[, date := as.Date(date)]
  
  # order by symbol
  prices = prices[order(symbol)]
  
  # find the start and end date of every symbol and name to identify any inconsistencies
  check = data.table(prices[,.(start = min(date), end = max(date)), by = .(name, symbol)])
  
  # count how many symbols that belong to each name becuase there are more symbols than names currently
  check[, count := sapply(1:nrow(check), function(i) length(unique(unname(unlist(check[name == check$name[i], .(symbol)])))))]
  
  # extract the subset of check that has repeated names
  check.name = data.table(check[count > 1])
  
  # compute the number of days that each name has with each symbol
  check.name[, days := as.numeric(difftime(end, start, units = "days"))]
  
  # get all unique names from check.name
  names = unique(check.name$name)
  
  # loop through each name and only extract the symbol associated with the largest number of days
  keep.symbol = unname(sapply(names, function(n)
  {
    # find which symbol has the max number of days
    position = which.max(unlist(check.name[name == n, .(days)]))
    
    # get the symbol
    sym = unname(unlist(check.name[name == n, .(symbol)]))[position]  
    
    return(sym)
  }))
  
  # get the symbols to remove
  remove.symbol = unname(unlist(check.name[!(symbol %in% keep.symbol), .(symbol)]))
  
  # remove these symbols from prices
  prices = prices[!(symbol %in% remove.symbol)]
  
  # update the name column to have no spacing or punctuation and be all capitalized
  prices[, name2 := toupper(gsub(" ", "", removeNumbers(removePunctuation(name)), fixed = TRUE))]
  
  # find the start and end date of every symbol and name2 to identify any inconsistencies
  check = data.table(prices[,.(start = min(date), end = max(date)), by = .(name2, symbol)])
  
  # count how many symbols that belong to each name2 becuase there are more symbols than names currently
  check[, count := sapply(1:nrow(check), function(i) length(unique(unname(unlist(check[name2 == check$name2[i], .(symbol)])))))]
  
  # extract the subset of check that has repeated names
  check.name2 = data.table(check[count > 1])
  
  # compute the number of days that each name2 has with each symbol
  check.name2[, days := as.numeric(difftime(end, start, units = "days"))]
  
  # get all unique names from check.name2
  names = unique(check.name2$name2)
  
  # loop through each name2 and only extract the symbol associated with the largest number of days
  keep.symbol = unname(sapply(names, function(n)
  {
    # find which symbol has the max number of days
    position = which.max(unlist(check.name2[name2 == n, .(days)]))
    
    # get the symbol
    sym = unname(unlist(check.name2[name2 == n, .(symbol)]))[position]  
    
    return(sym)
  }))
  
  # get the symbols to remove
  remove.symbol = unname(unlist(check.name2[!(symbol %in% keep.symbol), .(symbol)]))
  
  # remove these symbols from prices
  prices = prices[!(symbol %in% remove.symbol)]
  
  # update the name column of prices
  prices[, name := name2]
  prices[, name2 := NULL]
  
  # update the date column to have no dashes
  prices[, date := gsub("-", "", date)]
  
  # write out yahoo
  fwrite(prices, "yahoo.csv")
  
  # remove unneded objects
  rm(check, check.name, check.name2, keep.symbol, remove.symbol, names)
}

# do we need to update the mean computations?
update.means = FALSE

if(update.means)
{
  # remove the means column
  prices[, meanLogDiff := NULL]
  
  # give prices an id column to maintain its original order
  prices[, id := 1:nrow(prices)]
  
  # compute annual averages of logDiff for each company
  avg = data.table(prices[, .(meanLogDiff = mean(logDiff)), by = .(symbol, year)])
  
  # join avg onto prices
  setkey(avg, symbol, year)
  setkey(prices, symbol, year)
  prices = avg[prices]
  
  # order prices by id
  prices = prices[order(id)]
  prices[, id := NULL]
  
  # write out yahoo
  fwrite(prices, "yahoo.csv")
  
  # remove unneded objects
  rm(avg)
}

}

# -----------------------------------------------------------------------------------
# ---- Export Yahoo Data for MAD LP -------------------------------------------------
# -----------------------------------------------------------------------------------

{

# should we export the yahoo data for the semi mad lp?
export.yahoo = FALSE

if(export.yahoo)
{
  # extract all of the years from prices
  yrs = unique(prices$year)
  
  # set the desired profit margins (ie. 0.3 = 30% annual return)
  margins = seq(0.1, 1, 0.05)
  
  # create all portfolio combinations
  ports = data.table(expand.grid(year = yrs, margin = margins))
  
  # write out an ampl data file for each year in prices
  lapply(1:nrow(ports), function(i)
  {
    # get the year from ports
    y = ports$year[i]
    
    # get the margin from ports
    m = ports$margin[i]
    
    # extract year y from prices
    dat = data.table(prices[year == y])
    
    # create a subset of dat with just name and meanLogDiff
    dat.mean = data.table(dat[,.(name, meanLogDiff)])
    
    # remove all duplicates in dat.mean
    dat.mean = dat.mean[!duplicated(dat.mean)]
    
    # write out the set of days for year y
    write.table(ampl(unique(dat$date), object = "set", name = "time"), 
                file = paste0("mad-yahoo-", y, "-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE)
    
    # write out the set of names for year y
    write.table(ampl(unique(dat$name), object = "set", name = "stocks"), 
                file = paste0("mad-yahoo-", y, "-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of diff(log(price)) for year y
    write.table(ampl(dat[,.(name, date, logDiff)], object = "param", name = "diffLog"), 
                file = paste0("mad-yahoo-", y, "-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of mean(diff(log(price))) for year y
    write.table(ampl(dat.mean, object = "param", name = "meanDiffLog"), 
                file = paste0("mad-yahoo-", y, "-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
    
    # write out the parameter of profit margin for year y
    write.table(ampl(log(m + 1), object = "param", name = "logMarginPlusOne"), 
                file = paste0("mad-yahoo-", y, "-", m * 100, ".dat"), 
                quote = FALSE,
                row.names = FALSE,
                append = TRUE)
  })
}

}

# -----------------------------------------------------------------------------------
# ---- Prepare MAD Results ----------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# set the work directory
setwd(mad.wd)

# set the years being evaluated
yrs = 2009:2017

# set the margins being evaluated
margins = seq(0.1, 1, 0.05)

# create all portfolio combinations
ports = data.table(expand.grid(year = yrs, margin = margins))

# import the results data from each work directory
dat = lapply(1:nrow(ports), function(i)
{
  # get the year from ports
  y = ports$year[i]
  
  # get the margin from ports
  m = ports$margin[i]
  
  # import the results file
  output = fread(paste0("mad-yahoo-", y, "-", m * 100, "-results.txt"), sep = ",")
  
  # rename the columns
  setnames(output, c("variable", "name", "value"))
  
  # add the year and margin
  output[, year := y]
  output[, margin := m * 100]
  
  return(output)
})

# combine the list of data tables into 1 table
dat = rbindlist(dat)

# export dat
setwd(yahoo.wd)
fwrite(dat, "mad.csv")

}











