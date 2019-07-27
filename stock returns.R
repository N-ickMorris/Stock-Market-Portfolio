# -----------------------------------------------------------------------------------
# ---- Functions --------------------------------------------------------------------
# -----------------------------------------------------------------------------------

{

# ---- prints the data types of each column in a data frame -------------------------

types = function(dat)
{
  require(data.table)
  
  # make dat into a data.table
  dat = data.table(dat)
  
  # get the column names
  column = names(dat)
  
  # get the class of the columns
  dataType = sapply(1:ncol(dat), function(i) class(unlist(dat[, i, with = FALSE])))
  
  # compute the number of levels for each column
  levels = sapply(1:ncol(dat), function(i) ifelse(dataType[i] == "factor", length(levels(droplevels(unlist(dat[, i, with = FALSE])))), 0))
  
  # compute the number of unique values for each column
  uniqueValues = sapply(1:ncol(dat), function(i) length(unique(unname(unlist(dat[, i, with = FALSE])))))
  
  # compute the portion of missing data
  missing = sapply(1:ncol(dat), function(i) nrow(na.omit(dat[, i, with = FALSE], invert = TRUE)) / nrow(dat))
  
  # build the output table 
  output = data.table(column, id = 1:length(column), dataType, levels, uniqueValues, missing)
  
  # order output by dataType
  output = output[order(dataType)]
  
  return(output)
}

# ---- converts all columns to a character data type --------------------------------

tochar = function(dat)
{
  require(data.table)
  
  # make dat into a data.frame
  dat = data.table(dat)
  
  # get the column names
  column = names(dat)
  
  # get the values in the columns and convert them to character data types
  values = lapply(1:ncol(dat), function(i) as.character(unname(unlist(dat[, i, with = FALSE]))))
  
  # combine the values back into a data.frame
  dat = data.table(do.call("cbind", values), stringsAsFactors = FALSE)
  
  # give dat its column names
  setnames(dat, column)
  
  return(dat)
}

# ---- a qualitative color scheme ---------------------------------------------------

qcolor = function(n, a = 1)
{
  require(grDevices)
  require(scales)
  return(alpha(colorRampPalette(c("#e41a1c", "#0099ff", "#4daf4a", "#984ea3", "#ff7f00", "#ff96ca", "#a65628"))(n), 
               a))
}

# ---- the ggplot2 color scheme -----------------------------------------------------

ggcolor = function(n, a = 1)
{
  require(grDevices)
  require(scales)
  return(alpha(hcl(h = seq(15, 375, length = n + 1), 
                   l = 65, c = 100)[1:n], 
               a))
}

}

# -----------------------------------------------------------------------------------
# ---- Script -----------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# load packages
require(ggplot2)
require(scales)
require(data.table)
require(quantmod)
# require(plot3D)

# this function emulates the default ggplot2 color scheme
ggcolor = function(n, alpha = 1)
{
  require(scales)
  hues = seq(15, 375, length = n + 1)
  alpha(hcl(h = hues, l = 65, c = 100)[1:n], alpha)
}

# this function prints the data types of each column
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

# disable the yahoo warning message
options("getSymbols.yahoo.warning" = FALSE)
options("getSymbols.warning4.0" = FALSE)

# open a graphics window
windows()

# only keep closing prices at the end of the week, every 2 weeks, for the past 4 years
keep.dates = seq.Date(from = as.Date("2012-10-12"), to = as.Date("2018-10-12"), by = "2 weeks")

# get stock prices for a set of companies
stock.prices = lapply(c("FB", "AMZN", "AAPL", "NFLX", "GOOGL", "MSFT", "BSX", "GDOT", "NOVT"), function(s)
{
  # get the closing prices for symbol s 
  DT = data.frame(Cl(getSymbols(s, src = "yahoo", auto.assign = FALSE)))
  
  # make the row names into a Date column for DT
  DT = cbind("Date" = as.Date(row.names(DT)), data.table(DT))
  
  # rename the columns of DT
  setnames(DT, c("Date", "Close"))
  
  # make a Symbol column for DT
  DT = cbind("Symbol" = s, DT)
  
  # only keep the dates of interest
  DT = DT[Date %in% keep.dates]
  
  # compute the difference between log prices
  DT[, diffLog := c(NA, diff(log(Close)))]
  
  # remove missing values from DT
  DT = na.omit(DT)
  
  # compute a rolling sum of diffLog at each date of interest
  rolling.sum = sapply(1:nrow(DT), function(i) sum(DT$diffLog[1:i]))
  
  # convert the sum(diff(log(Close))) into percent return values
  rolling.return = exp(rolling.sum) - 1
  
  # add return values to DT
  DT[, Return := rolling.return]
  
  # return DT without missing values
  return(na.omit(DT))
})

# combine the list of tables into one table
stock.prices = rbindlist(stock.prices)

# make Symbol into a factor
stock.prices[, Symbol := factor(Symbol, levels = c("FB", "AMZN", "AAPL", "NFLX", "GOOGL", "MSFT", "BSX", "GDOT", "NOVT"))]

# plot Close
close.plot = ggplot(data = stock.prices, aes(x = Date, y = Close, color = Symbol)) +
  # geom_point() + 
  geom_line(size = 2) + 
  scale_y_continuous(label = dollar_format()) +
  ggtitle("Closing Prices: Oct-2012 to Oct-2018") + 
  labs(x = "Years\n(Bi-Weekly Intervals)", y = "Friday Closing Price", color = "") + 
  theme_bw(25) + 
  theme(legend.position = "top", 
        legend.key.size = unit(0.25, "in"), 
        plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  guides(color = guide_legend(override.aes = list(size = 20, linetype = 1, alpha = 1), nrow = 1))

close.plot

# plot Return
return.plot = ggplot(data = stock.prices, aes(x = Date, y = Return, color = Symbol)) +
  # geom_point() + 
  geom_line(size = 2) + 
  scale_y_continuous(label = percent) +
  ggtitle("Return on Investment: Oct-2012 to Oct-2018") + 
  labs(x = "Years\n(Bi-Weekly Intervals)", y = "Return Since Oct-2012", color = "") + 
  theme_bw(25) + 
  theme(legend.position = "top", 
        legend.key.size = unit(0.25, "in"), 
        plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  guides(color = guide_legend(override.aes = list(size = 20, linetype = 1, alpha = 1), nrow = 1))

return.plot

# plot Return
return.plot = ggplot(data = stock.prices[Symbol != "NFLX"], aes(x = Date, y = Return, color = Symbol)) +
  # geom_point() + 
  geom_line(size = 2) + 
  scale_y_continuous(label = percent) +
  ggtitle("Return on Investment: Oct-2012 to Oct-2018") + 
  labs(x = "Years\n(Bi-Weekly Intervals)", y = "Return Since Oct-2012", color = "") + 
  theme_bw(25) + 
  theme(legend.position = "top", 
        legend.key.size = unit(0.25, "in"), 
        plot.title = element_text(hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  guides(color = guide_legend(override.aes = list(size = 20, linetype = 1, alpha = 1), nrow = 1))

return.plot
