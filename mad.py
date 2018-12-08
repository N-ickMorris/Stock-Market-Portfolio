# this model decides how your total investment should be portioned amoung different stocks to create a portfolio that is expected to acheive a minimum profit margin at minimal risk
# risk is defined by how much a stocks return moves above/below its expected return over time (ie. volitility)

# ---- setup the model ----

from pyomo.environ import *    # imports the pyomo environment
from pyomo.opt import SolverFactory    # imports the solver environment
from pyomo.core import Var    # imports the variable environment
import gc    # imports the garbage collection environment

model = AbstractModel()    # creates an abstract model
model.name = "Portfolio Selection LP"    # gives the model a name

# ---- define set(s) ----

model.time = Set()    # a set of time periods
model.stocks = Set()    # a set of stocks

# ---- define parameter(s) ----

model.diffLog = Param(model.stocks, model.time, initialize = 0, mutable = True)    # the observed value of diff(log('price')) for a stock at a time period (ie. observed log return)
model.meanDiffLog = Param(model.stocks, mutable = True)    # the expected value of diff(log('price')) for a stock during the current horizon (ie. expected log return)
model.logMarginPlusOne = Param()    # the log('minimum profit margin' + 1)

# ---- define variable(s) ----

model.portion = Var(model.stocks, domain = NonNegativeReals)    # the portion of your total investment put in a stock
model.risk = Var(model.time, domain = NonNegativeReals)    # the total amount of risk during a time period --> risk = how much every invested stock varies about their expected return (ie. volitility)

# ---- define objective function(s) ----

def obj(model):
    return sum(model.risk[t] for t in model.time)   # the total amount of risk during a horizon (ie. portfolio volitility)

model.obj = Objective(rule = obj, sense = minimize)    # a minimization problem of the function defined above

# ---- define constraint(s) ----

def NEGDEVIATION(model, t):
    return model.risk[t] >= sum((model.meanDiffLog[s] - model.diffLog[s,t]) * model.portion[s] for s in model.stocks)    # risk is how much the observed log return is less than the expected log return across all stocks

def POSDEVIATION(model, t):
    return model.risk[t] >= sum((model.diffLog[s,t] - model.meanDiffLog[s]) * model.portion[s] for s in model.stocks)    # risk is how much the observed log return is greater than the expected log return across all stocks

def PROFIT(model, s):
    return model.portion[s] * (sum(model.diffLog[s,t] for t in model.time) - model.logMarginPlusOne) >= 0    # the realized rate of return for each stock must satisfy the minimum rate of return

def INVEST(model):
    return sum(model.portion[s] for s in model.stocks) == 1    # use up the entire investment

model.st_NEGDEVIATION = Constraint(model.time, rule = NEGDEVIATION)    # apply the NEGDEVIATION constraint across all time periods
model.st_POSDEVIATION = Constraint(model.time, rule = POSDEVIATION)    # apply the POSDEVIATION constraint across all time periods
model.st_PROFIT = Constraint(model.stocks, rule = PROFIT)    # apply the PROFIT constraint across all stocks
model.st_INVEST = Constraint(rule = INVEST)    # apply the INVEST constraint

# ---- execute solver ----

opt = SolverFactory("glpk")    # call the glpk solver
# opt = SolverFactory("ipopt", solver_io = "nl")    # call the ipopt solver

# create a list of the dat file names
datfiles = ['mad-yahoo-5-100.dat','mad-yahoo-5-200.dat','mad-yahoo-5-350.dat','mad-yahoo-5-500.dat','mad-yahoo-8-200.dat','mad-yahoo-8-400.dat','mad-yahoo-8-700.dat','mad-yahoo-8-1000.dat','mad-yahoo-all-400.dat','mad-yahoo-all-800.dat','mad-yahoo-all-1400.dat','mad-yahoo-all-2000.dat']

# create a list of result files names (include the path)
resultfiles = ['C:/ ... /Portfolio/mad-yahoo-5-100-results.txt','C:/ ... /Portfolio/mad-yahoo-5-200-results.txt','C:/ ... /Portfolio/mad-yahoo-5-350-results.txt','C:/ ... /Portfolio/mad-yahoo-5-500-results.txt','C:/ ... /Portfolio/mad-yahoo-8-200-results.txt','C:/ ... /Portfolio/mad-yahoo-8-400-results.txt','C:/ ... /Portfolio/mad-yahoo-8-700-results.txt','C:/ ... /Portfolio/mad-yahoo-8-1000-results.txt','C:/ ... /Portfolio/mad-yahoo-all-400-results.txt','C:/ ... /Portfolio/mad-yahoo-all-800-results.txt','C:/ ... /Portfolio/mad-yahoo-all-1400-results.txt','C:/ ... /Portfolio/mad-yahoo-all-2000-results.txt']

# manually define a file number if you like (starts at 0)
# filenum = 0

# define a start number (starts at 0)
startnum = 0

# solve every dat file (ie. problem instance)
for filenum in range(startnum, len(datfiles)):
	instance = []    # reset the problem instance
	results = []    # reset the solution
	varobject = []    # reset the variable values
	gc.collect()    # collect the garbage from the problem instance, solution, and variable values
	
	instance = model.create_instance(datfiles[filenum])    # create a problem instance
	results = opt.solve(instance, tee = True)    # solve the problem instance
	results.write()    # print out the solution to the console
	instance.solutions.load_from(results)    # store the solution results
	
	# open the path to where the results will be written
	with open(resultfiles[filenum], "w") as myfile:
		
		# write out the value of the objective function
		# myfile.write ("{}, {}\n".format("objective", value(instance.obj)))
		
		# get the value of each variable
		for v in instance.component_objects(Var, active = True):
			varobject = getattr(instance, str(v))
			
			# write out the value of each variable
			for index in varobject:
				myfile.write ("{}, {}, {}\n".format(v, index, varobject[index].value))
