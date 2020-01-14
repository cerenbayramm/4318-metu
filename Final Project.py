#Esma Ceren Bayram 2012581
#Kerem Baloğlu 2076073
#Kübra Endez 2012722


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model as lm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot, lag_plot
import statsmodels

data = pd.read_csv("total_data_cs.csv")

import warnings
warnings.filterwarnings("ignore")

credit_to_gdp = data[["Credit_to_private/GSYH"]]
print(credit_to_gdp.head())
bist_100 = data[["BIST_100"]]
print(bist_100.head())
gdp = data[["GSYH"]]
print(gdp.head())
"""
def the_plotter(x , typ):
    if typ == "plot":
        plt.plot(x)
        plt.title(list(x.columns))
        plt.show()
    elif typ == "autcor":
        autocorrelation_plot(x)
        plt.title(list(x.columns))
        plt.show()
    else:
        lag_plot(x)
        plt.title(list(x.columns))
        plt.show()

plotset = [credit_to_gdp,bist_100,gdp]
for i in plotset:
    ty = ["plot","autcor","lag"]
    for k in ty:
        the_plotter(i, k)
"""
#credit and bist100 regress on gdp (with constant) in order to see degree and direction of relationship
#between them
Y = data[["GSYH"]]
X = data[["Credit_to_private/GSYH","BIST_100"]]
print("Credit and BIST100 index regress on GDP")
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print_model = model.summary()
print(print_model)

#Unit Root Tests for regular unit root
sets = [credit_to_gdp,bist_100,gdp]
for i in sets:
    kpss_test = sm.tsa.stattools.kpss( i ,regression = "ct") #H0: Series has no unit root
    print(kpss_test)

#since all series are not stationary we need to take diffrence
dif_gdp = gdp.diff()
print(dif_gdp.head())
dif_bist_100 = bist_100.diff()
print(dif_bist_100.head())
dif_credit_to_gdp = credit_to_gdp.diff()
print(dif_credit_to_gdp.head())


#all first difffrences are taken. lets try KPSS test again
sets = [dif_credit_to_gdp[1:],dif_bist_100[1:],dif_gdp[1:]]
for i in sets:
    kpss_test = sm.tsa.stattools.kpss( i ,regression = "ct") #H0: Series has no unit root
    print(kpss_test)

# Unfortunetly we cannot remove unit root by diffrencing. Thus we cannot use any ARIMA based method
# nevertheless we can (and will) use deterministic forecast method such as exponential smoothing

#Forecasts
#In order understand how well models work, we need to divide our data into train and test parts.


#simple exponential smoothing
def simple_smt(x, alpha=0.2):
    train = x[0:53]
    test = x[54:55]
    simp = np.asarray(train)
    estimate = sm.tsa.ExponentialSmoothing(simp).fit(smoothing_level=alpha,optimized=False).forecast(1)
    error = mean_squared_error(test,estimate)
    print(error)
    print(estimate)

setses = [credit_to_gdp,bist_100,gdp]
for i in setses:
    a = simple_smt(i)
    print("for", list(i.columns), " estimated simple exponential smoothing forecast and error")
    print(a)

#holt method
def holts_method(x, alpha=0.2, slope=0.1):
    train = x[0:53]
    test = x[54:55]
    values = np.asarray(train)
    holt_mod = Holt(values)
    fit = holt_mod.fit(alpha, slope, 1)
    fore = fit.forecast()[-1]

    #print ("METHOD ----------------------", type(test), type(fore))
    #print (test.values[0])
    #print ([fore])
    error = mean_squared_error(test.values[0],[fore])
    return fore, error

setholt = [credit_to_gdp,bist_100,gdp]
for i in setholt:
    a = holts_method(i)
    print("for",list(i.columns), "estimated Holts forecast")
    print(a)

#Winters'
def holt_winters(x):
    train = x[0:53]
    test = x[54:55]
    model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=4).fit()
    pred = model.predict(len(test))

    #print ("WINTERS ----------------------", type(test), type(pred))
    #print (test)
    #print (pred)
    error = mean_squared_error(test, pred.tail(1))
    return pred, error

setwint = [credit_to_gdp,bist_100,gdp]
for i in setwint:
    a = holt_winters(i)
    print("for",list(i.columns), "estimated Holt Winters'")
    print(a)






