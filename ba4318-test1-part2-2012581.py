# Esma Ceren BAYRAM 2012581

import pandas as pd
import numpy as np
import os

data_brazil = pd.read_csv("sudeste.csv", usecols = ["date", "temp"])
data_madrid = pd.read_csv("weather_madrid_LEMD_1997_2015.csv", usecols = ["CET", "Mean TemperatureC"])

df_brazil = pd.DataFrame(data_brazil)
df_madrid = pd.DataFrame(data_madrid)

df_madrid.columns = ["date", "Mean TemperatureC"]

df_brazil['date'] = pd.to_datetime(df_brazil.date)
df_madrid['date'] = pd.to_datetime(df_madrid.date)

df_brazil2 = df_brazil.groupby(['date']).mean().sort_values(by = ['date'])
df_madrid2 = df_madrid.sort_values(by = ['date'])

#print(df_brazil2)
#print(df_madrid2)

df_final  = pd.merge(df_madrid2, df_brazil2, how = "inner", on = "date")
df_final.columns = ["date", "temp-brazil", "temp-madrid" ]

print(df_final)

print(df_final.corr())


