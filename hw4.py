import pandas as pd
import numpy as np
import os
filename = os.getcwd() + "\\Plaza Coffee.txt"
df = pd.read_csv(filename)
print(df)

print (df.groupby(['Company', 'Order', 'Payment'])["Quantity"].sum())
   

       
# "From KPMG 2 people have bought stuff on discount and paid in cash, also assistants got 4 servings of coffee on credit.