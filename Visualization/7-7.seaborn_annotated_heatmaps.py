# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:15:01 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:04:33 2023

@author: Solero
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the example flights dataset and convert to long-form
flights_long = sns.load_dataset("flights")

#%%

# error
flights = flights_long.pivot("month", "year", "passengers")

#%%
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)