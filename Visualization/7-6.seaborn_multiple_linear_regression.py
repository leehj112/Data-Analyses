# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:14:44 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:00:56 2023

@author: Solero
"""

import seaborn as sns
sns.set_theme()

# Load the penguins dataset
penguins = sns.load_dataset("penguins")

# Plot sepal width as a function of sepal_length across days
g = sns.lmplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    height=5
)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Snoot length (mm)", "Snoot depth (mm)")