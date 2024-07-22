# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:35:41 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt

sample1 = 200 + np.random.randn(10)
print(sample1)

sample2 = 200 + np.random.randn(10)
print(sample2)

print(abs(sample1-sample2))

x = [x for x in range(len(sample2))]
plt.plot(x, sample1, '-')
plt.plot(x, sample2, '-')
plt.fill_between(x, sample1, np.maximum(sample1, sample2),
                 where=(abs(sample1-sample2) > 1), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()
