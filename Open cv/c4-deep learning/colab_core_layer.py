# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:35:23 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf 
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
 
x = Input(shape=(32,))
y = layers.Dense(16, activation='softmax')(x)
model = Model(x, y)

norm = tf.random.normal([1, 32], mean=-1, stddev=10, dtype="half")
model(norm)
