# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:35:04 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

input_shape = (3, 2, 1)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(
    filters=1, kernel_size=2, activation=None, kernel_initializer = 'ones',
    input_shape=input_shape[1:])(x)

print("input ", x)
print("\nconv1D : ", y)


# In[ ]:


input_shape = (3, 3, 1)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(
    filters=2, kernel_size=2, activation=None, kernel_initializer = 'ones',
    input_shape=input_shape[1:])(x)

print("input ", x)
print("\nconv1D : ", y)


# In[ ]:


input_shape = (3, 5, 5, 1)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
    filters=2, kernel_size=2, activation=None, kernel_initializer = 'ones',
    input_shape=input_shape[1:])(x)

print("input ", x)
print("\nconv1D : ", y)
