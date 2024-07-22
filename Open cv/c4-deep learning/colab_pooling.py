# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:36:39 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

x = tf.constant([1., 3., 5., 7., 9., 2., 4., 6., 8., 10.])
x1 = tf.reshape(x, [1, 5, 2])
maxpool1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1,
                                         padding='valid')(x1)

print("input : ", x1)
print("\nmax pooling 1d : " ,maxpool1d)


# In[ ]:


x2 = tf.reshape(x, [1, 2, 5])
averagepool1d = tf.keras.layers.AveragePooling1D(pool_size=2, strides=1,
                                             padding='same')(x2)
print("input : ", x2)
print("\nAverage pooling 1d : " ,averagepool1d)


# In[ ]:


globalmaxpool1d = tf.keras.layers.GlobalMaxPooling1D()(x2)
globalaveragepool1d = tf.keras.layers.GlobalAveragePooling1D()(x2)

print("Global max pooling 1d : " ,globalmaxpool1d)
print("\nGlobal average pooling 1d : " ,globalaveragepool1d)
