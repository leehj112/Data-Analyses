# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:37:28 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras.layers as keras_layers

layer1 = keras_layers.Dense(1,
                            kernel_initializer=tf.constant_initializer(5.))
layer1_out = layer1(tf.convert_to_tensor([[1., 2., 3.]]))
layer1.get_weights()


# In[ ]:


layer2 = keras_layers.Dense(1,
                            kernel_initializer=tf.constant_initializer(2.))
layer2_out = layer2(tf.convert_to_tensor([[10., 20., 30.]]))
layer2.get_weights()


# In[ ]:


layer2.set_weights(layer1.get_weights())
layer2.get_weights()
