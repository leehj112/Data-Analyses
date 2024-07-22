# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:36:14 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import initializers

norm_init1 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
norm_init2 = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=100)

for i in range(3):
  values1 = norm_init1(shape=(2, 2)).numpy()  
  values2 = norm_init2(shape=(2, 2)).numpy()  
  print(i, "initializer1 : ", values1)  
  print(i, "initializer2 : ", values2)  


# In[ ]:


from matplotlib import pyplot as plt

unif_init1 = tf.keras.initializers.RandomUniform(minval=0, maxval=1)
unif_init2 = tf.keras.initializers.RandomUniform(minval=-2, maxval=2)

for i in range(100):
  value3 = unif_init1(shape=(1,1)).numpy()
  value4 = unif_init2(shape=(1,1)).numpy()
  plt.plot(i, value3[0,0], 'o', color = 'black')
  plt.plot(i, value4[0,0], '*', color = 'gray')

plt.title("Results of RandomUniform")
plt.show()  


# In[ ]:


norm_init3 = tf.keras.initializers.RandomNormal(mean=0., stddev=5.)
tun_norm_init1 = tf.keras.initializers.TruncatedNormal(mean=0., stddev=5.)

for i in range(1000):
  value3 = norm_init3(shape=(1,1)).numpy()
  value4 = tun_norm_init1(shape=(1,1)).numpy()
  plt.plot(i, value3[0,0], 'o', color = 'black')
  plt.plot(i, value4[0,0], '*', color = 'gray')

plt.title("Difference of RandomUniform and TruncatedNormal")
plt.show()


# In[ ]:


init_ones = tf.keras.initializers.Ones()
init_zeros = tf.keras.initializers.Zeros()
init_constant = tf.keras.initializers.Constant(value=3)

print("one init : ", init_ones(shape=(1,5)).numpy())
print("\nzeros init : ", init_zeros(shape=(1,5)).numpy())
print("\nconstant init : ", init_constant(shape=(1,5)).numpy())

init_identity = tf.keras.initializers.identity()
print("one init : ", init_identity(shape=(2,5)).numpy())


# In[ ]:


layer = tf.keras.layers.Dense(5, kernel_initializer='ones',
                              kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))
tensor = tf.ones(shape=(5, 5)) * 2.0
out = layer(tensor)
print(tf.math.reduce_sum(layer.losses))
