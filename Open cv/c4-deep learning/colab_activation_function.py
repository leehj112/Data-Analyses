# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:34:38 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

norm = tf.random.normal([1, 5], mean=-1, stddev=10, dtype="half")
print("input : ", norm)

tf.keras.activations.relu(norm).numpy()


# In[ ]:


al = tf.keras.activations.relu(norm, alpha=0.5)
max_v = tf.keras.activations.relu(norm, max_value=1)
thres = tf.keras.activations.relu(norm, threshold=norm[0,0])
print("alpha : ", al)
print("\nmax_value : ", max_v)
print("\nthreshold : ", thres)


# In[ ]:


from matplotlib import pyplot as plt

in1 = tf.constant([-10, -5.0, -2.0, 0.0, 2.0, 5.0, 10], dtype = tf.float32)
out1 = tf.keras.activations.sigmoid(in1)
print("sigmoid : ", out1)

plt.plot(in1 ,out1)
plt.title("Sigmoid Visualization")


# In[ ]:


in2 = tf.constant([[0.0, 1.0, 5.0, -2.0, 4.0, 8.0, 3.0]], dtype = tf.float32)
plt.plot(in2[0,:])
plt.title("Input Visualization")
plt.show()

out2 = tf.keras.activations.softmax(in2, axis=-1).numpy()
print("softmax : ", out2)
print("\nsum of softmax : ", out2.sum())

x = [x for x in range(7)]
plt.plot(x, out2[0,x], '-')

plt.title("Softmax Visualization")
plt.show()
