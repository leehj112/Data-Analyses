# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:35:56 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers

class exDense(layers.Layer):

  def __init__(self, units=32):
      super(exDense, self).__init__()
      self.units = units

  def build(self, input_shape):
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'), trainable=True, name="weight")
    b_init = tf.ones_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=False, name="bias")

  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b


# In[ ]:


# 레이어 인스턴스 생성.
exDense_layer = exDense(4)

# build() 호출 전의 가중치 확인
before_w = exDense_layer.weights
print("all weights(before)", before_w)

# 입력 데이터 대입 및 결과 반환
out = exDense_layer(tf.ones((3, 3)))
print("\noutputs:", out)

# build() 호출 후의 가중치 확인
after_w = exDense_layer.weights
print("\nall weights(after)", after_w)


# In[ ]:


t_w = exDense_layer.trainable_weights
print("\ntraninable_weights", t_w)

nt_w = exDense_layer.non_trainable_weights
print("\nnon_traninable_weights", nt_w)

t_v = exDense_layer.trainable_variables
print("\ntraninable_variables", t_v)

nt_v = exDense_layer.non_trainable_variables
print("\nnon_traninable_variables", nt_v)



# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers

class exDense(layers.Layer):

  def __init__(self, units=32):
      super(exDense, self).__init__()
      self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                          initializer='random_normal',
                          trainable=True, name="weight")
    self.b = self.add_weight(shape=(self.units,),
                              initializer='random_normal',
                              trainable=True, name="bias")
  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b
