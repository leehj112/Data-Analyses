# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:48:14 2024

@author: leehj
"""

import tensorflow
# import cv2
import pandas as pd
from sklearn import datasets

print(tensorflow.__version__) # 2.16.1
# print(cv2.__version__)

#%%
## iris 데이터 불러오기
iris = datasets.load_iris()

iris_X = iris.data
iris_y = pd.get_dummies(iris.target).to_numpy()

#%%

# ## 싸이킷런을 이용한 구현

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, 
                                                    test_size=0.3,
                                                    random_state=1)

#%%

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,30), 
                    activation="logistic", 
                    solver="adam",
                    max_iter=1000) 

mlp.fit(train_X, train_y)
mlp.score(test_X, test_y)
pred = mlp.predict(test_X)

#%%

import numpy as np
#교차분류표
pd.crosstab(np.argmax(pred, axis=1), np.argmax(test_y, axis=1)) 


#%%
##########################################################
# ## 텐서플로우를 이용한 구현

# In[14]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[15]:


x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])


# In[16]:


W1 = tf.Variable(tf.zeros([4, 50]))
b1 = tf.Variable(tf.zeros([50]))
h1 = tf.nn.sigmoid(tf.matmul(x, W1)+b1)


# In[17]:


W2 = tf.Variable(tf.zeros([50, 30]))
b2 = tf.Variable(tf.zeros([30]))
h2 = tf.nn.sigmoid(tf.matmul(h1, W2)+b2)


# In[18]:


W3 = tf.Variable(tf.zeros([30, 3]))
b3 = tf.Variable(tf.zeros([3]))
h3 = tf.nn.softmax(tf.matmul(h2, W3)+b3)


# In[19]:


cross_entropy = -tf.reduce_sum(y*tf.log(h3), reduction_indices=[1])
loss = tf.reduce_mean(cross_entropy)
train = tf.train.AdamOptimizer().minimize(loss)


# In[20]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, feed_dict={x:train_X, y:train_y})
    if i%100 == 0:
        tr_loss = sess.run(loss, feed_dict={x:train_X, y:train_y})
        print(i, tr_loss)


# In[21]:


correct_prediction = tf.equal(tf.argmax(h3, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:test_X, y:test_y}))

sess.close()


#%%

##############################################################
## 케라스를 이용한 구현

from tensorflow.keras.models import Sequential
model = Sequential()


#%%

from tensorflow.keras.layers import InputLayer, Dense
model.add(InputLayer(4))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax'))

#%%

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=1000)

# 출력값은 loss와 accuracy
model.evaluate(test_X, test_y)

#%%