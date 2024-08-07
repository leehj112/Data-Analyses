# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:28:12 2024

@author: leehj
"""

#%%

# https://keras.io/examples/nlp/bidirectional_lstm_imdb/

"""
Bidirectional LSTM on IMDB
Author: fchollet
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Train a 2-layer bidirectional LSTM on the IMDB movie review sentiment classification dataset.
"""

#%%

import numpy as np
import keras
from keras import layers

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

#%%

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)

# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)

# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

#%%

# Load the IMDB movie review sentiment data
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

# Use pad_sequence to standardize sequence length:
# this will truncate sequences longer than 200 words and zero-pad sequences shorter than 200 words.
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

#%%

# Train and evaluate the model
# You can use the trained model hosted on Hugging Face Hub and try the demo on Hugging Face Spaces.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

#%%

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()