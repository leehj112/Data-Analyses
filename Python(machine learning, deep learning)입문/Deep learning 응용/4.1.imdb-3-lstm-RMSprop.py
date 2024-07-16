# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:48:24 2024

@author: leehj
"""

# LSTM과 GRU 셀

# LSTM(Long Short-Term Momory)
# 단기 기억을 오래 기억하기 위한 모델
# 셀은 타임스텝이 긴 데이터를 효과적으로 학습하기 위해 고안된 순환층이다.
# 입력게이트, 삭제게이트, 출력게이트 역할을 하는 작은 셀이 포함되어 있다.
# LSTM에는 순환되는 상태가 2개 : 은닉상태, 셀상태(cell state)
# 셀은 은닉상태 외에 셀 상태를 출력, 셀 상태는 다음 층으로 전달되지 않으며 현 셀에서만 순환 된다.
#
# RNN은 레이어를 여러 개 거치면서 처음에 입력했던 단어에 대한 정보를 조금씩 잃어 버린다.
# 최근에 학습한 단어가 모델의 최종 예측에 더 큰 영향을 주게 된다.
# RNN은 단기 의존성이 높다.

# LSTM은 기존 정보 중에서 중요한 정보를 다음 단계로 전달하는 구조이다.
# 모델의 장기 기억 성능을 개선했기 때문에 길이기 긴 시퀀스 데이터를 학습하는데 적합하다.

#%%


# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다.
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

#%%

# LSTM 신경망 훈련하기

#%%


from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

#%%


from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

#%%


from tensorflow import keras

model = keras.Sequential()

model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#%%


rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('imdb-best-lstm-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])


#%%


import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


#%%
# 순환 층에 드롭아웃 적용하기
# 과대적합 예방

#%%


model2 = keras.Sequential()

model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3)) # 드롭아웃 30%
model2.add(keras.layers.Dense(1, activation='sigmoid'))

#%%

#%%
"""
RMSprop은 신경망에서 사용되는 경사 하강법 최적화 알고리즘 중 하나이다. 
주로 확률적 경사 하강법(SGD)의 변형으로 사용되며, 
학습률을 조절하기 위한 기법 중 하나이다. 
RMSprop의 이름은 "Root Mean Square Propagation"의 약자이다.
"""
#%%

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('imdb-best-lstm-dropout-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])

#%%

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

#%%

# 예측 : 테스트 데이터
test_seq = pad_sequences(test_input, maxlen=100)
preds_test = model2.predict(test_seq[0:10])
print(preds_test)