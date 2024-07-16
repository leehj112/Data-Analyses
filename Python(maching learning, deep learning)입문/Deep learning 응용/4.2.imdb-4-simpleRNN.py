# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:49:18 2024

@author: leehj
"""

# 피드포워드 신명망 vs 순환 신경망

# 피드포워드 신명망(FFNN: Feed Forward Neural Network)
# 이전에 처리했던 샘플을 다음 샘플을 처리하는데 재사용 하지 않음
# 데이터를 섞으면 효율이 좋아짐

# 순환 신경망으로 IMDB 리뷰 분류하기
# 다음 샘플을 위해서 이전 데이터가 신경망 층에 순환될 필요가 있다.
# 순서가 의미가 있다.
# 데이터를 섞으면 안된다.
# 언어, 날짜에 따른 연관, 날씨, 일자별 판매 실적
# 예시)
# "별로지만 추천해요", "기능은 단순하지만 편리해요", "편리하지만 기능이 부족해요"

# 순경 신경망에서 순환층을 셀이라 부른다. 
# 하나의 셀은 여러 개의 뉴런으로 구성
# 셀의 출력을 특별히 은닉 상태라고 부른다.
# 은닉상태는 다음층으로 전달될 뿐 아니라 셀이 
# 다음 타임스텝의 데이터를 처리할 때 재사용된다.
# 타임스텝(Timestep): 이전 샘플에 대한 기억을 가지고 있다.
#                     이렇게 샘플을 처리하는 한 단계를 타임스텝

#%%

# IMDB : 인터넷 영화 데이터베이스(imdb.com)에서 수집한 리뷰 감상평에 따라
# 긍정과 부정으로 분류해 놓은 데이터셋
# 총 50,000개의 샘플로 구성, 훈련데이터와 테스트 데이터가 25,000씩 분할

# 자연어처리(NLP: Natural Language Processing)와 말뭉치(corpus)
# 음성인식, 기계번역, 감성분석
# IMDB: 감성분석

# 토큰(Token) : 공백을 기준으로 단어를 분리
# 토큰을 분리 후 각 단어들에 정수로 값을 할당
# 토큰값 : 0(패딩), 1(문장의 시작), 2(어휘 사전에 없는 단어)

# 어휘사전 : 훈련 세트에서 고유한 단어를 뽑아 만든 목록

#%%
# [샘플]
# 영어로 된 문장으로 단어를 정수로 바꾼 데이터셋 
# 전체 데이터셋에서 가장 자주 등장하는 단어 300개 사용
# 매개변수 : num_words = 300

#%%

# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 
# 텐서플로 연산을 결정적으로 만듭니다.
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

#%%
# IMDB 리뷰 데이터셋

# 총 50,000개의 샘플로 구성, 훈련데이터와 테스트 데이터가 25,000씩 분할    
# 타겟 : 0(부정), 1(긍정)    

from tensorflow.keras.datasets import imdb

# 총 50,000 = 훈련(25000) + 테스트(25000)
# 300단어를 선택
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=300)

#%%

print(train_input.shape, test_input.shape) # (25000,) (25000,)

#%%

print(len(train_input[0])) # 218, 영화평 첫 번째 데이터 218개 단어로 구성
print(len(train_input[1])) # 189, 영화평 두 번째 데이터 189개 단어로 구성

#%%


print(train_input[0]) # 영화평 첫 번째 데이터 출력

#%%


print(train_target[:20]) # 영화평 0부터 19까지 정답 20개 출력

#%%


from sklearn.model_selection import train_test_split

# 훈련데이터 25,000건을 훈련세트와 검증세트로 랜덤분할(섞음)
# 훈련세트(80%) : 20,000개
# 검증세트(20%) : 5,000개
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

#%%


import numpy as np

# 행(20,000)을 하나씩 읽어서 각 행의 토큰(단어)의 갯수 구해서
# 리스트 만든 후 넘파이 배열로 생성
print(len(train_input[0])) # 259
lengths = np.array([len(x) for x in train_input])

#%%

# 토근의 평균 갯수, 중간값 
print(np.mean(lengths), np.median(lengths)) # 239.00925 178.0

#%%

# 데이터가 한쪽으로 치우쳐 있음 : 대부분 300단어 미만
# 평균값(239)이 중간값(178)보다 큰 이유는 오른 끝에 아주 큰 데이터가 있다.
# 어떤 리뷰가 1000단어 이상이다.
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

#%%

# 축소 : 100단어로 축소(maxlen=100)
# 패딩 : 100단어 안되는 데이터는 0으로 채움
# 배열 : 20000 * 100
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 뒤에서부터 100개를 선택
# truncating : 'pre', 'post'
# 'pre' : 기본값, 앞쪽에서 자름
# 'post' : 뒤쪽에서 자름
# 유용한 정보가 뒷쪽에 있을 것이라 기대, 결정적인 소감을 말할 가능성이 높다.
# train_seq = pad_sequences(train_input, maxlen=100, truncating='post')
train_seq = pad_sequences(train_input, maxlen=100)

#%%


print(train_seq.shape) # (20000, 100)

#%%


print(train_seq[0])

#%%

# 뒤에서부터 10개 출력
print(train_input[0][-10:])

#%%

# 다섯번째 출력
print(train_seq[5])

#%%

# 점증용 세트도 100개 단어로 자르고 패딩 처리
val_seq = pad_sequences(val_input, maxlen=100)

#%%
# ## 순환 신경망 만들기
# SimpleRNN 모델

#%%


from tensorflow import keras

model = keras.Sequential()

# input_shape=(100, 300)
# 100 : 샘플의 길이
# 300 : 어휘는 300단어
# activation='sigmoid' : 2진분류
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 300)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#%%

# 훈련세트    
# 원-핫 인코딩 : 
# 300단어 중에 하나만 1이고 나머지는 0으로 만든 정수
train_oh = keras.utils.to_categorical(train_seq)


#%%


print(train_oh.shape)

#%%


print(train_oh[0][0][:12])

#%%


print(np.sum(train_oh[0][0]))

#%%

# 검증세트    
# 원-핫 인코딩 : 
# 300단어 중에 하나만 1이고 나머지는 0으로 만든 정수
val_oh = keras.utils.to_categorical(val_seq)

#%%


model.summary()
"""
뉴런의 갯수 : 8
원핫인코딩 된 갯수: 300

300 * 8 = 2400
8 * 8   =   64
8       =    8 절편(각 뉴런마다)
----------------
          2472
________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn_2 (SimpleRNN)    (None, 8)                 2472      
                                                                 
 dense_2 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 2481 (9.69 KB)
Trainable params: 2481 (9.69 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

#%%

# ## 순환 신경망 훈련하기

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('imdb-best-simplernn-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

# 1 에포크 : 313번 훈련/검증 반복, 313 = 20000/64
# 100 에포크 : 3번 연속 변화가 없으면 중단 
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

#%%


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


#%%
# ## 단어 임베딩을 사용하기

# 단어 이베딩(Word Embedding)
# 각 단어를 고정된 크기의 실수 벡터로 변경
# 자연어 처리에서 더 좋은 성능을 낸다.
# 케라스에서 Embedding() 클래스 제공
# 원-핫 인코딩을 할 필요가 없다.
# 메모리를 효율적으로 사용한다.

#%%


model2 = keras.Sequential()

# 300 -> 16으로 줄임
model2.add(keras.layers.Embedding(300, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()

#%%


rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('imdb-best-simplernn-embedding-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

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

# 예측 : 검증 데이터
preds_val = model2.predict(val_seq[0:10])
print(preds_val)


#%%

# 예측 : 테스트 데이터

test_seq = pad_sequences(test_input, maxlen=100)
preds_test = model2.predict(test_seq[0:10])
print(preds_test)