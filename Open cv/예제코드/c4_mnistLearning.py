# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:12:57 2024

@author: leehj
"""

# MNIST 학습 예제 (c4_ministLearning.py)

# 관련 라이브러리 선언
from tensorflow.keras import datasets, layers, models
from createFolder import createFolder

# 학습 데이터 불러오기 및 전처리
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 은닉층 설계
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 모델 표시
model.summary()

# 출력층 설계
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# 학습과정 설정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습 및 평가
model.fit(train_images, train_labels, epochs=1)
model.evaluate(test_images, test_labels)

# 결과 저장
save_dir = './code_res_imgs/c4_minist'
createFolder(save_dir)
# model.save_weights(save_dir + '/mnist_checkpoint')
model.save(save_dir + '/mnist_model.h5')