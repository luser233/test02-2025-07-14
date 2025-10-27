import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, Sequential, Dense
import matplotlib.pyplot as plt
model = Sequential(
    [
        Dense(3, activation='sigmoid', input_shape=(2, ), name='input layer1'),
        Dense(4, activation='sigmoid', name='layer2'),
        Dense(4, activation='sigmoid', name='layer3'),
        Dense(1, activation='sigmoid', name='output layer4'),
        # 输出层的激活函数为sigmoid函数，用于二分类问题
    ]
)


model.fit(X, y, epochs = 20)

model.compile(
    loss = keras.losses.BinaryCrossentropy(), #二分交叉熵损失
    optimizer = keras.optimizers.Adam(learning_rate=0.01), #adam优化器
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'loss: {loss:.4f}')
print(f'accuracy: {accuracy:.4f}')

predictions = model.predict(X_new)
print(f'predictions: {predictions}')
