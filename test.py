import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
import matplotlib.pyplot as plt

# 生成随机二分类数据集
np.random.seed(42)  # 设置随机种子，保证结果可复现
num_samples = 1000  # 样本数量
X = np.random.randn(num_samples, 2)  # 生成2维特征的样本（符合正态分布）

# 定义分类规则：根据特征的线性组合生成标签（模拟可学习的模式）
# 这里使用简单的线性边界：x1 + x2 > 0.5 为类别1，否则为类别0
y = (X[:, 0] + X[:, 1] > 0.5).astype(int).reshape(-1, 1)  # 转换为二维数组（符合模型输出格式）

# 划分训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 生成新的测试数据（用于预测演示）
X_new = np.random.randn(5, 2)  # 5个新样本

# 定义模型（完全通过tf.keras调用组件）
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(3, activation='sigmoid', input_shape=(2, ), name='inputlayer1'),
        tf.keras.layers.Dense(4, activation='sigmoid', name='layer2'),
        tf.keras.layers.Dense(4, activation='sigmoid', name='layer3'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='outputlayer4'),
        # 输出层的激活函数为sigmoid函数，用于二分类问题
    ]
)

# 编译模型（需在训练前完成）
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),  # 二分交叉熵损失
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Adam优化器
    metrics=['accuracy']  # 监控准确率
)

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)  # 保留10%训练数据作为验证集

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'test loss: {loss:.4f}')
print(f'test accuracy: {accuracy:.4f}')

# 预测新数据
predictions = model.predict(X_new)
# 转换为类别（阈值0.5）
predictions_class = (predictions > 0.5).astype(int)
print(f'预测概率: {predictions.flatten()}')
print(f'预测类别: {predictions_class.flatten()}')