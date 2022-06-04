# mnist_test
import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import os

# データを読み込む
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 　データの前処理
train_images = train_images.reshape((60000, 28, 28, 1))  # 6/7は学習データ、sizeは28*28,1はグレースケール
test_images = test_images.reshape((10000, 28, 28, 1))  # 1/7をテストデータ

# 値を0から1までの範囲にスケールする。そのためには、画素の値を255で割る
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images.shape, test_images.shape, train_labels.shape, test_labels.shape

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 卷积层1，卷积核3*3
    layers.MaxPooling2D((2, 2)),  # 池化层1，2*2采样
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样

    layers.Flatten(),  # Flatten层，连接卷积层与全连接层
    layers.Dense(64, activation='relu'),  # 全连接层，特征进一步提取
    layers.Dense(10)  # 输出层，输出预期结果
])
# 打印网络结构
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# # 测试
# print("\n Testing -------- ")
# loss, accuracy = model.evaluate(test_images, test_labels)
#
# #
# print("test loss:", loss)
# print("test accuracy:", accuracy)
# #
# plt.figure(1)
# plt.plot(history.history['loss'])
# plt.legend(['training'])
# plt.title('Loss')
# plt.xticks(range(0, 10))
# plt.xlabel('epoch')
# #
# plt.figure(2)
# plt.plot(history.history['accuracy'])
# plt.legend(['training'])
# plt.title('accuracy')
# plt.xticks(range(0, 10))
# plt.xlabel('epoch')
# plt.show()

