import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

path = 'Data'
label = 'label.csv'

# 画像を読み込む
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in myList:
    myPicList = os.listdir(path + "/" + x)
    for y in range(1, 401):
        curImg = cv2.imread(path + "/" + x + "/" + str(y) + ".jpg")
        images.append(curImg)
        classNo.append(count)
    print(count, x)
    count += 1
images = np.array(images)
classNo = np.array(classNo)
imageSize = (32, 32, 3)

# データを分ける
# 全部データの80%は訓練データ, 20%はテストデータ
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
# 訓練データの80%は訓練データ, 20%は検証データ
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

# データ配置を示す
print("--------------------------")
print("Data Shapes")
print("Train", X_train.shape, y_train.shape)  # Train (5120, 32, 32, 3) (5120,)
print("Validation", X_validation.shape, y_validation.shape)  # Validation (1280, 32, 32, 3) (1280,)
print("Test", X_test.shape, y_test.shape)  # Test (1600, 32, 32, 3) (1600,)


def preprocessing(img):  # 画像の前処理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケールに変換
    img = cv2.equalizeHist(img)
    img = img / 255  # 値を0から1までの範囲にスケールする
    return img


X_train = np.array(list(map(preprocessing, X_train)))  # すべての画像を前処理する
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# 1の深さを追加する
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# リアルタイムにデータ拡張する
dataGen = ImageDataGenerator(width_shift_range=0.1,  # ランダムに水平シフトする範囲
                             height_shift_range=0.1,  # ランダムに垂直シフトする範囲
                             zoom_range=0.2,  # ランダムにズームする範囲
                             shear_range=0.1,  # 反時計回りのシアー角度
                             rotation_range=10)  # 画像をランダムに回転する回転範囲
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train,
                       batch_size=20)

# One-Hotベクトルに変換
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# 畳み込みニューラルネットワークモデル
model = Sequential()
# 畳み込み層1，3*3畳み込みカーネル
model.add((Conv2D(32, (3, 3), input_shape=(imageSize[0], imageSize[1], 1), activation='relu')))
model.add(Conv2D(32, (3, 3), activation='relu'))
# マックスプーリング層1，2*2カーネル
model.add(MaxPooling2D(pool_size=(2, 2)))
# 畳み込み層1，3*3畳み込みカーネル
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# マックスプーリング層1，2*2カーネル
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())  # Flatten層，3次元配列を1次元に直す
model.add(Dense(256, activation='relu'))  # 全結合層
model.add(Dropout(0.5))
model.add(Dense(noOfClasses, activation='softmax'))  # 出力層,noOfClasses（20）個出力
# COMPILE MODEL
model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())  # ネットワークの構造をプリントする

# モデルの訓練
history = model.fit(dataGen.flow(X_train, y_train, batch_size=20),
                    steps_per_epoch=len(X_train) // 20, epochs=20,
                    validation_data=(X_validation, y_validation), shuffle=1)

# loss 成功率
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))
ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].legend(['training', 'validation'])
ax[0].set_title('loss')
ax[0].set_label('epoch')
ax[1].plot(history.history['accuracy'])
ax[1].plot(history.history['val_accuracy'])
ax[1].legend(['training', 'validation'])
ax[1].set_title('Accuracy')
ax[1].set_label('epoch')
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test Accuracy:', accuracy)

# モデルを保存
# model.save("my_model")
print("Generate a prediction")
print(X_test.shape)
prediction = model.predict(X_test)
print("prediction shape:", prediction.shape)
cv2.waitKey(0)
