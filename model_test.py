import cv2
from tensorflow import keras

model = keras.models.load_model('my_model')
print(model.summary())
img = cv2.imread("Data/no_parking/1.jpg")
# cv2.imshow("Image", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)
img = img / 255
# cv2.imshow("Image", img)
img = img.reshape(1, 32, 32, 1)
prediction = model.predict(img)
print("prediction shape:", prediction.shape)
cv2.waitKey(0)

