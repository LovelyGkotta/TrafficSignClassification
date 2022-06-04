import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

data = os.listdir('Data')
label = pd.read_csv('label.csv')
noOfClasses = len(data)
a = 1
print(label)
for i in data:
    img = cv2.imread("Data" + "/" + i + "/" + "1" + ".jpg")
    plt.subplot(4, 5, a)
    plt.imshow(img)
    plt.title(i, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    a += 1
plt.show()
