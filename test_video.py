import numpy as np
import cv2
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = keras.models.load_model('my_model')  # モデルを読み込む
cap = cv2.VideoCapture('video.MP4')

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

def getClassName(classNo):  # クラス
    if classNo == 0: return '通行止め'
    elif classNo == 1: return '自転車通行止め'
    elif classNo == 2: return '駐車禁止'
    elif classNo == 3: return '一方通行(左)'
    elif classNo == 4: return '一方通行(前)'
    elif classNo == 5: return '駐車可'
    elif classNo == 6: return '横断歩道'
    elif classNo == 7: return '優先道路'
    elif classNo == 8: return '道路工事中'
    elif classNo == 9: return '環状の交差点'
    elif classNo == 10: return '徐行'
    elif classNo == 11: return '最高速度40'
    elif classNo == 12: return '最高速度50'
    elif classNo == 13: return '一時停止'
    elif classNo == 14: return '直進以外進行禁止'
    elif classNo == 15: return '信号機あり'
    elif classNo == 16: return '左折以外進行禁止'
    elif classNo == 17: return '右折以外進行禁止'
    elif classNo == 18: return '二方向交通'
    elif classNo == 19: return '車両進入禁止'


# 日本語描画（cv2.putTextは日本語を使えない）
def addText(img, text, left, top, textcolor=(0, 255, 0), textsize=50):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            # linux  /usr/share/fonts/opentype/noto/NotoSansCJK-Ragular.ttc
            # windows  font/simsun.ttc
            # Macos  ~/Library/Fonts/NotoSansCJKjp-Regular.ttf
            "font/simsun.ttc", textsize, encoding="utf-8")
        draw.text((left, top), text, textcolor, font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


while True:
    # カメラから画像を読み込む
    success, imgOriginal = cap.read()

    # 画像前処理
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = img / 255
    img = img.reshape(1, 32, 32, 3)

    cv2.namedWindow("Result", 0)
    cv2.resizeWindow("Result", int(width/3), int(height/3))
    imgOriginal = addText(imgOriginal, "標識種類:", 10, 35, (18, 11, 222), 200)
    imgOriginal = addText(imgOriginal, "正確率:", 10, 300, (18, 11, 222), 200)
    # 標識を識別
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    if probabilityValue > 0.75:
        imgOriginal = addText(imgOriginal, str(getClassName(classIndex)), 1000, 35, (18, 11, 222), 200)
        imgOriginal = addText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", 800, 300, (18, 11, 222), 200)
    cv2.imshow("Result", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
