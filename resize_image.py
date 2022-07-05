import cv2
import numpy as np

for i in range(1, 21):
    path = "C:/Users/cyjjjj/Desktop/Data/vehicles_prohibited/(%s).jpg" % (i)
    img = cv2.imread(path)
    for j in range(1, 31):
        new_path = "C:/Users/cyjjjj/Desktop/Data/vehicles_prohibited/%s.jpg" % ((i-1)*30+j)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        if j < 6:  # resize
            cv2.imwrite(new_path, cv2.resize(img, (32, 32)))

        elif j in [6, 7]:  # 左回りに 10 度回転
            M = cv2.getRotationMatrix2D(center, 10, 1.0)
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [8, 9]:  # 左回りに 20 度回転
            M = cv2.getRotationMatrix2D(center, 20, 1.0)
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [10, 11]:  # 右回りに 10 度回転
            M = cv2.getRotationMatrix2D(center, -10, 1.0)
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [12, 13]:  # 左回りに 20 度回転
            M = cv2.getRotationMatrix2D(center, -20, 1.0)
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [14, 15]:  # 明るくになる
            res = np.uint8(np.clip((cv2.add(1.5 * img, 30)), 0, 255))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [16, 17]:  # 暗くになる
            res = np.uint8(np.clip((cv2.add(0.6 * img, 0)), 0, 255))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [18, 19]:  # ぼやけた
            res = cv2.blur(img, (5, 5))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [20, 21]:  # 下に30ピクセル移動
            M = np.float32([[1, 0, 0], [0, 1, (h//8)]])
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [22, 23]:  # 上に30ピクセル移動
            M = np.float32([[1, 0, 0], [0, 1, -(h//8)]])
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [24, 25]:  # 右に30ピクセル移動
            M = np.float32([[1, 0, (w//8)], [0, 1, 0]])
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [26, 27]:  # 左に30ピクセル移動
            M = np.float32([[1, 0, -(w//8)], [0, 1, 0]])
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [28, 29]:  # 轻度ノイズ増加
            noise_level = 80
            noise = np.random.normal(0, noise_level, (h, w, 3))
            res = img + noise
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j == 30:  # 重度ノイズ増加
            noise_level = 150
            noise = np.random.normal(0, noise_level, (h, w, 3))
            res = img + noise
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))
