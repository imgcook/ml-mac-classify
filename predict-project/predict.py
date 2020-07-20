import allspark
import io
import numpy as np
import json
from PIL import Image
import requests
import threading
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

model = load_model('./train/cnn.model.h5')
# pred的输入应该是一个images的数组，而且图片都已经转为numpy数组的形式
# pred = model.predict(['./validation/button/button-demoplus-20200216-16615.png'])

#这个顺序一定要与label.json顺序相同，模型输出是一个数组，取最大值索引为预测值
Label = [
    "button",
    "keyboard",
    "searchbar",
    "switch"
    ]
testPath = "./test/button.png"

images = []
image = cv2.imread(testPath)
# image = cv2.cvtColor(image,cv2.CLOLR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image,(256,256))
images.append(image)
images = np.asarray(images)

pred = model.predict(images)

print(pred)

max_ = np.argmax(pred)
print('test.jpg的预测结果为：',Label[max_])
