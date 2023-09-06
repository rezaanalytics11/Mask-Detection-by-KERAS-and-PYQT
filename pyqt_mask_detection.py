from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QFormLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget,
  QPushButton, QVBoxLayout, QHBoxLayout,QGridLayout,QLineEdit)
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        hbox0 = QHBoxLayout()
        self.label0 = QLabel('Please enter the picture URL and press the key', self)
        hbox0.addWidget(self.label0)

        self.file_name = QLineEdit(self)
        hbox0.addWidget(self.file_name)


        hbox1 = QHBoxLayout()
        self.label1 = QLabel('', self)
        hbox1.addWidget(self.label1)


        # self.label2 = QLabel('', self)
        # hbox1.addWidget(self.label2)

        self.label3 = QLabel('', self)
        hbox1.addWidget(self.label3)

        Button = QPushButton('Key')
        hbox0.addWidget(Button)
        Button.clicked.connect(self.addurl)


        vbox = QVBoxLayout()
        vbox.addLayout(hbox0)
        vbox.addLayout(hbox1)
        #vbox.setAlignment(Qt.AlignHCenter)
        #vbox.setAlignment(Qt.AlignVCenter)
        self.setLayout(vbox)

        #self.setGeometry(400, 400, 300, 150)
        self.setWindowTitle('Box layout example, QHBoxLayout, QVBoxLayout')
        self.show()

    def addurl(self):

        a=self.file_name.text()
        print(a)
        self.draw(a)


    def draw(self,file_name):
        print(2*file_name)





        new_model = tf.keras.models.load_model(r'C:\Users\Ariya Rayaneh\Desktop\mask_detector (1).model')

        img = cv2.imread(r'C:\Users\Ariya Rayaneh\Desktop\{}.jpg'.format(file_name))
        image = img.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h, w) = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        haar_cascade_face = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop/haarcascade_frontalface_default.xml')
        faces_rects = haar_cascade_face.detectMultiScale(
            image_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(1, 1),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print('Faces found: ', len(faces_rects))

        image = cv2.resize(image, (224, 224))

        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        (mask, withoutMask) = new_model.predict(image)[0]

        if mask < withoutMask:
            color = (0, 0, 255)
            label = 'Not Wearing Mask'
            t='Not Wearing Mask'
        else:
            color = (0, 255, 0)
            label = 'Wearing Mask'
            t = 'Wearing Mask'

        for (x, y, w, h) in faces_rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y+130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        print(mask, withoutMask)
        cv2.imwrite(r"C:\Users\Ariya Rayaneh\Desktop\human20_output.jpg", img)
        self.label1.setPixmap(QPixmap(r"C:\Users\Ariya Rayaneh\Desktop\human20_output.jpg"))
        self.label3.setText(t)

if __name__ == '__main__':
 app = QApplication(sys.argv)
 ex = Example()
 sys.exit(app.exec_())
