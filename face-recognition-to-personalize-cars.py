## @Author: OMAR LEFRERE
## @Date:   03/2023
## @Last Modified by: OMAR LEFRERE

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget, QDesktopWidget, QLineEdit, QDialog, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QMovie
from PyQt5.QtCore import Qt, QSize,QThread, pyqtSignal, QTimer
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QIcon, QPixmap, QPainter
import qdarkstyle
import sys
import threading
import cv2 
import numpy as np
import mtcnn
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import os 
import pickle 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Lambda, add, GlobalAveragePooling2D, Convolution2D, LocallyConnected2D, ZeroPadding2D, concatenate, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import datetime
import time
import csv
import gtts
import pygame
 

class MainWindow(QMainWindow):

    
    def __init__(self):
        super().__init__()
        self.w = None

        # Personnalisation de la fenêtre
        self.setWindowTitle('Mon application')
        
        self.setGeometry(600, 100, 700, 700)
        #remove title barre 
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setMouseTracking(True)
        self.window_moving = False
        self.window_pos = None
        # Ajout de l'image de fond
        bg_label = QLabel(self)
        bg_label.setGeometry(0, 0, 700, 700)
        bg_label.setPixmap(QPixmap("UI\background.jpg"))

        # Ajout du nom de l'application au centre
        app_name_label = QLabel('Customizing Your Car Experience', self)
        app_name_label.setGeometry(100, 130, 500, 500)
        app_name_label.setStyleSheet('font-size: 30px; font-weight: bold; color: #191970;')
        

        # Ajout du bouton "Start"
        start_button = QPushButton('Start', self)
        #start_button.clicked.connect(self.show_new_window)
        start_button.clicked.connect(self.execute_code)
        start_button.setGeometry(310, 550, 80, 80)
        renderer = QSvgRenderer("UI\curseur-doigt.svg")
        icon_size = QSize(24, 24)
        icon_pixmap = QPixmap(icon_size)
        icon_pixmap.fill(Qt.transparent)
        painter = QPainter(icon_pixmap)
        painter.setBrush(Qt.red)
        renderer.render(painter)
        painter.end()

        start_button.setIcon(QIcon(icon_pixmap))
        start_button.setIconSize(icon_size)
        #start_button.clicked.connect(self.show_new_window)

        self.setWindowIcon(QIcon('UI\logo.png'))
        
        # Create label to display icon
        icon_label = QLabel(self)
        icon_label.setGeometry(235, 100, 250, 250)
        icon_label.setPixmap(QPixmap('UI\logo.png').scaled(icon_label.size(), Qt.KeepAspectRatio))
         # Ajout du bouton de mode light/dark
        self.dark_mode = True
        self.dark_mode_button = QPushButton(self)
        self.dark_mode_button.setGeometry(10, 10, 50, 50)
        self.dark_mode_button.setText("Light")
        self.dark_mode_button.setStyleSheet("QPushButton {""border-radius: 25px;"
                                            "font-size: 10px;"
                                            "}")  
                                            
         # Set the border radius to create a circular shape
        self.dark_mode_button.clicked.connect(self.switch_mode)
        renderer_dark = QSvgRenderer("UI\soleil_1.svg")
        icon_size_dark = QSize(24, 24)
        icon_pixmap_dark = QPixmap(icon_size_dark)
        icon_pixmap_dark.fill(Qt.transparent)
        painter_dark = QPainter(icon_pixmap_dark)
        painter_dark.setBrush(Qt.red)
        renderer_dark.render(painter_dark)
        painter_dark.end()
        self.dark_mode_button.setIcon(QIcon(icon_pixmap_dark))
        self.dark_mode_button.setIconSize(icon_size_dark)

        button_exit = QPushButton(self)
        button_exit.setText("")
        button_exit.setGeometry(650, 0, 30, 30)
        button_exit.setStyleSheet("QPushButton {""border-radius: 25px;""font-size: 10px;""}")  
        # Set the border radius to create a circular shape

        renderer3 = QSvgRenderer("UI\croix-cercle.svg")  # Replace with the path to your icon file in the resource file
        icon_size3 = QSize(24, 24)  # Set the desired size of the icon
        icon_pixmap3 = QPixmap(icon_size3)
        icon_pixmap3.fill(Qt.transparent)  # Use Qt.transparent instead of just transparent
        painter3 = QPainter(icon_pixmap3)
        # Customize the color of the icon
        painter3.setBrush(Qt.blue)
        renderer3.render(painter3)
        painter3.end()

        # Set the Flaticon icon as the button's icon
        button_exit.setIcon(QIcon(icon_pixmap3))
        button_exit.setIconSize(icon_size3)

        # Connect button's clicked signal to the custom slot
        button_exit.clicked.connect(self.close)
    
    
    def switch_mode(self):
        # Switch between dark and light mode
        if self.dark_mode:
            self.setStyleSheet('')
            self.dark_mode = False
            renderer_dark = QSvgRenderer("UI\soleil_1.svg")
            icon_size_dark = QSize(24, 24)
            icon_pixmap_dark = QPixmap(icon_size_dark)
            icon_pixmap_dark.fill(Qt.transparent)
            painter_dark = QPainter(icon_pixmap_dark)
            painter_dark.setBrush(Qt.red)
            renderer_dark.render(painter_dark)
            painter_dark.end()
            self.dark_mode_button.setText("Light")

            self.dark_mode_button.setIcon(QIcon(icon_pixmap_dark))
            self.dark_mode_button.setIconSize(icon_size_dark)

        else:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            self.dark_mode = True
            renderer_dark = QSvgRenderer("UI\lune.svg")
            icon_size_dark = QSize(24, 24)
            icon_pixmap_dark = QPixmap(icon_size_dark)
            icon_pixmap_dark.fill(Qt.transparent)
            painter_dark = QPainter(icon_pixmap_dark)
            painter_dark.setBrush(Qt.red)
            renderer_dark.render(painter_dark)
            painter_dark.end()
            self.dark_mode_button.setText("Dark")

            self.dark_mode_button.setIcon(QIcon(icon_pixmap_dark))
            self.dark_mode_button.setIconSize(icon_size_dark)
            bg_label = QLabel(self)
            pixmap = QPixmap("UI\darkmode.jpg")  # Replace with the path to your background image
            bg_label.setPixmap(pixmap)
            bg_label.setGeometry(100, 0, self.width(), self.height())
        
    
    #button start connection
    def execute_code(self):
        
        print("Button start clicked!")
        pygame.init()
        def scaling(x, scale):
            return x * scale
        
        def InceptionResNetV2():
            
            inputs = Input(shape=(160, 160, 3))
            x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_1a_3x3') (inputs)
            x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
            x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)
            x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_2a_3x3') (x)
            x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
            x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)
            x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name= 'Conv2d_2b_3x3') (x)
            x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
            x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)
            x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
            x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False, name= 'Conv2d_3b_1x1') (x)
            x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
            x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)
            x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_4a_3x3') (x)
            x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
            x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)
            x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_4b_3x3') (x)
            x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
            x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)
            
            # 5x Block35 (Inception-ResNet-A block):
            branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block35_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_1_Conv2d_0b_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
            branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0a_1x1') (x)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0b_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0c_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
            branches = [branch_0, branch_1, branch_2]
            mixed = Concatenate(axis=3, name='Block35_1_Concatenate')(branches)
            up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_1_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
            x = add([x, up])
            x = Activation('relu', name='Block35_1_Activation')(x)
            
            branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block35_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_1_Conv2d_0b_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
            branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0a_1x1') (x)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0b_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0c_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
            branches = [branch_0, branch_1, branch_2]
            mixed = Concatenate(axis=3, name='Block35_2_Concatenate')(branches)
            up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_2_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
            x = add([x, up])
            x = Activation('relu', name='Block35_2_Activation')(x)
            
            branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block35_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_1_Conv2d_0b_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
            branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0a_1x1') (x)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0b_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0c_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
            branches = [branch_0, branch_1, branch_2]
            mixed = Concatenate(axis=3, name='Block35_3_Concatenate')(branches)
            up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_3_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
            x = add([x, up])
            x = Activation('relu', name='Block35_3_Activation')(x)
            
            branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block35_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_1_Conv2d_0b_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
            branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0a_1x1') (x)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0b_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0c_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
            branches = [branch_0, branch_1, branch_2]
            mixed = Concatenate(axis=3, name='Block35_4_Concatenate')(branches)
            up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_4_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
            x = add([x, up])
            x = Activation('relu', name='Block35_4_Activation')(x)
            
            branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block35_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_1_Conv2d_0b_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
            branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0a_1x1') (x)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0b_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
            branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0c_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
            branches = [branch_0, branch_1, branch_2]
            mixed = Concatenate(axis=3, name='Block35_5_Concatenate')(branches)
            up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_5_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
            x = add([x, up])
            x = Activation('relu', name='Block35_5_Activation')(x)
        
            # Mixed 6a (Reduction-A block):
            branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_0_Conv2d_1a_3x3') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0b_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
            branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_1a_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
            branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
            branches = [branch_0, branch_1, branch_pool]
            x = Concatenate(axis=3, name='Mixed_6a')(branches)
        
            # 10x Block17 (Inception-ResNet-B block):
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_1_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_1_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_1_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_2_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_2_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_2_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_3_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_3_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_3_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_4_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_4_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_4_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_5_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_5_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_5_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_6_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_6_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_6_Activation')(x)	
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_7_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_7_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_7_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_7_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_8_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_8_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_8_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_8_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_9_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_9_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_9_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_9_Activation')(x)
            
            branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block17_10_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0b_1x7') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0b_1x7_Activation')(branch_1)
            branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0c_7x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0c_7x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block17_10_Concatenate')(branches)
            up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_10_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
            x = add([x, up])
            x = Activation('relu', name='Block17_10_Activation')(x)
        
            # Mixed 7a (Reduction-B block): 8 x 8 x 2080	
            branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_0a_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
            branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_1a_3x3') (branch_0)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
            branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_1a_3x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
            branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0a_1x1') (x)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
            branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0b_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
            branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_1a_3x3') (branch_2)
            branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
            branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
            branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
            branches = [branch_0, branch_1, branch_2, branch_pool]
            x = Concatenate(axis=3, name='Mixed_7a')(branches)
        
            # 5x Block8 (Inception-ResNet-C block):
            
            branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0b_1x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
            branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0c_3x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block8_1_Concatenate')(branches)
            up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_1_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
            x = add([x, up])
            x = Activation('relu', name='Block8_1_Activation')(x)
            
            branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block8_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0b_1x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0b_1x3_Activation')(branch_1)
            branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0c_3x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0c_3x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block8_2_Concatenate')(branches)
            up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_2_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
            x = add([x, up])
            x = Activation('relu', name='Block8_2_Activation')(x)
            
            branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block8_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0b_1x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0b_1x3_Activation')(branch_1)
            branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0c_3x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0c_3x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block8_3_Concatenate')(branches)
            up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_3_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
            x = add([x, up])
            x = Activation('relu', name='Block8_3_Activation')(x)
            
            branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block8_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0b_1x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0b_1x3_Activation')(branch_1)
            branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0c_3x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0c_3x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block8_4_Concatenate')(branches)
            up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_4_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
            x = add([x, up])
            x = Activation('relu', name='Block8_4_Activation')(x)
            
            branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block8_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0b_1x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0b_1x3_Activation')(branch_1)
            branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0c_3x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0c_3x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block8_5_Concatenate')(branches)
            up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_5_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
            x = add([x, up])
            x = Activation('relu', name='Block8_5_Activation')(x)
            
            branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_0_Conv2d_1x1') (x)
            branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
            branch_0 = Activation('relu', name='Block8_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
            branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0a_1x1') (x)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
            branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0b_1x3') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
            branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0c_3x1') (branch_1)
            branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
            branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
            branches = [branch_0, branch_1]
            mixed = Concatenate(axis=3, name='Block8_6_Concatenate')(branches)
            up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_6_Conv2d_1x1') (mixed)
            up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 1})(up)
            x = add([x, up])
            
            # Classification block
            x = GlobalAveragePooling2D(name='AvgPool')(x)
            x = Dropout(1.0 - 0.8, name='Dropout')(x)
            # Bottleneck
            x = Dense(128, use_bias=False, name='Bottleneck')(x)
            x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)
        
            # Create model
            model = Model(inputs, x, name='inception_resnet_v1')
        
            return model
        
        ####################################################################################################################pathsandvairables#########
        
        face_data = 'Faces'
        required_shape = (160,160)
        face_encoder = InceptionResNetV2()
        path = "facenet_keras_weights.h5"
        face_encoder.load_weights(path)
        #face_detector = mtcnn()
        face_detector = mtcnn.MTCNN()
        encodes = []
        encoding_dict = dict()
        l2_normalizer = Normalizer('l2')
        
        def normalize(img):
            mean, std = img.mean(), img.std()
            return (img - mean) / std
        
        
        for face_names in os.listdir(face_data):
            person_dir = os.path.join(face_data,face_names)
        
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir,image_name)
        
                img_BGR = cv2.imread(image_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        
                x = face_detector.detect_faces(img_RGB)
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1) , abs(y1)
                x2, y2 = x1+width , y1+height
                face = img_RGB[y1:y2 , x1:x2]
                
                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = face_encoder.predict(face_d)[0]
                encodes.append(encode)
        
            if encodes:
                encode = np.sum(encodes, axis=0 )
                encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                encoding_dict[face_names] = encode
            
        path = 'encodings.pkl'
        with open(path, 'wb') as file:
            pickle.dump(encoding_dict, file)
        
        ########################################################################################################
        
        confidence_t=0.99
        recognition_t=0.5
        required_size = (160,160)
        
        def get_face(img, box):
            x1, y1, width, height = box
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img[y1:y2, x1:x2]
            return face, (x1, y1), (x2, y2)
        
        def get_encode(face_encoder, face, size):
            face = normalize(face)
            face = cv2.resize(face, size)
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            return encode
        
        
        def load_pickle(path):
            with open(path, 'rb') as f:
                encoding_dict = pickle.load(f)
            return encoding_dict
            
   
        
        def apply_settings(user_name,image):
            with open('user_settings.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == user_name:
                        seat_position = tuple(map(int, row[1].split(',')))
                        mirror_settings = tuple(row[2].split(','))
                        #cv2.putText(image, user_name +  "SP=" + str(seat_position) + "MP=" + str(mirror_settings), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
                        #cv2.imshow('resulat', image)
                        #print("welcome back",user_name,"your seat_position is",seat_position,"your mirror_settings is ",mirror_settings,"hope you a nice driving experience")
                        if not os.path.exists(user_name + ".mp3"):
                            tts = gtts.gTTS("welcomve back"+ user_name + "your seat position is"+ str(seat_position) +"your mirror settings is " + str(mirror_settings) +"hope you a nice driving experience")
                            tts.save(user_name + ".mp3")
                            sound = pygame.mixer.Sound(user_name + ".mp3")
                            sound.play()
                            pygame.time.wait(int(sound.get_length() * 1000))
                
                        else:
                            sound = pygame.mixer.Sound(user_name + ".mp3")
                            sound.play()
                            pygame.time.wait(int(sound.get_length() * 1000))
                        dialog = DialogWindow()
                        dialog.show_last_message()
                return
        
        def detect(img, detector, encoder, encoding_dict):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)
            for res in results:
                if res['confidence'] < confidence_t:
                    continue
                face, pt_1, pt_2 = get_face(img_rgb, res['box'])
                encode = get_encode(encoder, face, required_size)
                encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in encoding_dict.items():
                    dist = cosine(db_encode, encode)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist
        
                if name == 'unknown':
                    
                    
                    print("Please enter your name  ")
                    
                    dialog = DialogWindow()
                    dialog.exec_()
                   
                    var1, var2, var3 = dialog.get_information()
                    user_name=var1
                    seat_position=var2
                    mirror_settings=var3

                    # Create a folder with the user's name if it doesn't exist
                    if not os.path.exists(user_name):
                        path = os.path.join('Faces', user_name)
                        os.mkdir(path)
                    else:
                        dialog.show_error_message
                        dialog.exec_()
                        var1, var2, var3 = dialog.get_information()
                        user_name=var1

                    # Save the image of the detected face in the user's folder with a unique name
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    camera = cv2.VideoCapture(0)
                    while True:
                        ret, img = camera.read()
                        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        faces = face_cascade.detectMultiScale(img, 1.3, 5)
                        for (x, y, w, h) in faces:
                            roi_gray = img[y:y + h, x:x + w]
                            roi_color = img[y:y + h, x:x + w]
                            eyes = eye_cascade.detectMultiScale(roi_gray)
                            if len(eyes) == 2:  # Only save if two eyes are detected
                                face_img = cv2.resize(roi_gray, (1200, 1000))  # Resize face image
                                image_name = f"{user_name}_{current_time}.jpg"
                                cv2.imwrite(os.path.join(path, image_name), face_img)
                                print("your Face",user_name,"is saved thanks.")
                                settings = {}
                                settings['user_name'] = user_name
                                settings['seat_position'] = seat_position
                                settings['mirror_settings'] = mirror_settings
                                save_settings(user_name,seat_position,mirror_settings)
                                dialog.show_ok_message()
                                #print("setting saved thanks:",settings)
                                cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
                                apply_settings(user_name,img)
                        break
                    break
            
                else:
                    apply_settings(name,img)
                        
                    break
  
                    #return
            
                #return img
            
        #create csv file
        def create_csv_file():
            headers = ['user name','seat postion', 'mirror position',]
            with open('user_settings.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        
        def save_settings(user_name,seat_position,mirror_settings):
            if not os.path.isfile('user_settings.csv'):
                # create the CSV file if it doesn't exist
                create_csv_file()
            
            # write the user's settings to the CSV file
            with open('user_settings.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([user_name,seat_position,mirror_settings])
        
        if __name__ == "__main__":
            required_shape = (160,160)
            face_encoder = InceptionResNetV2()
            path_m = "facenet_keras_weights.h5"
            face_encoder.load_weights(path_m)
            encodings_path = 'encodings.pkl'
            face_detector = mtcnn.MTCNN()
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            encoding_dict = load_pickle(encodings_path)
            
            cap = cv2.VideoCapture(0)
        
            while cap.isOpened():
                ret, frame = cap.read()
        
                frame= detect(frame , face_detector , face_encoder , encoding_dict)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
    
 

class DialogWindow(QDialog):
    def __init__(self):
        super().__init__()
        

        # Set window size
        self.setGeometry(600, 100, 700, 700)
        
        bg_label = QLabel(self)
        bg_label.setGeometry(0, 0, 700, 700)
        bg_label.setPixmap(QPixmap("UI\background.jpg"))
        
        # Remove title bar and close icon
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setMouseTracking(True)
        self.window_moving = False
        self.window_pos = None
        
        # Set window icon
        self.setWindowIcon(QIcon('UI\visage.png'))
        
        # Create label to display icon
        icon_label = QLabel(self)
        icon_label.setGeometry(500, 30, 150, 150)
        icon_label.setPixmap(QPixmap('UI\visage.png').scaled(icon_label.size(), Qt.KeepAspectRatio))
        
        
        # Create a QLabel  add user name
        label = QLabel(self)
        
        # Create an icon from an image file
        icon = QIcon("UI\contact.png")  # Replace "icon.png" with the path to your icon file
        
        # Set the icon on the label
        label.setPixmap(icon.pixmap(52, 52))  # Set the pixmap size as desired
        
        # Set the label's position and size
        label.setGeometry(50, 200, 52, 52)
        
        # Create a QLabel  add seat position
        label = QLabel(self)
        
        # Create an icon from an image file
        icon = QIcon("UI\car.png")  # Replace "icon.png" with the path to your icon file
        
        # Set the icon on the label
        label.setPixmap(icon.pixmap(52, 52))  # Set the pixmap size as desired
        
        # Set the label's position and size
        label.setGeometry(50, 300, 52, 52)
        
        # Create a QLabel  add mirror position
        label = QLabel(self)
        
        # Create an icon from an image file
        icon = QIcon("UI\mirror-removebg.png")  # Replace "icon.png" with the path to your icon file
        
        # Set the icon on the label
        label.setPixmap(icon.pixmap(52, 52))  # Set the pixmap size as desired
        
        # Set the label's position and size
        label.setGeometry(50, 400, 52, 52)
 
        
        # Set the QLineEdits
        self.name_input = QLineEdit(self)
        self.name_input.setGeometry(300, 200, 300, 40)
        self.name_input.setPlaceholderText("Enter your name")

        self.seat_input = QLineEdit(self)
        self.seat_input.setGeometry(300, 300, 300, 40)
        self.seat_input.setPlaceholderText("Enter seat position")

        self.miror_input = QLineEdit(self)
        self.miror_input.setGeometry(300, 400, 300, 40)
        self.miror_input.setPlaceholderText("Enter mirror position")

        # ...

        # Create a submit button
        submit_button = QPushButton("Submit", self)
        submit_button.setGeometry(300, 550, 70, 70)
        icon = QIcon("UI\submit.png")  # Replace "submit_icon.png" with the path to your icon file
        submit_button.setIcon(icon)
        icon_size = submit_button.sizeHint().height() - 10  # Adjust the size as desired
        #submit_button.setIconSize(Qt.QSize(icon_size, icon_size))
        submit_button.clicked.connect(self.submit)
        

        
        # Initialize instance variables
        self.var1 = None
        self.var2 = None
        self.var3 = None


        # Add a button to switch between dark and light mode
        self.dark_mode = True
        self.dark_mode_button = QPushButton(self)
        self.dark_mode_button.setGeometry(10, 10, 50, 50)
        self.dark_mode_button.setText("Light")
        self.dark_mode_button.setStyleSheet("QPushButton {""border-radius: 25px;""font-size: 15px;""}")  # Set the border radius to create a circular shape
        self.dark_mode_button.clicked.connect(self.switch_mode)
        renderer_dark = QSvgRenderer("UI\soleil_1.svg")
        icon_size_dark = QSize(24, 24)
        icon_pixmap_dark = QPixmap(icon_size_dark)
        icon_pixmap_dark.fill(Qt.transparent)
        painter_dark = QPainter(icon_pixmap_dark)
        painter_dark.setBrush(Qt.red)
        renderer_dark.render(painter_dark)
        painter_dark.end()
        self.dark_mode_button.setIcon(QIcon(icon_pixmap_dark))
        self.dark_mode_button.setIconSize(icon_size_dark)
        
        button_exit = QPushButton(self)
        button_exit.setText("Back")
        button_exit.setGeometry(30, 630, 100, 50)
        button_exit.setStyleSheet("QPushButton {""border-radius: 25px;""font-size: 15px;""}")  # Set the border radius to create a circular shape
        
        renderer3 = QSvgRenderer("UI\Back.svg")  # Replace with the path to your icon file in the resource file
        icon_size3 = QSize(30, 30)  # Set the desired size of the icon
        icon_pixmap3 = QPixmap(icon_size3)
        icon_pixmap3.fill(Qt.transparent)  # Use Qt.transparent instead of just transparent
        painter3 = QPainter(icon_pixmap3)
        # Customize the color of the icon
        painter3.setBrush(Qt.blue)
        renderer3.render(painter3)
        painter3.end()
        
        # Set the Flaticon icon as the button's icon
        button_exit.setIcon(QIcon(icon_pixmap3))
        button_exit.setIconSize(icon_size3)
        
        # Connect button's clicked signal to the custom slot
        button_exit.clicked.connect(self.close)
        #self.show()
        self.user_name = None
        self.seat_position = None
        self.mirror_settings = None
        
        
    def submit(self):
        self.var1 = self.name_input.text()
        self.var2 = self.seat_input.text()
        self.var3 = self.miror_input.text()
        self.accept()
    
 
        
    def get_information(self):
        return self.var1, self.var2, self.var3
    
    
    
    def show_ok_message(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("information")
        message_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }" "QLabel { color: #333333; font-size: 16px; }")
        message_box.setIcon(QMessageBox.Information)
        message_box.setTextFormat(Qt.RichText)
        message_box.setText("<b>Information entered successfully:</b> {}".format(self.var1))
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.button(QMessageBox.Ok).setStyleSheet("QPushButton { background-color: #337ab7; color: #ffffff; }""QPushButton:hover { background-color: #23527c; }""QPushButton:pressed { background-color: #1d4566; }")
        message_box.exec_()
    

        
    def show_error_message(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("Error")
        message_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }" "QLabel { color: #333333; font-size: 16px; }")
        message_box.setTextFormat(Qt.RichText)
        message_box.setText("<b>incorrect Information try again:</b> {}".format(self.var1))
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.button(QMessageBox.Ok).setStyleSheet("QPushButton { background-color: #337ab7; color: #ffffff; }""QPushButton:hover { background-color: #23527c; }""QPushButton:pressed { background-color: #1d4566; }")
        message_box.exec_()
        
    def show_last_message(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("Information")
        message_box.setStyleSheet("QMessageBox { background-color: #f0f0f0; }" "QLabel { color: #333333; font-size: 16px; }")
        message_box.setTextFormat(Qt.RichText)
        message_box.setText("<strong>loading your setting ...,goodbye hope you a nice driving experience </strong>")
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.button(QMessageBox.Ok).setStyleSheet("QPushButton { background-color: #337ab7; color: #ffffff; }""QPushButton:hover { background-color: #23527c; }""QPushButton:pressed { background-color: #1d4566; }")
        message_box.exec_()
        sys.exit()
        

    
    def switch_mode(self):
    # Switch between dark and light mode
        if self.dark_mode:
            self.setStyleSheet('')
            self.dark_mode = False
            renderer_dark = QSvgRenderer("UI\soleil_1.svg")
            icon_size_dark = QSize(24, 24)
            icon_pixmap_dark = QPixmap(icon_size_dark)
            icon_pixmap_dark.fill(Qt.transparent)
            painter_dark = QPainter(icon_pixmap_dark)
            painter_dark.setBrush(Qt.red)
            renderer_dark.render(painter_dark)
            painter_dark.end()
            self.dark_mode_button.setText("Light")
            self.dark_mode_button.setIcon(QIcon(icon_pixmap_dark))
            self.dark_mode_button.setIconSize(icon_size_dark)
        else:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            self.dark_mode = True
            renderer_dark = QSvgRenderer("UI\lune.svg")
            icon_size_dark = QSize(24, 24)
            icon_pixmap_dark = QPixmap(icon_size_dark)
            icon_pixmap_dark.fill(Qt.transparent)
            painter_dark = QPainter(icon_pixmap_dark)
            painter_dark.setBrush(Qt.red)
            renderer_dark.render(painter_dark)
            painter_dark.end()
            self.dark_mode_button.setText("Dark")
    
            self.dark_mode_button.setIcon(QIcon(icon_pixmap_dark))
            self.dark_mode_button.setIconSize(icon_size_dark)



class LoadingDialog(QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Loading")
        self.setFixedSize(700, 200)
        
        self.loading_label = QLabel(self)
        self.movie = QMovie("setting.gif")
        self.loading_label.setMovie(self.movie)
        
        self.movie.start()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
