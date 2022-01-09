import sys
import os
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import ImageQt
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
from functions import LK_opticalFlow, LK_opticalFlow_openCV

class displayLabel(QLabel):
    def customInit(self):        
        self.haveImage = False
        self.setStyleSheet('background-color: rgb(255, 255, 255);')
        self.setScaledContents(True)
        self.selectedPoint = []
        self.parent().btn_startTransform.setEnabled(False)

    def setImage(self, filePath):
        self.customInit()
        self.image = QPixmap(filePath)
        self.haveImage = True

    def paintEvent(self, event): 
        if not self.haveImage:
            return       

        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

        #
        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
        for x, y in self.selectedPoint:            
            painter.drawPoint(x, y) 


    def mousePressEvent(self, event):
        if not self.haveImage:
            return

        pos = event.pos()

        self.selectedPoint.append([pos.x(), pos.y()])

        if len(self.selectedPoint) > 0:
            self.parent().btn_startTransform.setEnabled(True)
        else:
            self.parent().btn_startTransform.setEnabled(False)

        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(1500, 800)
        self.setWindowTitle('tracking')

        # param
        self.trackingPoints = []
        self.processedImage = None          
        
        # Text
        self.label = QLabel(self)        
        self.label.setText("Select image first")
        self.label.setGeometry(100, 150, 600, 50)
        self.label.setAlignment(Qt.AlignCenter) 
        self.label.setFont(QFont('Arial', 20)) 

        # Buttom
        self.btn_selectPrevImage = QPushButton(self)
        self.btn_selectPrevImage.setText("Select pre Image")
        self.btn_selectPrevImage.setFont(QFont('Arial', 20)) 
        self.btn_selectPrevImage.setGeometry(100, 100, 600, 50)        
        self.btn_selectPrevImage.clicked.connect(self.open_prev_image)

        self.btn_selectNextImage = QPushButton(self)
        self.btn_selectNextImage.setText("Select next Image")
        self.btn_selectNextImage.setFont(QFont('Arial', 20)) 
        self.btn_selectNextImage.setGeometry(800, 100, 600, 50)        
        self.btn_selectNextImage.clicked.connect(self.open_next_image)

        self.btn_startTransform = QPushButton(self)
        self.btn_startTransform.setText('start')
        self.btn_startTransform.setFont(QFont('Arial', 20))
        self.btn_startTransform.setGeometry(100, 650, 600, 50)
        self.btn_startTransform.setEnabled(False)
        self.btn_startTransform.clicked.connect(self.start_tracking)

        self.btn_saveImage = QPushButton(self)
        self.btn_saveImage.setText('Save')
        self.btn_saveImage.setFont(QFont('Arial', 20))
        self.btn_saveImage.setGeometry(800, 650, 600, 50)
        self.btn_saveImage.setEnabled(False)
        self.btn_saveImage.clicked.connect(self.save_image)

        # Display
        self.displayLabel = displayLabel(self)    
        self.displayLabel.customInit()       
        self.displayLabel.setGeometry(100, 200, 600, 400)

        self.disployLabel_processed = QLabel(self)
        self.disployLabel_processed.setGeometry(800, 200, 600, 400)
        self.disployLabel_processed.setStyleSheet("background-color: white")
    
    def open_prev_image(self):
        self.fileName_prev = QFileDialog.getOpenFileName(self, \
            'Open file', 'D:/Documents/GitHub/ImageProcessing_HW4_LucasKanadeFlow/src',"Image files (*.jpg *.bmp *.png)")[0]
        if self.fileName_prev == '':
            return

        self.displayLabel.setImage(QPixmap(self.fileName_prev))

        self.label.setText('XX')
    
    def open_next_image(self):
        self.fileName_next = QFileDialog.getOpenFileName(self, \
            'Open file', 'D:/Documents/GitHub/ImageProcessing_HW4_LucasKanadeFlow/src',"Image files (*.jpg *.bmp *.png)")[0]
        if self.fileName_next == '':
            return

        # self.disployLabel_processed.setPixmap(QPixmap(self.fileName_next))
        frame = cv2.imread(self.fileName_next)     
        bytesPerLine = 3 * 600
        frame = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        qImg = QImage(frame, 600, 400, bytesPerLine, QImage.Format_RGB888)
        self.disployLabel_processed.setPixmap(QPixmap.fromImage(qImg))

    def save_image(self):
        saveFileName = os.path.splitext(self.fileName_prev.split('/')[-1])[0]        
        cv2.imwrite(f'./results/{saveFileName}.jpg', self.processedImage)
    

    def start_tracking(self):
        img_prev = cv2.imread(self.fileName_prev)
        img_next = cv2.imread(self.fileName_next)
        height, width, channel = img_next.shape

        window_size = [25, 25]
        trackingPoint = self.displayLabel.selectedPoint        
        trackingPoint = [[y / 400. * height, x / 600. * width] for x, y in trackingPoint]

        iter_points = LK_opticalFlow(img_prev, img_next, trackingPoint, window_size)
        # iter_points = LK_opticalFlow_openCV(img_prev, img_next, trackingPoint, window_size)

        
        output_frame = img_next.copy()
        for ps in iter_points:            
            for s in ps[1:-1]:
                output_frame = cv2.circle(output_frame, [s[1], s[0]], 1, (0, 0, 255), -1)
            output_frame = cv2.circle(output_frame, [ps[0, 1], ps[0, 0]], 4, (0, 255, 0), -1)
            output_frame = cv2.circle(output_frame, [ps[-1, 1], ps[-1, 0]], 4, (255, 0, 0), -1)
        
        self.processedImage = output_frame.copy()
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        bytesPerLine = 3 * 600
        output_frame = cv2.resize(output_frame, (600, 400), interpolation=cv2.INTER_AREA)
        qImg = QImage(output_frame, 600, 400, bytesPerLine, QImage.Format_RGB888)
        self.disployLabel_processed.setPixmap(QPixmap.fromImage(qImg))


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())