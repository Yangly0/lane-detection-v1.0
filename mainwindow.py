import sys
import time

import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from lanedetection import LaneDetection


class Ui_MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.format = 0                        # 0 空格式 1 图片格式 2 视频格式
        self.data = None                       # 数据

        self.lanedetection = LaneDetection()   # 车道线检测类
        
        self.timer1 = VideoTimer()             # 视频显示多线程
        self.timer1._signal.connect(self.show_input_video)

        self.timer2 = VideoTimer()             # 视频显示多线程
        self.timer2._signal.connect(self.show_output_video)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(1000, 600)
        MainWindow.setFixedSize(1000, 600)
        self.centralwidget = QWidget(MainWindow)

        # 图片控件
        self.label_inputs = QLabel(self.centralwidget)
        self.label_inputs.setGeometry(QRect(50, 70, 400, 400))
        self.label_inputs.setFrameShape(QFrame.Box)
        self.label_inputs.setObjectName("label_inputs")

        self.label_inputsName = QLabel(self.centralwidget)
        self.label_inputsName.setGeometry(QRect(175, 25, 150, 50))
        self.label_inputsName.setObjectName("label_inputsname")
        
        self.label_outputs = QLabel(self.centralwidget)
        self.label_outputs.setGeometry(QRect(550, 70, 400, 400))
        self.label_outputs.setFrameShape(QFrame.Box)
        self.label_outputs.setObjectName("label_outputs")

        self.label_outputsName = QLabel(self.centralwidget)
        self.label_outputsName.setGeometry(QRect(685, 25, 150, 50))
        self.label_outputsName.setObjectName("label_outputsname")
        
        # 按钮控件
        # 打开图片或视频
        self.pushButton_open = QPushButton(self.centralwidget)
        self.pushButton_open.setGeometry(QRect(150, 500, 150, 50))
        self.pushButton_open.setObjectName("pushButton_open")
        self.pushButton_open.clicked.connect(self.open)
        self.pushButton_open.setEnabled(True)
        
        # 运行图片或视频
        self.pushButton_run = QPushButton(self.centralwidget)
        self.pushButton_run.setGeometry(QRect(665, 500, 150, 50))
        self.pushButton_run.setObjectName("pushButton_run")
        self.pushButton_run.clicked.connect(self.run)
        self.pushButton_run.setEnabled(False)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.label_inputs.setText(_translate("MainWindow", "原始图片或视频"))
        self.label_inputsName.setText(_translate("MainWindow", "原始图片或视频"))
        self.label_outputs.setText(_translate("MainWindow", "处理图片或视频"))
        self.label_outputsName.setText(_translate("MainWindow", "处理图片或视频"))
        
        self.pushButton_open.setText(_translate("MainWindow", "打开图片或视频"))
        self.pushButton_run.setText(_translate("MainWindow", "处理图片或视频"))
        
    def open(self):
        # 打开图片文件
        file_name = QFileDialog.getOpenFileName(self.centralwidget, "选择图片或视频", "./assets", "ALL Files(*.*)")
        if not file_name[0]:
            result=QMessageBox.question(
                QWidget(),
                '问题提示', 
                '路径未选择', 
                QMessageBox.Yes,
                QMessageBox.Yes  # 默认关闭界面选择No
            )
            return
        
        if file_name[0].endswith('.jpg'):
            self.format = 1
            # 读取图片
            self.data = cv2.imread(file_name[0], 1)
            self.show_input_image(self.data)
            self.pushButton_open.setEnabled(False)
            self.pushButton_run.setEnabled(True)
            
        elif file_name[0].endswith('.mp4'):
            self.format = 2
            # 创建一个视频读写类
            self.data = cv2.VideoCapture(file_name[0])
            # self.outputs = cv.VideoCapture(0)
            if not self.data.isOpened():
                result=QMessageBox.question(
                QWidget(),
                '问题提示', 
                '打开mp4文件失败', 
                QMessageBox.Yes,
                QMessageBox.Yes  # 默认关闭界面选择No
            )
                return
            self.pushButton_open.setEnabled(False)
            self.timer1.start()
            self.pushButton_run.setEnabled(True)
            
        else:
            result=QMessageBox.question(
                QWidget(),
                '问题提示', 
                '文件格式不存在，请重新选择', 
                QMessageBox.Yes,
                QMessageBox.Yes  # 默认关闭界面选择No
            )
            return
                
    def run(self):
        if self.data is None:
            result=QMessageBox.question(
                QWidget(),
                '问题提示', 
                '数据未打开', 
                QMessageBox.Yes,
                QMessageBox.Yes  # 默认关闭界面选择No
            )  
            return
        
        if self.format is 1 and self.data is not None:
            res = self.lanedetection(self.data)
            self.show_output_image(res)
            
            self.pushButton_open.setEnabled(True)
            self.pushButton_run.setEnabled(False)
            
        elif self.format is 2 and self.data is not None:
            self.pushButton_run.setEnabled(False)
            self.timer2.start()
            self.pushButton_open.setEnabled(True)
            
    def show_input_video(self):
        if self.format is 2 and self.data.isOpened():
            success, frame = self.data.read()
            if success:
              self.show_input_image(frame)
            else:
                self.timer1.stop()
                # 从视频头开始
                self.data.set(cv2.cv2.CAP_PROP_POS_AVI_RATIO, 0)
                
    def show_output_video(self):
        if self.format is 2 and self.data.isOpened():
            success, frame = self.data.read()
            if success:
                res = self.lanedetection(frame)
                self.show_output_image(res)
            else:
                self.timer2.stop()
                self.data.release()
       
    def show_input_image(self, image):
        # 等比例缩放图片
        img = cv2.resize(image, (self.label_inputs.size().height(), self.label_inputs.size().width()))
        # 图片格式从BGR 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 显示图片：先从opencv转换到pyqt5种QImage格式，再通过setPixmap设置图片
        QImg = QImage(
            img.data, 
            img.shape[1], 
            img.shape[0],
            img.shape[1] * 3, 
            QImage.Format_RGB888
        )
        self.label_inputs.setPixmap(QPixmap.fromImage(QImg))

    def show_output_image(self, image):
        # 等比例缩放图片
        img = cv2.resize(image, (self.label_inputs.size().height(), self.label_inputs.size().width()))
        # 图片格式从BGR 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 显示图片：先从opencv转换到pyqt5种QImage格式，再通过setPixmap设置图片
        QImg = QImage(
            img.data, 
            img.shape[1], 
            img.shape[0],
            img.shape[1] * 3, 
            QImage.Format_RGB888
        )
        self.label_outputs.setPixmap(QPixmap.fromImage(QImg))

      
class VideoTimer(QThread):
    # refetence: https://github.com/fengtangzheng/pyqt5-opencv-video
    _signal = pyqtSignal()
    def __init__(self):
        QThread.__init__(self)
        self.stopped = False
        self.fps = 50  # 设置显示快慢
        
        self.mutex = QMutex()
        
    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self._signal.emit()
            time.sleep(1 / self.fps)
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True
            
  
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())

