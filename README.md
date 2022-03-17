# LaneDetectionV1.0

[![github](https://img.shields.io/badge/Github-%E8%AF%B7%E7%82%B9%E4%B8%AAStar%EF%BC%8C%E6%84%9F%E8%B0%A2%EF%BC%81-red?style=for-the-badge&logo=github&logoColor=white)](https://github.com/liuyangly1/LaneDetectionV1.0) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![pyQt5](https://img.shields.io/badge/pyQt5-%23217346.svg?style=for-the-badge&logo=Qt&logoColor=white)

![1_Out](assets/1_out.jpg)

**基于Opencv的车道线检测**：1. 图像加载；2.图像预处理：图片灰度化，高斯滤波；3.Cany边缘检测；4.感兴趣区域检测；5.霍夫直线检测 ；6.直线拟合；7.车道线叠加；8.图片和视频测试；9.可视化界面pyqt5 （可选）。

## Requirements - 必要条件

- python 3.x
- numpy
- matplotlib
- opencv-python
- pyqt5 (option)

## Usage - 用法

1. Clone and install.

```bash
$ git clone git@github.com:yangliuly1/LaneDetectionV1.0.git
$ cd LaneDetectionV1.0
$ pip install -r requirements.txt
```

2. Test.

```bash
# picture test
$ python .\lanedetection.py -i ./assets/1.jpg -o ./assets/1_out.jpg
# video test 
$ python .\lanedetection.py -i ./assets/project_video.mp4 -o ./assets/project_video_out.mp4
```

3. Visualization.

```bash
# install package
$ pip install pyqt5
# run
$ python mainwindow.py
```

## Changelog - 更新日志

- Done：可视化界面。

## License - 版权信息

[MIT](https://choosealicense.com/licenses/mit/)

## Reference - 参考

[01-陈光-无人驾驶技术入门（十四）| 初识图像之初级车道线检测-知乎](https://zhuanlan.zhihu.com/p/52623916)

