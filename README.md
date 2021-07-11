---
@Author: liuyangly1
@Date  : 2021-07-07 23:33:43
@Blog  : https://blog.csdn.net/liuyang_1106
@Github: https://github.com/liuyangly1
@Email : 522927317@qq.com
---

[toc]

# LaneDetectionV1.0

[<img src="https://img.shields.io/badge/Github-%E8%AF%B7%E7%82%B9%E4%B8%AAStar%EF%BC%8C%E6%84%9F%E8%B0%A2%EF%BC%81-red" />](https://github.com/liuyangly1/LaneDetectionV1.0) [<img src="https://img.shields.io/badge/CSDN-%E8%AF%B7%E7%82%B9%E4%B8%80%E4%B8%AA%E5%85%B3%E6%B3%A8%EF%BC%8C%E6%84%9F%E8%B0%A2%EF%BC%81-brightgreen" />](https://blog.csdn.net/liuyang_1106)

**基于Opencv的车道线检测**：

1. 图像加载；
2. 图像预处理：
3. 图片灰度化，高斯滤波；
4. Cany边缘检测；
5. 感兴趣区域检测；
6. 霍夫直线检测 ；
7. 直线拟合；
8. 车道线叠加；
9. 图片和视频测试；
10. 可视化界面pyqt5。

![](Assets/1_out.jpg)

## Requirements - 必要条件

- python 3.x
- numpy
- matplotlib
- opencv-python

## Usage - 用法

```bash
$ git clone git@github.com:liuyangly1/LaneDetectionV1.0.git
$ cd LaneDetectionV1.0
$ pip install -r requirements.txt
# 图片测试
$ python .\LaneDetectionV1.0.py -i ./Assets/1.jpg -o ./Assets/1_out.jpg
# 视频测试
$ python .\LaneDetectionV1.0.py -i ./Assets/project_video.mp4 -o ./Assets/project_video_out.mp4
```

## Changelog - 更新日志

- Todo：可视化界面

## License - 版权信息

[MIT](https://choosealicense.com/licenses/mit/)

## Reference - 参考

[01-陈光-无人驾驶技术入门（十四）| 初识图像之初级车道线检测-知乎](https://zhuanlan.zhihu.com/p/52623916)

