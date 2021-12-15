#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
################################################################################
@File              :   LaneDetectionV1.0.py
@Time              :   2021/07/08 22:49:31
@Author            :   liuyangly
@Email             :   522927317@qq.com
@Desc              :   基于Opencv的车道线检测
################################################################################
"""
# Built-in modules
import argparse
# import os
# import sys

# Third-party modules
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Customized Modules
# import tooltik


class LaneDetection:
    r"""车道线检测

    基于Opencv的车道线检测:
        1. 图像加载；
        2. 图像预处理：图片灰度化，高斯滤波；
        3. Cany边缘检测；
        4. 感兴趣区域检测；
        5. 霍夫直线检测；
        6. 直线拟合；
        7. 车道线叠加;

    .. Warning::
        1. 车的目标方向最好在图片中心，否则需要调整ROI区域；
        2. 目前该算法还无法识别弯的车道线，这是霍夫直线拟合的弊端；

    .. Note::
        如果没用识别到车道线，会默认跳过。

    Args:
        # 高斯滤波参数
        ksize (tuple): 高斯核函数大小，默认[5, 5]。
        sigma (tuple): 高斯核函数sigmaX和sigmaY，默认[0, 0]。
        # 边缘检测参数
        threshold1 (int): Cany边缘检测的低阈值，默认100。
        threshold2 (int): Cany边缘检测的高阈值，默认200。
        aperture_size (int): Sobel边缘检测的核大小，默认3。
        # 感兴趣区域参数
        direction_point (tuple): 车的朝向，默认是图片的中心点，默认None，会自动设置[w//2, h//2]。
        # 霍夫直线检测参数
        rho (int): 霍夫网格的像素距离分辨率，默认1。
        theta (float): 霍夫网格的弧度角分辨率，默认pi/180。
        threshold (int): 最小投票数（霍夫网格单元中的交叉点），默认50。
        min_line_len (int): 组成一条线的最小像素数，默认200。
        max_line_gap (int): 组成一条线的最大像素数，默认400。
        # 直线拟合点参数
        x1L (int):  左车道线，插值点位置1，默认None，会自动设置w*0.1。
        x2L (int):  左车道线，插值点位置2，默认None，会自动设置w*0.4。
        x1R (int):  右车道线，插值点位置1，默认None，会自动设置w*0.6。
        x2R (int):  右车道线，插值点位置2，默认None，会自动设置w*0.9。

    Returns:
        res (opencv-python image): 叠加车道线的图片。

    Example::
        >>> lanedetection = LaneDetection()
        >>> img = cv2.imread("./Assets/1.jpg", 1)
        >>> res = lanedetection(img)
        >>> cv2.imwrite("./Assets/1_out.jpg", res)
    """

    def __init__(
        self,
        ksize=(5, 5),
        sigma=(0, 0),
        threshold1=100,
        threshold2=200,
        aperture_size=3,
        direction_point=None,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        min_line_len=200,
        max_line_gap=400,
        x1L=None,
        x2L=None,
        x1R=None,
        x2R=None,
    ):
        self.ksize = ksize
        self.sigma = sigma
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.direction_point = direction_point
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.x1L = x1L
        self.x2L = x2L
        self.x1R = x1R
        self.x2R = x2R

    def __call__(self, img):

        gauss = self._image_preprocess(img)

        edge = self._edge_canny(gauss)

        roi = self._roi_trapezoid(edge)

        lines = self._Hough_line_fitting(roi)

        line_img = self._lane_line_fitting(img, lines)

        res = self._weighted_img_lines(img, line_img)

        return res

    def _image_preprocess(self, img):
        r"""预处理

        预处理，包括灰度化和高斯滤波。

        Args:
            img (np.array): 原始图像。
        Returns:
            gauss (np.array): 灰度化和高斯滤波图片
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, self.ksize, self.sigma[0], self.sigma[1])

        return gauss

    def _edge_canny(self, img):
        r"""Canny边缘检测

        先计算sobel边缘检测，再非极大值抑制边缘，即利用高阈值划分边缘区域，然后根据低阈值和连通性的
        边缘信息来确定模糊的边界。

        Args:
            img (np.array): 原始图像。
        Returns:
            edge (np.array): 边缘检测图片
        """
        edge = cv2.Canny(img, self.threshold1, self.threshold2, self.aperture_size)

        return edge

    def _roi_trapezoid(self, img):
        r"""感兴趣区域

        根据车的方向，构建一个梯形和三角形区域，消除四周的背景干扰。

        Args:
            img (np.array): 原始图像。
        Returns:
            roi (np.array): 边缘检测的目标区域图片
        """

        h, w = img.shape[:2]

        # 车方向的中心点
        if self.direction_point is None:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            left_top = [w//2, h//2]
            right_top = [w//2, h//2]
        else:
            left_top = self.direction_point
            right_top = self.direction_point

        left_down = [int(w * 0.1), h]
        right_down = [int(w * 0.9), h]
        self.roi_points = np.array([left_down, left_top, right_top, right_down])

        # 填充梯形区域
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.roi_points, 255)

        # 目标区域提取：逻辑与
        roi = cv2.bitwise_and(img, mask)

        return roi

    def _Hough_line_fitting(self, img):
        r"""霍夫直线拟合

        根据极坐标空间y=xcos\fai+xsin\fai，从0度到180度搜索每个点的半径长度，然后累加，投票
        判断满足条件的极坐标，即对应的直线。

        Args:
            img (np.array): 原始图像。
            lines (list): 霍夫直线结果线段。
            color (tuple)：车道线颜色。
            thickness (int): 车道线厚度。
        Returns:
            line_img (np.array): 车道线图片
        """
        lines = cv2.HoughLinesP(
            img, self.rho, self.theta, self.threshold, np.array([]),
            minLineLength=self.min_line_len, maxLineGap=self.max_line_gap
        )

        return lines

    def _lane_line_fitting(self, img, lines, color=(0, 255, 0), thickness=8):
        r"""直线拟合

        根据边缘检测点，然后拟合最小二乘进行直线拟合

        Args:
            img (np.array): 原始图像。
            lines (list): 霍夫直线结果线段。
            color (tuple)：车道线颜色。
            thickness (int): 车道线厚度。
        Returns:
            line_img (np.array): 车道线图片
        """

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        right_x = []
        right_y = []
        left_x = []
        left_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2-y1)/(x2-x1))
                if slope <= -0.2:
                    left_x.extend((x1, x2))
                    left_y.extend((y1, y2))

                elif slope >= 0.2:
                    right_x.extend((x1, x2))
                    right_y.extend((y1, y2))

        if left_x and left_y:
            left_fit = np.polyfit(left_x, left_y, 1)
            left_line = np.poly1d(left_fit)
            if not self.x1L:
                x1L = int(img.shape[1] * 0.1)
            y1L = int(left_line(x1L))
            if not self.x2L:
                x2L = int(img.shape[1] * 0.4)
            y2L = int(left_line(x2L))
            cv2.line(line_img, (x1L, y1L), (x2L, y2L), color, thickness)

        if right_x and right_y:
            right_fit = np.polyfit(right_x, right_y, 1)
            right_line = np.poly1d(right_fit)
            if not self.x1R:
                x1R = int(img.shape[1] * 0.6)
            y1R = int(right_line(x1R))
            if not self.x2R:
                x2R = int(img.shape[1] * 0.9)
            y2R = int(right_line(x2R))
            cv2.line(line_img, (x1R, y1R), (x2R, y2R), color, thickness)

        return line_img

    def _weighted_img_lines(self, img, line_img, α=1, β=1, λ=0.):
        r"""加权图片和车道线

        根据像素，图片叠加

        Args:
            img (np.array): 原始图像。
            line_img (np.array): 车道线图像。
            α (int): 原图像权重。
            β (int): 车道线权重。
            λ (float): 偏差值。
        Returns:
            res (np.array): 车道线叠加原始图像。

        """
        res = cv2.addWeighted(img, α, line_img, β, λ)
        return res


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detection V1.0")
    parser.add_argument("-i", "--input_path", type=str, default="./assets/1.jpg", help="Input path of image.")
    parser.add_argument("-o", "--output_path", type=str, default="./assets/1_out.jpg", help="Ouput path of image.")
    return parser.parse_args()


def main():
    args = parse_args()

    lanedetection = LaneDetection()
    # jpg图片检测
    if args.input_path.endswith('.jpg'):
        img = cv2.imread(args.input_path, 1)
        res = lanedetection(img)
        # 拼接显示原图和结果图
        x = np.hstack([img, res])
        cv2.imwrite(args.output_path, x),

    # mp4视频检测
    elif args.input_path.endswith('.mp4'):
        # 创建一个视频读写类
        video_capture = cv2.VideoCapture(args.input_path)
        # video_capture = cv.VideoCapture(0)q
        if not video_capture.isOpened():
            print('Open is fasle!')
            exit()
        # 读取视频的fps, size
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("fps: {} \nsize: {}".format(fps, size))
        # print(help(cv2.VideoWriter))
        out = cv2.VideoWriter(args.output_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), apiPreference=0, fps=fps, frameSize=size)

        # 读取视频时长（帧总数）
        total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[INFO] {} total frames in video".format(total))

        # 设定从视频的第几帧开始读取
        frameToStart = 0
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

        # 视频播放
        while(True):

            ret, frame = video_capture.read()  # 读取每帧视频
            if not ret:  # 视频读取完， 则跳出循环
                break
            res = lanedetection(frame)
            out.write(res)

            cv2.imshow("video", res)
            # 键盘控制视频
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
