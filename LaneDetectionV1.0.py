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
        7. 车道线叠加；


    .. Warning::
        context

    .. Note::
        context

    Args:
        variable (type): introduction, size=[N, M]
    Returns:
        variable (type): introduction, size=[N, M]

    Example::
        >>> lanedetection = LaneDetection()
        >>> img = cv2.imread("./Assets/1.jpg", 1)
        >>> res = lanedetection(img)
        >>> cv2.imwrite("./Assets/1_out.jpg", res)
    """

    def __init__(
        self,
        ksize=(5, 5),
        sigmaX=0,
        sigmaY=0,
        threshold1=100,
        threshold2=200,
        aperture_size=300,
        direction_point=None,  # 车的朝向方向
        rho=1,  # 霍夫网格的像素距离分辨率
        theta=np.pi/180,  # 霍夫网格的弧度角分辨率
        threshold=50,  # 最小投票数（霍夫网格单元中的交叉点）
        min_line_len=200,  # 组成一条线的最小像素数
        max_line_gap=400,  # 组成一条线的最大像素数
    ):
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.direction_point = direction_point
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap

    def __call__(self, img):

        gauss = self._image_preprocess(img)

        edge = self._edge_canny(gauss)

        roi = self._roi_trapezoid(edge)

        lines = self._Hough_line_fitting(roi)

        line_img = self._lane_line_fitting(img, lines)

        res = self._weighted_img_lines(img, line_img)

        return res

    def _image_preprocess(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, self.ksize, self.sigmaX, self.sigmaY)

        # cv2.imshow('gauss', gauss)
        # cv2.waitKey(0)
        return gauss

    def _edge_canny(self, img):
        edge = cv2.Canny(img, self.threshold1, self.threshold2, self.aperture_size)

        # cv2.imshow('edge', edge)
        # cv2.waitKey(0)

        return edge

    def _roi_trapezoid(self, img):
        # 4. 感兴趣区域: 梯形区域

        h, w = img.shape[:2]

        # 梯形的四个顶点
        if self.direction_point is None:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            left_top = [w//2, h//2]
            right_top = [w//2, h//2]
        else:
            left_top = self.direction_point
            right_top = self.direction_point

        left_down = [0, h]
        right_down = [w, h]
        self.roi_points = np.array([left_down, left_top, right_top, right_down])

        # 填充梯形区域
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.roi_points, 255)

        # 目标区域提取：逻辑与
        roi = cv2.bitwise_and(img, mask)

        # cv2.imshow('mask', mask)
        # cv2.imshow('roi', roi)
        # cv2.waitKey(0)
        return roi

    def _Hough_line_fitting(self, img):
        lines = cv2.HoughLinesP(
            img, self.rho, self.theta, self.threshold, np.array([]),
            minLineLength=self.min_line_len, maxLineGap=self.max_line_gap
        )

        
        # 绘制拟合的直线
        # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(line_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        # cv2.imshow("img_line", line_img)
        return lines

    def _lane_line_fitting(self, img, lines, color=[0, 255, 0], thickness=8):

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        right_x = []
        right_y = []
        left_x = []
        left_y = []
        # left_slope = []
        # right_slope = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2-y1)/(x2-x1))
                if slope >= 0.2:
                    # right_slope.extend(int(slope))
                    right_x.extend((x1, x2))
                    right_y.extend((y1, y2))

                elif slope <= -0.2:
                    # left_slope.extend(int(slope))
                    left_x.extend((x1, x2))
                    left_y.extend((y1, y2))
        if right_x and right_y:
            right_fit = np.polyfit(right_x, right_y, 1)
            right_line = np.poly1d(right_fit)
            x1R = int(img.shape[1] * 0.6)  # 550
            y1R = int(right_line(x1R))
            x2R = int(img.shape[1] * 0.9)  # 850
            y2R = int(right_line(x2R))
            cv2.line(line_img, (x1R, y1R), (x2R, y2R), color, thickness)
        if left_x and left_y:
            left_fit = np.polyfit(left_x, left_y, 1)
            
            left_line = np.poly1d(left_fit)
            x1L = int(img.shape[1] * 0.1)  # 120
            y1L = int(left_line(x1L))
            x2L = int(img.shape[1] * 0.4)  # 425
            y2L = int(left_line(x2L))
            cv2.line(line_img, (x1L, y1L), (x2L, y2L), color, thickness)
        # cv2.imshow("img_line", line_img)
        return line_img

    def _weighted_img_lines(self, img, line_img, α=1, β=1, λ=0.):
        return cv2.addWeighted(img, α, line_img, β, λ)


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detection V1.0")
    parser.add_argument("--input_path", type=str, default="./Assets/2.jpg", help="Input path of image.")
    parser.add_argument("--output_path", type=str, default="./Assets/2_out.jpg", help="Ouput path of image.")
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

            ret, frame = video_capture.read()
            if not ret:
                break
            # frame = cv2.flip(frame,1)
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
