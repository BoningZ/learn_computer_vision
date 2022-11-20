import cv2
import numpy as np
import scipy

from myHarris import *
from mySIFT import *
from myMatchPics import *
from myDrawer import *

if __name__ == '__main__':
    image1 = cv2.imread("cv_cover.jpg").astype('single')
    image2 = scipy.ndimage.rotate(image1, 20).astype('single')
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    image1_bw = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    feature_width = 16

    # Harris选取兴趣点
    x1, y1 = get_interest_points(image1_bw, feature_width)
    x2, y2 = get_interest_points(image2_bw, feature_width)

    # SIFT计算特征
    image1_features = get_features(image1_bw, x1, y1, feature_width)
    image2_features = get_features(image2_bw, x2, y2, feature_width)

    # 欧拉距离匹配
    matches, confidences = match_features(image1_features, image2_features)

    # 可视化
    num_pts_to_visualize = 30
    x1_m = x1[matches[:num_pts_to_visualize, 0].astype(np.int32)]
    y1_m = y1[matches[:num_pts_to_visualize, 0].astype(np.int32)]
    x2_m = x2[matches[:num_pts_to_visualize, 1].astype(np.int32)]
    y2_m = y2[matches[:num_pts_to_visualize, 1].astype(np.int32)]
    show_correspondence(image1, image2, x1_m, y1_m, x2_m, y2_m, 'match_result')
