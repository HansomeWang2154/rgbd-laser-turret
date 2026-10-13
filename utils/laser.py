from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import time
import serial
from utils.utils import *


EXPOSURE_FACTOR_INIT = 0.6          # 曝光缩放初值
BRIGHTNESS_THRESH_INIT = 150        # 亮度阈值初值



def reduce_exposure(frame, f):
    tmp = frame.astype(np.float32)*f
    return np.clip(tmp,0,255).astype(np.uint8)

def brightness_binary(frame, t):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return b

def detect_highlight(roi_color, ef, bt):
    """高亮点掩码、可视化、质心"""
    red = reduce_exposure(roi_color, ef)
    bin_ = brightness_binary(red, bt)
    cl = close_mask(bin_, ksize=(1,1))
    cen = find_center(cl)
    vis = roi_color.copy()
    filtered_center = highlight_filter.update(cen)
    if filtered_center:
        cv2.circle(vis, filtered_center, 7, (0,255,255), 2)
        cv2.putText(vis, 'HL', (filtered_center[0]+10,filtered_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    return cl, vis, filtered_center


from collections import deque

class MedianFilter:
    def __init__(self, window_size=10):
        self.window = deque(maxlen=window_size)
        self.median_cache = None
        
    def update(self, point):
        """更新滤波窗口并返回中值"""
        if point is None:
            return self.median_cache
            
        self.window.append(point)
        if len(self.window) == self.window.maxlen:
            # 按x坐标排序取中值
            sorted_x = sorted(self.window, key=lambda p: p[0])
            sorted_y = sorted(self.window, key=lambda p: p[1])
            self.median_cache = (sorted_x[len(sorted_x)//2][0], 
                                sorted_y[len(sorted_y)//2][1])
        return self.median_cache
highlight_filter = MedianFilter(window_size=5)  # 放在while循环之前