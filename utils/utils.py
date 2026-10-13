from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import time
import serial

def close_mask(mask, ksize=(7,7)):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

# def find_center(mask):
#     """质心 (cx,cy)"""
#     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts: return None
#     c = max(cnts, key=cv2.contourArea)
#     M = cv2.moments(c)
#     if M['m00']==0: return None
#     return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

def find_center(mask, min_area_threshold=10):
    """质心 (cx,cy) - 仅当最大轮廓面积超过阈值时返回中心"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return None
    
    # 找到最大轮廓并计算其面积
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    # 检查面积是否超过阈值
    if area < min_area_threshold:
        return None
    
    # 计算并返回中心
    M = cv2.moments(c)
    if M['m00'] == 0: 
        return None
    
    return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))    