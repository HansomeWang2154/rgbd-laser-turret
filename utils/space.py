from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import time
import serial
from utils import *




def get_roi_center_depth(dpt, roi_mask):
    """
    计算ROI区域中心的深度值
    
    参数:
        dpt: 原始深度图(单位:mm)
        roi_mask: ROI区域的二值掩码
        
    返回:
        float: ROI中心点的深度值(mm)
        tuple: ROI中心点坐标(x,y)
    """
    # 找到ROI区域的非零像素坐标
    ys, xs = np.nonzero(roi_mask)
    
    if len(xs) == 0:
        return None, None
    
    # 计算ROI的中心坐标
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    
    # 获取中心点的深度值
    depth_value = dpt[center_y, center_x]
    
    return depth_value, (center_x, center_y)

def pixel_to_real_coords_old(pixel_coords, roi_center, pixel_scale=1.0):
    """
    将像素坐标转换为相对于ROI中心的实际坐标
    
    参数:
        pixel_coords: 像素坐标 (x,y)
        roi_center: ROI中心点 (x,y)
        pixel_scale: 像素到实际距离的比例 (mm/pixel)
    
    返回:
        tuple: (dx_mm, dy_mm) 实际坐标偏移量(mm)
    """
    if pixel_coords is None or roi_center is None:
        return (0, 0)
    
    # 计算像素偏移量
    dx_pixel = pixel_coords[0] - roi_center[0]
    dy_pixel = pixel_coords[1] - roi_center[1]
    
    # 转换为实际距离 (注意Y轴方向相反)
    dx_mm = dx_pixel * pixel_scale
    dy_mm = -dy_pixel * pixel_scale  # 图像Y轴向下为正，实际坐标系Y轴向上为正
    
    return (dx_mm, dy_mm)

def pixel_to_real_coords(pixel_coords, roi_center, pixel_scale=1.0):
    """
    将像素坐标转换为相对于ROI中心的实际坐标（带类型检查和溢出处理）
    
    参数:
        pixel_coords: 像素坐标 (x,y) - 可接受None/int/float/tuple/np.array
        roi_center: ROI中心点 (x,y) - 可接受None/int/float/tuple/np.array
        pixel_scale: 像素到实际距离的比例 (mm/pixel) - 必须为正数
    
    返回:
        tuple: (dx_mm, dy_mm) 实际坐标偏移量(mm)，保证在int16范围内
    """
    # 1. 输入验证
    if pixel_scale <= 0:
        raise ValueError("pixel_scale必须是正数")
    
    # 2. 处理None输入
    if pixel_coords is None or roi_center is None:
        return (0, 0)
    
    # 3. 强制类型转换（防止float和溢出）
    def safe_int(v):
        """安全转换为int，处理大数和None"""
        if v is None:
            return 0
        # 先限制在int16范围（-32768~32767）
        v = max(-32768, min(32767, int(round(float(v)))))
        return v
    
    # 转换输入坐标
    try:
        x = safe_int(pixel_coords[0])
        y = safe_int(pixel_coords[1])
        cx = safe_int(roi_center[0])
        cy = safe_int(roi_center[1])
    except (TypeError, IndexError) as e:
        print(f"坐标转换错误: {e}")
        return (0, 0)
    
    # 4. 计算像素偏移量（使用int16范围计算）
    dx_pixel = np.int16(x) - np.int16(cx)  # 使用np.int16处理溢出
    dy_pixel = np.int16(y) - np.int16(cy)
    
    # 5. 转换为实际距离（Y轴反转）
    dx_mm = float(dx_pixel) * pixel_scale
    dy_mm = -float(dy_pixel) * pixel_scale
    
    # 6. 最终输出限制在合理范围（单位：mm）
    max_mm = 1000  # 假设最大1米偏移
    dx_mm = max(-max_mm, min(max_mm, dx_mm))
    dy_mm = max(-max_mm, min(max_mm, dy_mm))
    
    return (float(dx_mm), float(dy_mm))  # 保持float精度

def calculate_pixel_scale(dpt, pixel_diameter, real_diameter_mm=100.0):
    """
    根据深度ROI的实际直径计算像素比例尺
    
    参数:
        dpt: 深度图
        roi_mask: ROI掩码
        real_diameter_mm: ROI区域的实际直径(mm)
    
    返回:
        float: 像素比例尺 (mm/pixel)
    """
    # 找到ROI轮廓
    
    # 计算比例尺
    if pixel_diameter > 0:
        return real_diameter_mm / pixel_diameter
    return None



import math

def spherical_to_plane(pitch, yaw, distance):
    """
    将球坐标系的pitch和yaw角度转换为平面上的坐标
    
    参数:
    pitch (float): 俯仰角（弧度），向上为正
    yaw (float): 偏航角（弧度），向右为正
    distance (float): 光源到平面的距离
    
    返回:
    tuple: 平面上的坐标(x, y)
    """
    x = distance * math.tan(yaw)
    y = distance * math.tan(pitch) / math.cos(yaw)
    return x, y

def plane_to_spherical(x, y, distance):
    """
    将平面上的坐标转换为球坐标系的pitch和yaw角度
    
    参数:
    x (float): 平面上的x坐标
    y (float): 平面上的y坐标
    distance (float): 光源到平面的距离
    
    返回:
    tuple: 球坐标系的角度(pitch, yaw)（弧度）
    """
    yaw = math.atan2(x, distance)
    pitch = math.atan2(y * math.cos(yaw), distance)
    return pitch, yaw

# 示例使用
if __name__ == "__main__":
    # 设置光源到平面的距离
    D = 882.0-30
    
    # 测试从球坐标到平面坐标的转换
    pitch_rad = math.radians(0)  # 30度俯仰角
    yaw_rad = math.radians(0)    # 45度偏航角
    x, y = spherical_to_plane(pitch_rad, yaw_rad, D)
    print(f"Pitch: {math.degrees(pitch_rad):2.2f}°, Yaw: {math.degrees(yaw_rad):.2f}° -> 平面坐标: ({x:.2f}, {y:.2f})")
    
    # 测试从平面坐标到球坐标的转换
    x = 68
    y = 156
    pitch_rad, yaw_rad = plane_to_spherical(x, y, D)
    print(f"平面坐标: ({x}, {y}) -> Pitch: {math.degrees(pitch_rad*10):+03.0f}°, Yaw: {math.degrees(yaw_rad*10):+03.0f}°")    