
from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import time
import serial
from utils.utils import *


# 深度图尺寸
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480

def get_color_masks(hsv, roi):
    """HSV 分割红绿蓝，并与 ROI 相交"""
    lr1, ur1 = np.array([0,120,70]),  np.array([10,255,255])
    lr2, ur2 = np.array([160,120,70]), np.array([180,255,255])
    mr = cv2.inRange(hsv, lr1, ur1) | cv2.inRange(hsv, lr2, ur2)
    mg = cv2.inRange(hsv, np.array([30,20,20]), np.array([85,255,255]))
    mb = cv2.inRange(hsv, np.array([100,50,50]), np.array([130,255,255]))
    return mr & roi, mg & roi, mb & roi

def morphological_close(mask, ksize=(7,7)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def compute_deltas(hc, color_cs):
    """hc:(x,y), color_cs:dict{name:(x,y)} → dict{name:(dx,dy)}"""
    ds = {}
    if hc is None: return ds
    hx, hy = hc
    for name, c in color_cs.items():
        if c:
            x,y = c
            ds[name] = (hx-x, hy-y)
    return ds


def three_region(color_roi,roi_mask):
    #从此处到高亮检测前的代码可以封装为一个区域检测的函数，可以写多种方法实现的three_regions_detect的代码。
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    mr, mg, mb = get_color_masks(hsv, roi_mask)
    cr = find_center(morphological_close(mr))
    cg = find_center(morphological_close(mg))
    cb = find_center(morphological_close(mb))

    # 合并显示三个闭运算后的掩码
    cm_vis = np.hstack([
        cv2.cvtColor(morphological_close(mr), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morphological_close(mg), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morphological_close(mb), cv2.COLOR_GRAY2BGR),
    ])
    cv2.imshow('closed_masks', cm_vis)

    # 在 ROI 上标注质心
    cent_vis = color_roi.copy()
    for c, col, name in [(cr,(0,0,255),'R'), (cg,(0,255,0),'G'), (cb,(255,0,0),'B')]:
        if c:
            cv2.circle(cent_vis, c, 5, col, -1)
            cv2.putText(cent_vis, name, (c[0]+5,c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imshow('centers', cent_vis)

    return cr,cg,cb

def three_region2(color_roi, roi_mask):
    """
    检测ROI区域内的红、绿、蓝三色中心点
    修改后流程: 闭运算 -> 膨胀20像素 -> 腐蚀50像素
    """
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    mr, mg, mb = get_color_masks(hsv, roi_mask)
    
    # 定义形态学操作核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    # 对每种颜色掩码进行处理
    def process_mask1(mask):
        # 1. 初始闭运算
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # # 2. 膨胀20像素
        # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # dilated = cv2.dilate(closed, dilate_kernel, iterations=1)
        
        # # 3. 腐蚀50像素
        # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        # eroded = cv2.erode(dilated, erode_kernel, iterations=1)
        
        return closed    
    # 对每种颜色掩码进行处理
    def process_mask2(mask):
        # 1. 初始闭运算
        # closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. 膨胀20像素
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilated = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        # 3. 腐蚀50像素
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        eroded = cv2.erode(dilated, erode_kernel, iterations=1)
        
        return eroded
    
    # 处理三种颜色掩码
    mr = process_mask1(mr)
    mg = process_mask2(mg)
    mb = process_mask1(mb)
    
    # 查找中心点
    cr = find_center(mr)
    cg = find_center(mg)
    cb = find_center(mb)

    # 合并显示处理后的掩码
    cm_vis = np.hstack([
        cv2.cvtColor(mr, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mg, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mb, cv2.COLOR_GRAY2BGR),
    ])
    cv2.imshow('processed_masks', cm_vis)

    # 在ROI上标注质心
    cent_vis = color_roi.copy()
    for c, col, name in [(cr,(0,0,255),'R'), (cg,(0,255,0),'G'), (cb,(255,0,0),'B')]:
        if c:
            cv2.circle(cent_vis, c, 5, col, -1)
            cv2.putText(cent_vis, name, (c[0]+5,c[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imshow('centers', cent_vis)

    return cr, cg, cb
# 在代码开头或 main 前，添加五个权重的全局初值
weight_h, weight_s, weight_v, weight_x, weight_y = 1.0, 0, 0, 0.5, 0.5

def three_region_shape_weight_old(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': np.array([0, 180, 150], dtype=np.float32),
        'Green': np.array([100, 130, 130], dtype=np.float32),
        'Blue': np.array([120, 150, 150], dtype=np.float32)
    }
    
    # HSV分量权重 (H, S, V)
    WEIGHTS = np.array([1.5, 0, 0])  # 色相最重要，饱和度次之，亮度最不重要
    
    # 1. 预处理
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        masked_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,
        param1=200,
        param2=70,
        minRadius=20,
        maxRadius=100
    )
    
    # 初始化结果
    cr, cg, cb = None, None, None
    cent_vis = color_roi.copy()
    
    if circles is not None:
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:3]  # 取前3大圆
        
        # 存储所有圆和它们的颜色距离信息
        circle_info = []
        for (x, y, r) in circles:
            circle_mask = np.zeros_like(roi_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            ys, xs = np.where(circle_mask)
            
            if len(xs) > 0:
                mean_hsv = hsv[ys, xs].mean(axis=0)
                distances = {
                    color: np.linalg.norm((mean_hsv - ref) * WEIGHTS)
                    for color, ref in REF_COLORS.items()
                }
                circle_info.append({
                    'center': (x, y),
                    'radius': r,
                    'distances': distances
                })
        
        # 最佳颜色分配（确保每个颜色只分配一个圆）
        if len(circle_info) >= 3:
            # 生成所有可能的颜色分配组合
            from itertools import permutations
            color_names = list(REF_COLORS.keys())
            
            min_total_distance = float('inf')
            best_assignment = None
            
            # 尝试所有可能的颜色分配组合
            for perm in permutations(color_names, 3):
                total_dist = sum(
                    circle_info[i]['distances'][perm[i]] 
                    for i in range(3)
                )
                if total_dist < min_total_distance:
                    min_total_distance = total_dist
                    best_assignment = perm
            
            # 应用最佳分配
            color_centers = {}
            for i in range(3):
                color_centers[best_assignment[i]] = circle_info[i]['center']
                
                # 绘制结果
                color_bgr = {
                    'Red': (0, 0, 255),
                    'Green': (0, 255, 0),
                    'Blue': (255, 0, 0)
                }[best_assignment[i]]
                
                x, y = circle_info[i]['center']
                r = circle_info[i]['radius']
                cv2.circle(cent_vis, (x, y), r, color_bgr, 2)
                cv2.circle(cent_vis, (x, y), 5, color_bgr, -1)
                cv2.putText(cent_vis, best_assignment[i][0], (x+5, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            cr = color_centers.get('Red')
            cg = color_centers.get('Green')
            cb = color_centers.get('Blue')
    
    cv2.imshow('centers', cent_vis)
    return cr, cg, cb


def three_region_shape_weight_old(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': np.array([0, 150, 150], dtype=np.float32),
        'Green': np.array([60, 80, 130], dtype=np.float32),
        'Blue': np.array([120, 200, 150], dtype=np.float32)
    }
    
    # 白色/高亮区域的HSV范围 (需要排除)
    WHITE_HSV_MIN = np.array([30, 5, 120])    # 低饱和度、高亮度
    WHITE_HSV_MAX = np.array([100, 30, 190])
    
    # HSV分量权重 (H, S, V)
    WEIGHTS = np.array([1.5, 0.5, 0])  # 色相最重要，饱和度次之，亮度最不重要
    
    # 1. 预处理
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        masked_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,
        param1=200,
        param2=70,
        minRadius=20,
        maxRadius=100
    )
    
    # 初始化结果
    cr, cg, cb = None, None, None
    cent_vis = color_roi.copy()
    
    if circles is not None:
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:3]  # 取前3大圆
        
        # 存储所有圆和它们的颜色距离信息
        circle_info = []
        for (x, y, r) in circles:
            circle_mask = np.zeros_like(roi_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            ys, xs = np.where(circle_mask)
            
            if len(xs) > 0:
                # 创建临时图像用于白色像素过滤
                temp_hsv = np.zeros_like(hsv)
                temp_hsv[ys, xs] = hsv[ys, xs]
                
                # 创建白色像素掩码
                white_mask = cv2.inRange(temp_hsv, WHITE_HSV_MIN, WHITE_HSV_MAX)
                
                # 获取非白色像素的HSV值
                non_white_pixels = hsv[ys, xs][white_mask[ys, xs] == 0]
                
                if len(non_white_pixels) > 0:
                    mean_hsv = non_white_pixels.mean(axis=0)
                else:
                    # 如果全部都是白色像素，则使用原始均值
                    mean_hsv = hsv[ys, xs].mean(axis=0)
                
                distances = {
                    color: np.linalg.norm((mean_hsv - ref) * WEIGHTS)
                    for color, ref in REF_COLORS.items()
                }
                circle_info.append({
                    'center': (x, y),
                    'radius': r,
                    'distances': distances
                })
        
        # 最佳颜色分配（确保每个颜色只分配一个圆）
        if len(circle_info) >= 3:
            # 生成所有可能的颜色分配组合
            from itertools import permutations
            color_names = list(REF_COLORS.keys())
            
            min_total_distance = float('inf')
            best_assignment = None
            
            # 尝试所有可能的颜色分配组合
            for perm in permutations(color_names, 3):
                total_dist = sum(
                    circle_info[i]['distances'][perm[i]] 
                    for i in range(3)
                )
                if total_dist < min_total_distance:
                    min_total_distance = total_dist
                    best_assignment = perm
            
            # 应用最佳分配
            color_centers = {}
            for i in range(3):
                color_centers[best_assignment[i]] = circle_info[i]['center']
                
                # 绘制结果
                color_bgr = {
                    'Red': (0, 0, 255),
                    'Green': (0, 255, 0),
                    'Blue': (255, 0, 0)
                }[best_assignment[i]]
                
                x, y = circle_info[i]['center']
                r = circle_info[i]['radius']
                cv2.circle(cent_vis, (x, y), r, color_bgr, 2)
                cv2.circle(cent_vis, (x, y), 5, color_bgr, -1)
                cv2.putText(cent_vis, best_assignment[i][0], (x+5, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            cr = color_centers.get('Red')
            cg = color_centers.get('Green')
            cb = color_centers.get('Blue')
    
    cv2.imshow('centers', cent_vis)
    return cr, cg, cb

def three_region_shape_weight(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': [np.array([0, 80, 150], dtype=np.float32),    # 低H值红色
                np.array([179, 150, 150], dtype=np.float32)], # 高H值红色
        'Green': [np.array([90, 80, 130], dtype=np.float32)],
        'Blue': [np.array([120, 200, 150], dtype=np.float32)]
    }
    
    # 白色/高亮区域的HSV范围 (需要排除)
    WHITE_HSV_MIN = np.array([30, 5, 120])    # 低饱和度、高亮度
    WHITE_HSV_MAX = np.array([100, 30, 190])
    
    # HSV分量权重 (H, S, V)
    WEIGHTS = np.array([1.5, 0, 0])  # 色相最重要，饱和度次之，亮度最不重要
    
    # 1. 预处理
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        masked_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,
        param1=200,
        param2=70,
        minRadius=20,
        maxRadius=100
    )
    
    # 初始化结果
    cr, cg, cb = None, None, None
    cent_vis = color_roi.copy()
    
    if circles is not None:
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:3]  # 取前3大圆
        
        # 存储所有圆和它们的颜色距离信息
        circle_info = []
        for (x, y, r) in circles:
            circle_mask = np.zeros_like(roi_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            ys, xs = np.where(circle_mask)
            
            if len(xs) > 0:
                # 创建临时图像用于白色像素过滤
                temp_hsv = np.zeros_like(hsv)
                temp_hsv[ys, xs] = hsv[ys, xs]
                
                # 创建白色像素掩码
                white_mask = cv2.inRange(temp_hsv, WHITE_HSV_MIN, WHITE_HSV_MAX)
                
                # 获取非白色像素的HSV值
                non_white_pixels = hsv[ys, xs][white_mask[ys, xs] == 0]
                
                if len(non_white_pixels) > 0:
                    mean_hsv = non_white_pixels.mean(axis=0)
                else:
                    # 如果全部都是白色像素，则使用原始均值
                    mean_hsv = hsv[ys, xs].mean(axis=0)
                
                # 计算与各颜色的距离（红色特殊处理）
                distances = {}
                for color, refs in REF_COLORS.items():
                    if color == 'Red':
                        # 红色取两个参考向量的最小距离
                        dist1 = np.linalg.norm((mean_hsv - refs[0]) * WEIGHTS)
                        dist2 = np.linalg.norm((mean_hsv - refs[1]) * WEIGHTS)
                        distances[color] = min(dist1, dist2)
                    else:
                        # 其他颜色正常计算
                        distances[color] = np.linalg.norm((mean_hsv - refs[0]) * WEIGHTS)
                
                circle_info.append({
                    'center': (x, y),
                    'radius': r,
                    'distances': distances
                })
        
        # 最佳颜色分配（确保每个颜色只分配一个圆）
        if len(circle_info) >= 3:
            # 生成所有可能的颜色分配组合
            from itertools import permutations
            color_names = list(REF_COLORS.keys())
            
            min_total_distance = float('inf')
            best_assignment = None
            
            # 尝试所有可能的颜色分配组合
            for perm in permutations(color_names, 3):
                total_dist = sum(
                    circle_info[i]['distances'][perm[i]] 
                    for i in range(3)
                )
                if total_dist < min_total_distance:
                    min_total_distance = total_dist
                    best_assignment = perm
            
            # 应用最佳分配
            color_centers = {}
            for i in range(3):
                color_centers[best_assignment[i]] = circle_info[i]['center']
                
                # 绘制结果
                color_bgr = {
                    'Red': (0, 0, 255),
                    'Green': (0, 255, 0),
                    'Blue': (255, 0, 0)
                }[best_assignment[i]]
                
                x, y = circle_info[i]['center']
                r = circle_info[i]['radius']
                cv2.circle(cent_vis, (x, y), r, color_bgr, 2)
                cv2.circle(cent_vis, (x, y), 5, color_bgr, -1)
                cv2.putText(cent_vis, best_assignment[i][0], (x+5, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            cr = color_centers.get('Red')
            cg = color_centers.get('Green')
            cb = color_centers.get('Blue')
    
    cv2.imshow('centers', cent_vis)
    return cr, cg, cb


def three_region_shape_weight_in_old(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': [np.array([0, 80, 150], dtype=np.float32),    # 低H值红色
                np.array([179, 150, 150], dtype=np.float32)], # 高H值红色
        'Green': [np.array([90, 80, 130], dtype=np.float32)],
        'Blue': [np.array([120, 200, 150], dtype=np.float32)]
    }
    
    # 白色/高亮区域的HSV范围 (需要排除)
    WHITE_HSV_MIN = np.array([30, 0, 0])    # 低饱和度、高亮度
    WHITE_HSV_MAX = np.array([100, 60, 255])
    
    # HSV分量权重 (H, S, V)
    WEIGHTS = np.array([1.5, 0, 0])  # 色相最重要，饱和度次之，亮度最不重要
    
    # 1. 预处理
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        masked_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,
        param1=200,
        param2=70,
        minRadius=20,
        maxRadius=100
    )
    
    # 初始化结果
    cr, cg, cb = None, None, None
    cent_vis = color_roi.copy()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:3]  # 取前3大圆
        
        # 存储所有圆和它们的颜色距离信息
        circle_info = []
        for (x, y, r) in circles:
            # 创建圆形掩码
            circle_mask = np.zeros_like(roi_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # 获取圆内所有像素点
            ys, xs = np.where(circle_mask)
            
            if len(xs) > 0:
                # 获取圆内所有像素的HSV值
                circle_hsv = hsv[ys, xs]
                
                # 正确创建白色像素掩码
                white_mask = np.zeros(len(xs), dtype=np.uint8)
                for i in range(len(xs)):
                    pixel = circle_hsv[i]
                    if (WHITE_HSV_MIN[0] <= pixel[0] <= WHITE_HSV_MAX[0] and
                        WHITE_HSV_MIN[1] <= pixel[1] <= WHITE_HSV_MAX[1] and
                        WHITE_HSV_MIN[2] <= pixel[2] <= WHITE_HSV_MAX[2]):
                        white_mask[i] = 255
                
                # 计算非白色像素的HSV均值
                non_white_pixels = circle_hsv[white_mask == 0]
                if len(non_white_pixels) > 0:
                    mean_hsv = np.mean(non_white_pixels, axis=0)
                else:
                    # 如果全部都是白色像素，则跳过该圆
                    continue
                
                # 计算与各颜色的距离（红色特殊处理）
                distances = {}
                for color, refs in REF_COLORS.items():
                    if color == 'Red':
                        # 红色取两个参考向量的最小距离
                        dist1 = np.linalg.norm((mean_hsv - refs[0]) * WEIGHTS)
                        dist2 = np.linalg.norm((mean_hsv - refs[1]) * WEIGHTS)
                        distances[color] = min(dist1, dist2)
                    else:
                        # 其他颜色正常计算
                        distances[color] = np.linalg.norm((mean_hsv - refs[0]) * WEIGHTS)
                
                circle_info.append({
                    'center': (x, y),
                    'radius': r,
                    'mean_hsv': mean_hsv,
                    'distances': distances
                })
        
        # 最佳颜色分配（确保每个颜色只分配一个圆）
        if len(circle_info) >= 3:
            # 生成所有可能的颜色分配组合
            from itertools import permutations
            color_names = list(REF_COLORS.keys())
            
            min_total_distance = float('inf')
            best_assignment = None
            
            # 尝试所有可能的颜色分配组合
            for perm in permutations(color_names, 3):
                total_dist = sum(
                    circle_info[i]['distances'][perm[i]] 
                    for i in range(3)
                )
                if total_dist < min_total_distance:
                    min_total_distance = total_dist
                    best_assignment = perm
            
            # 应用最佳分配
            color_centers = {}
            for i in range(3):
                color_centers[best_assignment[i]] = circle_info[i]['center']
                
                # 绘制结果
                color_bgr = {
                    'Red': (0, 0, 255),
                    'Green': (0, 255, 0),
                    'Blue': (255, 0, 0)
                }[best_assignment[i]]
                
                x, y = circle_info[i]['center']
                r = circle_info[i]['radius']
                cv2.circle(cent_vis, (x, y), r, color_bgr, 2)
                cv2.circle(cent_vis, (x, y), 5, color_bgr, -1)
                cv2.putText(cent_vis, best_assignment[i][0], (x+5, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            cr = color_centers.get('Red')
            cg = color_centers.get('Green')
            cb = color_centers.get('Blue')
    
    cv2.imshow('centers', cent_vis)
    return cr, cg, cb

def three_region_shape_weight_in(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': [np.array([0, 80, 150], dtype=np.float32),    # 低H值红色
                np.array([179, 150, 150], dtype=np.float32)], # 高H值红色
        'Green': [np.array([90, 80, 130], dtype=np.float32)],
        'Blue': [np.array([120, 200, 150], dtype=np.float32)]
    }
    
    # 白色/高亮区域的HSV范围 (需要排除)
    WHITE_HSV_MIN = np.array([30, 0, 0])    # 低饱和度、高亮度
    WHITE_HSV_MAX = np.array([100, 60, 255])
    
    # HSV分量权重 (H, S, V)
    WEIGHTS = np.array([1.5, 0, 0])  # 色相最重要，饱和度次之，亮度最不重要
    
    # 1. 预处理
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        masked_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,
        param1=200,
        param2=70,
        minRadius=20,
        maxRadius=100
    )
    
    # 初始化结果
    cr, cg, cb = None, None, None
    cent_vis = color_roi.copy()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:3]  # 取前3大圆
        
        # 存储所有圆和它们的颜色距离信息
        circle_info = []
        for (x, y, r) in circles:
            # 创建圆形掩码
            circle_mask = np.zeros_like(roi_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # 获取圆内所有像素点
            ys, xs = np.where(circle_mask)
            
            if len(xs) > 0:
                # 获取圆内所有像素的HSV值
                circle_hsv = hsv[ys, xs]
                
                # 初始化每个像素到各颜色的距离
                pixel_distances = {
                    'Red': np.zeros(len(xs)),
                    'Green': np.zeros(len(xs)),
                    'Blue': np.zeros(len(xs))
                }
                
                # 逐个像素计算距离
                for i in range(len(xs)):
                    pixel = circle_hsv[i]
                    
                    # 跳过白色像素
                    if (WHITE_HSV_MIN[0] <= pixel[0] <= WHITE_HSV_MAX[0] and
                        WHITE_HSV_MIN[1] <= pixel[1] <= WHITE_HSV_MAX[1] and
                        WHITE_HSV_MIN[2] <= pixel[2] <= WHITE_HSV_MAX[2]):
                        continue
                    
                    # 计算到各颜色的距离（红色特殊处理）
                    for color, refs in REF_COLORS.items():
                        if color == 'Red':
                            # 红色取两个参考向量的最小距离
                            dist1 = np.linalg.norm((pixel - refs[0]) * WEIGHTS)
                            dist2 = np.linalg.norm((pixel - refs[1]) * WEIGHTS)
                            pixel_distances[color][i] = min(dist1, dist2)
                        else:
                            # 其他颜色正常计算
                            pixel_distances[color][i] = np.linalg.norm((pixel - refs[0]) * WEIGHTS)
                
                # 计算非白色像素的平均距离
                valid_pixels = np.sum([pixel_distances[color] > 0 for color in pixel_distances], axis=0) > 0
                if np.any(valid_pixels):
                    mean_distances = {
                        color: np.mean(pixel_distances[color][valid_pixels]) 
                        for color in pixel_distances
                    }
                else:
                    # 如果全部都是白色像素，则跳过该圆
                    continue
                
                circle_info.append({
                    'center': (x, y),
                    'radius': r,
                    'distances': mean_distances
                })
        
        # 最佳颜色分配（确保每个颜色只分配一个圆）
        if len(circle_info) >= 3:
            # 生成所有可能的颜色分配组合
            from itertools import permutations
            color_names = list(REF_COLORS.keys())
            
            min_total_distance = float('inf')
            best_assignment = None
            
            # 尝试所有可能的颜色分配组合
            for perm in permutations(color_names, 3):
                total_dist = sum(
                    circle_info[i]['distances'][perm[i]] 
                    for i in range(3)
                )
                if total_dist < min_total_distance:
                    min_total_distance = total_dist
                    best_assignment = perm
            
            # 应用最佳分配
            color_centers = {}
            for i in range(3):
                color_centers[best_assignment[i]] = circle_info[i]['center']
                
                # 绘制结果
                # color_bgr = {
                #     'Red': (0, 0, 255),
                #     'Green': (0, 255, 0),
                #     'Blue': (255, 0, 0)
                # }[best_assignment[i]]
                
                # x, y = circle_info[i]['center']
                # r = circle_info[i]['radius']
                # cv2.circle(cent_vis, (x, y), r, color_bgr, 2)
                # cv2.circle(cent_vis, (x, y), 1, color_bgr, -1)
                # cv2.putText(cent_vis, best_assignment[i][0], (x+5, y), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            cr = color_centers.get('Red')
            cg = color_centers.get('Green')
            cb = color_centers.get('Blue')
    
    cv2.imshow('centers', cent_vis)
    return cr, cg, cb


def three_region_kmeans(color_roi, roi_mask, weights):
    """
    1) 用 HSV+XY(5维)做 KMeans(k=4)
    2) 计算每簇的 HSV 中心
    3) 与 4 个标准颜色一一匹配，保证一一对应
    4) 返回红/绿/蓝 三色的空间中心，并在窗口 'cluster_result' 可视化
    """
    # 标准 HSV 参考 (H:0-179, S,V:0-255)
    standards = {
        'Red':   np.array([  5,180,150], dtype=np.float32),
        'Green': np.array([ 60,130,130], dtype=np.float32),
        'Blue':  np.array([120,150,150], dtype=np.float32),
        'White': np.array([  0,  0,255], dtype=np.float32),
    }

    ys, xs = np.nonzero(roi_mask)
    if len(xs)==0:
        return None, None, None

    # 提取 ROI 内像素 HSV
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    pix_hsv = hsv[ys, xs]  # (N,3)

    # 构造带权特征
    h = (pix_hsv[:,0]/179.0)*weights[0]
    s = (pix_hsv[:,1]/255.0)*weights[1]
    v = (pix_hsv[:,2]/255.0)*weights[2]
    x = (xs.astype(np.float32)/DEPTH_WIDTH)*weights[3]
    y = (ys.astype(np.float32)/DEPTH_HEIGHT)*weights[4]
    feats = np.stack([h,s,v,x,y], axis=1)

    # KMeans = 4
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    _, lb, ctrs = cv2.kmeans(feats.astype(np.float32), 4, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = lb.flatten()

    # 计算每簇的 HSV 中心
    cluster_hsv = []
    for i in range(4):
        idxs = np.where(labels==i)[0]
        if len(idxs)==0:
            cluster_hsv.append(None)
        else:
            cluster_hsv.append(pix_hsv[idxs].mean(axis=0))

    # 一一匹配：构建所有 (簇,标准色,距离) 三元组并排序
    matches = []
    for i,ch in enumerate(cluster_hsv):
        if ch is None: continue
        for cname, ref in standards.items():
            dist = np.linalg.norm(ch - ref)
            matches.append((dist, i, cname))
    matches.sort(key=lambda x: x[0])

    # 贪心分配
    assigned_cluster = set()
    assigned_color   = set()
    cluster_to_color = {}
    for dist, ci, cname in matches:
        if ci not in assigned_cluster and cname not in assigned_color:
            cluster_to_color[ci] = cname
            assigned_cluster.add(ci)
            assigned_color.add(cname)
        if len(assigned_color)==4:
            break

    # 把每个像素按它所属簇的映射颜色上色
    vis = np.zeros_like(color_roi)
    color_map = {'Red':(0,0,255),'Green':(0,255,0),'Blue':(255,0,0),'White':(255,255,255)}
    for idx, (x0,y0) in enumerate(zip(xs, ys)):
        cls = cluster_to_color.get(int(labels[idx]))
        if cls:
            vis[y0, x0] = color_map[cls]
    cv2.imshow('cluster_result', vis)

    # 聚合三色算质心
    def compute_center_for(color_name):
        pts = [i for i,lab in enumerate(labels) if cluster_to_color.get(lab)==color_name]
        if not pts:
            return None
        xs2 = xs[pts]; ys2 = ys[pts]
        return (int(xs2.mean()), int(ys2.mean()))

    cr = compute_center_for('Red')
    cg = compute_center_for('Green')
    cb = compute_center_for('Blue')
    # 在 ROI 上标注质心
    cent_vis = color_roi.copy()
    for c, col, name in [(cr,(0,0,255),'R'), (cg,(0,255,0),'G'), (cb,(255,0,0),'B')]:
        if c:
            cv2.circle(cent_vis, c, 5, col, -1)
            cv2.putText(cent_vis, name, (c[0]+5,c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imshow('centers', cent_vis)

    return cr, cg, cb


# 定义ROI外的颜色范围（可以与ROI内的不同）
outside_color_ranges = {
    'Red': ([0, 100, 100], [10, 255, 255]),  # 红色范围
    'Green': ([48, 130, 80], [85, 255, 150]), # 绿色范围
    'Blue': ([111, 150, 50], [125, 255, 100])  # 蓝色范围
}


    # lr1, ur1 = np.array([0,120,70]),  np.array([10,255,255])
    # lr2, ur2 = np.array([160,120,70]), np.array([180,255,255])
    # mr = cv2.inRange(hsv, lr1, ur1) | cv2.inRange(hsv, lr2, ur2)
    # mg = cv2.inRange(hsv, np.array([30,20,20]), np.array([85,255,255]))
    # mb = cv2.inRange(hsv, np.array([100,50,50]), np.array([130,255,255]))

def detect_color_regions_outside_roi(frame, roi_mask, color_ranges):
    """
    检测ROI之外的颜色区域，基于HSV颜色范围
    
    参数:
        frame: 输入彩色图像
        roi_mask: ROI掩码，用于排除ROI内的区域
        color_ranges: 字典，定义每种颜色的HSV范围
            {'Red': ([lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v]), ...}
    
    返回:
        dict: 每种颜色的中心点坐标 {'Red': (x,y), ...}
    """
    # 反转ROI掩码，得到ROI之外的区域
    outside_mask = cv2.bitwise_not(roi_mask)
    
    # 转换为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 初始化结果字典
    centers = {}
    
    # 为每种颜色创建可视化图像
    vis = frame.copy()
    
    # 检测每种颜色
    for color_name, (lower, upper) in color_ranges.items():
        # 创建颜色掩码
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # 只考虑ROI之外的区域
        color_mask = cv2.bitwise_and(color_mask, outside_mask)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算中心点
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers[color_name] = (cx, cy)
                
                # 在可视化图像上绘制
                cv2.drawContours(vis, [largest_contour], -1, (0,255,255), 2)
                cv2.circle(vis, (cx, cy), 5, (0,255,255), -1)
                cv2.putText(vis, color_name, (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    cv2.imshow('Color Regions Outside ROI', vis)
    return centers

def detect_rectangles_by_color(frame, roi_mask=None, min_area=50, max_area=800):
    """
    检测 frame 中面积在 min_area~max_area 的矩形轮廓，
    计算每个矩形的平均 HSV，与标准红绿蓝比距离，
    为每种颜色选最相似的一个，返回 { 'Red':(x,y), ... }，
    并在窗口 'rect_by_color' 中可视化。
    
    参数:
        frame: 输入彩色图像
        roi_mask: 可选，ROI掩码。如果提供，则会在ROI之外检测矩形
        min_area: 最小矩形面积
        max_area: 最大矩形面积
    """
    # 1) 标准 HSV - 使用与ROI内不同的参数
    standards = {
        'Red':   np.array([  0, 150, 50],  dtype=np.float32),  # 更宽松的红色范围
        'Green': np.array([ 60, 100, 40],  dtype=np.float32),  # 更宽松的绿色范围
        'Blue':  np.array([120, 150, 40],  dtype=np.float32),  # 更宽松的蓝色范围
    }

    h, w = frame.shape[:2]
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 2) 边缘 + 轮廓
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        # 用 boundingRect 面积比过滤非矩形
        x,y,ww,hh = cv2.boundingRect(approx)
        if abs(area - ww*hh) / float(ww*hh) > 0.3:
            continue

        # 如果提供了ROI掩码，检查矩形是否在ROI之外
        if roi_mask is not None:
            # 计算矩形中心
            M = cv2.moments(approx)
            cx = int(M['m10'] / (M['m00'] + 1e-6))
            cy = int(M['m01'] / (M['m00'] + 1e-6))
            
            # 如果中心点在ROI内，跳过这个矩形
            if 0 <= cx < w and 0 <= cy < h and roi_mask[cy, cx] > 0:
                continue

        # 有效矩形——算平均 HSV
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [approx], -1, 255, -1)
        ys, xs = np.nonzero(mask)
        mean_hsv = hsv_img[ys, xs].mean(axis=0)

        # 计算质心
        M = cv2.moments(approx)
        cx = int(M['m10'] / (M['m00'] + 1e-6))
        cy = int(M['m01'] / (M['m00'] + 1e-6))

        rects.append({'pts': approx, 'hsv': mean_hsv, 'center': (cx, cy)})

    # 3) 为每个标准色找最相似矩形
    chosen = {}
    for cname, ref in standards.items():
        best = None
        bd = float('inf')
        for r in rects:
            d = np.linalg.norm(r['hsv'] - ref)
            if d < bd:
                bd = d
                best = r
        if best:
            chosen[cname] = best['center']

    # 4) 可视化
    vis = frame.copy()
    # 全部矩形—灰色
    for r in rects:
        cv2.drawContours(vis, [r['pts']], -1, (128,128,128), 1)
    # 选中用各自颜色
    cmap = {'Red':(0,0,255), 'Green':(0,255,0), 'Blue':(255,0,0)}
    for cname, center in chosen.items():
        # 找对应 pts 并画
        for r in rects:
            if r['center'] == center:
                cv2.drawContours(vis, [r['pts']], -1, cmap[cname], 2)
                cv2.circle(vis, center, 5, cmap[cname], -1)
                cv2.putText(vis, cname, (center[0]+5, center[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmap[cname], 2)
                break

    cv2.imshow('rect_by_color', vis)
    return chosen

MEDIAN_FILTER_WINDOW_CENTER = 5  # 中值滤波窗口大小

class CenterFilter:
    """针对three_region_shape_weight_in输出的中心点专用滤波器"""
    def __init__(self, window_size=MEDIAN_FILTER_WINDOW_CENTER):
        self.window_size = window_size
        # 使用字典存储各颜色历史点，初始化为空列表
        self.history = {'Red': [], 'Green': [], 'Blue': []}
        # 存储上一次有效中心点（用于插值）
        self.last_valid = {'Red': None, 'Green': None, 'Blue': None}
        
    def update(self, centers):
        """输入为(cr, cg, cb)元组，返回滤波后的(cr, cg, cb)"""
        cr, cg, cb = centers
        colors = ['Red', 'Green', 'Blue']
        inputs = [cr, cg, cb]
        outputs = []
        
        for color, center in zip(colors, inputs):
            if center is not None:
                # 更新历史记录
                self.history[color].append(center)
                self.last_valid[color] = center
                
                # 保持窗口大小
                if len(self.history[color]) > self.window_size:
                    self.history[color].pop(0)
                
                # 计算中值（需要至少3个点才滤波）
                if len(self.history[color]) >= 3:
                    xs = [p[0] for p in self.history[color]]
                    ys = [p[1] for p in self.history[color]]
                    filtered = (int(np.median(xs)), int(np.median(ys)))
                    outputs.append(filtered)
                else:
                    outputs.append(center)
            else:
                # 无检测时使用最后一次有效值（可选）
                outputs.append(self.last_valid[color])
                
        return tuple(outputs)
    
