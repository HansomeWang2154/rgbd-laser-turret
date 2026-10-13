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
DEPTH_MIN, DEPTH_MAX = 500, 1500    # mm

# 映射：数字到颜色名称和绘制用 BGR 颜色
COLOR_MAP = {
    '1': ('Red',   (0,   0, 255)),
    '2': ('Green', (0, 255,   0)),
    '3': ('Blue',  (255, 0,   0)),
}




def read_depth_frame(ds):
    """读取深度帧，返回深度矩阵与可视化灰度图"""
    f = ds.read_frame()
    buf = np.array(f.get_buffer_as_triplet()).reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 2)
    dpt = buf[...,0].astype(np.float32) + buf[...,1].astype(np.float32)*255
    dpt = dpt[:, ::-1]
    vis = cv2.convertScaleAbs(dpt, alpha=255.0/DEPTH_MAX)
    return dpt, vis

def compute_depth_roi_old(dpt, vis, dmin, dmax, morph_ksize=(7,7)):
    """
    根据深度阈值生成二值图，做形态学去噪，
    找最大连通域并拟合最小外接圆，返回 roi_mask、depth_bin、depth_roi_vis
    """
    # 阈值分割
    depth_bin = cv2.inRange(dpt, dmin, dmax).astype(np.uint8)
    # 闭开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
    depth_bin = cv2.morphologyEx(depth_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    depth_bin = cv2.morphologyEx(depth_bin, cv2.MORPH_OPEN,  kernel, iterations=2)
    # 找最大连通域并拟合圆
    contours, _ = cv2.findContours(depth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(depth_bin)
    depth_roi_vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if contours:
        target = "ellipse_jiaozheng"
        c = max(contours, key=cv2.contourArea)
        if target == "circle":
            # 圆拟合
            (x, y), r = cv2.minEnclosingCircle(c)
            center, radius = (int(x),int(y)), int(r)
            # 如果要用椭圆拟合，替换为 fitEllipse:
            # ellipse = cv2.fitEllipse(c)
            # cv2.ellipse(depth_roi_vis, ellipse, (0,255,0),2)
            cv2.circle(depth_roi_vis, center, radius, (0,255,0), 2)
            # 构造 ROI 掩码
            cv2.circle(roi_mask, center, radius, 255, -1)
        elif target == "ellipse":
            # 椭圆拟合
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(depth_roi_vis, ellipse, (0,255,0), 2)
            # 构造 ROI 掩码
            cv2.ellipse(roi_mask, ellipse, 255, -1)
        elif target == "ellipse_jiaozheng":
            # 椭圆拟合
            ellipse = cv2.fitEllipse(c)
            # 定义椭圆掩码的固定偏移 (x偏移, y偏移)
            offset_x = -25  # 根据实际情况调整
            offset_y = -25   # 根据实际情况调整
            # 从原始椭圆中提取参数
            (center_x, center_y), (width, height), angle = ellipse
            # 应用偏移到中心点
            shifted_center = (int(center_x + offset_x), int(center_y + offset_y))
            # 创建带偏移的新椭圆参数
            shifted_ellipse = (shifted_center, (width-15, height-15), angle)
            # 在深度图上绘制原始椭圆（用于可视化）
            cv2.ellipse(depth_roi_vis, ellipse, (0, 255, 0), 2)
            # 构造带偏移的ROI掩码
            cv2.ellipse(roi_mask, shifted_ellipse, 255, -1)
            # 在深度图上绘制带偏移的椭圆（仅用于调试，实际应用中可注释掉）
            cv2.ellipse(depth_roi_vis, shifted_ellipse, (0, 0, 255), 2)

    return roi_mask, depth_bin, depth_roi_vis


def compute_depth_roi_old(dpt, vis, dmin, dmax, morph_ksize=(7,7), depth_threshold=10):
    """
    根据深度阈值生成二值图，做形态学去噪，
    检测深度差在30mm以内的最大连通域图层并拟合椭圆
    返回 roi_mask、depth_bin、depth_roi_vis
    """
    # 1. 初始阈值分割
    depth_bin = cv2.inRange(dpt, dmin, dmax).astype(np.uint8)
    
    # 2. 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
    depth_bin = cv2.morphologyEx(depth_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    depth_bin = cv2.morphologyEx(depth_bin, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. 找出所有连通域
    contours, _ = cv2.findContours(depth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化输出
    roi_mask = np.zeros_like(depth_bin)
    depth_roi_vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    if contours:
        # 4. 按深度分层处理连通域
        layer_contours = []
        current_layer = []
        
        # 按y坐标排序轮廓(假设深度方向大致对应y轴)
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        for i, c in enumerate(sorted_contours):
            if not current_layer:
                current_layer.append(c)
                continue
                
            # 计算当前轮廓与前一个轮廓的深度差(使用y坐标差作为近似)
            prev_rect = cv2.boundingRect(sorted_contours[i-1])
            curr_rect = cv2.boundingRect(c)
            depth_diff = abs(curr_rect[1] - prev_rect[1])
            
            if depth_diff <= depth_threshold:
                current_layer.append(c)
            else:
                layer_contours.append(current_layer)
                current_layer = [c]
        
        if current_layer:
            layer_contours.append(current_layer)
            
        # 5. 找出最大图层的轮廓(按面积和)
        if layer_contours:
            max_layer = max(layer_contours, key=lambda layer: sum(cv2.contourArea(c) for c in layer))
            
            # 6. 合并图层内的所有轮廓
            merged_contour = np.vstack(max_layer)
            
            # 7. 拟合椭圆
            if len(merged_contour) >= 5:  # 拟合椭圆至少需要5个点
                ellipse = cv2.fitEllipse(merged_contour)
                
                # 应用偏移校正
                (center_x, center_y), (width, height), angle = ellipse
                offset_x = -25  # 根据实际情况调整
                offset_y = -25   # 根据实际情况调整
                shifted_center = (int(center_x + offset_x), int(center_y + offset_y))
                shifted_ellipse = (shifted_center, (width-15, height-15), angle)
                
                # 可视化
                cv2.ellipse(depth_roi_vis, ellipse, (0, 255, 0), 2)  # 原始椭圆(绿色)
                cv2.ellipse(depth_roi_vis, shifted_ellipse, (0, 0, 255), 2)  # 校正后椭圆(红色)
                
                # 生成ROI掩码
                cv2.ellipse(roi_mask, shifted_ellipse, 255, -1)
    
    return roi_mask, depth_bin, depth_roi_vis

def compute_depth_roi_old(dpt, vis, dmin, dmax, morph_ksize=(7,7), depth_threshold=5):
    """
    改进版深度ROI检测：
    1. 先利用深度信息进行分层
    2. 在每个深度层内计算连通域
    3. 找出最大连通域并拟合椭圆
    返回 roi_mask、depth_bin、depth_roi_vis
    """
    # 1. 初始深度阈值分割
    depth_bin = cv2.inRange(dpt, dmin, dmax).astype(np.uint8)
    
    # 2. 形态学预处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
    processed = cv2.morphologyEx(depth_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. 深度分层处理
    depth_values = dpt[(processed == 255) & (dpt >= dmin) & (dpt <= dmax)]
    if len(depth_values) == 0:
        return np.zeros_like(depth_bin), depth_bin, cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # 计算深度直方图用于分层
    hist = cv2.calcHist([depth_values.astype(np.float32)], [0], None, [256], [dmin, dmax])
    
    # 找出主要深度层 (峰值检测)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 10:  # 简单的峰值检测
            peaks.append(i)
    
    # 4. 处理每个深度层
    layer_masks = []
    for peak in peaks:
        layer_min = max(dmin, peak - depth_threshold/2)
        layer_max = min(dmax, peak + depth_threshold/2)
        
        # 创建当前深度层掩码
        layer_mask = cv2.inRange(dpt, layer_min, layer_max)
        layer_mask = cv2.bitwise_and(layer_mask, processed)
        
        # 在当前层内找连通域
        contours, _ = cv2.findContours(layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 找出当前层的最大连通域
            max_contour = max(contours, key=cv2.contourArea)
            layer_masks.append((cv2.contourArea(max_contour), max_contour, layer_mask))
    
    # 5. 选择最大连通域
    roi_mask = np.zeros_like(depth_bin)
    depth_roi_vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    if layer_masks:
        # 按面积排序选择最大的连通域
        largest_layer = max(layer_masks, key=lambda x: x[0])
        max_contour = largest_layer[1]
        
        # 拟合椭圆
        if len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            
            # 应用偏移校正
            (center_x, center_y), (width, height), angle = ellipse
            offset_x = -25
            offset_y = -25
            shifted_center = (int(center_x + offset_x), int(center_y + offset_y))
            shifted_ellipse = (shifted_center, (width-15, height-15), angle)
            
            # 可视化
            cv2.ellipse(depth_roi_vis, ellipse, (0, 255, 0), 2)  # 原始椭圆(绿色)
            cv2.ellipse(depth_roi_vis, shifted_ellipse, (0, 0, 255), 2)  # 校正后椭圆(红色)
            
            # 生成ROI掩码
            cv2.ellipse(roi_mask, shifted_ellipse, 255, -1)
    
    return roi_mask, depth_bin, depth_roi_vis

def compute_depth_roi_old(dpt, vis, dmin, dmax, morph_ksize=(7,7)):
    """
    改进版深度ROI检测，解决depth_bin有值但无ROI的问题
    返回 roi_mask, depth_bin, depth_roi_vis
    """
    # 1. 深度阈值分割
    depth_bin = cv2.inRange(dpt, dmin, dmax).astype(np.uint8)
    print(f"depth_bin有效像素数: {np.count_nonzero(depth_bin)}")  # 调试输出

    # 如果没有有效像素，直接返回空结果
    if np.count_nonzero(depth_bin) == 0:
        return np.zeros_like(depth_bin), depth_bin, cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 2. 形态学处理（先闭后开）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
    processed = cv2.morphologyEx(depth_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=2)
    # print(f"形态学处理后有效像素数: {np.count_nonzero(processed)}")  # 调试输出

    # 3. 轮廓检测（只检测外部轮廓）
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"找到轮廓数量: {len(contours)}")  # 调试输出

    # 初始化输出
    roi_mask = np.zeros_like(depth_bin)
    depth_roi_vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if contours:
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        print(f"最大轮廓面积: {cv2.contourArea(max_contour)}")  # 调试输出

        # 椭圆拟合（至少需要5个点）
        if len(max_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(max_contour)
                print(f"拟合椭圆参数: {ellipse}")  # 调试输出

                # 应用偏移校正
                (center_x, center_y), (width, height), angle = ellipse
                offset_x = -25  # 水平偏移补偿
                offset_y = -25  # 垂直偏移补偿
                shifted_center = (int(center_x + offset_x), int(center_y + offset_y))
                shifted_ellipse = (shifted_center, (width-15, height-15), angle)

                # 可视化
                cv2.ellipse(depth_roi_vis, ellipse, (0, 255, 0), 2)  # 原始椭圆(绿色)
                cv2.ellipse(depth_roi_vis, shifted_ellipse, (0, 0, 255), 2)  # 校正后椭圆(红色)

                # 生成ROI掩码（使用校正后的椭圆）
                cv2.ellipse(roi_mask, shifted_ellipse, 255, -1)
                print(f"ROI掩码有效像素数: {np.count_nonzero(roi_mask)}")  # 调试输出

            except Exception as e:
                print(f"椭圆拟合错误: {e}")
                # 回退方案：使用最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(max_contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(roi_mask, center, radius, 255, -1)
                cv2.circle(depth_roi_vis, center, radius, (0, 255, 255), 2)
        else:
            print("轮廓点数不足5个，无法拟合椭圆")
            # 回退方案：使用边界矩形
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(roi_mask, (x, y), (x+w, y+h), 255, -1)
            cv2.rectangle(depth_roi_vis, (x, y), (x+w, y+h), (0, 255, 255), 2)
    else:
        print("未找到有效轮廓，尝试使用原始二值图作为ROI")
        roi_mask = processed.copy()

    return roi_mask, depth_bin, depth_roi_vis


def compute_depth_roi(dpt, vis, dmin, dmax, morph_ksize=(3,3), depth_interval=40):
    """
    直接深度分层+圆拟合：
    1. 按固定深度间隔(depth_interval)切片
    2. 选择最大切片
    3. 拟合最小外接圆
    返回 roi_mask, depth_bin, depth_roi_vis
    """
    # 1. 初始深度阈值分割
    depth_bin = cv2.inRange(dpt, dmin, dmax).astype(np.uint8)
    valid_pixels = np.count_nonzero(depth_bin)
    # print(f"初始有效深度像素: {valid_pixels} (范围: {dmin}-{dmax}mm)")

    if valid_pixels == 0:
        return np.zeros_like(depth_bin), depth_bin, cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 2. 创建深度切片
    slices = []
    current_min = dmin
    while current_min < dmax:
        current_max = min(current_min + depth_interval, dmax)
        slice_mask = cv2.inRange(dpt, current_min, current_max)
        slice_mask = cv2.bitwise_and(slice_mask, depth_bin)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
        processed = cv2.morphologyEx(slice_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        pixel_count = np.count_nonzero(processed)
        if pixel_count > 0:
            slices.append({
                'min': current_min,
                'max': current_max,
                'mask': processed,
                'pixels': pixel_count
            })
            # print(f"切片{depth_interval}mm: {current_min}-{current_max}mm, 像素={pixel_count}")
        
        current_min += depth_interval

    # 3. 选择最大切片
    roi_mask = np.zeros_like(depth_bin)
    depth_roi_vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    if slices:
        main_slice = max(slices, key=lambda x: x['pixels'])
        # print(f"选择主切片: {main_slice['min']}-{main_slice['max']}mm, 像素={main_slice['pixels']}")

        # # 在选中的切片中找最大连通域
        # contours, _ = cv2.findContours(main_slice['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if contours:
        #     max_contour = max(contours, key=cv2.contourArea)
        #     (x, y), radius = cv2.minEnclosingCircle(max_contour)
        #     area = cv2.contourArea(max_contour)
        #     circularity = 4 * np.pi * area / (np.pi * radius)**2 if radius > 0 else 0
            
            # print(f"最大连通域: 中心=({x:.1f},{y:.1f}), 半径={radius:.1f}, 圆形度={circularity:.2f}")
            

    # 1. 找到最大连通域
    contours, _ = cv2.findContours(main_slice['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        # 2. 创建单独的连通域掩码
        contour_mask = np.zeros_like(main_slice['mask'])
        cv2.drawContours(contour_mask, [max_contour], -1, 255, -1)  # 填充轮廓
        
        # 3. 腐蚀10像素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))  # 核大小=2 * 5+1
        eroded_mask = cv2.erode(contour_mask, kernel, iterations=1)
        
        # 4. 从腐蚀后的掩码重新提取轮廓
        eroded_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if eroded_contours:
            max_eroded_contour = max(eroded_contours, key=cv2.contourArea)
            
            # 后续处理（如拟合圆/椭圆）
            (x, y), radius = cv2.minEnclosingCircle(max_eroded_contour)

            # 应用偏移校正
            adj_center = (int(x-17) , int(y-30) )  # 中心偏移
            adj_radius = max(10, int(radius+10))   # 半径缩减
            
            # 可视化
            cv2.circle(depth_roi_vis, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # 原始圆(绿色)
            cv2.circle(depth_roi_vis, adj_center, adj_radius, (0, 0, 255), 2)        # 校正圆(红色)
            
            # 生成ROI掩码
            cv2.circle(roi_mask, adj_center, adj_radius, 255, -1)

    else:
        print("未找到有效切片，使用全局最大外接圆")
        contours, _ = cv2.findContours(depth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            cv2.circle(roi_mask, (int(x), int(y)), int(radius), 255, -1)

    # print(f"最终ROI像素数: {np.count_nonzero(roi_mask)}")
    return roi_mask, depth_bin, depth_roi_vis,(radius+15)*2,adj_center




