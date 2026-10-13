"""
    实现了包括二维码识别在内的完整流程
    还需要做的就是优化识别区域中心那部分
    可以考虑相对位置，聚类等
    继承了多种算法。
    可以用深度信息去推测比例尺，结合距离等尺度信息
"""

from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import time
import os
# 深度图尺寸
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480

# 全局参数
# DEPTH_MIN, DEPTH_MAX = 500, 1500    # mm

#离线参数
DEPTH_MIN, DEPTH_MAX =  20, 40    # mm


EXPOSURE_FACTOR_INIT = 0.6          # 曝光缩放初值
BRIGHTNESS_THRESH_INIT = 150        # 亮度阈值初值

# 映射：数字到颜色名称和绘制用 BGR 颜色
COLOR_MAP = {
    '1': ('Red',   (0,   0, 255)),
    '2': ('Green', (0, 255,   0)),
    '3': ('Blue',  (255, 0,   0)),
}

def init_streams(depth_w, depth_h, fps=30):
    """初始化深度与彩色流"""
    openni2.initialize()
    dev = openni2.Device.open_any()
    ds = dev.create_depth_stream()
    ds.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=depth_w, resolutionY=depth_h, fps=fps
    ))
    ds.start()
    cap = cv2.VideoCapture(1)
    time.sleep(2)
    return ds, cap, dev

def morphological_close(mask, ksize=(7,7)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def read_depth_frame(ds):
    """读取深度帧，返回深度矩阵与可视化灰度图"""
    f = ds.read_frame()
    buf = np.array(f.get_buffer_as_triplet()).reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 2)
    dpt = buf[...,0].astype(np.float32) + buf[...,1].astype(np.float32)*255
    dpt = dpt[:, ::-1]
    vis = cv2.convertScaleAbs(dpt, alpha=255.0/DEPTH_MAX)
    return dpt, vis

def compute_depth_roi(dpt, vis, dmin, dmax, morph_ksize=(7,7)):
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
            shifted_ellipse = (shifted_center, (width-10, height-10), angle)
            # 在深度图上绘制原始椭圆（用于可视化）
            cv2.ellipse(depth_roi_vis, ellipse, (0, 255, 0), 2)
            # 构造带偏移的ROI掩码
            cv2.ellipse(roi_mask, shifted_ellipse, 255, -1)
            # 在深度图上绘制带偏移的椭圆（仅用于调试，实际应用中可注释掉）
            cv2.ellipse(depth_roi_vis, shifted_ellipse, (0, 0, 255), 2)

    return roi_mask, depth_bin, depth_roi_vis

def get_color_masks(hsv, roi):
    """HSV 分割红绿蓝，并与 ROI 相交"""
    lr1, ur1 = np.array([0,120,70]),  np.array([10,255,255])
    lr2, ur2 = np.array([160,120,70]), np.array([180,255,255])
    mr = cv2.inRange(hsv, lr1, ur1) | cv2.inRange(hsv, lr2, ur2)
    mg = cv2.inRange(hsv, np.array([30,20,20]), np.array([85,255,255]))
    mb = cv2.inRange(hsv, np.array([100,50,50]), np.array([130,255,255]))
    return mr & roi, mg & roi, mb & roi

def close_mask(mask, ksize=(7,7)):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

def find_center(mask):
    """质心 (cx,cy)"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00']==0: return None
    return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

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
    cl = close_mask(bin_, ksize=(5,5))
    cen = find_center(cl)
    vis = roi_color.copy()
    if cen:
        cv2.circle(vis, cen, 7, (0,255,255), 2)
        cv2.putText(vis, 'HL', (cen[0]+10,cen[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    return cl, vis, cen

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

def qr_scan(cap):
    """启动时扫描二维码，返回字符串，例如 '312'"""
    det = cv2.QRCodeDetector()
    print("请将包含数字1/2/3的二维码移入画面...")
    while True:
        ret, f = cap.read()
        if not ret: continue
        data, bbox, _ = det.detectAndDecode(f)
        cv2.imshow('QR Scan', f)
        if data and all(ch in '123' for ch in data):
            print("检测到二维码内容:", data)
            cv2.destroyWindow('QR Scan')
            return list(data)
        if cv2.waitKey(1) == ord('q'):
            print("退出 QR 扫描")
            cv2.destroyWindow('QR Scan')
            return ['1']  # 默认回退

def three_region_color(color_roi,roi_mask):
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


def three_region_shape_weight(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': np.array([5, 180, 150], dtype=np.float32),
        'Green': np.array([60, 130, 130], dtype=np.float32),
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

def three_region_shape(color_roi, roi_mask):
    # 预设HSV参考值 (H:0-179, S,V:0-255)
    REF_COLORS = {
        'Red': np.array([5, 180, 150], dtype=np.float32),
        'Green': np.array([60, 130, 130], dtype=np.float32),
        'Blue': np.array([120, 150, 150], dtype=np.float32)
    }
    
    # 1. 预处理 - 使用ROI掩码获取有效区域
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    
    # 2. 霍夫圆检测
    circles = cv2.HoughCircles(
        masked_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,  # 圆之间的最小距离
        param1=200,  # Canny边缘检测阈值
        param2=70,   # 圆心检测阈值
        minRadius=20, 
        maxRadius=100
    )
    
    # 初始化结果
    cr, cg, cb = None, None, None
    cent_vis = color_roi.copy()
    
    if circles is not None:
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        
        # 取前3个最大的圆(按半径排序)
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)[:3]
        
        # 为每个圆确定颜色
        color_centers = {'Red': None, 'Green': None, 'Blue': None}
        
        for (x, y, r) in circles:
            # 创建圆形掩码
            circle_mask = np.zeros_like(roi_mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # 计算圆内像素的HSV均值
            ys, xs = np.where(circle_mask)
            if len(xs) == 0:
                continue
                
            mean_hsv = hsv[ys, xs].mean(axis=0)
            
            # 计算与参考颜色的距离
            distances = {
                color: np.linalg.norm(mean_hsv - ref)
                for color, ref in REF_COLORS.items()
            }
            
            # 找到最接近的颜色
            closest_color = min(distances.items(), key=lambda x: x[1])[0]
            
            # 保存圆心位置
            color_centers[closest_color] = (x, y)
            
            # 在可视化图像上绘制
            color_bgr = {
                'Red': (0, 0, 255),
                'Green': (0, 255, 0),
                'Blue': (255, 0, 0)
            }[closest_color]
            
            cv2.circle(cent_vis, (x, y), r, color_bgr, 2)
            cv2.circle(cent_vis, (x, y), 5, color_bgr, -1)
            cv2.putText(cent_vis, closest_color[0], (x+5, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        
        # 分配检测到的颜色中心
        cr = color_centers['Red']
        cg = color_centers['Green']
        cb = color_centers['Blue']
    
    # 显示结果
    cv2.imshow('centers', cent_vis)
    
    return cr, cg, cb


# 在代码开头或 main 前，添加五个权重的全局初值
weight_h, weight_s, weight_v, weight_x, weight_y = 1.0, 0, 0, 0.5, 0.5

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


# 在全局变量部分修改为以下内容
SAVE_IMAGE = True  # 是否保存当前帧图像
LOAD_IMAGE = False  # 是否加载保存的图像
# IMAGE_SAVE_PATH = "./saved_images/"  # 图像保存路径
# SAVED_COLOR_PATH = os.path.join(IMAGE_SAVE_PATH, "saved_color.png")
# SAVED_DEPTH_PATH = os.path.join(IMAGE_SAVE_PATH, "saved_depth.png")

# 全局路径设置
IMAGE_SAVE_PATH = "./saved_images/"
SAVED_COLOR_PATH = os.path.join(IMAGE_SAVE_PATH, "saved_color.png")
SAVED_DEPTH_PATH = os.path.join(IMAGE_SAVE_PATH, "saved_depth.png")  # 使用TIFF格式保存深度数据

def save_current_frame(color_frame, depth_frame):
    """正确保存当前帧的彩色图和深度图"""
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)
    
    # 保存彩色图（正常保存）
    cv2.imwrite(SAVED_COLOR_PATH, color_frame)
    
    # 保存深度图（保留原始精度）
    if depth_frame.dtype == np.float32:
        # 如果是float32类型，保存为16位TIFF（毫米单位）
        depth_to_save = (depth_frame * 1000).astype(np.uint16)  # 转换为毫米单位的整数
    else:
        # 如果是uint16类型，直接保存
        depth_to_save = depth_frame
        
    cv2.imwrite(SAVED_DEPTH_PATH, depth_to_save)
    print(f"已保存当前帧: {SAVED_COLOR_PATH} 和 {SAVED_DEPTH_PATH}")

def load_saved_frame():
    """正确加载保存的图像"""
    if not os.path.exists(SAVED_COLOR_PATH) or not os.path.exists(SAVED_DEPTH_PATH):
        print("未找到保存的图像文件")
        return None, None
    
    # 加载彩色图
    color_frame = cv2.imread(SAVED_COLOR_PATH)
    
    # 加载深度图
    depth_frame = cv2.imread(SAVED_DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    
    # 将深度图转换回原始范围
    if depth_frame.dtype == np.uint16:
        depth_frame = depth_frame.astype(np.float32) / 1000.0  # 转换回米单位的浮点数
    
    print(f"已加载保存的图像: {SAVED_COLOR_PATH} 和 {SAVED_DEPTH_PATH}")
    return color_frame, depth_frame

def read_depth_frame(ds):
    """读取深度帧的示例实现"""
    f = ds.read_frame()
    buf = np.array(f.get_buffer_as_triplet()).reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 2)
    
    # 原始深度数据（毫米单位）
    dpt_mm = buf[...,0].astype(np.float32) + buf[...,1].astype(np.float32)*255
    
    # 转换为米单位（可选）
    dpt = dpt_mm / 1000.0
    
    # 可视化图像
    dvis = cv2.convertScaleAbs(dpt_mm, alpha=255.0/DEPTH_MAX)
    
    return dpt, dvis

def main():
    global weight_h, weight_s, weight_v, weight_x, weight_y
    global SAVE_IMAGE, LOAD_IMAGE
    
    # 状态变量
    mode = 0  # 0:矩形中心模式, 1:ROI中心模式
    
    # 根据是否加载图像决定是否初始化摄像头
    if not LOAD_IMAGE:
        # 初始化摄像头和深度传感器
        ds, cap, dev = init_streams(DEPTH_WIDTH, DEPTH_HEIGHT)
    else:
        # 加载图像模式，不打开摄像头
        ds, cap, dev = None, None, None
        print("运行在图像加载模式，不打开摄像头设备")

    # 首先扫描二维码，获取顺序列表
    # order = ['1','2','3']
    order = qr_scan(cap) if not LOAD_IMAGE else ['1','2','3']  # 加载模式使用默认顺序
    idx = 0  # 当前颜色索引

    # 创建窗口
    for w in ['depth_bin','depth_roi','color_roi','closed_masks','centers',
              'highlight_mask','highlight_vis']:
        cv2.namedWindow(w)

    exposure_factor = EXPOSURE_FACTOR_INIT
    brightness_thresh = BRIGHTNESS_THRESH_INIT


    while True:
        # 1. 获取图像
        if LOAD_IMAGE:
            # 加载保存的图像
            frm, dvis = load_saved_frame()
            if frm is None or dvis is None:
                LOAD_IMAGE = False
                continue
            dpt = dvis  # 深度图已经是正确范围
        else:
            # 从摄像头获取图像
            dpt, dvis = read_depth_frame(ds)
            ret, frm = cap.read()
            if not ret: 
                print("无法从摄像头获取图像")
                break
            
            # 保存当前帧
            if SAVE_IMAGE:
                save_current_frame(frm, dpt)
                SAVE_IMAGE = False

        # 2. 深度 ROI
        roi_mask, dbin, droi_vis = compute_depth_roi(dpt, dvis, DEPTH_MIN, DEPTH_MAX)
        cv2.imshow('depth_bin', dbin)
        cv2.imshow('depth_roi', droi_vis)

        # 3. 彩色 ROI
        color_roi = cv2.bitwise_and(frm, frm, mask=roi_mask)
        cv2.imshow('color_roi', color_roi)

        # 4. 检测矩形中心 (只在矩形中心模式下进行)
        rect_centers = {}
        if mode == 0:
            rect_centers = detect_color_regions_outside_roi(frm, roi_mask, outside_color_ranges)
        
        # 5. 检测ROI区域中心 (只在ROI中心模式下进行)
        roi_centers = {}
        if mode == 1:
            cr, cg, cb = three_region_shape_weight(color_roi, roi_mask)
            roi_centers = {
                'Red': cr,
                'Green': cg,
                'Blue': cb
            }

        # 6. 高亮点检测 (两种模式都需要)
        hl_mask, hl_vis, hcen = detect_highlight(color_roi, exposure_factor, brightness_thresh)
        cv2.imshow('highlight_mask', hl_mask)
        
        # 获取当前颜色信息
        current_color = order[idx]
        color_name, color_bgr = COLOR_MAP[current_color]
        
        # 在高亮图上标注当前模式和状态
        mode_text = "Rect Center" if mode == 0 else "ROI Center"
        cv2.putText(hl_vis, f"Color: {color_name} - {mode_text}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2)
        cv2.imshow('highlight_vis', hl_vis)

        # 7. 根据当前模式和颜色计算并打印差值
        if hcen:
            if mode == 0:  # 矩形中心模式
                if color_name in rect_centers and rect_centers[color_name]:
                    rect_center = rect_centers[color_name]
                    dx, dy = hcen[0] - rect_center[0], hcen[1] - rect_center[1]
                    print(f"[{color_name}-Rect] Highlight{hcen} - RectCenter{rect_center} = (dx={dx}, dy={dy})")



            else:  # ROI中心模式
                if color_name in roi_centers and roi_centers[color_name]:
                    roi_center = roi_centers[color_name]
                    dx, dy = hcen[0] - roi_center[0], hcen[1] - roi_center[1]
                    print(f"[{color_name}-ROI] Highlight{hcen} - ROICenter{roi_center} = (dx={dx}, dy={dy})")


        # 8. 键盘交互
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):  # 切换模式或颜色
            if mode == 0:
                # 从矩形中心模式切换到ROI中心模式
                mode = 1
                print(f"切换到ROI中心模式，颜色: {color_name}")
            else:
                # 从ROI中心模式切换到下一个颜色的矩形中心模式
                mode = 0
                idx = (idx + 1) % len(order)
                color_name = COLOR_MAP[order[idx]][0]
                print(f"切换到矩形中心模式，颜色: {color_name}")
        elif key == ord('s'):  # 保存当前帧
            SAVE_IMAGE = True
            print("下一帧将被保存")
        elif key == ord('l'):  # 加载保存的图像
            LOAD_IMAGE = not LOAD_IMAGE
            print(f"图像加载模式: {'开启' if LOAD_IMAGE else '关闭'}")
        # 其他键盘控制保持不变...
        elif key == ord('+'):
            brightness_thresh = min(255, brightness_thresh + 5)
        elif key == ord('-'):
            brightness_thresh = max(0, brightness_thresh - 5)
        elif key == ord('e'):
            exposure_factor = min(1.0, exposure_factor + 0.01)
        elif key == ord('d'):
            exposure_factor = max(0.1, exposure_factor - 0.01)


        elif key == ord('h'):
            weight_h = max(0, weight_h - 0.1)
        elif key == ord('H'):
            weight_h += 0.1
        elif key == ord('s'):
            weight_s = max(0, weight_s - 0.1)
        elif key == ord('S'):
            weight_s += 0.1
        elif key == ord('v'):
            weight_v = max(0, weight_v - 0.1)
        elif key == ord('V'):
            weight_v += 0.1
        elif key == ord('x'):
            weight_x = max(0, weight_x - 0.1)
        elif key == ord('X'):
            weight_x += 0.1
        elif key == ord('y'):
            weight_y = max(0, weight_y - 0.1)
        elif key == ord('Y'):
            weight_y += 0.1
        # 打印当前权重
        if key in (ord('h'),ord('H'),ord('s'),ord('S'),
                   ord('v'),ord('V'),ord('x'),ord('X'),ord('y'),ord('Y')):
            print(f"Weights H,S,V,X,Y = {weight_h:.1f},{weight_s:.1f},{weight_v:.1f},{weight_x:.1f},{weight_y:.1f}")


    cap.release()
    ds.stop()
    dev.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
