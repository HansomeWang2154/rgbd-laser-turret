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
import serial
from utils.serial_send import *
from utils.roi import *
from utils.laser import *
from utils.space import *
from utils.detect import *
# 全局参数



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




def main():
    global weight_h, weight_s, weight_v, weight_x, weight_y
    hen=(0,0)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"串口 {SERIAL_PORT} 已连接")
    except serial.SerialException as e:
        print(f"无法打开串口 {SERIAL_PORT}: {e}")
        ser = None

    # 状态变量
    mode = 0  # 0:矩形中心模式, 1:ROI中心模式
    ds, cap, dev = init_streams(DEPTH_WIDTH, DEPTH_HEIGHT)

    # 首先扫描二维码，获取顺序列表
    order = qr_scan(cap)
    idx = 0  # 当前颜色索引

    # 创建窗口
    for w in ['depth_bin','depth_roi','color_roi','closed_masks','centers',
              'highlight_mask','highlight_vis']:
        cv2.namedWindow(w)

    exposure_factor = EXPOSURE_FACTOR_INIT
    brightness_thresh = BRIGHTNESS_THRESH_INIT

    time.sleep(1)
#只用做一次的静态部分
        # 1. 深度 ROI
    dpt, dvis = read_depth_frame(ds)
    roi_mask, dbin, droi_vis , pixel_diameter,roi_center_global = compute_depth_roi(dpt, dvis, DEPTH_MIN, DEPTH_MAX)
    cv2.imshow('depth_bin', dbin)
    cv2.imshow('depth_roi', droi_vis)

    # 2. 彩色 ROI
    ret, frm = cap.read()
    # if not ret: break
    color_roi = cv2.bitwise_and(frm, frm, mask=roi_mask)
    cv2.imshow('color_roi', color_roi)

    # 获取ROI中心深度和全局坐标
    roi_depth, _ = get_roi_center_depth(dpt, roi_mask)
    if roi_depth is not None:
        print(f"ROI中心点全局坐标: {roi_center_global}, 深度值: {roi_depth}mm")
        
        # 在深度图上标记中心点
        cv2.circle(droi_vis, roi_center_global, 5, (0, 0, 255), -1)
        cv2.putText(droi_vis, f"{roi_depth}mm", 
                   (roi_center_global[0]+10, roi_center_global[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 在彩色图上标记相同的中心点（使用全局坐标）
        cv2.circle(frm, roi_center_global, 5, (0, 0, 255), -1)
        cv2.putText(frm, f"{roi_depth}mm",
                   (roi_center_global[0]+10, roi_center_global[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示标记后的图像
        cv2.imshow('depth_roi', droi_vis)
        cv2.imshow('color_frame', frm)  # 添加这个窗口显示完整彩色图
    # 3. 检测矩形中心
    rect_centers = {}
    # if mode == 0:
    rect_centers = detect_color_regions_outside_roi(frm, roi_mask, outside_color_ranges)

    # 在初始化后添加比例尺计算
    pixel_scale = 0.5  # 默认值 (mm/pixel)，可根据实际调整
    calculated_scale = calculate_pixel_scale(dpt, pixel_diameter, real_diameter_mm=350.0)
    if calculated_scale is not None:
        pixel_scale = calculated_scale
        print(f"计算得到的像素比例尺: {pixel_scale:.4f} mm/pixel")
    else:
        print(f"使用默认像素比例尺: {pixel_scale:.4f} mm/pixel")

    center_filter = CenterFilter(window_size=3)  # 可调整窗口大小

    while True:

        # 2. 彩色 ROI
        ret, frm = cap.read()
        # if not ret: break
        color_roi = cv2.bitwise_and(frm, frm, mask=roi_mask)
        cv2.imshow('color_roi', color_roi)

        # 4. 检测ROI区域中心 (只在ROI中心模式下进行)
        roi_centers = {}
        if mode == 1:
            
            raw_centers = three_region_shape_weight_in(color_roi, roi_mask)
            # 应用中值滤波
            filtered_centers = center_filter.update(raw_centers)
            # cr, cg, cb = filtered_centers
            cr, cg, cb = raw_centers
            roi_centers = {'Red': cr, 'Green': cg, 'Blue': cb}

            # 可视化原始点和滤波点
            cent_vis = color_roi.copy()
            for color, center in zip(['Red', 'Green', 'Blue'], raw_centers):
                if center:
                    cv2.circle(cent_vis, center, 2, (0,255,255), -1)  # 黄色原始点
            for color, center in roi_centers.items():
                if center:
                    cv2.circle(cent_vis, center, 2, (0,0,255), -1)    # 红色滤波点
                    cv2.putText(cent_vis, f"{color[0]}", 
                               (center[0]+10, center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow('centers', cent_vis)

            # cr, cg, cb = three_region_shape_weight_in(color_roi, roi_mask)
            # roi_centers = {
            #     'Red': cr,
            #     'Green': cg,
            #     'Blue': cb
            # }


        # 5. 高亮点检测 (两种模式都需要)
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

        # 6. 根据当前模式和颜色计算并打印差值
        if hcen:
            if mode == 0:  # 矩形中心模式
                if color_name in rect_centers and rect_centers[color_name]:
                    rect_center = rect_centers[color_name]
                    dx, dy = hcen[0] - rect_center[0], hcen[1] - rect_center[1]
                    print(f"[{color_name}-Rect] Highlight{hcen} - RectCenter{rect_center} = (dx={dx}, dy={dy})")

                    # state = {'Red': 1, 'Green': 2, 'Blue': 3}[color_name]
                    state = 0x0e
                    # send_serial_data(roi_depth, rect_center[0], rect_center[1], state,ser=ser)


            else:  # ROI中心模式
                if color_name in roi_centers and roi_centers[color_name]:
                    roi_center = roi_centers[color_name]
                    print(hcen)
                    # dx, dy = hcen[0] - roi_center[0], hcen[1] - roi_center[1]
                    dx = -int(np.int16(hcen[0] - roi_center[0]))  # 先转np.int16处理溢出，再转Python int
                    dy = -int(np.int16(hcen[1] - roi_center[1]))

                    dx_mm, dy_mm = pixel_to_real_coords(roi_center, roi_center_global, pixel_scale)
                    dy_mm = dy_mm+140
                    dx_mm = dx_mm
                    print(pixel_scale)

                    # print(f"[{color_name}-ROI] Highlight{hcen} - ROICenter{roi_center} = (dx={dx}, dy={dy})")
                    print(dx_mm,dy_mm)
                    # state = {'Red': 4, 'Green': 5, 'Blue': 6}[color_name]
                    state = "0x0f"

                    # send_serial_data(roi_depth,dx_mm, dy_mm, state,ser=ser) 
                    pitch_rad, yaw_rad = plane_to_spherical(dx_mm, dy_mm, roi_depth)
                    pitch_rad = -pitch_rad
                    yaw_rad = -yaw_rad

                    print(f"平面坐标: ({dx_mm}, {dy_mm}) -> Pitch: {math.degrees(pitch_rad):+06.2f}°, Yaw: {math.degrees(yaw_rad):+06.02f}°")    

                    send_serial_data_angle(roi_depth,yaw_rad,pitch_rad,state,ser)
                    # time.sleep(1)                 



        # 7. 键盘交互
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
        
        # 其他键盘控制保持不变...
        elif key == ord('+'):
            brightness_thresh = min(255, brightness_thresh + 1)
        elif key == ord('-'):
            brightness_thresh = max(0, brightness_thresh - 1)
        elif key == ord('e'):
            exposure_factor = min(1.0, exposure_factor + 0.002)
        elif key == ord('d'):
            exposure_factor = max(0.1, exposure_factor - 0.002)

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
