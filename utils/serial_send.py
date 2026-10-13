import serial
import struct
import math

# 在全局变量部分添加串口初始化
SERIAL_PORT = "COM11"  # 根据实际情况修改串口号
BAUDRATE = 115200
# ser = None


def send_serial_data_old(depth, dx, dy, state_str,ser):
    """
    通过串口发送固定长度字符串格式的数据:
    格式: "DddddX+ddddY+ddddsd\n" (总长度20字节)
    - Ddddd: 深度值，固定5位数字，前面补零 (如 01234)
    - X+dddd: X偏移，符号+4位数字 (如 X-0123)
    - Y+dddd: Y偏移，符号+4位数字 (如 Y+0045)
    - sd: 状态，1位符号+1位数字 (如 s2)
    - 结束符: \n
    示例: "01234X-0123Y+0045s2\n"
    """
    if ser is None:
        print("未发送")
        return
    
    # 格式化各部分为固定长度
    depth_str = f"{int(round(depth)):03d}"      # 5位数字，前面补零
    dx_str = f"{dx:+03.0f}"          # 符号+4位数字
    dy_str = f"{dy:+03.0f}"          # 符号+4位数字
    # state_str = f"{state_str}"                    # 状态标记
    print(len(dx_str))
    print(len(dy_str))
    print(len(state_str))
    # 组合成固定格式字符串
    # data_str = f"{depth_str}X{dx_str}Y{dy_str}{state_str}\n"
    data_str = f"{state_str}{dx_str}{dy_str}"
    
    try:
        ser.write(data_str.encode('ascii'))
        print(f"发送数据: {data_str}")
        print(len(data_str))
    except serial.SerialException as e:
        print(f"串口发送失败: {e}")

def send_serial_data(depth, dx, dy, state, ser):
    """
    通过串口发送固定9位字符串格式的数据:
    格式: "sXXXYYY" (总长度7字节)
    - s: 状态值(1位数字)
    - XXX: X偏移量(符号+2位数字)
    - YYY: Y偏移量(符号+2位数字)
    示例: "2-12+45" 表示状态2，X偏移-12，Y偏移+45
    """
    if ser is None:
        print("未发送")
        return
    
    # 确保状态是1位数字 (0-9)
    # state_str = f"{int(state) % 10}"  # 强制转换为1位数字
    state_str = "\x0f"
    
    # 格式化X/Y偏移量为符号+2位数字
    # 限制范围在-99到+99之间
    dx = max(-999, min(999, int(round(dx))))
    dy = max(-999, min(999, int(round(dy))))
    
    dx_str = f"{dx:+04d}"  # 例如 +12 或 -05
    dy_str = f"{dy:+04d}"  # 例如 +45 或 -12
    
    # 组合成固定7位字符串
    data_str = f"{state_str}{dx_str}{dy_str}"
    
    # 验证长度是否为7字节
    if len(data_str) != 9:
        print(f"错误：数据长度应为7字节，实际为{len(data_str)}字节")
        print(f"生成的数据: {data_str!r}")
        return
    
    try:
        ser.write(data_str.encode('ascii'))
        print(f"发送数据: {data_str}")
    except serial.SerialException as e:
        print(f"串口发送失败: {e}")



def send_serial_data_angle(depth, yaw_rad, pitch_rad, state, ser):
    """
    通过串口发送固定9位字符串格式的数据:
    格式: "sXXXYYY" (总长度7字节)
    - s: 状态值(1位数字)
    - XXX: X偏移量(符号+2位数字)
    - YYY: Y偏移量(符号+2位数字)
    示例: "2-12+45" 表示状态2，X偏移-12，Y偏移+45
    """
    if ser is None:
        print("未发送")
        return
    
    # 确保状态是1位数字 (0-9)
    # state_str = f"{int(state) % 10}"  # 强制转换为1位数字
    state_str = "\x0f"
    

    dx_str = f"{math.degrees(yaw_rad*100):+05.0f}"  # 例如 +12 或 -05
    dy_str = f"{math.degrees(pitch_rad*100):+05.0f}"  # 例如 +45 或 -12
    
    # 组合成固定7位字符串
    data_str = f"{dx_str}{dy_str}"
    
    # 验证长度是否为7字节
    if len(data_str) != 10:
        print(f"错误：数据长度应为9字节，实际为{len(data_str)}字节")
        print(f"生成的数据: {data_str!r}")
        return
    
    try:
        ser.write(data_str.encode('ascii'))
        print(f"发送数据: {data_str}")
    except serial.SerialException as e:
        print(f"串口发送失败: {e}")
