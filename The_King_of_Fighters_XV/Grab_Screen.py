import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

if __name__ == "__main__":

    

    # 调用grab_screen()函数获取截图
    window_img = grab_screen(region=(960,0,1920,700))

    img = grab_screen(region=(960+58,149,960+428,153))
    # 创建一个名为'Screen'的窗口
    cv2.namedWindow('Screen')
    # 将BGR格式的图像转换为RGB格式
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将彩色图像转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 将灰度图像二值化
    threshold_value = 127 # 阈值可以根据应用程序的需求进行调整
    max_value = 255 # 像素值的最大值
    ret, bw_img = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY)

    print(bw_img)
    # 显示图像
    cv2.imshow('Screen', bw_img)

    total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
    num_white_pixels = np.count_nonzero(bw_img == 255) # 计算像素值为255的点的数量
    white_ratio = num_white_pixels / total_pixels # 计算像素值为255的点占所有点的比率

    print(white_ratio)


    # # 定义一个回调函数来获取鼠标坐标并在图像中显示
    # def show_coordinates(event, x, y, flags, param):
    #     if event == cv2.EVENT_MOUSEMOVE:
    #         # 在控制台输出鼠标坐标
    #         print(f'x: {x}, y: {y}')
    #         # 在图像上绘制坐标信息
    #         cv2.putText(img, f'x: {x}, y: {y}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # # # 将回调函数与窗口绑定
    # cv2.setMouseCallback('Screen', show_coordinates)
    # 等待按下任意键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()