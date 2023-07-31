import cv2
from utils import grab_screen

if __name__ == "__main__":

    
    # 调用grab_screen()函数获取截图
    img = grab_screen(region=(0,360,1279,1079))
    
    # 创建一个名为'Screen'的窗口
    cv2.namedWindow('Screen')

    cv2.imshow('Screen', img)

    # 定义一个回调函数来获取鼠标坐标并在图像中显示
    def show_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # 在控制台输出鼠标坐标
            print(f'x: {x}, y: {y}')
            # 将图片转换成Umat的格式
            img_ptr = cv2.UMat(img)
            # 在图像上绘制坐标信息
            cv2.putText(img_ptr, f'x: {x}, y: {y}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # # # 将回调函数与窗口绑定
    cv2.setMouseCallback('Screen', show_coordinates)
    # 等待按下任意键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()