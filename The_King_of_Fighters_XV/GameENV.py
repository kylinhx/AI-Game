import cv2
import numpy as np

from Grab_Screen import grab_screen

# 交互环境类
class ENV(object):

    # 初始化函数
    def __init__(self, 
                 window_size = (960,0,1920,700), 
                 self_blood_size = (960,0,1920,700),
                 enemy_blood_size = (960,0,1920,700),
                 yolo_net = 'yolov8n',
                 ) -> None:
        # 窗口位置坐标
        self.window_size = window_size
        # 血量位置坐标
        self.me_blood_size = self_blood_size
        self.enemy_blood_size = enemy_blood_size

        self.net = yolo_net
        pass
    
    # step 继续读取图像，获得下一个state
    def step(self) -> None:
        pass
    
    # 根据图像获得state
    def get_state(self, x) -> None:
        output = self.net
        # 进行后处理
        pass
        # 返回两个人之间的距离，以AI为中心，右边为正，左边为负
        return x
    
    # 获得图像
    def get_img(self) -> None:
        window_img = grab_screen(region=self.window_size)
        self_blood_img = grab_screen(region=self.me_blood_size)
        enemy_blood_img = grab_screen(region=self.enemy_blood_size)

        self_blood_state = self.handle_blood_(self_blood_img)
        enemy_blood_state = self.handle_blood_(enemy_blood_img)
        
        other_state = self.handle_state_(window_img)

        return (self_blood_state, enemy_blood_state, other_state)

    # 初始化环境d
    def init_env(self) -> None:
        pass

    # check函数
    def check_env(self) -> None:
        pass
    
    # 根据截取的blood，返回blood情况
    def handle_blood_(self, blood_image) -> None:

        # 将彩色图像转换为灰度图像
        gray_img = cv2.cvtColor(blood_image, cv2.COLOR_BGR2GRAY)

        # 将灰度图像二值化
        threshold_value = 127 # 阈值可以根据应用程序的需求进行调整
        max_value = 255 # 像素值的最大值
        ret, bw_img = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY)

        # bw_img即为转换后的灰度图像
        # bw_img是二值化图像
        total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
        num_white_pixels = np.count_nonzero(bw_img == 255) # 计算像素值为255的点的数量
        white_ratio = num_white_pixels / total_pixels # 计算像素值为255的点占所有点的比率
        
        # white_ratio 即可看作是百分比血量
        return white_ratio
    
    # 根据截取的window，返回神经网络预测的feature
    def handle_state_(self, window_image) -> None:
        pass