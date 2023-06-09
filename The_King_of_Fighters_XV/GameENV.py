import cv2
import torch
import random
import numpy as np

from Grab_Screen import grab_screen
from direct_keys import (Jump, Left, Right, Down, attack_heavy_legs, attack_heavy_rist, attack_light_legs, attack_light_rist,
                        super)

# 交互环境类
class ENV(object):

    # 初始化函数
    def __init__(self, 
                 window_size = (960,0,1920,700), 
                 self_blood_size = (960,0,1920,700),
                 enemy_blood_size = (960,0,1920,700),
                 yolo_net = None,
                 ) -> None:
        
        # 窗口位置坐标
        self.window_size = window_size
        # 血量位置坐标
        self.me_blood_size = self_blood_size
        self.enemy_blood_size = enemy_blood_size

        self.net = yolo_net

        self.observation_space =  5
        self.action_space = [Jump,Left,Right,Down,attack_light_rist,attack_light_legs,attack_heavy_rist,attack_heavy_legs,super]

        # 初始化一些与reword相关的变量
        self.reword = 0.0
        self.self_blood = 1.0
        self.enemy_blood = 1.0

        pass
    
    # step 继续读取图像，获得下一个state
    def step(self, action) -> None:
        '''
            action: 一个索引
        '''

        done = self.check_env()

        # 采取行动
        self.action_space[action]()

        # 获得新的状态
        new_state = self.get_state()

        reword = self.reword

        return new_state, reword, done
    
    # 根据图像获得state
    def get_state(self) -> None:
        # 获得图像
        window_img, self_blood_img, enemy_blood_img = self.get_img()
        # 获得所有状态
        self_blood_state = self.handle_blood_(self_blood_img)
        enemy_blood_state = self.handle_blood_(enemy_blood_img)

        distance, Jing_bbox, Ann_bbox = self.handle_state_(window_img)

        # 根据血量变化来设置reword
        self.reword = 0.6 * (self_blood_state - self.self_blood) + (self.enemy_blood - enemy_blood_state) * 0.4
        # 更新自己和敌人的血量值
        self.self_blood = self_blood_state
        self.enemy_blood = enemy_blood_state


        return (self_blood_state, enemy_blood_state, distance, Jing_bbox, Ann_bbox)
    
    # 获得图像
    def get_img(self) -> None:
        window_img = grab_screen(region=self.window_size)
        self_blood_img = grab_screen(region=self.me_blood_size)
        enemy_blood_img = grab_screen(region=self.enemy_blood_size)

        return window_img, self_blood_img, enemy_blood_img
        
    # 初始化环境d
    def init_env(self) -> None:
        pass

    # check函数
    def check_env(self) -> None:
        
        return True
    
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

        '''
        如果检测出来的不是一个Ann和一个Jing，就再次截图继续找，直到找到一个Jing和一个Ann
        '''

        # 获得预测结果
        result = self.net(window_image)

        # 检测是否是一个ann一个jing
        while True:
            if result[0].boxes.xyxy.shape[0] == 2 & result[0].boxes.cls.shape[0] == 2:
                if result[0].boxes.cls[0] != result[0].boxes.cls[1]:
                    break
            else:
                window_img = grab_screen(region=self.window_size)
                result = self.net(window_img)

        cls_result = result[0].boxes.cls
        xyxy_bbox_result = result[0].boxes.xyxy

        # 获得Ann和Jing的bbox的索引
        Ann_index = torch.nonzero(cls_result)
        Jing_index = torch.nonzero(cls_result==0)

        # 获得Ann和JIng的bbox (xyxy格式)
        Ann_bbox = xyxy_bbox_result[Ann_index][0].view([4])
        Jing_bbox = xyxy_bbox_result[Jing_index][0].view([4])

        # 计算Ann和Jing之间的距离（以Jing为原点，直接计算差值并取平均值），正数的话就是ann在右，负数的话就是ann在左
        distance = torch.sum(Ann_bbox-Jing_bbox) / 2

        return distance.view([1]), Jing_bbox, Ann_bbox
    
    def select_randomAction(self):
        action_index = random.randint(0, 9)
        return action_index


if __name__ == "__main__":
    environment = ENV()
    window_img, self_blood_img, enemy_blood_img = environment.get_img()

    print(type(window_img))