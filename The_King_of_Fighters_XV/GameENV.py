import cv2
import torch
import random
import numpy as np
import gym
from gym import spaces
from Grab_Screen import grab_screen
from direct_keys import *


# 交互环境类
class ENV(gym.Env):

    # 初始化函数
    def __init__(self, 
                 window_size = (0, 366, 1279, 1079),
                # window_size = (960,0,1920,700), 
                #  self_blood_size = (960,0,1920,700),
                #  enemy_blood_size = (960,0,1920,700),
                #  self_defend_size = (960,0,1920,700),
                #  enemy_defend_size = (960,0,1920,700),
                #  self_energy_size = (960,0,1920,700),
                #  enemy_energy_size = (960,0,1920,700),
                
                 yolo_net = None,
                 ) -> None:
        #------------------------战斗界面ROI位置信息-----------------------#
        # [w, h, xmin, ymin]
        self_blood_roi = [0.3747, 0.0153, 0.0858, 0.0589]
        enemy_blood_roi = [0.3747, 0.0153, 0.5404, 0.0589]
        self_defend_roi = [0.1476, 0.0113, 0.2770, 0.0985]
        enemy_defend_roi = [0.1476, 0.0113, 0.5754, 0.0985]
        self_energy_roi = [0.1544, 0.0137, 0.0954, 0.9177]
        enemy_energy_roi = [0.1544, 0.0137, 0.7498, 0.9177]
        # 胜负判定图标有些模糊, 暂时搁置
        #----------------------------------------------------------------#

        # 窗口位置坐标
        self.window_size = window_size
        # 血量位置坐标
        self.self_blood_size = self.get_relative_size(self_blood_roi)
        self.enemy_blood_size = self.get_relative_size(enemy_blood_roi)
        # 防御条位置坐标
        self.self_defend_size = self.get_relative_size(self_defend_roi)
        self.enemy_defend_size = self.get_relative_size(enemy_defend_roi)
        # 能量条位置坐标
        self.self_energy_size = self.get_relative_size(self_energy_roi)
        self.enemy_energy_size = self.get_relative_size(enemy_energy_roi)
        # 胜负图标位置坐标, 根据window_size取相对位置
        self.win_icon_size = window_size
        # 游戏是否结束
        self.done = False

        # 生成数字到键盘输入的动作映射列表
        self.action_map = self.get_action_map()
        # 定义动作空间
        self.action_space = self.get_action_space() # {0,1,2...,action_num-1}

        # 定义状态空间
        self.observation_space = self.get_observation_space()

        # # 初始化一些与reward相关的变量
        # self.reward = 0.0
        # self.self_blood = 1.0
        # self.enemy_blood = 1.0

        # 计数器
        self.step_num = 0

    def get_relative_size(self, relative_pos:list):
    # 解析相对位置向量,返回left,top,right,bottom格式的元组(屏幕全局坐标系下)
        window_w = self.window_size[2]-self.window_size[0]
        window_h = self.window_size[3]-self.window_size[1]
        relative_size = (int(np.floor(self.window_size[0]+window_w*relative_pos[2])),
                            int(np.floor(self.window_size[1]+window_h*relative_pos[3])),
                            int(np.ceil(self.window_size[0]+window_w*relative_pos[2]+window_w*relative_pos[0])),
                            int(np.ceil(self.window_size[1]+window_h*relative_pos[3]+window_h*relative_pos[1])))
        return relative_size
    
    # 生成状态空间,选手可自行重写
    def get_observation_space(self):
        
        # 最大距离取窗口对角线长   
        max_distance = np.sqrt(np.power(self.window_size[2]-self.window_size[0], 2) + np.power(self.window_size[3]-self.window_size[1], 2))
        # 最大x，y坐标取窗口宽、高
        max_box_x = self.window_size[2] - self.window_size[0]
        max_box_y = self.window_size[3] - self.window_size[1]

        #-------------------------------------------------------------------#
        # 2023/6/19 Mshiro00
        # 如果yolo有一瞬间没有检测到目标（比如开始和结束时），那么bbox返回值是什么？
        # 以下low,high待定
        #-------------------------------------------------------------------#

        return spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, -max_distance, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                          high=np.array([1, 1, 1, 1, 1, 1, 
                                         max_distance, 
                                         max_box_x, max_box_y, max_box_x, max_box_y, 
                                         max_box_x, max_box_y, max_box_x, max_box_y], dtype=np.float32))

    # 生成动作空间,选手可自行重写
    def get_action_space(self):
        return spaces.Discrete(len(self.action_map))
    
    # 生成动作-键盘映射列表,选手可自行重写
    def get_action_map(self):
        return [Jump, 
                Down, 
                Left, 
                Right, 
                attack_light_rist, 
                attack_heavy_rist, 
                attack_light_legs,
                attack_heavy_legs,
                super]
    
    # 执行动作
    def apply_action(self, action):
        self.action_map[action]()

    # 当前reward函数, 暂时只返回1
    def get_reward(self):
        return 1

    # step 继续读取图像，获得下一个state
    def step(self, action):
        '''
            action: 一个索引
        '''

        # 采取行动
        self.apply_action(action)

        # 获得新的状态
        new_state = self.get_state()

        # 获得reward
        reward = self.get_reward()

        # done = self.check_env()
        done = self.is_done()

        # 要返回的信息，没有可以不写
        info = {}
        
        return new_state, reward, done, info
    
    # 根据图像获得state
    def get_state(self):
        # 获得图像
        (window_img, 
         self_blood_img, 
         enemy_blood_img, 
         self_defend_img, 
         enemy_defend_img, 
         self_energy_img,
         enemy_energy_img,
         win_icon_img) = self.get_img()
        # 获得所有状态
        self_blood_state = self.handle_blood_(self_blood_img) # 自身血量
        enemy_blood_state = self.handle_blood_(enemy_blood_img) # 对手血量
        self_defend_state = self.handle_blood_(self_defend_img) # 自身防御条
        enemy_defend_state = self.handle_blood_(enemy_defend_img) # 对手防御条
        self_energy_state = self.handle_blood_(self_energy_img) # 自身能量条
        enemy_energy_state = self.handle_blood_(enemy_energy_img) # 对手能量条        
        
        distance, Jing_bbox, Ann_bbox = self.handle_state_(window_img) # 角色距离和bbox
        # game_result = self.handle_result_(win_icon_img) # 游戏是否胜利

        return (self_blood_state, 
                enemy_blood_state, 
                self_defend_state,
                enemy_defend_state,
                self_energy_state,
                enemy_energy_state,
                distance, 
                Jing_bbox, # (4,)
                Ann_bbox # (4,)
                # game_result # 暂时不在获取状态时输出游戏结果
                )
    
    # 获得图像
    def get_img(self):
        window_img = grab_screen(region=self.window_size)
        self_blood_img = grab_screen(region=self.self_blood_size)
        enemy_blood_img = grab_screen(region=self.enemy_blood_size)
        self_defend_img = grab_screen(region=self.self_defend_size)
        enemy_defend_img = grab_screen(region=self.enemy_defend_size)
        self_energy_img = grab_screen(region=self.self_energy_size)
        enemy_energy_img = grab_screen(region=self.enemy_energy_size)
        win_icon_img = grab_screen(region=self.win_icon_size)

        return (window_img, 
                self_blood_img, 
                enemy_blood_img, 
                self_defend_img, 
                enemy_defend_img, 
                self_energy_img,
                enemy_energy_img,
                win_icon_img)
        
    # 重置环境d
    def reset(self):
        self.done = False
        return # 初始环境

    # check函数
    def check_env(self):
        pass

    # 判断结束
    #------------------------------------------------#
    # 2023/6/19 Mshiro00
    # 如何判断结束待定，先固定done=False
    #------------------------------------------------#
    def is_done(self):
        return False
    
    # 根据截取的blood，返回blood情况
    @staticmethod
    def handle_blood_(blood_image):
        # 判断绿色通道灰度值计算血量
        image_g = blood_image[:,:,1]
        threshold_value = 110 # 阈值可以根据应用程序的需求进行调整
        ret, bw_img = cv2.threshold(image_g, threshold_value, 255, cv2.THRESH_TOZERO)

        # bw_img是二值化图像
        total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
        num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
        white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
        
        # white_ratio 即可看作是百分比血量
        return white_ratio
    
    # 获取防御条信息
    @staticmethod
    def handle_defend_(defend_image):
        # 判断蓝色通道灰度值计算血量
        image_b = defend_image[:,:,0]
        threshold_value = 150 # 阈值可以根据应用程序的需求进行调整
        ret, bw_img = cv2.threshold(image_b, threshold_value, 255, cv2.THRESH_TOZERO)

        # bw_img是二值化图像
        total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
        num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
        white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
        
        # white_ratio 即可看作是百分比血量
        return white_ratio

    # 获取能量条信息
    @staticmethod
    def handle_energy_(energy_image):
        # 判断红色通道灰度值计算血量
        image_r = energy_image[:,:,2]
        threshold_value = 200 # 阈值可以根据应用程序的需求进行调整
        ret, bw_img = cv2.threshold(image_r, threshold_value, 255, cv2.THRESH_TOZERO)

        # bw_img是二值化图像
        total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
        num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
        white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
        
        # white_ratio 即可看作是百分比血量
        return white_ratio
    
    # 根据截取的window，返回神经网络预测的feature
    def handle_state_(self, window_image) -> None:

        '''
        如果检测出来的不是一个Ann和一个Jing，就再次截图继续找，直到找到一个Jing和一个Ann
        '''
        #--------------------------------------------------------------#
        # 2023/6/19 Mshiro00
        # xyxy形式的bbox不适合作为状态输出，建议改为中心坐标+宽高的local形式
        #--------------------------------------------------------------#

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
        action_index = random.randint(0, len(self.action_map))
        return action_index


if __name__ == "__main__":
    environment = ENV()
    # (window_img, 
    #  self_blood_img, 
    #  enemy_blood_img, 
    #  self_defend_img, 
    #  enemy_defend_img, 
    #  self_energy_img, 
    #  enemy_energy_img, 
    #  win_icon_img) = environment.get_img()
    # cv2.imshow('window_img', window_img)
    # cv2.imshow('self_blood_img', self_blood_img)
    # cv2.imshow('enemy_blood_img', enemy_blood_img)
    # cv2.imshow('self_defend_img', self_defend_img)
    # cv2.imshow('enemy_defend_img', enemy_defend_img)
    # cv2.imshow('self_energy_img', self_energy_img)
    # cv2.imshow('enemy_energy_img', enemy_energy_img)
    # cv2.waitKey(0)
    # print(type(window_img))

    print(environment.get_observation_space().sample())