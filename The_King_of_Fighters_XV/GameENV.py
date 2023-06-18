import cv2
import torch
import numpy as np
import gym
from gym import spaces
from Grab_Screen import grab_screen
from direct_keys import *


# 交互环境类
class ENV(gym.Env):

    # 初始化函数
    def __init__(self, 
                 window_size = (65, 350, 1079, 923),
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
        self_blood_roi = [0.3810, 0.0444, 0.0831, 0.0468]
        enemy_blood_roi = [0.3810, 0.0444, 0.5381, 0.0468]
        self_defend_roi = [0.1526, 0.0291, 0.2752, 0.0896]
        enemy_defend_roi = [0.1526, 0.0291, 0.5740, 0.0896]
        self_energy_roi = [0.1580, 0.0484, 0.0940, 0.9096]
        enemy_energy_roi = [0.1580, 0.0484, 0.7498, 0.9096]
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

        self.net = yolo_net
        pass

    # 生成状态空间,选手可自行重写
    def get_observation_space(self):
        pass

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

    # step 继续读取图像，获得下一个state
    def step(self, action) -> None:
        pass
    
    # 根据图像获得state
    def get_state(self, x):
        # 获得图像
        window_img, self_blood_img, enemy_blood_img, win_icon_img = self.get_img()
        # 获得所有状态
        self_blood_state = self.handle_blood_(self_blood_img) # 自身血量
        enemy_blood_state = self.handle_blood_(enemy_blood_img) # 对手血量
        
        distance, Jing_bbox, Ann_bbox = self.handle_state_(window_img) # 角色距离和bbox
        game_result = self.handle_result_(win_icon_img) # 游戏是否胜利

        return (self_blood_state, enemy_blood_state, distance, Jing_bbox, Ann_bbox, game_result)
    
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
    def reset(self) -> None:
        self.done = False
        return # 初始环境

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

        # 获得预测结果
        result = self.net(window_image)

        # 获得cls和bbox的预测结果
        cls_result = result[0].boxes.cls
        xyxy_bbox_result = result[0].boxes.xyxy

        # 获得Ann和Jing的bbox的索引
        Ann_index = torch.nonzero(cls_result)
        Jing_index = torch.nonzero(cls_result==0)

        # 获得Ann和JIng的bbox (xyxy格式)
        Ann_bbox = xyxy_bbox_result[Ann_index].view([4])
        Jing_bbox = xyxy_bbox_result[Jing_index].view([4])

        # 计算Ann和Jing之间的距离（以Jing为原点，直接计算差值并取平均值），正数的话就是ann在右，负数的话就是ann在左
        distance = torch.sum(Ann_bbox-Jing_bbox) / 2

        return distance.view([1]), Jing_bbox, Ann_bbox
    
    def get_relative_size(self, relative_pos:list):
        # 解析相对位置向量,返回left,top,right,bottom格式的元组
        window_w = self.window_size[2]-self.window_size[0]
        window_h = self.window_size[3]-self.window_size[1]
        relative_size = (int(np.floor(self.window_size[0]+window_w*relative_pos[2])),
                         int(np.floor(self.window_size[1]+window_h*relative_pos[3])),
                         int(np.ceil(self.window_size[0]+window_w*relative_pos[2]+window_w*relative_pos[0])),
                         int(np.ceil(self.window_size[1]+window_h*relative_pos[3]+window_h*relative_pos[1])))
        return relative_size


    def handle_result_(self, win_icon_img):
        # 用opencv模板匹配的方式判断胜败
        template_win = cv2.imread(r'./template/win.png', cv2.IMREAD_GRAYSCALE) # 胜败图标用程序截取，便于统一模板和观测图的大小
        template_lose = cv2.imread(r'./template/lose.png', cv2.IMREAD_GRAYSCALE)
        win_icon_img_gray = cv2.cvtColor(win_icon_img, cv2.COLOR_BGRA2GRAY)
        match_threshold = 0.9
        res_win = cv2.matchTemplate(win_icon_img_gray, template_win, cv2.TM_CCOEFF_NORMED)
        res_lose = cv2.matchTemplate(win_icon_img_gray, template_lose, cv2.TM_CCOEFF_NORMED)
        _, max_val_win, _, _ = cv2.minMaxLoc(res_win)
        _, max_val_lose, _, _ = cv2.minMaxLoc(res_lose)

        # 胜利返回1，失败返回-1
        if max_val_win > match_threshold:
            self.done = True
            return 1
        if max_val_lose > match_threshold:
            self.done = True
            return -1
        # 平局返回0，不知道能不能平局，先写在这里
        # if max_val_draw > match_threshold:
        #     self.done = True
        #     return 0
    
    # 获取防御条信息
    def handle_defend_(self):
        pass

    # 获取能量条信息
    def handle_energy_(self):
        pass
    

if __name__ == "__main__":
    environment = ENV()
    (window_img, 
     self_blood_img, 
     enemy_blood_img, 
     self_defend_img, 
     enemy_defend_img, 
     self_energy_img, 
     enemy_energy_img, 
     win_icon_img) = environment.get_img()
    cv2.imshow('window_img', window_img)
    cv2.imshow('self_blood_img', self_blood_img)
    cv2.imshow('enemy_blood_img', enemy_blood_img)
    cv2.imshow('self_defend_img', self_defend_img)
    cv2.imshow('enemy_defend_img', enemy_defend_img)
    cv2.imshow('self_energy_img', self_energy_img)
    cv2.imshow('enemy_energy_img', enemy_energy_img)
    cv2.waitKey(0)
    print(type(window_img))