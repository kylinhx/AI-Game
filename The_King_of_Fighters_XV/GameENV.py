import cv2
import torch
import random
import numpy as np
import os
from skimage import morphology
import gym
from gym import spaces
from Grab_Screen import grab_screen
from direct_keys import *
from process_control import *


# 交互环境类
class ENV(gym.Env):

    # 初始化函数
    def __init__(self, 
                 window_size = (0,0,1050,740),
                # window_size = (960,0,1920,700), 
                # self_blood_size = (960,0,1920,700),
                # enemy_blood_size = (960,0,1920,700),
                # self_defend_size = (960,0,1920,700),
                # enemy_defend_size = (960,0,1920,700),
                # self_energy_size = (960,0,1920,700),
                # enemy_energy_size = (960,0,1920,700),
                bbox_detector = None,
                action_detector = None,
                ) -> None:
        #----------------------------CV模型信息----------------------------#
        self.box_detector = bbox_detector
        self.action_detector = action_detector

        #------------------------战斗界面ROI位置信息-----------------------#
        # [w, h, xmin, ymin]
        self_blood_roi = [0.3742857142857143, 0.005555555555555556, 0.0838095238095238, 0.16805555555555557]
        enemy_blood_roi = [0.3742857142857143, 0.005555555555555556, 0.540952380952381, 0.16805555555555557]
        self_defend_roi = [0.1457142857142857, 0.004166666666666667, 0.2780952380952381, 0.19722222222222222]
        enemy_defend_roi = [0.1457142857142857, 0.004166666666666667, 0.5771428571428572, 0.19722222222222222]
        self_energy_roi = [0.15619047619047619, 0.006944444444444444, 0.09428571428571429, 0.8694444444444445]
        self_energy_number_roi = [0.0380952380952381, 0.08055555555555556, 0.04857142857142857, 0.8291666666666667]
        enemy_energy_roi = [0.15333333333333332, 0.006944444444444444, 0.7514285714285714, 0.8694444444444445]
        enemy_energy_number_roi = [0.0380952380952381, 0.08055555555555556, 0.9152380952380952, 0.8291666666666667]
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
        # 能量条数字位置
        self.self_energy_number_size = self.get_relative_size(self_energy_number_roi)
        self.enemy_energy_number_size = self.get_relative_size(enemy_energy_number_roi)       
        # 胜负图标位置坐标, 根据window_size取相对位置
        self.win_icon_size = window_size
        # 游戏是否结束
        self.done = False
        # 劣势方flag(-1表示己方劣势，1表示敌方劣势，0表示相当)
        # 用于协助胜负判断
        self.situation = 0
        # 是否获胜
        self.win = None

        # 生成数字到键盘输入的动作映射列表
        self.action_map = self.get_action_map()
        # 定义动作空间
        self.action_space = self.get_action_space() # {0,1,2...,action_num-1}

        # 定义状态空间的维度（可以自定义）
        self.observation_space_dim = 15
        # 定义状态空间
        self.observation_space = self.get_observation_space()
        
        self.old_blood_state = 0.0
        self.old_enemy_blood_state = 0.0
        # # 初始化一些与reward相关的变量
        # self.reward = 0.0
        # self.self_blood = 1.0
        # self.enemy_blood = 1.0

        # 初始化能量条数字模板
        self.template_list = self.load_energy_template('./The_King_of_Fighters_XV/template')

        # 计数器
        self.step_num = 0

        self.self_blood_state = 0.0
        self.enemy_blood_state = 0.0
        
    def get_relative_size(self, relative_pos:list):
    # 解析相对位置向量,返回left,top,right,bottom格式的元组(屏幕全局坐标系下)
        window_w = self.window_size[2]-self.window_size[0]
        window_h = self.window_size[3]-self.window_size[1]
        relative_size = (int(relative_pos[2]*window_w), 
                         int(relative_pos[3]*window_h),
                         int(relative_pos[0]*window_w) + int(relative_pos[2]*window_w),
                         int(relative_pos[1]*window_h) + int(relative_pos[3]*window_h))
        print(relative_size)
        return relative_size
    
    # 加载能量条数字模板
    def load_energy_template(self, template_dir):
        
        # 先计算屏幕中能量数字区域大小(w, h)
        roi_size = (self.self_energy_number_size[2] - self.self_energy_number_size[0], self.self_energy_number_size[3] - self.self_energy_number_size[1])
        template_list = []
        for i in range(6):
            template_img = cv2.imread(os.path.join(template_dir, ("energy_template%d.png" % i)))
            template_img = cv2.resize(template_img, roi_size)
            # template_img = cv2.copyMakeBorder(template_img, boarder_size, boarder_size, boarder_size, boarder_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            skeleton0 = morphology.skeletonize(template_img)   # 骨架提取
            skeleton = skeleton0.astype(np.uint8)*255
            template_list.append(skeleton)

        return template_list
    
    # 生成状态空间,选手可自行重写
    def get_observation_space(self):

        #-------------------------------------------------------------------#
        # 2023/6/19 Mshiro00
        # 如果yolo有一瞬间没有检测到目标（比如开始和结束时），那么bbox返回值是什么？
        # 以下low,high待定
        #-------------------------------------------------------------------#

        return np.array(self.observation_space_dim, np.float32)

    # 生成动作空间
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
                attack_1,
                attack_2,
                attack_3,
                attack_4,
                attack_5,
                attack_6]
    
    # 执行动作
    def apply_action(self, action):
        self.action_map[action]()
        time.sleep(0.05)

    # 当前reward函数, 暂时只返回1
    def get_reward(self):
        self.reward = 0
        if self.self_blood_state < self.old_blood_state:
            self.reward += -10
        else:
            self.reward += 1
        if self.old_enemy_blood_state > self.enemy_blood_state:
            self.reward += 15
        return self.reward

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

        # 要返回的信息，没有可以不写
        # info = {}
        # done = self.check_env()
        
        done = self.is_done()
        self.done = done
        
        # 判断是否获胜
        if done:
            self.win = is_win(self.situation) # 只有结束时才会判断胜利，否则self.win初始化为None
            
        # 如获胜可以对reward进行处理
        # ...

        return new_state, reward, done
    
    # 根据图像获得state
    def get_state(self):
        # 获得图像
        (window_img, 
         self_blood_img, 
         enemy_blood_img, 
         self_defend_img, 
         enemy_defend_img, 
         self_energy_img,
         self_energy_number_img,
         enemy_energy_img,
         enemy_energy_number_img,
         win_icon_img) = self.get_img()
        # 获得所有状态
        # self_blood_state = self.handle_blood_(self_blood_img) # 自身血量
        self.self_blood_state = self.handle_blood_(self_blood_img) # 自身血量
        # enemy_blood_state = self.handle_blood_(enemy_blood_img) # 对手血量
        self.enemy_blood_state = self.handle_blood_(enemy_blood_img) # 对手血量
        self_defend_state = self.handle_defend_(self_defend_img) # 自身防御条
        enemy_defend_state = self.handle_defend_(enemy_defend_img) # 对手防御条
        self_energy_state = self.handle_energy_(self_energy_img, self_energy_number_img) # 自身能量条
        enemy_energy_state = self.handle_energy_(enemy_energy_img, enemy_energy_number_img) # 对手能量条        
        
        # 更新优劣势flag(该flag仅用于胜负判断，不应出现在返回值中), 血量相等或一方血量低于2%则保持原situation
        if self.self_blood_state > self.enemy_blood_state and self.self_blood_state > 0.02 and self.enemy_blood_state > 0.02:
            self.situation = 1
        elif self.self_blood_state < self.enemy_blood_state and self.self_blood_state > 0.02 and self.enemy_blood_state > 0.02:
            self.situation = -1
        
        distance, Jing_bbox, Ann_bbox = self.handle_state_(window_img) # 角色距离和bbox

        Ann_action = self.handle_action(window_img)

        if len(Ann_action) == 0:
            Ann_action = torch.tensor([99], dtype=torch.float32, device='cuda')
        elif len(Ann_action) > 1:
            Ann_action = Ann_action[0]
        print(Ann_action)
        # game_result = self.handle_result_(win_icon_img) # 游戏是否胜利

        # ind1 = torch.tensor([0], dtype=torch.float32, device='cuda')
        # ind2 = torch.tensor([0], dtype=torch.float32, device='cuda')
        data =  (self.self_blood_state, 
                self.enemy_blood_state, 
                self_defend_state,
                enemy_defend_state,
                self_energy_state,
                enemy_energy_state,
                distance, 
                Jing_bbox, # (4,)
                Ann_bbox, # (4,)
                # Ann_action,
                # ind1,
                # ind2
                # game_result # 暂时不在获取状态时输出游戏结果
                )
        
        # 将所有的torch.Tensor对象转换为NumPy数组
        data = tuple(x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x for x in data)

        # 将所有的0维数组转换为一维数组
        data = tuple(np.array([x]) if np.ndim(x) == 0 else x for x in data)

        # 将数据拼接成一个一维的NumPy数组
        np_data = np.concatenate(data)

        return np_data
    
    # 获得图像
    def get_img(self):
        window_img = grab_screen(region=self.window_size)
        self_blood_img = grab_screen(region=self.self_blood_size)
        enemy_blood_img = grab_screen(region=self.enemy_blood_size)
        self_defend_img = grab_screen(region=self.self_defend_size)
        enemy_defend_img = grab_screen(region=self.enemy_defend_size)
        self_energy_img = grab_screen(region=self.self_energy_size)
        self_energy_number_img = grab_screen(region=self.self_energy_number_size)
        enemy_energy_img = grab_screen(region=self.enemy_energy_size)
        enemy_energy_number_img = grab_screen(region=self.enemy_energy_number_size)
        win_icon_img = grab_screen(region=self.win_icon_size)

        return (window_img, 
                self_blood_img, 
                enemy_blood_img, 
                self_defend_img, 
                enemy_defend_img, 
                self_energy_img,
                self_energy_number_img,
                enemy_energy_img,
                enemy_energy_number_img,
                win_icon_img)
        
    # 重置环境d
    # 一局判断结束后开始执行重置，键盘不断输入ENTER直到加载界面出现
    # 加载界面结束后return
    def reset(self):
        self.done = False
        self.situation = 0
        self.win = None
        while True:
            window_img = grab_screen(region=self.window_size)
            PressKey(ENTER)
            time.sleep(0.02)
            ReleaseKey(ENTER)
            if is_loading(window_img):
                break
        time.sleep(1)
        while True:
            window_img = grab_screen(region=self.window_size)
            if not is_loading(window_img):
                break
            time.sleep(0.02)
        # if self.self_blood_state <= 0.0001:
        #     self.done = True
        # elif self.enemy_blood_state <=0.0001:
        #     self.done = True
        # else:
        #     self.done = False
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
        if self.self_blood_state <= 0.012:
            print(self.self_blood_state)
            return True
        elif self.enemy_blood_state <=0.012:
            return True
        else:
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
        # 判断蓝色通道灰度值计算防御条
        image_b = defend_image[:,:,0]
        threshold_value = 150 # 阈值可以根据应用程序的需求进行调整
        ret, bw_img = cv2.threshold(image_b, threshold_value, 255, cv2.THRESH_TOZERO)

        # bw_img是二值化图像
        total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
        num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
        white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率
        
        # white_ratio 即可看作是百分比防御条
        return white_ratio

    # 获取能量条信息
    def handle_energy_(self, energy_image, number_image):
        
        # 识别能量条数
        energy_int = self.energy_number_rec(number_image)
        # 5条能量满, 不再计算小数部分
        if energy_int != 5:
            # 判断红色通道灰度值计算能量
            image_r = energy_image[:,:,2]
            threshold_value = 200 # 阈值可以根据应用程序的需求进行调整
            ret, bw_img = cv2.threshold(image_r, threshold_value, 255, cv2.THRESH_TOZERO)

            # bw_img是二值化图像
            total_pixels = bw_img.shape[0] * bw_img.shape[1] # 计算图像中所有像素的数量
            num_white_pixels = np.count_nonzero(bw_img) # 计算像素值不为0的点的数量
            white_ratio = num_white_pixels / total_pixels # 计算像素值不为0的点占所有点的比率

            # white_ratio 即可看作是百分比能量, 是总能量的小数部分
            return (white_ratio + energy_int)
        else:
            return 0.0 + energy_int
    
    # 模板匹配识别能量条数
    def energy_number_rec(self, energy_number_img):
        gray_img = cv2.cvtColor(energy_number_img, cv2.COLOR_RGB2GRAY)
        _, img_thresh = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY_INV)
        img_thresh[img_thresh==255] = 1
        skeleton0 = morphology.skeletonize(img_thresh)   # 骨架提取
        skeleton = skeleton0.astype(np.uint8)*255
        max_val_idx = 0
        max_val = 0
        for i in range(len(self.template_list)):
            res = cv2.matchTemplate(skeleton, self.template_list[i], cv2.TM_CCOEFF_NORMED)
            # cv2.imshow("skeleton%d" % i, skeleton)
            # cv2.imshow("skeleton_tem%d" % i, self.template_list[i])
            # cv2.waitKey(0)
            temp = cv2.minMaxLoc(res)[1]
            if temp > max_val:
                max_val_idx = i
                max_val = temp

        return max_val_idx

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
        result = self.box_detector(window_image)

        # 检测是否是一个ann一个jing
        while True:
            if result[0].boxes.xyxy.shape[0] == 2 & result[0].boxes.cls.shape[0] == 2:
                if result[0].boxes.cls[0] != result[0].boxes.cls[1]:
                    break
                else:
                    window_img = grab_screen(region=self.window_size)
                    result = self.box_detector(window_img)
            else:
                window_img = grab_screen(region=self.window_size)
                result = self.box_detector(window_img)

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
        action_index = random.randint(0, len(self.action_map)-1)
        return action_index
    
    # 返回一个值
    def handle_action(self, windows_img):
        result = self.action_detector(windows_img)

        cls_result = result[0].boxes.cls
        index = torch.nonzero(cls_result).view(-1)
        return index.float()

from ultralytics import YOLO
bbox_detector_path = './The_King_of_Fighters_XV/model/bbox_detector.pt'
Action_detector_path = './The_King_of_Fighters_XV/model/action_detector.pt'

if __name__ == "__main__":
    Box_detector = YOLO(bbox_detector_path)
    Action_detector = YOLO(Action_detector_path)
    
    env = ENV(
        bbox_detector = Box_detector,
        action_detector = Action_detector,
    )


    state = env.self_blood_size
    print(state)

    # state
    # (0.013433551895090356, 0.0007996161842315689, 0.007853403141361256, 0.0, 4.0, 1.1425, 
    #  tensor([344.0627], device='cuda:0'), 
    #  tensor([ 721.1611,    0.0000, 1138.0284,  637.3776], device='cuda:0'), 
    #  tensor([1182.8895,   95.0364, 1278.3524,  628.4142], device='cuda:0'))