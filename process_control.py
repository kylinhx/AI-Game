import cv2 as cv
import numpy as np
import time


# 这个函数kylinhx在环境中定义过了，弃置不用
def is_end(img:np.ndarray) -> bool:
    """一方被击败时屏幕会变白，因此通过平均灰度值来判断是否结束

    Args:
        img (np.ndarray): 输入一帧图像(RGB or GRAYSCALE)

    Returns:
        bool: 游戏结束返回True，否则返回False
    """
    if len(img.shape) != 2:
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    else:
        img_gray = img
    mean_grayscale = np.mean(img_gray)
    if mean_grayscale >= 220:
        return True
    else:
        return False
    
def is_win(situation) -> bool:
    """结束后根据先前优劣势判断胜负

    Args:
        situation: 1表示敌方劣势，-1表示己方劣势，一般不会有0出现

    Returns:
        bool: 己方是否胜利
    """
    if situation > 0:
        return True
    else:
        return False

def is_loading(img:np.ndarray) -> bool:
    """加载阶段是黑屏画面，利用平均灰度判断

    Args:
        img (np.ndarray): 输入一帧图像(RGB or GRAYSCALE)

    Returns:
        bool: 正在加载返回True，否则返回False
    """
    if len(img.shape) != 2:
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    else:
        img_gray = img
    mean_grayscale = np.mean(img_gray)
    if mean_grayscale <= 2:
        return True
    else:
        return False      


if __name__ == "__main__":
    img = cv.imread("The_King_of_Fighters_XV\dev_imgs\laoding.png", cv.IMREAD_GRAYSCALE)
    time1 = time.time()
    print(np.mean(img))
    time2 = time.time()
    print(time2-time1) # 计算均值产生的延迟约2ms