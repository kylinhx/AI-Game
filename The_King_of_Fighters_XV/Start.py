import time
from direct_keys import *

if __name__ == "__main__":
    print('开始游戏')
    time.sleep(1)
    
    # 进入游戏界面
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    # 选择对战
    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)
    
    # 两次down，一次ENTER
    time.sleep(2)
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)

    time.sleep(2)
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    # 一次down，一次enter
    time.sleep(2)
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    # 按下键盘的SPACE键
    time.sleep(2)
    PressKey(SPACE)
    time.sleep(0.4)
    ReleaseKey(SPACE)
    
    time.sleep(5)
    for i in range(9):
        time.sleep(1)
        PressKey(A)
        time.sleep(0.1)
        ReleaseKey(A)
    
    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    for i in range(5):
        time.sleep(1)
        PressKey(A)
        time.sleep(0.1)
        ReleaseKey(A)
    
    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

    time.sleep(2)
    PressKey(W)
    time.sleep(0.1)
    ReleaseKey(W)

    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)