import ctypes
import time
from utils import *

# 所有的action，选手重新写

# 跳跃
def Jump():
    PressKey(W)
    time.sleep(0.015)
    ReleaseKey(W)
    #time.sleep(0.1)

# 下蹲
def Down():
    PressKey(S)
    time.sleep(0.015)
    ReleaseKey(S)

# 左移动
def Left():
    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

# 右移动
def Right():
    PressKey(D)
    time.sleep(0.015)
    ReleaseKey(D)

# 轻拳
def attack_light_rist():
    PressKey(U)
    time.sleep(0.015)
    ReleaseKey(U)

# 重拳
def attack_heavy_rist():
    PressKey(I)
    time.sleep(0.015)
    ReleaseKey(I)

# 轻腿
def attack_light_legs():
    PressKey(J)
    time.sleep(0.015)
    ReleaseKey(J)

# 重腿
def attack_heavy_legs():
    PressKey(K)
    time.sleep(0.015)
    ReleaseKey(K)

def attack_1():
    PressKey(D)
    time.sleep(0.015)
    ReleaseKey(D)

    PressKey(W)
    time.sleep(0.015)
    ReleaseKey(W)

    PressKey(J)
    time.sleep(0.015)
    ReleaseKey(J)

def attack_2():
    PressKey(D)
    time.sleep(0.015)
    ReleaseKey(D)

    PressKey(S)
    time.sleep(0.015)
    ReleaseKey(S)

    PressKey(J)
    time.sleep(0.015)
    ReleaseKey(J)

def attack_3():
    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

    PressKey(S)
    time.sleep(0.015)
    ReleaseKey(S)

    PressKey(J)
    time.sleep(0.015)
    ReleaseKey(J)

def attack_4():
    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

    PressKey(W)
    time.sleep(0.015)
    ReleaseKey(W)

    PressKey(J)
    time.sleep(0.015)
    ReleaseKey(J)

def attack_5():
    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

    PressKey(W)
    time.sleep(0.015)
    ReleaseKey(W)

    PressKey(U)
    time.sleep(0.015)
    ReleaseKey(U)

def attack_6():
    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

    PressKey(S)
    time.sleep(0.015)
    ReleaseKey(S)

    PressKey(U)
    time.sleep(0.015)
    ReleaseKey(U)

def supper_1():
    PressKey(S)
    time.sleep(0.015)
    ReleaseKey(S)

    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

    PressKey(D)
    time.sleep(0.015)
    ReleaseKey(D)

    PressKey(U)
    time.sleep(0.015)
    ReleaseKey(U)

def supper_2():
    PressKey(S)
    time.sleep(0.015)
    ReleaseKey(S)

    PressKey(A)
    time.sleep(0.015)
    ReleaseKey(A)

    PressKey(D)
    time.sleep(0.015)
    ReleaseKey(D)

    PressKey(K)
    time.sleep(0.015)
    ReleaseKey(K)

if __name__ == '__main__':
    # while True:
    #     supper_2()
    #     time.sleep(0.015*4)
    pass