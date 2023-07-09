# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:50 2020

@author: pang
"""

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

SPACE = 0x39

UP = 0xC8
LEFT = 0xCB
RIGHT = 0xCD
DOWN = 0xD0

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

ENTER = 0x9C

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
V = 0x2F
U = 0x16
Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21

up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
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

if __name__ == '__main__':
    time.sleep(5)