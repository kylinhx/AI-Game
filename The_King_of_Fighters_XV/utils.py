import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
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
R = 0x13
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
    

# 截图函数
def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)
    img = img[:, :, :3]
    
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


# 选择对战模式、选择人物、选择进入对战
def init_game():
    print('Start Game')
    # time.sleep(1)
    # PressKey(ENTER)
    # time.sleep(0.4)
    # ReleaseKey(ENTER)
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
    time.sleep(4)

    PressKey(W)
    time.sleep(0.1)
    ReleaseKey(W)
    
    time.sleep(2)
    PressKey(ENTER)
    time.sleep(0.4)
    ReleaseKey(ENTER)

def restart_game():
    print('Restart Game')

    time.sleep(5)
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


# 这个函数kylinhx在环境中定义过了，弃置不用
def is_end(img:np.ndarray) -> bool:
    """一方被击败时屏幕会变白，因此通过平均灰度值来判断是否结束

    Args:
        img (np.ndarray): 输入一帧图像(RGB or GRAYSCALE)

    Returns:
        bool: 游戏结束返回True，否则返回False
    """
    if len(img.shape) != 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    mean_grayscale = np.mean(img_gray)
    if mean_grayscale <= 2:
        return True
    else:
        return False      


if __name__ == "__main__":
    img = cv2.imread("The_King_of_Fighters_XV\dev_imgs\loading.png", cv2.IMREAD_GRAYSCALE)
    time1 = time.time()
    print(np.mean(img))
    time2 = time.time()
    print(time2-time1) # 计算均值产生的延迟约2ms