import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

if __name__ == "__main__":

    while True:
        # 检查键盘输入
        keys = key_check()
        if len(keys) > 0:
            # 如果有键被按下，则输出被按下的键
            print(keys)
        # 等待0.1秒再次检查键盘输入
        time.sleep(0.1)