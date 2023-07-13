import time
from direct_keys import * 
from utils import init_game, restart_game

class Start():
    def __init__(self) -> None:
        
        self.time = 0
    
    # 刚开始进入游戏时运行
    def init_game(self):
        init_game()
        return
    
    # 每次结束后都运行
    def restart_game(self):
        restart_game()


if __name__ == "__main__":
    starter = Start()
    starter.init_game()
    starter.restart_game()