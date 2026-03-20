# core_physics.py
import pygame
from typing import List
from config import Config

class Platform:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.rect = pygame.Rect(x, y, width, height)

class Player:
    def __init__(self, x: float, y: float):
        # 核心物理状态使用浮点数，防止精度丢失卡死
        self.true_x: float = x
        self.true_y: float = y
        self.vy: float = 0.0          
        self.is_grounded: bool = False
        
        self.rect = pygame.Rect(int(self.true_x), int(self.true_y), Config.PLAYER_SIZE, Config.PLAYER_SIZE)

    def update(self, keys: pygame.key.ScancodeWrapper, platforms: List[Platform]) -> None:
        # 1. 水平运动学
        vx = 0.0
        current_speed = Config.BASE_SPEED
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            current_speed += Config.DASH_BONUS
            
        if keys[pygame.K_a]:
            vx = -current_speed
        if keys[pygame.K_d]:
            vx = current_speed
            
        self.true_x += vx
        
        # 边界约束
        if self.true_x < 0: 
            self.true_x = 0.0
        if self.true_x > Config.WIDTH - Config.PLAYER_SIZE: 
            self.true_x = float(Config.WIDTH - Config.PLAYER_SIZE)
            
        self.rect.x = int(self.true_x)

        # 2. 垂直运动学
        self.vy += Config.GRAVITY
        self.true_y += self.vy
        self.rect.y = int(self.true_y)
        self.is_grounded = False

        # 3. 单向离散碰撞检测 (One-way Collision)
        # 仅当角色下落时，且是从台阶上方落入时才发生物理阻挡
        if self.vy > 0:
            for plat in platforms:
                if self.rect.colliderect(plat.rect):
                    # 穿透容差判定：防止侧面平移导致的错误判定
                    if self.rect.bottom - self.vy <= plat.rect.top + Config.GRAVITY * 2: 
                        self.true_y = float(plat.rect.top - Config.PLAYER_SIZE)
                        self.rect.y = int(self.true_y)
                        self.vy = 0.0
                        self.is_grounded = True
                        break

    def jump(self) -> None:
        if self.is_grounded:
            self.vy = -Config.JUMP_VELOCITY
            self.is_grounded = False