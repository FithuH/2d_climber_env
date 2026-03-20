# core_physics.py
import pygame
from typing import List
from config import Config

class Platform:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.rect = pygame.Rect(int(x), int(y), int(width), int(height))

class Player:
    def __init__(self, x: float, y: float):
        self.true_x: float = x
        self.true_y: float = y
        self.vy: float = 0.0          
        self.is_grounded: bool = False
        self.rect = pygame.Rect(int(self.true_x), int(self.true_y), Config.PLAYER_SIZE, Config.PLAYER_SIZE)

    def update(self, move_dir: int, do_jump: bool, do_dash: bool, platforms: List[Platform]) -> None:
        if do_jump and self.is_grounded:
            self.vy = -Config.JUMP_VELOCITY
            self.is_grounded = False

        current_speed = Config.BASE_SPEED
        if do_dash:
            current_speed += Config.DASH_BONUS
            
        vx = move_dir * current_speed
        self.true_x += vx
        
        if self.true_x < 0: self.true_x = 0.0
        if self.true_x > Config.WIDTH - Config.PLAYER_SIZE: self.true_x = float(Config.WIDTH - Config.PLAYER_SIZE)
        self.rect.x = int(self.true_x)

        self.vy += Config.GRAVITY
        self.true_y += self.vy
        self.rect.y = int(self.true_y)
        self.is_grounded = False

        if self.vy > 0:
            for plat in platforms:
                if self.rect.colliderect(plat.rect):
                    if self.rect.bottom - self.vy <= plat.rect.top + Config.GRAVITY * 3: 
                        self.true_y = float(plat.rect.top - Config.PLAYER_SIZE)
                        self.rect.y = int(self.true_y)
                        self.vy = 0.0
                        self.is_grounded = True
                        break