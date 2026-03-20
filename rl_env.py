# rl_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import json
import os
import math
from typing import Tuple, Dict, Any, List

from config import Config
from core_physics import Player, Platform

class ClimberEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.K_PLATFORMS = 5  
        self.FRAME_SKIP = 4
        
        self.action_space = spaces.MultiDiscrete([3, 2, 2])
        obs_dim = 3 + 2 * self.K_PLATFORMS
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        
        # 加载真实物理极限查找表
        json_path = os.path.join(os.path.dirname(__file__), "reachability.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError("找不到 reachability.json！请先运行预计算脚本。")
            
        with open(json_path, "r") as f:
            self.reach_map = {int(k): float(v) for k, v in json.load(f).items()}
        self.min_dy = min(self.reach_map.keys()) 
        
        self.window = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None: random.seed(seed)
            
        self.start_y = float(Config.HEIGHT - 100)
        self.player = Player(Config.WIDTH // 2, self.start_y)
        self.platforms = self._generate_platforms(count=20) 
        
        self.highest_y = self.player.true_y
        self.highest_ground_y = self.player.true_y 
        self.camera_y = 0.0
        self.step_count = 0
        
        self.steps_on_current_plat = 0         
        self.last_target_plat = None           
        self.min_dist_to_target = float('inf') 
        
        if self.render_mode == "human": self.render()
        return self._get_obs(), {}

    def _get_target_platform(self) -> Platform:
        for plat in self.platforms:
            if plat.rect.top < self.highest_ground_y - 10:
                return plat
        return self.platforms[0]

    def _get_distance_to_target(self, target: Platform) -> float:
        tx = target.rect.centerx
        ty = target.rect.top
        px = self.player.rect.centerx
        py = self.player.rect.bottom
        return math.hypot(tx - px, ty - py)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self.steps_on_current_plat += 1 
        
        move_dir = action[0] - 1  
        do_jump = bool(action[1])
        do_dash = bool(action[2])

        reward = 0.0
        terminated = False
        truncated = False
        
        # 锁定当前目标
        current_target = self._get_target_platform()
        if current_target != self.last_target_plat:
            self.last_target_plat = current_target
            self.min_dist_to_target = self._get_distance_to_target(current_target)

        current_dist = self._get_distance_to_target(current_target)

        # ==========================================
        # 🔪 策略三：斯巴达模式 (烫脚的地板)
        # ==========================================
        # 取消原本 -0.01 的固定惩罚。现在距离越远，每步扣分越狠！
        # 假设距离为 400 像素，这一步就会扣掉 -1.0 分。如果站着不动，几十步就会扣成重伤。
        reward -= (current_dist / 400.0)

        for _ in range(self.FRAME_SKIP):
            self.player.update(move_dir, do_jump, do_dash, self.platforms)
            
            # ==========================================
            # 🔪 策略一：通货紧缩 (白嫖势能分暴砍十倍)
            # ==========================================
            if self.player.true_y < self.highest_y:
                reward += (self.highest_y - self.player.true_y) * 0.01 # 从 0.1 降至 0.01
                self.highest_y = self.player.true_y
                
            # 登顶新台阶重赏 (上调至 100，抵消斯巴达模式跑路过程中的扣分成本)
            if self.player.is_grounded and self.player.true_y < self.highest_ground_y - 5.0:
                reward += 100.0 
                self.highest_ground_y = self.player.true_y
                self.steps_on_current_plat = 0 
                break 
                
            # ==========================================
            # 🔪 策略二：严刑峻法 (高额死亡惩罚)
            # ==========================================
            if self.player.rect.top - self.camera_y > Config.HEIGHT:
                reward -= 30.0 # 从 0 / -5.0 暴涨至 -30.0
                terminated = True
                break 
                
            if self.player.rect.y - self.camera_y < Config.HEIGHT // 2:
                self.camera_y = self.player.rect.y - Config.HEIGHT // 2

        # ==========================================
        # 🔪 策略一：通货紧缩 (白嫖距离分暴砍十倍)
        # ==========================================
        if not terminated:
            current_dist = self._get_distance_to_target(current_target)
            if current_dist < self.min_dist_to_target:
                reward += (self.min_dist_to_target - current_dist) * 0.01 # 从 0.1 降至 0.01
                self.min_dist_to_target = current_dist

        # ==========================================
        # 🔪 策略二：严刑峻法 (高额超时惩罚)
        # ==========================================
        if self.steps_on_current_plat > 150:
            reward -= 30.0 # 超时同样扣除 -30.0 分
            terminated = True

        self._update_platforms()
        
        if self.step_count >= 3000: 
            truncated = True

        if self.render_mode == "human": self.render()

        info = {"highest_reached": float(-self.highest_ground_y)}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = self.player.true_x / Config.WIDTH
        obs[1] = np.clip(self.player.vy / Config.JUMP_VELOCITY, -2.0, 2.0)
        obs[2] = 1.0 if self.player.is_grounded else 0.0  
        
        future_plats = [p for p in self.platforms if p.rect.bottom < self.player.true_y + 50]
        future_plats.sort(key=lambda p: abs(p.rect.y - self.player.true_y))
        
        for i in range(self.K_PLATFORMS):
            idx = 3 + i * 2 
            if i < len(future_plats):
                plat = future_plats[i]
                dx = (plat.rect.centerx - (self.player.true_x + Config.PLAYER_SIZE/2)) / Config.WIDTH
                dy = (plat.rect.top - self.player.true_y) / Config.HEIGHT
                obs[idx] = dx
                obs[idx+1] = dy
            else:
                obs[idx] = 0.0
                obs[idx+1] = -1.0 
        return obs

    def _get_safe_random_pos(self, prev_x: float, prev_y: float) -> Tuple[float, float, float]:
        # 平滑缓冲成长曲线 (保留上一轮的优秀设计)
        linear_progress = max(0.0, min(1.0, (self.start_y - prev_y) / 30000.0))
        progress = linear_progress ** 2 
        
        dy = random.uniform(self.min_dy * 0.85, -40.0)
        closest_dy = min(self.reach_map.keys(), key=lambda k: abs(k - dy))
        max_air_dx = self.reach_map[closest_dy]
        
        STEP_SIZE = (Config.BASE_SPEED + Config.DASH_BONUS) * self.FRAME_SKIP
        safe_max_dx = max_air_dx - STEP_SIZE
        
        min_gap = 10.0 + (safe_max_dx * 0.85 - 10.0) * progress
        max_gap = max(10.0, safe_max_dx * (0.5 + 0.5 * progress))
        min_gap = min(min_gap, max_gap * 0.9)
        
        dx_mag = random.uniform(min_gap, max_gap)
        plat_width = Config.PLATFORM_WIDTH - (Config.PLATFORM_WIDTH * 0.5) * progress
            
        direction = random.choice([-1, 1])
        
        if direction == 1:
            jump_origin_x = prev_x + Config.PLATFORM_WIDTH - Config.PLAYER_SIZE
            nx = jump_origin_x + dx_mag - (plat_width / 2)
        else:
            jump_origin_x = prev_x
            nx = jump_origin_x - dx_mag + (plat_width / 2)
            
        margin = 15.0
        max_x = Config.WIDTH - plat_width - margin
        
        if nx < margin or nx > max_x:
            direction *= -1
            if direction == 1:
                jump_origin_x = prev_x + Config.PLATFORM_WIDTH - Config.PLAYER_SIZE
                nx = jump_origin_x + dx_mag - (plat_width / 2)
            else:
                jump_origin_x = prev_x
                nx = jump_origin_x - dx_mag + (plat_width / 2)
            nx = max(margin, min(nx, max_x))
            
        ny = prev_y + dy
        return nx, ny, plat_width

    def _generate_platforms(self, count: int) -> List[Platform]:
        platforms = []
        cx = float(Config.WIDTH // 2 - Config.PLATFORM_WIDTH // 2)
        cy = float(Config.HEIGHT - 50)
        platforms.append(Platform(cx, cy, Config.PLATFORM_WIDTH, Config.PLATFORM_HEIGHT))
        
        for _ in range(count):
            nx, ny, w = self._get_safe_random_pos(cx, cy)
            platforms.append(Platform(nx, ny, w, Config.PLATFORM_HEIGHT))
            cx, cy = nx, ny
        return platforms

    def _update_platforms(self):
        self.platforms = [p for p in self.platforms if p.rect.top < self.camera_y + Config.HEIGHT + 100]
        if len(self.platforms) > 0:
            highest_plat = min(self.platforms, key=lambda p: p.rect.y)
            cx, cy = float(highest_plat.rect.x), float(highest_plat.rect.y)
            while highest_plat.rect.y > self.camera_y - Config.HEIGHT * 2:
                nx, ny, w = self._get_safe_random_pos(cx, cy)
                new_plat = Platform(nx, ny, w, Config.PLATFORM_HEIGHT)
                self.platforms.append(new_plat)
                highest_plat = new_plat
                cx, cy = nx, ny

    def render(self): ...
    def close(self): ...