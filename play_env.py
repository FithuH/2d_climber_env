# play_env.py
import pygame
import sys
import random
from typing import List
from config import Config
from core_physics import Player, Platform

class ClimberGame:
    def __init__(self):
        pygame.init()
        # 强制关闭文本输入法，确保动作独占
        pygame.key.stop_text_input() 
        
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pygame.display.set_caption("RL Environment PoC: Climber Score System")
        self.clock = pygame.time.Clock()
        
        # 初始化 UI 字体管线
        # 使用系统默认无衬线字体，字号 32
        self.font = pygame.font.SysFont('arial', 32, bold=True) 
        
        self.reset()

    def generate_platforms(self, count: int = 50) -> List[Platform]:
        platforms = []
        current_x = Config.WIDTH // 2 - Config.PLATFORM_WIDTH // 2
        current_y = Config.HEIGHT - 50
        platforms.append(Platform(current_x, current_y, Config.PLATFORM_WIDTH, Config.PLATFORM_HEIGHT))

        for _ in range(count):
            dx = random.uniform(-Config.MAX_JUMP_DIST, Config.MAX_JUMP_DIST)
            dy = random.uniform(-Config.MAX_JUMP_HEIGHT, -Config.PLATFORM_HEIGHT * 2.5)
            
            next_x = current_x + dx
            next_y = current_y + dy
            
            next_x = max(0, min(Config.WIDTH - Config.PLATFORM_WIDTH, next_x))
            
            platforms.append(Platform(next_x, next_y, Config.PLATFORM_WIDTH, Config.PLATFORM_HEIGHT))
            current_x, current_y = next_x, next_y
            
        return platforms

    def reset(self) -> None:
        self.platforms = self.generate_platforms()
        start_plat = self.platforms[0]
        
        self.player = Player(
            start_plat.rect.centerx - Config.PLAYER_SIZE // 2, 
            start_plat.rect.top - Config.PLAYER_SIZE
        )
        self.camera_y = 0.0 
        
        # [新增] 计分状态初始化
        self.start_y = self.player.true_y
        self.highest_y = self.player.true_y
        self.score = 0

    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(Config.FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.player.jump()

            keys = pygame.key.get_pressed()
            self.player.update(keys, self.platforms)
            
            # [新增] 计算分数 (势能差)
            # 因为越往上 Y 越小，所以取 min
            if self.player.true_y < self.highest_y:
                self.highest_y = self.player.true_y
                
            # 每向上攀爬 10 像素计 1 分
            self.score = max(0, int((self.start_y - self.highest_y) / 10.0))

            if self.player.rect.top - self.camera_y > Config.HEIGHT:
                print(f"[Env Log] Agent fallen. Final Score: {self.score}. Resetting...")
                self.reset()

            if self.player.rect.y - self.camera_y < Config.HEIGHT // 2:
                self.camera_y = self.player.rect.y - Config.HEIGHT // 2

            # 渲染管线
            self.screen.fill((30, 30, 40)) 
            
            for plat in self.platforms:
                render_rect = pygame.Rect(
                    plat.rect.x, 
                    plat.rect.y - self.camera_y, 
                    plat.rect.width, 
                    plat.rect.height
                )
                if render_rect.bottom > 0 and render_rect.top < Config.HEIGHT: 
                    pygame.draw.rect(self.screen, (100, 200, 100), render_rect, border_radius=3)
            
            player_render_rect = pygame.Rect(
                self.player.rect.x, 
                self.player.rect.y - self.camera_y, 
                Config.PLAYER_SIZE, 
                Config.PLAYER_SIZE
            )
            pygame.draw.rect(self.screen, (250, 80, 80), player_render_rect)

            # [新增] 渲染分数 UI (绘制在右上角)
            score_surface = self.font.render(f"Score: {self.score}", True, (255, 215, 0)) # 金色字体
            score_rect = score_surface.get_rect()
            score_rect.topright = (Config.WIDTH - 20, 20)
            self.screen.blit(score_surface, score_rect)

            pygame.display.flip()
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ClimberGame()
    game.run()