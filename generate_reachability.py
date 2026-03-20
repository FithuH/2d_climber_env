# generate_reachability.py
import json
import math
from config import Config
from core_physics import Player

def compute_reachability():
    print("开始进行物理引擎实机模拟...")
    
    # 初始化玩家在原点
    player = Player(0, 0)
    player.is_grounded = True
    
    points = []
    
    # 第一帧：按下跳跃 + 右 + 冲刺
    player.update(move_dir=1, do_jump=True, do_dash=True, platforms=[])
    points.append((player.true_x, player.true_y))
    
    # 后续帧：在空中保持 右 + 冲刺，直到掉出屏幕底部
    # 模拟 200 帧（绝对足够落地）
    for _ in range(200):
        player.update(move_dir=1, do_jump=False, do_dash=True, platforms=[])
        points.append((player.true_x, player.true_y))
        if player.true_y > Config.HEIGHT:
            break

    # 解析模拟数据
    reach_map = {}
    min_y = min(p[1] for p in points)  # 物理极限最高点 (负数)
    
    print(f"实机测试最高可达相对高度: {-min_y:.2f} 像素")
    
    # 我们以 5 像素为一个区间 (bin)，记录对应的极限水平距离
    start_y = int(math.floor(min_y))
    end_y = Config.HEIGHT + 10
    
    for dy in range(start_y, end_y, 5):
        max_x = 0.0
        # 遍历轨迹，找到经过该高度时的 X 坐标
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            
            # 如果轨迹穿过或等于该高度
            if min(y1, y2) <= dy <= max(y1, y2):
                if y1 == y2:
                    x = max(x1, x2)
                else:
                    # 线性插值找到精确的 X
                    x = x1 + (x2 - x1) * (dy - y1) / (y2 - y1)
                if x > max_x:
                    max_x = x
                    
        reach_map[dy] = float(max_x)

    # 导出到 JSON
    with open("reachability.json", "w") as f:
        json.dump(reach_map, f, indent=4)
        
    print(f"数据已成功保存至 reachability.json！(记录了 {len(reach_map)} 个高度切片)")

if __name__ == "__main__":
    compute_reachability()