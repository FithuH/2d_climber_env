# test_model.py
import os
import time
from stable_baselines3 import PPO
from rl_env import ClimberEnv

def main():
    # ==========================================
    # 1. 路径配置 (确保与训练脚本一致)
    # ==========================================
    base_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models", "ppo_climber")
    
    # 你可以修改这里来加载不同的权重文件
    # 如果训练还没结束，你可以加载中途保存的 checkpoint，例如 "ppo_model_100000_steps.zip"
    # 如果训练结束了，就加载最终的模型 "ppo_climber_final.zip"
    model_name = "ppo_climber_final" 
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path + ".zip"):
        print(f"❌ 找不到模型文件: {model_path}.zip")
        print("请检查模型名称是否正确，或者等待训练脚本保存第一个 Checkpoint。")
        return

    print(f"✅ 成功找到模型: {model_name}.zip，正在加载...")

    # ==========================================
    # 2. 初始化环境 (开启人类可视渲染模式)
    # ==========================================
    env = ClimberEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    # ==========================================
    # 3. 运行测试循环 (看 AI 玩游戏)
    # ==========================================
    episodes = 5  # 测试 5 局
    
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        score = 0.0
        
        print(f"\n--- 第 {ep + 1} 局开始 ---")
        
        while not (terminated or truncated):
            # 使用模型预测下一步动作
            # deterministic=True 表示使用确定的最优策略，不再加入随机探索
            action, _states = model.predict(obs, deterministic=True)
            
            # 与环境交互
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            
            # 如果觉得 AI 跑得太快看不清，可以取消下面这行的注释来强制减速
            # time.sleep(0.01) 
            
        print(f"第 {ep + 1} 局结束！")
        print(f"获得总奖励: {score:.2f}")
        print(f"最高到达相对高度: {info.get('highest_reached', 0):.2f} 像素")
        time.sleep(1) # 局与局之间停顿 1 秒

    env.close()
    print("测试结束。")

if __name__ == "__main__":
    main()