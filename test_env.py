# test_env.py
from rl_env import ClimberEnv
from stable_baselines3.common.env_checker import check_env

# 1. 无头模式 (Headless) 校验：这对在服务器云端训练至关重要
env = ClimberEnv(render_mode=None)
print("正在校验环境规范...")
check_env(env, warn=True)
print("校验通过！环境符合 RL 接口标准。")

# 2. 随机策略测试 (Random Agent Test)
obs, info = env.reset()
score = 0
for _ in range(500):
    # 从 Action Space 随机采样合法动作
    random_action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(random_action)
    score += reward
    if terminated or truncated:
        print(f"Episode 结束。总 Reward: {score:.2f}")
        break