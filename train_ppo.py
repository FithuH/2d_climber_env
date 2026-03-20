# train_ppo.py
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

from rl_env import ClimberEnv

def make_env():
    """环境创建工厂函数"""
    def _init():
        env = ClimberEnv(render_mode=None)
        env = Monitor(env) 
        return env
    return _init

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models", "ppo_climber")
    logs_dir = os.path.join(base_dir, "tb_logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    device = "cpu"
    print(f"正在使用计算设备: {device.upper()}")

    num_cpu = 8 
    vec_env = SubprocVecEnv([make_env() for i in range(num_cpu)])

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,          
        tensorboard_log=logs_dir, 
        n_steps=2048,           
        batch_size=256,         
        n_epochs=10,            
        gamma=0.99,             
        ent_coef=0.08,          
        learning_rate=3e-4,     
        policy_kwargs=policy_kwargs
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // num_cpu, 1), 
        save_path=models_dir,
        name_prefix="ppo_model"
    )

    total_timesteps = 3_000_000 
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print(f"开始训练，日志将保存在: {logs_dir}")
    print("你可以新开一个终端输入 `tensorboard --logdir tb_logs` 查看实时曲线。")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name=f"PPO_{current_time}" 
    )

    model.save(os.path.join(models_dir, "ppo_climber_final"))
    print("训练完成并已保存！")