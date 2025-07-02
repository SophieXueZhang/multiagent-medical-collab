#!/usr/bin/env python3
"""
增强版多智能体医疗系统强化学习训练脚本

基于真实MIMIC-III数据训练PPO智能体
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强化学习相关导入
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import torch

# 项目组件导入
from env.enhanced_multi_agent_env import EnhancedMultiAgentHealthcareEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn

class TrainingCallback(BaseCallback):
    """训练过程监控回调"""
    
    def __init__(self, save_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.training_logs = []
    
    def _on_step(self) -> bool:
        # 每save_freq步记录一次
        if self.n_calls % self.save_freq == 0:
            info = {
                'step': self.n_calls,
                'total_timesteps': self.locals.get('total_timesteps', 0),
                'fps': getattr(self.model, '_last_log_fps', 0),
                'episode_reward_mean': getattr(self.model, '_last_ep_reward_mean', 0),
                'episode_length_mean': getattr(self.model, '_last_ep_length_mean', 0),
            }
            self.training_logs.append(info)
            
            if self.verbose > 0:
                print(f"步骤 {self.n_calls:,}: 平均奖励 {info['episode_reward_mean']:.3f}, "
                      f"平均长度 {info['episode_length_mean']:.1f}, FPS {info['fps']:.1f}")
        
        return True
    
    def save_logs(self, filepath: str):
        """保存训练日志"""
        with open(filepath, 'w') as f:
            json.dump(self.training_logs, f, indent=2)

def create_single_agent_env():
    """创建单智能体包装的环境"""
    # 创建增强版环境
    base_env = EnhancedMultiAgentHealthcareEnv(
        render_mode="rgb_array", 
        use_real_data=True
    )
    
    # 为训练包装成单智能体环境 - 只训练医生智能体
    class SingleAgentWrapper(gym.Env):
        def __init__(self, multi_agent_env):
            super().__init__()
            self.env = multi_agent_env
            self.observation_space = multi_agent_env.observation_space("doctor")
            self.action_space = multi_agent_env.action_space("doctor")
            self.current_agent = "doctor"
            
        def reset(self, **kwargs):
            observations, infos = self.env.reset(**kwargs)
            doctor_obs = observations.get("doctor", np.zeros(self.observation_space.shape))
            doctor_info = infos.get("doctor", {})
            return doctor_obs, doctor_info
        
        def step(self, action):
            # 为医生智能体执行动作
            if self.env.agent_selection == "doctor":
                self.env.step(action)
            
            # 为其他智能体生成简单动作
            while self.env.agent_selection != "doctor" and not all(self.env.terminations.values()):
                if self.env.agent_selection == "patient":
                    # 患者配合治疗
                    patient_action = [5, 6, 3, 5]  # 中等配合度
                elif self.env.agent_selection == "insurance":
                    # 保险适度批准
                    insurance_action = [3, 2, 1, 5]  # 标准审批
                else:
                    patient_action = [0, 0, 0, 0]
                    insurance_action = [0, 0, 0, 0]
                
                self.env.step(patient_action if self.env.agent_selection == "patient" else insurance_action)
            
            # 获取医生的观察和奖励
            obs = self.env.observations.get("doctor", np.zeros(self.observation_space.shape))
            reward = self.env.rewards.get("doctor", 0)
            terminated = self.env.terminations.get("doctor", False)
            truncated = self.env.truncations.get("doctor", False)
            info = {}
            
            return obs, reward, terminated, truncated, info
        
        def close(self):
            self.env.close()
    
    return SingleAgentWrapper(base_env)

def train_single_agent(config: Dict[str, Any]):
    """训练单个PPO智能体（医生）"""
    print("🏥 开始训练医生智能体（PPO算法）")
    
    # 创建训练目录
    models_dir = Path("models")
    logs_dir = Path("logs") 
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # 创建环境
    print("📋 创建训练环境...")
    env = create_single_agent_env()
    
    # 配置日志
    try:
        logger = configure(str(logs_dir / "ppo_doctor"), ["stdout", "csv", "tensorboard"])
    except Exception:
        logger = configure(str(logs_dir / "ppo_doctor"), ["stdout", "csv"])
    
    # 创建PPO模型
    print("🧠 初始化PPO模型...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        verbose=1,
        device=config.get("device", "auto")
    )
    
    # 设置日志记录器
    model.set_logger(logger)
    
    # 创建回调
    training_callback = TrainingCallback(save_freq=config.get("log_freq", 10000))
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("save_freq", 50000),
        save_path=str(models_dir),
        name_prefix="ppo_doctor"
    )
    
    # 开始训练
    print(f"🚀 开始训练 {config['total_timesteps']:,} 步...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[training_callback, checkpoint_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"✅ 训练完成！耗时: {training_time:.1f}秒")
    
    # 保存最终模型
    final_model_path = models_dir / "ppo_doctor_final"
    model.save(str(final_model_path))
    print(f"💾 模型已保存到: {final_model_path}")
    
    # 保存训练日志
    training_callback.save_logs(str(logs_dir / "training_logs.json"))
    
    # 清理
    env.close()
    
    return model, training_callback.training_logs

def evaluate_trained_agent(model_path: str, n_episodes: int = 10):
    """评估训练好的智能体"""
    print(f"\n📊 评估训练好的智能体 ({n_episodes} episodes)")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建评估环境
    env = create_single_agent_env()
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_length += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {episode + 1}: 奖励 {total_reward:.3f}, 长度 {episode_length}")
    
    env.close()
    
    # 计算统计信息
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episodes": episode_rewards
    }
    
    print(f"\n📈 评估结果:")
    print(f"  平均奖励: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"  平均长度: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    
    return stats

def main():
    """主训练流程"""
    print("🚀 启动增强版多智能体医疗系统训练")
    print("=" * 60)
    
    # 训练配置
    config = {
        "total_timesteps": 100000,  # 训练步数
        "learning_rate": 3e-4,      # 学习率
        "n_steps": 2048,            # 每次收集的步数
        "batch_size": 64,           # 批次大小
        "n_epochs": 10,             # 每次更新的轮数
        "gamma": 0.99,              # 折扣因子
        "gae_lambda": 0.95,         # GAE参数
        "clip_range": 0.2,          # PPO裁剪范围
        "ent_coef": 0.01,           # 熵系数
        "vf_coef": 0.5,             # 价值函数系数
        "max_grad_norm": 0.5,       # 梯度裁剪
        "log_freq": 5000,           # 日志频率
        "save_freq": 25000,         # 保存频率
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"🖥️  设备: {config['device']}")
    print(f"📊 训练配置: {config['total_timesteps']:,} 步")
    
    try:
        # 训练智能体
        model, training_logs = train_single_agent(config)
        
        # 评估智能体
        model_path = "models/ppo_doctor_final"
        eval_stats = evaluate_trained_agent(model_path, n_episodes=5)
        
        # 保存评估结果
        results_dir = Path("data/training_results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "evaluation_results.json", "w") as f:
            json.dump(eval_stats, f, indent=2)
        
        print(f"\n🎉 训练和评估完成！")
        print(f"📁 结果保存在: {results_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 