#!/usr/bin/env python3
"""
Learning Curve Visualization Script

Analyze PPO training learning curves and performance metrics
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_training_logs(log_dir: str = "logs/ppo_doctor") -> Dict[str, Any]:
    """Load training logs"""
    log_path = Path(log_dir)
    
    # Try to read CSV log
    progress_csv = log_path / "progress.csv"
    if progress_csv.exists():
        df = pd.read_csv(progress_csv)
        return {"csv_data": df, "source": "csv"}
    
    # Try to read JSON log
    training_logs_json = Path("logs/training_logs.json")
    if training_logs_json.exists():
        with open(training_logs_json, 'r') as f:
            data = json.load(f)
        return {"json_data": data, "source": "json"}
    
    print(f"⚠️  Training log file not found")
    return {"source": "none"}

def plot_reward_curves(data: Dict[str, Any], save_dir: str = "data/training_results"):
    """Plot reward curves"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO Training Learning Curves - Based on Real MIMIC-III Data', fontsize=16, fontweight='bold')
    
    if data["source"] == "csv":
        df = data["csv_data"]
        
        # Reward curve
        reward_col = 'rollout/ep_rew_mean'
        timesteps_col = 'time/total_timesteps'
        
        if reward_col in df.columns and timesteps_col in df.columns:
            axes[0, 0].plot(df[timesteps_col], df[reward_col], color='#2E86AB', linewidth=2, marker='o', markersize=3)
            axes[0, 0].set_title('📈 Average Reward Over Time')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0.8, 1.2)
        
        # Episode length
        length_col = 'rollout/ep_len_mean'
        if length_col in df.columns and timesteps_col in df.columns:
            axes[0, 1].plot(df[timesteps_col], df[length_col], color='#A23B72', linewidth=2, marker='s', markersize=3)
            axes[0, 1].set_title('📏 Average Episode Length')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Average Length')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        lr_col = 'train/learning_rate'
        if lr_col in df.columns and timesteps_col in df.columns:
            axes[1, 0].plot(df[timesteps_col], df[lr_col], color='#F18F01', linewidth=2, marker='^', markersize=3)
            axes[1, 0].set_title('🎯 Learning Rate Over Time')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Value function loss
        loss_col = 'train/value_loss'
        if loss_col in df.columns and timesteps_col in df.columns:
            axes[1, 1].plot(df[timesteps_col], df[loss_col], color='#C73E1D', linewidth=2, marker='d', markersize=3)
            axes[1, 1].set_title('💥 Value Function Loss')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].grid(True, alpha=0.3)
    
    elif data["source"] == "json":
        json_data = data["json_data"]
        
        if json_data:
            steps = [item["step"] for item in json_data]
            rewards = [item["episode_reward_mean"] for item in json_data]
            lengths = [item["episode_length_mean"] for item in json_data]
            fps_values = [item["fps"] for item in json_data]
            
            # Reward curve
            axes[0, 0].plot(steps, rewards, color='#2E86AB', linewidth=2, marker='o', markersize=4)
            axes[0, 0].set_title('📈 Average Reward Over Time')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Episode length
            axes[0, 1].plot(steps, lengths, color='#A23B72', linewidth=2, marker='s', markersize=4)
            axes[0, 1].set_title('📏 Average Episode Length')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Average Length')
            axes[0, 1].grid(True, alpha=0.3)
            
            # FPS
            axes[1, 0].plot(steps, fps_values, color='#F18F01', linewidth=2, marker='^', markersize=4)
            axes[1, 0].set_title('⚡ Training Speed (FPS)')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Frames Per Second')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Training progress
            progress = [step / max(steps) for step in steps]
            axes[1, 1].plot(steps, progress, color='#C73E1D', linewidth=2, marker='d', markersize=4)
            axes[1, 1].set_title('📊 Training Progress')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Completion')
            axes[1, 1].grid(True, alpha=0.3)
    
    else:
        # Blank chart shows waiting for training
        for i, ax in enumerate(axes.flat):
            ax.text(0.5, 0.5, '⏳ Waiting for training data...', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Chart {i+1}')
    
    plt.tight_layout()
    
    # Save charts
    plt.savefig(save_path / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / 'learning_curves.pdf', bbox_inches='tight')
    
    print(f"📊 Learning curve charts saved to: {save_path}")
    return fig

def plot_reward_distribution(data: Dict[str, Any], save_dir: str = "data/training_results"):
    """Plot reward distribution"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    if data["source"] == "csv":
        df = data["csv_data"]
        reward_col = 'rollout/ep_rew_mean'
        timesteps_col = 'time/total_timesteps'
        
        if reward_col in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Reward distribution histogram
            ax1.hist(df[reward_col], bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
            ax1.set_title('📊 Reward Distribution Histogram')
            ax1.set_xlabel('Average Reward')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Reward trend
            rolling_mean = df[reward_col].rolling(window=10, min_periods=1).mean()
            if timesteps_col in df.columns:
                x_axis = df[timesteps_col]
                ax2.plot(x_axis, df[reward_col], alpha=0.5, color='gray', label='Raw Reward', linewidth=1)
                ax2.plot(x_axis, rolling_mean, color='#2E86AB', linewidth=2, label='10-episode Moving Average')
                ax2.set_xlabel('Training Steps')
            else:
                ax2.plot(df.index, df[reward_col], alpha=0.5, color='gray', label='Raw Reward', linewidth=1)
                ax2.plot(df.index, rolling_mean, color='#2E86AB', linewidth=2, label='10-episode Moving Average')
                ax2.set_xlabel('Training Epoch')
            
            ax2.set_title('📈 Reward Trend Analysis')
            ax2.set_ylabel('Average Reward')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path / 'reward_analysis.png', dpi=300, bbox_inches='tight')
            
            print(f"📈 Reward analysis chart saved to: {save_path}")
            return fig

def analyze_training_performance(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze training performance"""
    if data["source"] == "csv":
        df = data["csv_data"]
        
        timesteps_col = 'time/total_timesteps'
        reward_col = 'rollout/ep_rew_mean'
        length_col = 'rollout/ep_len_mean'
        
        analysis = {
            "total_timesteps": df[timesteps_col].max() if timesteps_col in df.columns else 0,
            "episodes_completed": len(df),
            "final_mean_reward": df[reward_col].iloc[-1] if reward_col in df.columns and len(df) > 0 else 0,
            "best_mean_reward": df[reward_col].max() if reward_col in df.columns else 0,
            "average_episode_length": df[length_col].mean() if length_col in df.columns else 0,
            "training_stability": 1.0 - (df[reward_col].std() / abs(df[reward_col].mean())) if reward_col in df.columns and df[reward_col].mean() != 0 else 0
        }
        
    elif data["source"] == "json":
        json_data = data["json_data"]
        
        if json_data:
            rewards = [item["episode_reward_mean"] for item in json_data]
            lengths = [item["episode_length_mean"] for item in json_data]
            
            analysis = {
                "total_timesteps": json_data[-1]["step"] if json_data else 0,
                "episodes_completed": len(json_data),
                "final_mean_reward": rewards[-1] if rewards else 0,
                "best_mean_reward": max(rewards) if rewards else 0,
                "average_episode_length": np.mean(lengths) if lengths else 0,
                "training_stability": 1.0 - (np.std(rewards) / abs(np.mean(rewards))) if rewards and np.mean(rewards) != 0 else 0
            }
        else:
            analysis = {
                "total_timesteps": 0,
                "episodes_completed": 0,
                "final_mean_reward": 0,
                "best_mean_reward": 0,
                "average_episode_length": 0,
                "training_stability": 0
            }
    else:
        analysis = {
            "total_timesteps": 0,
            "episodes_completed": 0,
            "final_mean_reward": 0,
            "best_mean_reward": 0,
            "average_episode_length": 0,
            "training_stability": 0
        }
    
    return analysis

def create_training_report(save_dir: str = "data/training_results"):
    """Create training report"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Load data
    data = load_training_logs()
    analysis = analyze_training_performance(data)
    
    # Create report
    report = f"""
# Multi-Agent Medical System PPO Training Report

## 🏥 Project Overview
- **Training Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: Enhanced Multi-Agent Medical Collaboration Environment
- **Data Source**: Real MIMIC-III Medical Data
- **Agents**: Doctor, Patient, Insurance Auditor

## 📊 Training Statistics

### Basic Metrics
- **Total Training Steps**: {analysis['total_timesteps']:,}
- **Completed Episodes**: {analysis['episodes_completed']:,}
- **Final Average Reward**: {analysis['final_mean_reward']:.3f}
- **Best Average Reward**: {analysis['best_mean_reward']:.3f}
- **Average Episode Length**: {analysis['average_episode_length']:.1f}
- **Training Stability**: {analysis['training_stability']:.2%}

### Performance Evaluation

#### 🎯 Training Effectiveness
- {'✅ Excellent' if analysis['training_stability'] > 0.7 else '⚠️  Average' if analysis['training_stability'] > 0.4 else '❌ Needs Improvement'} **Stability**: {analysis['training_stability']:.2%}
- {'✅ Excellent' if analysis['final_mean_reward'] > 1.0 else '⚠️  Average' if analysis['final_mean_reward'] > 0.5 else '❌ Needs Improvement'} **Final Reward**: {analysis['final_mean_reward']:.3f}

#### 📈 Improvement Suggestions
"""
    
    if analysis['training_stability'] < 0.5:
        report += "- Consider lowering learning rate to improve training stability\n"
    if analysis['final_mean_reward'] < 0.5:
        report += "- Consider adjusting reward function weights\n"
    if analysis['average_episode_length'] < 10:
        report += "- Episode may terminate prematurely, check termination conditions\n"
    
    report += f"""
## 🔧 Technical Details

### Environment Configuration
- **Observation Space**: 85-dimensional real vector
- **Action Space**: Multi-Discrete Action Space
- **Maximum Episode Length**: 50 Steps
- **Reward Function**: Optimized Based on MIMIC-III Data

### Model Configuration
- **Network Architecture**: MlpPolicy
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Collect Steps**: 2048
- **Training Epochs**: 10

## 📋 Data-Driven Characteristics

### MIMIC-III Integration
- ✅ **Patient Model**: Based on 100 Real Patients
- ✅ **Diagnosis Mapping**: 581 Real Diagnoses
- ✅ **Drug Data**: 100 Common Drugs
- ✅ **Cost Data**: 297 DRG Cost Classification

### Reward Optimization
- 🎯 **Death Risk Reduction**: 20% Weight
- 🏥 **Treatment Success**: 17% Weight
- 💰 **Cost Optimization**: 11% Weight
- 📊 **Symptom Improvement**: 10% Weight

## 🚀 Next Steps
1. Train Other Agents (Patient, Insurance)
2. Multi-Agent Joint Training
3. Hyperparameter Optimization
4. Performance Baseline Test

---
*Report Generated Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report
    with open(save_path / "training_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"📋 Training Report Generated: {save_path / 'training_report.md'}")
    return report

def main():
    """Main function"""
    print("📊 Starting Learning Curve Analysis")
    print("=" * 50)
    
    # Create results directory
    results_dir = "data/training_results"
    Path(results_dir).mkdir(exist_ok=True)
    
    # Load training data
    print("📥 Loading Training Logs...")
    data = load_training_logs()
    
    if data["source"] != "none":
        print(f"✅ Found {data['source']} format training logs")
        
        # Plot learning curves
        print("📈 Generating Learning Curves...")
        plot_reward_curves(data, results_dir)
        
        # Plot reward analysis
        if data["source"] == "csv":
            print("📊 Generating Reward Analysis...")
            plot_reward_distribution(data, results_dir)
        
        # Create training report
        print("📋 Generating Training Report...")
        create_training_report(results_dir)
        
        print(f"\n�� Analysis Completed!")
        print(f"📁 All results saved to: {results_dir}")
        
    else:
        print("⏳ Waiting for training logs, training may still be in progress...")
        print("💡 Please wait for training to start or complete before running this script")
        
        # Still create basic report
        create_training_report(results_dir)

if __name__ == "__main__":
    main() 