#!/usr/bin/env python3
"""
增强版多智能体医疗环境

集成真实MIMIC-III数据的多智能体医疗协作环境，
使用真实患者模型和基于数据优化的奖励函数
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.communication import CommunicationProtocol, AgentRole, MessageType, Message
from utils.config_loader import config
from env.reward_optimizer import RealDataRewardOptimizer, RewardComponents

# 导入患者模型
sys.path.append(str(Path(__file__).parent.parent / "data"))
from patient_models import RealPatientModelGenerator, PatientCondition

class EnhancedMultiAgentHealthcareEnv(AECEnv):
    """增强版多智能体医疗环境"""
    
    metadata = {"render_modes": ["human"], "name": "enhanced_healthcare_v1"}
    
    def __init__(self, render_mode=None, use_real_data=True):
        super().__init__()
        
        # 环境配置
        self.use_real_data = use_real_data
        self.render_mode = render_mode
        
        # 智能体配置
        self.agents = ["doctor", "patient", "insurance"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        
        # 初始化组件
        self.communication_protocol = CommunicationProtocol()
        
        # 真实数据组件
        if self.use_real_data:
            self.patient_generator = RealPatientModelGenerator()
            self.reward_optimizer = RealDataRewardOptimizer()
            self.reward_weights = self.reward_optimizer.get_optimized_weights()
        else:
            self.patient_generator = None
            self.reward_optimizer = None
            self.reward_weights = self._get_default_weights()
        
        # 当前状态
        self.current_patient: Optional[PatientCondition] = None
        self.episode_step = 0
        self.max_steps = 50
        self.episode_history = []
        
        # 智能体状态
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = None
        
        # 观察和动作空间
        self._setup_spaces()
        
        # 性能统计
        self.episode_stats = {
            'treatment_effectiveness': 0.0,
            'cost_efficiency': 0.0,
            'patient_satisfaction': 0.0,
            'communication_quality': 0.0,
            'total_reward': 0.0
        }
        
        print("🏥 Enhanced multi-agent healthcare environment initialization completed")
        if self.use_real_data:
            print("   ✅ Using real MIMIC-III data")
            print("   ✅ Optimized reward function enabled")
        else:
            print("   ⚠️  Using simulated data")
    
    def _setup_spaces(self):
        """设置观察和动作空间"""
        # 观察空间：患者状态 + 通信历史 + 环境状态
        if self.use_real_data:
            # 基于真实数据的观察空间
            self.observation_spaces = {
                agent: spaces.Box(
                    low=-1.0, high=10.0, shape=(85,), dtype=np.float32
                ) for agent in self.agents
            }
        else:
            # 原始观察空间
            self.observation_spaces = {
                agent: spaces.Box(
                    low=-1.0, high=10.0, shape=(80,), dtype=np.float32
                ) for agent in self.agents
            }
        
        # 动作空间：增强的多离散动作
        self.action_spaces = {
            "doctor": spaces.MultiDiscrete([
                15,  # 诊断动作 (扩展)
                20,  # 治疗动作 (扩展)
                10,  # 检查动作
                15   # 通信动作
            ]),
            "patient": spaces.MultiDiscrete([
                10,  # 症状报告
                8,   # 反馈动作
                5,   # 配合程度
                10   # 通信动作
            ]),
            "insurance": spaces.MultiDiscrete([
                5,   # 审批动作
                8,   # 调查动作
                6,   # 谈判动作
                12   # 通信动作
            ])
        }
    
    def _get_default_weights(self) -> Dict[str, float]:
        """获取默认奖励权重"""
        return {
            'treatment_success': 0.20,
            'treatment_efficiency': 0.10,
            'cost_optimization': 0.15,
            'patient_safety': 0.20,
            'communication_efficiency': 0.10,
            'resource_utilization': 0.10,
            'insurance_optimization': 0.05,
            'delay_penalty': 0.05,
            'error_penalty': 0.05
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 重置智能体选择器
        self.agent_selector.reset()
        self.agent_selection = self.agent_selector.next()
        
        # 生成新患者
        if self.use_real_data and self.patient_generator:
            self.current_patient = self.patient_generator.generate_realistic_patient(seed=seed)
        else:
            self.current_patient = self._generate_simulated_patient()
        
        # 重置环境状态
        self.episode_step = 0
        self.episode_history = []
        self.communication_protocol.clear_history()
        
        # 重置性能统计
        self.episode_stats = {
            'treatment_effectiveness': 0.0,
            'cost_efficiency': 0.0,
            'patient_satisfaction': 0.0,
            'communication_quality': 0.0,
            'total_reward': 0.0
        }
        
        # 生成初始观察
        self.observations = {agent: self._get_obs(agent) for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        return self.observations, self.infos
    
    def step(self, action):
        """执行一步"""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        
        # 处理动作
        reward = self._process_action(agent, action)
        
        # 更新环境状态
        self.episode_step += 1
        
        # 计算奖励
        self.rewards[agent] = reward
        
        # 检查终止条件
        if self.episode_step >= self.max_steps:
            self.truncations = {agent: True for agent in self.agents}
        
        # 检查治疗完成条件
        if self.use_real_data and self.current_patient:
            if self._is_treatment_complete():
                self.terminations = {agent: True for agent in self.agents}
        
        # 更新观察
        self.observations[agent] = self._get_obs(agent)
        
        # 选择下一个智能体
        self.agent_selection = self.agent_selector.next()
        
        # 累积性能统计
        self._update_episode_stats(agent, reward)
    
    def _generate_simulated_patient(self) -> PatientCondition:
        """生成模拟患者（fallback）"""
        from dataclasses import dataclass
        
        # 创建基本的模拟患者
        patient = PatientCondition(
            subject_id=np.random.randint(1000, 9999),
            age=np.random.randint(18, 90),
            gender=np.random.randint(0, 2),
            primary_diagnosis="simulated_diagnosis",
            diagnosis_codes=["sim_001"],
            severity_score=np.random.uniform(1.0, 5.0),
            mortality_risk=np.random.uniform(0.0, 0.5),
            treatment_complexity=np.random.uniform(1.0, 3.0),
            symptoms={
                'pain': np.random.uniform(0.0, 1.0),
                'fatigue': np.random.uniform(0.0, 1.0),
                'nausea': np.random.uniform(0.0, 1.0)
            },
            current_treatments=[],
            treatment_effectiveness=0.7,
            estimated_treatment_cost=np.random.uniform(5000, 20000),
            insurance_coverage=np.random.uniform(0.6, 0.9),
            length_of_stay_prediction=np.random.uniform(3, 15),
            urgency_level=np.random.randint(1, 6)
        )
        
        return patient
    
    def _get_obs(self, agent: str) -> np.ndarray:
        """获取智能体观察"""
        if not self.current_patient:
            return np.zeros(self.observation_spaces[agent].shape[0])
        
        # 基础患者信息
        patient_obs = [
            self.current_patient.age / 100.0,  # 标准化年龄
            float(self.current_patient.gender),
            self.current_patient.severity_score / 5.0,  # 标准化严重程度
            self.current_patient.mortality_risk,
            self.current_patient.treatment_complexity / 5.0,
            self.current_patient.treatment_effectiveness,
            self.current_patient.estimated_treatment_cost / 50000.0,  # 标准化成本
            self.current_patient.insurance_coverage,
            self.current_patient.length_of_stay_prediction / 30.0,  # 标准化住院时长
            float(self.current_patient.urgency_level) / 5.0
        ]
        
        # 症状信息（扩展到10维）
        symptoms = list(self.current_patient.symptoms.values())[:10]
        while len(symptoms) < 10:
            symptoms.append(0.0)
        patient_obs.extend(symptoms)
        
        # 治疗信息（扩展到15维）
        treatments = [1.0 if t else 0.0 for t in self.current_patient.current_treatments[:15]]
        while len(treatments) < 15:
            treatments.append(0.0)
        patient_obs.extend(treatments)
        
        # 环境状态
        env_obs = [
            self.episode_step / self.max_steps,  # 进度
            len(self.episode_history) / 100.0,  # 历史长度
            self.episode_stats['treatment_effectiveness'],
            self.episode_stats['cost_efficiency'],
            self.episode_stats['patient_satisfaction']
        ]
        patient_obs.extend(env_obs)
        
        # 通信历史（扩展）
        comm_obs = self._get_communication_obs(agent)
        patient_obs.extend(comm_obs)
        
        # 智能体特定观察
        agent_specific_obs = self._get_agent_specific_obs(agent)
        patient_obs.extend(agent_specific_obs)
        
        # 确保观察长度正确
        obs_array = np.array(patient_obs, dtype=np.float32)
        target_length = self.observation_spaces[agent].shape[0]
        
        if len(obs_array) < target_length:
            # 填充零
            padding = np.zeros(target_length - len(obs_array))
            obs_array = np.concatenate([obs_array, padding])
        elif len(obs_array) > target_length:
            # 截断
            obs_array = obs_array[:target_length]
        
        return obs_array
    
    def _get_communication_obs(self, agent: str) -> List[float]:
        """获取通信观察"""
        comm_obs = []
        
        # 最近的通信消息（简化版本）
        recent_messages = self.communication_protocol.get_recent_messages(count=10)
        
        for i in range(10):  # 固定10条消息
            if i < len(recent_messages):
                msg = recent_messages[i]
                comm_obs.extend([
                    float(msg.sender.value),
                    float(msg.receiver.value),
                    float(msg.message_type.value),
                    msg.priority.value / 4.0,  # 标准化优先级
                    msg.cost / 1.0  # 标准化成本
                ])
            else:
                comm_obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return comm_obs  # 50维
    
    def _get_agent_specific_obs(self, agent: str) -> List[float]:
        """获取智能体特定观察"""
        if agent == "doctor":
            return [
                1.0,  # 医生标识
                0.0, 0.0,
                # 医疗专业信息
                0.8,  # 专业水平
                0.9,  # 诊断准确性
            ]
        elif agent == "patient":
            return [
                0.0, 1.0, 0.0,
                # 患者状态信息
                0.7,  # 配合度
                0.6,  # 满意度
            ]
        elif agent == "insurance":
            return [
                0.0, 0.0, 1.0,
                # 保险信息
                0.8,  # 审批率
                0.85, # 覆盖率
            ]
        else:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _process_action(self, agent: str, action) -> float:
        """处理智能体动作"""
        if not isinstance(action, (list, tuple, np.ndarray)):
            action = [action]
        
        if agent == "doctor":
            return self._process_doctor_action(action)
        elif agent == "patient":
            return self._process_patient_action(action)
        elif agent == "insurance":
            return self._process_insurance_action(action)
        
        return 0.0
    
    def _process_doctor_action(self, action) -> float:
        """处理医生动作"""
        if len(action) < 4:
            action = list(action) + [0] * (4 - len(action))
        
        diagnosis_action, treatment_action, check_action, comm_action = action[:4]
        
        # 记录动作
        action_record = {
            'agent': 'doctor',
            'step': self.episode_step,
            'diagnosis': diagnosis_action,
            'treatment': treatment_action,
            'check': check_action,
            'communication': comm_action
        }
        self.episode_history.append(action_record)
        
        # 计算奖励
        reward = 0.0
        
        # 治疗动作奖励
        if self.use_real_data and self.reward_optimizer:
            # 使用真实数据计算奖励
            treatment_data = {
                'effectiveness': self.current_patient.treatment_effectiveness,
                'appropriateness': 0.8 + (treatment_action / 20) * 0.2,  # 基于动作的适当性
                'efficiency': 0.7 + (diagnosis_action / 15) * 0.3
            }
            
            outcome_data = {
                'actual_cost': self.current_patient.estimated_treatment_cost * (0.8 + (treatment_action / 40)),
                'baseline_cost': self.current_patient.estimated_treatment_cost,
                'symptom_improvement': 0.6 + (treatment_action / 30) * 0.4,
                'mortality_risk_reduction': self.current_patient.mortality_risk * 0.3
            }
            
            # 使用优化的奖励函数
            components = self.reward_optimizer.calculate_comprehensive_reward(
                patient_data=self._get_patient_data_dict(),
                treatment_data=treatment_data,
                outcome_data=outcome_data,
                process_data={'decision_time': 5, 'communication_quality': 0.8, 'resource_utilization': 0.7}
            )
            
            reward = components.total_reward(self.reward_weights)
        else:
            # 简化奖励计算
            reward = (diagnosis_action + treatment_action) * 0.05
        
        # 通信奖励
        if comm_action > 0:
            comm_reward = self._process_communication(AgentRole.DOCTOR, comm_action)
            reward += comm_reward
        
        return reward
    
    def _process_patient_action(self, action) -> float:
        """处理患者动作"""
        if len(action) < 4:
            action = list(action) + [0] * (4 - len(action))
        
        symptom_action, feedback_action, cooperation_action, comm_action = action[:4]
        
        # 记录动作
        action_record = {
            'agent': 'patient',
            'step': self.episode_step,
            'symptom_report': symptom_action,
            'feedback': feedback_action,
            'cooperation': cooperation_action,
            'communication': comm_action
        }
        self.episode_history.append(action_record)
        
        # 计算奖励
        reward = 0.0
        
        # 配合度奖励
        cooperation_reward = cooperation_action * 0.1
        reward += cooperation_reward
        
        # 症状报告准确性奖励
        symptom_accuracy = 1.0 - abs(symptom_action - 5) / 5.0  # 假设5是最准确的报告
        reward += symptom_accuracy * 0.2
        
        # 通信奖励
        if comm_action > 0:
            comm_reward = self._process_communication(AgentRole.PATIENT, comm_action)
            reward += comm_reward
        
        return reward
    
    def _process_insurance_action(self, action) -> float:
        """处理保险动作"""
        if len(action) < 4:
            action = list(action) + [0] * (4 - len(action))
        
        approval_action, investigation_action, negotiation_action, comm_action = action[:4]
        
        # 记录动作
        action_record = {
            'agent': 'insurance',
            'step': self.episode_step,
            'approval': approval_action,
            'investigation': investigation_action,
            'negotiation': negotiation_action,
            'communication': comm_action
        }
        self.episode_history.append(action_record)
        
        # 计算奖励
        reward = 0.0
        
        # 审批效率奖励
        if approval_action > 0:
            # 基于保险覆盖率和成本的审批奖励
            coverage_factor = self.current_patient.insurance_coverage
            cost_factor = 1.0 - (self.current_patient.estimated_treatment_cost / 50000.0)
            approval_reward = approval_action * 0.1 * coverage_factor * max(0.1, cost_factor)
            reward += approval_reward
        
        # 调查准确性奖励
        investigation_reward = investigation_action * 0.05
        reward += investigation_reward
        
        # 通信奖励
        if comm_action > 0:
            comm_reward = self._process_communication(AgentRole.INSURANCE, comm_action)
            reward += comm_reward
        
        return reward
    
    def _process_communication(self, sender_role: AgentRole, comm_action: int) -> float:
        """处理通信动作"""
        # 简化的通信处理
        comm_cost = 0.1
        comm_effectiveness = min(comm_action / 10.0, 1.0)
        
        return comm_effectiveness * 0.1 - comm_cost
    
    def _get_patient_data_dict(self) -> Dict:
        """获取患者数据字典"""
        if not self.current_patient:
            return {}
        
        return {
            'severity': self.current_patient.severity_score,
            'insurance_coverage': self.current_patient.insurance_coverage,
            'age': self.current_patient.age,
            'mortality_risk': self.current_patient.mortality_risk
        }
    
    def _is_treatment_complete(self) -> bool:
        """检查治疗是否完成"""
        # 更合理的完成条件 - 只在接近最大步数时才考虑完成
        if self.episode_step < self.max_steps * 0.8:  # 至少完成80%的步数
            return False
        
        # 检查是否有充分的协作和治疗
        doctor_actions = [h for h in self.episode_history if h['agent'] == 'doctor']
        patient_actions = [h for h in self.episode_history if h['agent'] == 'patient']
        insurance_actions = [h for h in self.episode_history if h['agent'] == 'insurance']
        
        # 需要每个智能体都有足够的参与
        if (len(doctor_actions) >= 5 and 
            len(patient_actions) >= 5 and 
            len(insurance_actions) >= 5 and
            self.episode_stats['treatment_effectiveness'] > 0.7):
            return True
        
        return False
    
    def _update_episode_stats(self, agent: str, reward: float):
        """Update episode statistics"""
        self.episode_stats['total_reward'] += reward
        
        # Update specific metrics based on agent actions
        if agent == 'doctor':
            self.episode_stats['treatment_effectiveness'] = min(
                self.episode_stats['treatment_effectiveness'] + reward * 0.1, 1.0
            )
            # Doctor contributes to communication quality through medical communication
            self.episode_stats['communication_quality'] = min(
                self.episode_stats['communication_quality'] + abs(reward) * 0.08, 1.0
            )
        elif agent == 'patient':
            self.episode_stats['patient_satisfaction'] = min(
                self.episode_stats['patient_satisfaction'] + reward * 0.2, 1.0
            )
            # Patient contributes to communication quality through cooperation and feedback
            self.episode_stats['communication_quality'] = min(
                self.episode_stats['communication_quality'] + abs(reward) * 0.12, 1.0
            )
        elif agent == 'insurance':
            self.episode_stats['cost_efficiency'] = min(
                self.episode_stats['cost_efficiency'] + reward * 0.15, 1.0
            )
            # Insurance contributes to communication quality through efficient coordination
            self.episode_stats['communication_quality'] = min(
                self.episode_stats['communication_quality'] + abs(reward) * 0.10, 1.0
            )
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            print(f"\n=== Episode Step {self.episode_step} ===")
            if self.current_patient:
                print(f"患者: {self.current_patient.age}岁, 严重程度: {self.current_patient.severity_score:.2f}")
                print(f"当前智能体: {self.agent_selection}")
            print(f"Episode统计: {self.episode_stats}")
    
    def close(self):
        """关闭环境"""
        pass

def test_enhanced_environment():
    """测试增强版环境"""
    print("🧪 测试增强版多智能体医疗环境")
    print("=" * 60)
    
    # 创建环境
    env = EnhancedMultiAgentHealthcareEnv(render_mode="human", use_real_data=True)
    
    # 重置环境
    observations, infos = env.reset(seed=42)
    
    print(f"\n🔄 环境重置完成")
    print(f"智能体: {env.agents}")
    print(f"当前患者ID: {env.current_patient.subject_id}")
    print(f"患者诊断: {env.current_patient.primary_diagnosis}")
    print(f"严重程度: {env.current_patient.severity_score:.2f}")
    
    # 运行几步
    for step in range(10):
        agent = env.agent_selection
        
        # 生成随机动作
        if agent == "doctor":
            action = [np.random.randint(0, 15), np.random.randint(0, 20), 
                     np.random.randint(0, 10), np.random.randint(0, 15)]
        elif agent == "patient":
            action = [np.random.randint(0, 10), np.random.randint(0, 8), 
                     np.random.randint(0, 5), np.random.randint(0, 10)]
        elif agent == "insurance":
            action = [np.random.randint(0, 5), np.random.randint(0, 8), 
                     np.random.randint(0, 6), np.random.randint(0, 12)]
        
        # 执行动作
        env.step(action)
        
        # 渲染
        if step % 3 == 0:
            env.render()
        
        # 检查终止
        if all(env.terminations.values()) or all(env.truncations.values()):
            break
    
    print(f"\n📊 最终Episode统计:")
    for key, value in env.episode_stats.items():
        print(f"   {key}: {value:.3f}")
    
    env.close()
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_enhanced_environment() 