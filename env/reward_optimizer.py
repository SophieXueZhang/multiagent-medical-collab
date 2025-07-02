#!/usr/bin/env python3
"""
基于真实数据的奖励函数优化器

使用MIMIC-III数据优化多智能体医疗协作环境的奖励函数，
确保奖励机制反映真实医疗环境的目标和约束
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import config

@dataclass
class RewardComponents:
    """奖励组件数据类"""
    # 治疗效果相关
    treatment_success: float = 0.0
    treatment_efficiency: float = 0.0
    symptom_improvement: float = 0.0
    
    # 成本效率相关
    cost_optimization: float = 0.0
    resource_utilization: float = 0.0
    length_of_stay_optimization: float = 0.0
    
    # 患者安全相关
    mortality_risk_reduction: float = 0.0
    complication_prevention: float = 0.0
    treatment_appropriateness: float = 0.0
    
    # 协作效率相关
    communication_efficiency: float = 0.0
    decision_speed: float = 0.0
    information_sharing: float = 0.0
    
    # 保险相关
    insurance_optimization: float = 0.0
    approval_efficiency: float = 0.0
    
    # 惩罚项
    delay_penalty: float = 0.0
    error_penalty: float = 0.0
    
    def total_reward(self, weights: Dict[str, float]) -> float:
        """计算总奖励"""
        components = {
            # 正向奖励
            'treatment_success': self.treatment_success,
            'treatment_efficiency': self.treatment_efficiency,
            'symptom_improvement': self.symptom_improvement,
            'cost_optimization': self.cost_optimization,
            'resource_utilization': self.resource_utilization,
            'length_of_stay_optimization': self.length_of_stay_optimization,
            'mortality_risk_reduction': self.mortality_risk_reduction,
            'complication_prevention': self.complication_prevention,
            'treatment_appropriateness': self.treatment_appropriateness,
            'communication_efficiency': self.communication_efficiency,
            'decision_speed': self.decision_speed,
            'information_sharing': self.information_sharing,
            'insurance_optimization': self.insurance_optimization,
            'approval_efficiency': self.approval_efficiency,
            
            # 负向惩罚
            'delay_penalty': -self.delay_penalty,
            'error_penalty': -self.error_penalty
        }
        
        total = sum(components[key] * weights.get(key, 0.0) for key in components)
        return total

class RealDataRewardOptimizer:
    """基于真实数据的奖励函数优化器"""
    
    def __init__(self, processed_data_path="data/processed/"):
        self.data_path = Path(processed_data_path)
        
        # 加载预处理数据
        self.patients_df = None
        self.episodes_df = None
        self.diagnoses_mapping = None
        self.drugs_mapping = None
        self.cost_mapping = None
        
        # 基准统计数据
        self.benchmarks = {}
        
        # 优化的奖励权重
        self.optimized_weights = {}
        
        print("🎯 初始化基于真实数据的奖励函数优化器")
        self.load_data()
        self.calculate_benchmarks()
        self.optimize_reward_weights()
    
    def load_data(self):
        """加载预处理数据"""
        print("\n📥 加载预处理数据...")
        
        try:
            self.patients_df = pd.read_csv(self.data_path / "patients.csv")
            self.episodes_df = pd.read_csv(self.data_path / "episodes.csv")
            self.diagnoses_mapping = pd.read_csv(self.data_path / "diagnoses_mapping.csv")
            self.drugs_mapping = pd.read_csv(self.data_path / "drugs_mapping.csv")
            self.cost_mapping = pd.read_csv(self.data_path / "cost_mapping.csv")
            
            print(f"   ✅ 数据加载完成")
        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            raise
    
    def calculate_benchmarks(self):
        """计算基准统计数据"""
        print("\n📊 计算基准统计数据...")
        
        # 治疗成功率基准
        self.benchmarks['avg_treatment_effectiveness'] = self.drugs_mapping['effectiveness_score'].mean()
        self.benchmarks['target_treatment_effectiveness'] = 0.85  # 目标85%有效性
        
        # 成本基准
        self.benchmarks['avg_treatment_cost'] = self.cost_mapping['estimated_cost'].mean()
        self.benchmarks['target_cost_reduction'] = 0.15  # 目标减少15%成本
        
        # 住院时长基准
        self.benchmarks['avg_length_of_stay'] = self.episodes_df['length_of_stay'].mean()
        self.benchmarks['target_los_reduction'] = 0.20  # 目标减少20%住院时长
        
        # 死亡率基准
        self.benchmarks['avg_mortality_rate'] = self.episodes_df['hospital_expire_flag'].mean()
        self.benchmarks['target_mortality_reduction'] = 0.25  # 目标减少25%死亡率
        
        # 严重程度基准
        self.benchmarks['avg_severity'] = self.diagnoses_mapping['severity_score'].mean()
        self.benchmarks['avg_complexity'] = self.diagnoses_mapping['treatment_complexity'].mean()
        
        # 保险覆盖基准
        self.benchmarks['avg_insurance_coverage'] = self.cost_mapping['insurance_coverage'].mean()
        self.benchmarks['target_insurance_optimization'] = 0.90  # 目标90%保险覆盖
        
        print(f"   ✅ 基准数据计算完成")
        for key, value in self.benchmarks.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.3f}")
    
    def optimize_reward_weights(self):
        """优化奖励权重"""
        print("\n⚖️ 优化奖励权重...")
        
        # 基于真实数据分析的权重优化
        total_weight = 1.0
        
        # 主要目标权重（基于医疗质量优先级）
        self.optimized_weights = {
            # 治疗效果 (40% - 最重要)
            'treatment_success': 0.20,
            'treatment_efficiency': 0.10,
            'symptom_improvement': 0.10,
            
            # 患者安全 (25% - 第二重要)
            'mortality_risk_reduction': 0.15,
            'complication_prevention': 0.05,
            'treatment_appropriateness': 0.05,
            
            # 成本效率 (20% - 第三重要)
            'cost_optimization': 0.10,
            'resource_utilization': 0.05,
            'length_of_stay_optimization': 0.05,
            
            # 协作效率 (10% - 支持目标)
            'communication_efficiency': 0.03,
            'decision_speed': 0.04,
            'information_sharing': 0.03,
            
            # 保险优化 (5% - 运营效率)
            'insurance_optimization': 0.03,
            'approval_efficiency': 0.02,
            
            # 惩罚权重
            'delay_penalty': 0.1,
            'error_penalty': 0.2
        }
        
        # 基于数据特征调整权重
        # 如果死亡率较高，增加安全相关权重
        if self.benchmarks['avg_mortality_rate'] > 0.25:
            self.optimized_weights['mortality_risk_reduction'] += 0.05
            self.optimized_weights['treatment_success'] -= 0.03
            self.optimized_weights['cost_optimization'] -= 0.02
        
        # 如果成本较高，增加成本优化权重
        if self.benchmarks['avg_treatment_cost'] > 15000:
            self.optimized_weights['cost_optimization'] += 0.03
            self.optimized_weights['length_of_stay_optimization'] += 0.02
            self.optimized_weights['treatment_efficiency'] -= 0.05
        
        # 如果住院时长较长，增加效率权重
        if self.benchmarks['avg_length_of_stay'] > 10:
            self.optimized_weights['length_of_stay_optimization'] += 0.03
            self.optimized_weights['decision_speed'] += 0.02
            self.optimized_weights['symptom_improvement'] -= 0.05
        
        print(f"   ✅ 权重优化完成")
        print(f"   📋 优化后的权重分布:")
        for component, weight in self.optimized_weights.items():
            if weight > 0.01:  # 只显示重要权重
                print(f"     {component}: {weight:.3f}")
    
    def calculate_treatment_success_reward(self, 
                                         treatment_effectiveness: float, 
                                         patient_severity: float,
                                         treatment_appropriateness: float = 1.0) -> float:
        """计算治疗成功奖励"""
        # 基础成功奖励
        base_reward = treatment_effectiveness
        
        # 严重程度调整（治疗严重疾病给更高奖励）
        severity_bonus = patient_severity / self.benchmarks['avg_severity'] * 0.2
        
        # 治疗适当性调整
        appropriateness_factor = treatment_appropriateness
        
        # 与基准比较
        benchmark_factor = treatment_effectiveness / self.benchmarks['avg_treatment_effectiveness']
        
        final_reward = (base_reward + severity_bonus) * appropriateness_factor * benchmark_factor
        return max(0, min(final_reward, 2.0))  # 限制在0-2范围
    
    def calculate_cost_optimization_reward(self, 
                                         actual_cost: float, 
                                         baseline_cost: float,
                                         insurance_coverage: float) -> float:
        """计算成本优化奖励"""
        # 成本节约比例
        if baseline_cost > 0:
            cost_savings_ratio = (baseline_cost - actual_cost) / baseline_cost
        else:
            cost_savings_ratio = 0
        
        # 基础成本奖励
        base_reward = cost_savings_ratio
        
        # 保险覆盖调整
        insurance_factor = insurance_coverage / self.benchmarks['avg_insurance_coverage']
        
        # 与目标比较
        target_factor = cost_savings_ratio / self.benchmarks['target_cost_reduction'] if self.benchmarks['target_cost_reduction'] > 0 else 1
        
        final_reward = base_reward * insurance_factor * target_factor
        return max(-1.0, min(final_reward, 1.0))  # 限制在-1到1范围
    
    def calculate_safety_reward(self, 
                               mortality_risk_reduction: float,
                               complication_risk: float = 0.0,
                               treatment_safety_score: float = 1.0) -> float:
        """计算患者安全奖励"""
        # 死亡风险降低奖励
        mortality_reward = mortality_risk_reduction * 2.0  # 高权重
        
        # 并发症风险惩罚
        complication_penalty = complication_risk * 0.5
        
        # 治疗安全性奖励
        safety_reward = (treatment_safety_score - 0.5) * 0.5
        
        # 与基准比较
        benchmark_factor = mortality_risk_reduction / (self.benchmarks['avg_mortality_rate'] + 0.01)
        
        final_reward = (mortality_reward + safety_reward - complication_penalty) * benchmark_factor
        return max(0, min(final_reward, 2.0))
    
    def calculate_efficiency_reward(self, 
                                  decision_time: float,
                                  communication_quality: float,
                                  resource_utilization: float) -> float:
        """计算效率奖励"""
        # 决策速度奖励（时间越短越好）
        max_decision_time = 10.0  # 假设最大10分钟
        decision_reward = max(0, (max_decision_time - decision_time) / max_decision_time)
        
        # 沟通质量奖励
        communication_reward = communication_quality
        
        # 资源利用率奖励
        utilization_reward = resource_utilization
        
        # 综合效率分数
        efficiency_score = (decision_reward + communication_reward + utilization_reward) / 3
        
        return max(0, min(efficiency_score, 1.0))
    
    def calculate_length_of_stay_reward(self, 
                                      predicted_los: float, 
                                      actual_los: float,
                                      patient_severity: float) -> float:
        """计算住院时长优化奖励"""
        # 基于严重程度的预期住院时长
        severity_factor = patient_severity / self.benchmarks['avg_severity']
        expected_los = self.benchmarks['avg_length_of_stay'] * severity_factor
        
        # 实际表现与预期比较
        if expected_los > 0:
            los_ratio = (expected_los - actual_los) / expected_los
        else:
            los_ratio = 0
        
        # 预测准确性奖励
        prediction_accuracy = 1.0 - abs(predicted_los - actual_los) / max(predicted_los, actual_los, 1.0)
        
        # 综合奖励
        final_reward = los_ratio * 0.7 + prediction_accuracy * 0.3
        
        return max(-0.5, min(final_reward, 1.0))
    
    def calculate_insurance_optimization_reward(self, 
                                              insurance_coverage: float,
                                              approval_time: float,
                                              claim_accuracy: float) -> float:
        """计算保险优化奖励"""
        # 覆盖率奖励
        coverage_reward = insurance_coverage / self.benchmarks['target_insurance_optimization']
        
        # 审批速度奖励
        max_approval_time = 24.0  # 24小时
        approval_reward = max(0, (max_approval_time - approval_time) / max_approval_time)
        
        # 准确性奖励
        accuracy_reward = claim_accuracy
        
        # 综合保险优化分数
        insurance_score = (coverage_reward + approval_reward + accuracy_reward) / 3
        
        return max(0, min(insurance_score, 1.0))
    
    def calculate_comprehensive_reward(self, 
                                     patient_data: Dict,
                                     treatment_data: Dict,
                                     outcome_data: Dict,
                                     process_data: Dict) -> RewardComponents:
        """计算综合奖励"""
        components = RewardComponents()
        
        # 治疗效果相关
        components.treatment_success = self.calculate_treatment_success_reward(
            treatment_data.get('effectiveness', 0.7),
            patient_data.get('severity', 1.0),
            treatment_data.get('appropriateness', 1.0)
        )
        
        components.treatment_efficiency = treatment_data.get('efficiency', 0.5)
        components.symptom_improvement = outcome_data.get('symptom_improvement', 0.5)
        
        # 成本效率相关
        components.cost_optimization = self.calculate_cost_optimization_reward(
            outcome_data.get('actual_cost', 10000),
            outcome_data.get('baseline_cost', 10000),
            patient_data.get('insurance_coverage', 0.8)
        )
        
        components.length_of_stay_optimization = self.calculate_length_of_stay_reward(
            outcome_data.get('predicted_los', 5),
            outcome_data.get('actual_los', 5),
            patient_data.get('severity', 1.0)
        )
        
        # 患者安全相关
        components.mortality_risk_reduction = self.calculate_safety_reward(
            outcome_data.get('mortality_risk_reduction', 0.1),
            outcome_data.get('complication_risk', 0.0),
            treatment_data.get('safety_score', 1.0)
        )
        
        # 协作效率相关
        efficiency = self.calculate_efficiency_reward(
            process_data.get('decision_time', 5),
            process_data.get('communication_quality', 0.8),
            process_data.get('resource_utilization', 0.7)
        )
        
        components.communication_efficiency = efficiency * 0.4
        components.decision_speed = efficiency * 0.6
        
        # 保险相关
        insurance_reward = self.calculate_insurance_optimization_reward(
            patient_data.get('insurance_coverage', 0.8),
            process_data.get('approval_time', 12),
            process_data.get('claim_accuracy', 0.95)
        )
        
        components.insurance_optimization = insurance_reward * 0.7
        components.approval_efficiency = insurance_reward * 0.3
        
        # 惩罚项
        components.delay_penalty = process_data.get('delays', 0) * 0.1
        components.error_penalty = process_data.get('errors', 0) * 0.2
        
        return components
    
    def get_optimized_weights(self) -> Dict[str, float]:
        """获取优化后的权重"""
        return self.optimized_weights.copy()
    
    def save_reward_optimization(self, output_file: str = "reward_optimization.json"):
        """保存奖励优化结果"""
        output_path = self.data_path / output_file
        
        optimization_data = {
            'benchmarks': self.benchmarks,
            'optimized_weights': self.optimized_weights,
            'optimization_metadata': {
                'data_source': 'MIMIC-III',
                'optimization_strategy': 'data_driven_medical_priorities',
                'creation_time': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(optimization_data, f, indent=2)
        
        print(f"💾 奖励优化配置已保存到: {output_path}")

def main():
    """主函数"""
    print("🚀 启动基于真实数据的奖励函数优化")
    print("=" * 60)
    
    # 创建奖励优化器
    optimizer = RealDataRewardOptimizer()
    
    # 保存优化结果
    optimizer.save_reward_optimization()
    
    # 演示奖励计算
    print(f"\n🎯 奖励函数演示:")
    
    # 模拟患者和治疗数据
    sample_patient = {
        'severity': 2.5,
        'insurance_coverage': 0.85,
        'age': 65
    }
    
    sample_treatment = {
        'effectiveness': 0.88,
        'appropriateness': 0.95,
        'efficiency': 0.80,
        'safety_score': 0.92
    }
    
    sample_outcome = {
        'actual_cost': 12000,
        'baseline_cost': 15000,
        'predicted_los': 7,
        'actual_los': 6,
        'symptom_improvement': 0.85,
        'mortality_risk_reduction': 0.15,
        'complication_risk': 0.05
    }
    
    sample_process = {
        'decision_time': 3.5,
        'communication_quality': 0.90,
        'resource_utilization': 0.85,
        'approval_time': 8.0,
        'claim_accuracy': 0.98,
        'delays': 0,
        'errors': 0
    }
    
    # 计算综合奖励
    reward_components = optimizer.calculate_comprehensive_reward(
        sample_patient, sample_treatment, sample_outcome, sample_process
    )
    
    total_reward = reward_components.total_reward(optimizer.get_optimized_weights())
    
    print(f"   患者严重程度: {sample_patient['severity']:.2f}")
    print(f"   治疗有效性: {sample_treatment['effectiveness']:.2%}")
    print(f"   成本节约: ${sample_outcome['baseline_cost'] - sample_outcome['actual_cost']}")
    print(f"   住院时长优化: {sample_outcome['predicted_los'] - sample_outcome['actual_los']} 天")
    print(f"   \n🏆 综合奖励分数: {total_reward:.3f}")
    
    print(f"\n📊 主要奖励组件:")
    print(f"   治疗成功: {reward_components.treatment_success:.3f}")
    print(f"   成本优化: {reward_components.cost_optimization:.3f}")
    print(f"   安全改善: {reward_components.mortality_risk_reduction:.3f}")
    print(f"   效率提升: {reward_components.communication_efficiency + reward_components.decision_speed:.3f}")
    
    print("\n🎉 奖励函数优化完成！")
    print("\n🔄 下一步建议:")
    print("   1. 将优化的奖励函数集成到多智能体环境")
    print("   2. 使用真实患者模型重新训练智能体")
    print("   3. 验证改进后的系统性能")

if __name__ == "__main__":
    main() 