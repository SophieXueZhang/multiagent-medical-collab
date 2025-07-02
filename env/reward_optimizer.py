#!/usr/bin/env python3
"""
Real Data-Based Reward Function Optimizer

Optimize reward mechanisms for multi-agent healthcare system based on real MIMIC-III data.
Ensures reward mechanisms reflect real medical environment goals and constraints.
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

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import config

@dataclass
class RewardComponents:
    """Reward components data class"""
    # Treatment effectiveness related
    treatment_success: float = 0.0
    treatment_efficiency: float = 0.0
    symptom_improvement: float = 0.0
    
    # Cost efficiency related
    cost_optimization: float = 0.0
    resource_utilization: float = 0.0
    length_of_stay_optimization: float = 0.0
    
    # Patient safety related
    mortality_risk_reduction: float = 0.0
    complication_prevention: float = 0.0
    treatment_appropriateness: float = 0.0
    
    # Collaboration efficiency related
    communication_efficiency: float = 0.0
    decision_speed: float = 0.0
    information_sharing: float = 0.0
    
    # Insurance related
    insurance_optimization: float = 0.0
    approval_efficiency: float = 0.0
    
    # Penalty items
    delay_penalty: float = 0.0
    error_penalty: float = 0.0
    
    def total_reward(self, weights: Dict[str, float]) -> float:
        """Calculate total reward"""
        components = {
            # Positive rewards
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
            
            # Negative penalties
            'delay_penalty': -self.delay_penalty,
            'error_penalty': -self.error_penalty
        }
        
        total = sum(components[key] * weights.get(key, 0.0) for key in components)
        return total

class RealDataRewardOptimizer:
    """Real data-based reward function optimizer"""
    
    def __init__(self, processed_data_path="data/processed/"):
        self.data_path = Path(processed_data_path)
        
        # Load preprocessed data
        self.patients_df = None
        self.episodes_df = None
        self.diagnoses_mapping = None
        self.drugs_mapping = None
        self.cost_mapping = None
        
        # Benchmark statistics
        self.benchmarks = {}
        
        # Optimized reward weights
        self.optimized_weights = {}
        
        print("🎯 Initializing real data-based reward function optimizer")
        self.load_data()
        self.calculate_benchmarks()
        self.optimize_reward_weights()
    
    def load_data(self):
        """Load preprocessed data"""
        print("\n📥 Loading preprocessed data...")
        
        try:
            self.patients_df = pd.read_csv(self.data_path / "patients.csv")
            self.episodes_df = pd.read_csv(self.data_path / "episodes.csv")
            self.diagnoses_mapping = pd.read_csv(self.data_path / "diagnoses_mapping.csv")
            self.drugs_mapping = pd.read_csv(self.data_path / "drugs_mapping.csv")
            self.cost_mapping = pd.read_csv(self.data_path / "cost_mapping.csv")
            
            print(f"   ✅ Data loading completed")
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            raise
    
    def calculate_benchmarks(self):
        """Calculate benchmark statistics"""
        print("\n📊 Calculating benchmark statistics...")
        
        # Treatment success rate benchmark
        self.benchmarks['avg_treatment_effectiveness'] = self.drugs_mapping['effectiveness_score'].mean()
        self.benchmarks['target_treatment_effectiveness'] = 0.85  # Target 85% effectiveness
        
        # Cost benchmark
        self.benchmarks['avg_treatment_cost'] = self.cost_mapping['estimated_cost'].mean()
        self.benchmarks['target_cost_reduction'] = 0.15  # Target 15% cost reduction
        
        # Length of stay benchmark
        self.benchmarks['avg_length_of_stay'] = self.episodes_df['length_of_stay'].mean()
        self.benchmarks['target_los_reduction'] = 0.20  # Target 20% LOS reduction
        
        # Mortality rate benchmark
        self.benchmarks['avg_mortality_rate'] = self.episodes_df['hospital_expire_flag'].mean()
        self.benchmarks['target_mortality_reduction'] = 0.25  # Target 25% mortality reduction
        
        # Severity benchmark
        self.benchmarks['avg_severity'] = self.diagnoses_mapping['severity_score'].mean()
        self.benchmarks['avg_complexity'] = self.diagnoses_mapping['treatment_complexity'].mean()
        
        # Insurance coverage benchmark
        self.benchmarks['avg_insurance_coverage'] = self.cost_mapping['insurance_coverage'].mean()
        self.benchmarks['target_insurance_optimization'] = 0.90  # Target 90% insurance coverage
        
        print(f"   ✅ Benchmark calculation completed")
        for key, value in self.benchmarks.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.3f}")
    
    def optimize_reward_weights(self):
        """Optimize reward weights"""
        print("\n⚖️ Optimizing reward weights...")
        
        # Weight optimization based on real data analysis
        total_weight = 1.0
        
        # Main objective weights (based on medical quality priorities)
        self.optimized_weights = {
            # Treatment effectiveness (40% - most important)
            'treatment_success': 0.20,
            'treatment_efficiency': 0.10,
            'symptom_improvement': 0.10,
            
            # Patient safety (25% - second most important)
            'mortality_risk_reduction': 0.15,
            'complication_prevention': 0.05,
            'treatment_appropriateness': 0.05,
            
            # Cost efficiency (20% - third most important)
            'cost_optimization': 0.10,
            'resource_utilization': 0.05,
            'length_of_stay_optimization': 0.05,
            
            # Collaboration efficiency (10% - supporting objective)
            'communication_efficiency': 0.03,
            'decision_speed': 0.04,
            'information_sharing': 0.03,
            
            # Insurance optimization (5% - operational efficiency)
            'insurance_optimization': 0.03,
            'approval_efficiency': 0.02,
            
            # Penalty weights
            'delay_penalty': 0.1,
            'error_penalty': 0.2
        }
        
        # Adjust weights based on data characteristics
        # If mortality rate is high, increase safety-related weights
        if self.benchmarks['avg_mortality_rate'] > 0.25:
            self.optimized_weights['mortality_risk_reduction'] += 0.05
            self.optimized_weights['treatment_success'] -= 0.03
            self.optimized_weights['cost_optimization'] -= 0.02
        
        # If cost is high, increase cost optimization weights
        if self.benchmarks['avg_treatment_cost'] > 15000:
            self.optimized_weights['cost_optimization'] += 0.03
            self.optimized_weights['length_of_stay_optimization'] += 0.02
            self.optimized_weights['treatment_efficiency'] -= 0.05
        
        # If length of stay is long, increase efficiency weights
        if self.benchmarks['avg_length_of_stay'] > 10:
            self.optimized_weights['length_of_stay_optimization'] += 0.03
            self.optimized_weights['decision_speed'] += 0.02
            self.optimized_weights['symptom_improvement'] -= 0.05
        
        print(f"   ✅ Weight optimization completed")
        print(f"   📋 Optimized weight distribution:")
        for component, weight in self.optimized_weights.items():
            if weight > 0.01:  # Only show significant weights
                print(f"     {component}: {weight:.3f}")
    
    def calculate_treatment_success_reward(self, 
                                         treatment_effectiveness: float, 
                                         patient_severity: float,
                                         treatment_appropriateness: float = 1.0) -> float:
        """Calculate treatment success reward"""
        # Base success reward
        base_reward = treatment_effectiveness
        
        # Severity adjustment (Higher reward for treating severe diseases)
        severity_bonus = patient_severity / self.benchmarks['avg_severity'] * 0.2
        
        # Treatment appropriateness adjustment
        appropriateness_factor = treatment_appropriateness
        
        # Compare with benchmark
        benchmark_factor = treatment_effectiveness / self.benchmarks['avg_treatment_effectiveness']
        
        final_reward = (base_reward + severity_bonus) * appropriateness_factor * benchmark_factor
        return max(0, min(final_reward, 2.0))  # Limit between 0-2
    
    def calculate_cost_optimization_reward(self, 
                                         actual_cost: float, 
                                         baseline_cost: float,
                                         insurance_coverage: float) -> float:
        """Calculate cost optimization reward"""
        # Cost savings ratio
        if baseline_cost > 0:
            cost_savings_ratio = (baseline_cost - actual_cost) / baseline_cost
        else:
            cost_savings_ratio = 0
        
        # Base cost reward
        base_reward = cost_savings_ratio
        
        # Insurance coverage adjustment
        insurance_factor = insurance_coverage / self.benchmarks['avg_insurance_coverage']
        
        # Compare with target
        target_factor = cost_savings_ratio / self.benchmarks['target_cost_reduction'] if self.benchmarks['target_cost_reduction'] > 0 else 1
        
        final_reward = base_reward * insurance_factor * target_factor
        return max(-1.0, min(final_reward, 1.0))  # Limit between -1 and 1
    
    def calculate_safety_reward(self, 
                               mortality_risk_reduction: float,
                               complication_risk: float = 0.0,
                               treatment_safety_score: float = 1.0) -> float:
        """Calculate patient safety reward"""
        # Mortality risk reduction reward
        mortality_reward = mortality_risk_reduction * 2.0  # High weight
        
        # Complication risk penalty
        complication_penalty = complication_risk * 0.5
        
        # Treatment safety reward
        safety_reward = (treatment_safety_score - 0.5) * 0.5
        
        # Compare with benchmark
        benchmark_factor = mortality_risk_reduction / (self.benchmarks['avg_mortality_rate'] + 0.01)
        
        final_reward = (mortality_reward + safety_reward - complication_penalty) * benchmark_factor
        return max(0, min(final_reward, 2.0))
    
    def calculate_efficiency_reward(self, 
                                  decision_time: float,
                                  communication_quality: float,
                                  resource_utilization: float) -> float:
        """Calculate efficiency reward"""
        # Decision speed reward (Lower is better)
        max_decision_time = 10.0  # Assume maximum 10 minutes
        decision_reward = max(0, (max_decision_time - decision_time) / max_decision_time)
        
        # Communication quality reward
        communication_reward = communication_quality
        
        # Resource utilization reward
        utilization_reward = resource_utilization
        
        # Overall efficiency score
        efficiency_score = (decision_reward + communication_reward + utilization_reward) / 3
        
        return max(0, min(efficiency_score, 1.0))
    
    def calculate_length_of_stay_reward(self, 
                                      predicted_los: float, 
                                      actual_los: float,
                                      patient_severity: float) -> float:
        """Calculate length of stay optimization reward"""
        # Based on severity, expected length of stay
        severity_factor = patient_severity / self.benchmarks['avg_severity']
        expected_los = self.benchmarks['avg_length_of_stay'] * severity_factor
        
        # Compare actual performance with expected
        if expected_los > 0:
            los_ratio = (expected_los - actual_los) / expected_los
        else:
            los_ratio = 0
        
        # Prediction accuracy reward
        prediction_accuracy = 1.0 - abs(predicted_los - actual_los) / max(predicted_los, actual_los, 1.0)
        
        # Overall reward
        final_reward = los_ratio * 0.7 + prediction_accuracy * 0.3
        
        return max(-0.5, min(final_reward, 1.0))
    
    def calculate_insurance_optimization_reward(self, 
                                              insurance_coverage: float,
                                              approval_time: float,
                                              claim_accuracy: float) -> float:
        """Calculate insurance optimization reward"""
        # Coverage reward
        coverage_reward = insurance_coverage / self.benchmarks['target_insurance_optimization']
        
        # Approval speed reward
        max_approval_time = 24.0  # 24 hours
        approval_reward = max(0, (max_approval_time - approval_time) / max_approval_time)
        
        # Accuracy reward
        accuracy_reward = claim_accuracy
        
        # Overall insurance optimization score
        insurance_score = (coverage_reward + approval_reward + accuracy_reward) / 3
        
        return max(0, min(insurance_score, 1.0))
    
    def calculate_comprehensive_reward(self, 
                                     patient_data: Dict,
                                     treatment_data: Dict,
                                     outcome_data: Dict,
                                     process_data: Dict) -> RewardComponents:
        """Calculate comprehensive reward"""
        components = RewardComponents()
        
        # Treatment effectiveness related
        components.treatment_success = self.calculate_treatment_success_reward(
            treatment_data.get('effectiveness', 0.7),
            patient_data.get('severity', 1.0),
            treatment_data.get('appropriateness', 1.0)
        )
        
        components.treatment_efficiency = treatment_data.get('efficiency', 0.5)
        components.symptom_improvement = outcome_data.get('symptom_improvement', 0.5)
        
        # Cost efficiency related
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
        
        # Patient safety related
        components.mortality_risk_reduction = self.calculate_safety_reward(
            outcome_data.get('mortality_risk_reduction', 0.1),
            outcome_data.get('complication_risk', 0.0),
            treatment_data.get('safety_score', 1.0)
        )
        
        # Collaboration efficiency related
        efficiency = self.calculate_efficiency_reward(
            process_data.get('decision_time', 5),
            process_data.get('communication_quality', 0.8),
            process_data.get('resource_utilization', 0.7)
        )
        
        components.communication_efficiency = efficiency * 0.4
        components.decision_speed = efficiency * 0.6
        
        # Insurance related
        insurance_reward = self.calculate_insurance_optimization_reward(
            patient_data.get('insurance_coverage', 0.8),
            process_data.get('approval_time', 12),
            process_data.get('claim_accuracy', 0.95)
        )
        
        components.insurance_optimization = insurance_reward * 0.7
        components.approval_efficiency = insurance_reward * 0.3
        
        # Penalty items
        components.delay_penalty = process_data.get('delays', 0) * 0.1
        components.error_penalty = process_data.get('errors', 0) * 0.2
        
        return components
    
    def get_optimized_weights(self) -> Dict[str, float]:
        """Get optimized weights"""
        return self.optimized_weights.copy()
    
    def save_reward_optimization(self, output_file: str = "reward_optimization.json"):
        """Save reward optimization results"""
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
        
        print(f"💾 Reward optimization configuration saved to: {output_path}")

def main():
    """Main function"""
    print("🚀 Starting real data-based reward function optimization")
    print("=" * 60)
    
    # Create reward optimizer
    optimizer = RealDataRewardOptimizer()
    
    # Save optimization results
    optimizer.save_reward_optimization()
    
    # Demonstrate reward calculation
    print(f"\n🎯 Reward function demonstration:")
    
    # Simulate patient and treatment data
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
    
    # Calculate comprehensive reward
    reward_components = optimizer.calculate_comprehensive_reward(
        sample_patient, sample_treatment, sample_outcome, sample_process
    )
    
    total_reward = reward_components.total_reward(optimizer.get_optimized_weights())
    
    print(f"    Patient severity: {sample_patient['severity']:.2f}")
    print(f"    Treatment effectiveness: {sample_treatment['effectiveness']:.2%}")
    print(f"    Cost savings: ${sample_outcome['baseline_cost'] - sample_outcome['actual_cost']}")
    print(f"    Length of stay optimization: {sample_outcome['predicted_los'] - sample_outcome['actual_los']} days")
    print(f"   \n🏆 Comprehensive reward score: {total_reward:.3f}")
    
    print(f"\n📊 Main reward components:")
    print(f"    Treatment success: {reward_components.treatment_success:.3f}")
    print(f"    Cost optimization: {reward_components.cost_optimization:.3f}")
    print(f"    Safety improvement: {reward_components.mortality_risk_reduction:.3f}")
    print(f"    Efficiency improvement: {reward_components.communication_efficiency + reward_components.decision_speed:.3f}")
    
    print("\n🎉 Reward function optimization completed!")
    print("\n🔄 Next steps suggestion:")
    print("   1. Integrate optimized reward function into multi-agent environment")
    print("   2. Retrain intelligent agents with real patient model")
    print("   3. Verify system performance after improvements")

if __name__ == "__main__":
    main() 