#!/usr/bin/env python3
"""
Real Data-Based Patient Model

Uses preprocessed MIMIC-III data to build realistic patient state models
for multi-agent healthcare collaboration environment
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
import random
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import config

@dataclass
class PatientCondition:
    """Patient condition data class"""
    # Basic information
    subject_id: int
    age: int
    gender: int  # 0: Female, 1: Male
    
    # Current diagnosis
    primary_diagnosis: str
    diagnosis_codes: List[str]
    
    # Disease severity
    severity_score: float
    mortality_risk: float
    treatment_complexity: float
    
    # Symptom presentation
    symptoms: Dict[str, float]
    
    # Treatment history
    current_treatments: List[str]
    treatment_effectiveness: float
    
    # Cost information
    estimated_treatment_cost: float
    insurance_coverage: float
    
    # Clinical indicators
    length_of_stay_prediction: float
    urgency_level: int  # 1-5
    
    # Metadata
    episode_id: Optional[int] = None
    admission_type: Optional[str] = None

class RealPatientModelGenerator:
    """Real data-based patient model generator"""
    
    def __init__(self, processed_data_path="data/processed/"):
        self.data_path = Path(processed_data_path)
        
        # Load preprocessed data
        self.patients_df = None
        self.episodes_df = None
        self.diagnoses_mapping = None
        self.drugs_mapping = None
        self.procedures_mapping = None
        self.cost_mapping = None
        
        # Probability distributions and statistical information
        self.patient_age_dist = None
        self.diagnosis_probabilities = None
        self.severity_statistics = None
        
        print("🏥 Initializing Real Patient Model Generator")
        self.load_processed_data()
        self.build_statistical_models()
    
    def load_processed_data(self):
        """Load preprocessed data"""
        print("\n📥 Loading preprocessed data...")
        
        try:
            self.patients_df = pd.read_csv(self.data_path / "patients.csv")
            self.episodes_df = pd.read_csv(self.data_path / "episodes.csv")
            self.diagnoses_mapping = pd.read_csv(self.data_path / "diagnoses_mapping.csv")
            self.drugs_mapping = pd.read_csv(self.data_path / "drugs_mapping.csv")
            self.procedures_mapping = pd.read_csv(self.data_path / "procedures_mapping.csv")
            self.cost_mapping = pd.read_csv(self.data_path / "cost_mapping.csv")
            
            print(f"   ✅ Patient data: {len(self.patients_df)} patients")
            print(f"   ✅ Medical episodes: {len(self.episodes_df)} episodes")
            print(f"   ✅ Diagnosis mapping: {len(self.diagnoses_mapping)} diagnoses")
            print(f"   ✅ Drug mapping: {len(self.drugs_mapping)} drugs")
            print(f"   ✅ Procedure mapping: {len(self.procedures_mapping)} procedures")
            print(f"   ✅ Cost mapping: {len(self.cost_mapping)} DRGs")
            
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            raise
    
    def build_statistical_models(self):
        """Build statistical models"""
        print("\n📊 Building statistical models...")
        
        # Age distribution
        self.patient_age_dist = {
            'mean': self.patients_df['age'].mean(),
            'std': self.patients_df['age'].std(),
            'min': self.patients_df['age'].min(),
            'max': self.patients_df['age'].max()
        }
        
        # Diagnosis probability distribution
        diagnosis_freq = self.diagnoses_mapping['frequency'].values
        total_freq = diagnosis_freq.sum()
        self.diagnosis_probabilities = diagnosis_freq / total_freq
        
        # Severity statistics
        self.severity_statistics = {
            'mean': self.diagnoses_mapping['severity_score'].mean(),
            'std': self.diagnoses_mapping['severity_score'].std(),
            'min': self.diagnoses_mapping['severity_score'].min(),
            'max': self.diagnoses_mapping['severity_score'].max()
        }
        
        print(f"   ✅ Age distribution: {self.patient_age_dist['mean']:.1f} ± {self.patient_age_dist['std']:.1f}")
        print(f"   ✅ Diagnosis probability distribution: {len(self.diagnosis_probabilities)} diagnoses")
        print(f"   ✅ Severity distribution: {self.severity_statistics['mean']:.2f} ± {self.severity_statistics['std']:.2f}")
    
    def generate_realistic_patient(self, seed: Optional[int] = None) -> PatientCondition:
        """Generate patient based on real data"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 1. Basic demographic characteristics
        age = self._generate_realistic_age()
        gender = np.random.choice([0, 1], p=[0.55, 0.45])  # Based on real gender distribution
        
        # 2. Select realistic diagnosis
        primary_diagnosis = self._select_realistic_diagnosis()
        diagnosis_info = self.diagnoses_mapping[
            self.diagnoses_mapping['icd9_code'] == primary_diagnosis
        ].iloc[0]
        
        # 3. Determine disease severity based on diagnosis
        severity_score = diagnosis_info['severity_score']
        mortality_risk = diagnosis_info['mortality_risk']
        treatment_complexity = diagnosis_info['treatment_complexity']
        
        # 4. Generate symptom presentation
        symptoms = self._generate_symptoms_from_diagnosis(primary_diagnosis, severity_score, age)
        
        # 5. Select treatment plan
        current_treatments = self._select_realistic_treatments(primary_diagnosis, severity_score)
        
        # 6. Calculate treatment effectiveness
        treatment_effectiveness = self._calculate_treatment_effectiveness(
            current_treatments, severity_score, age
        )
        
        # 7. Estimate costs
        estimated_cost, insurance_coverage = self._estimate_realistic_costs(
            primary_diagnosis, treatment_complexity, current_treatments
        )
        
        # 8. Predict length of stay and urgency level
        los_prediction = self._predict_length_of_stay(severity_score, treatment_complexity, age)
        urgency_level = self._determine_urgency_level(severity_score, mortality_risk)
        
        # Generate unique patient ID
        subject_id = np.random.randint(10000, 99999)
        
        patient_condition = PatientCondition(
            subject_id=subject_id,
            age=age,
            gender=gender,
            primary_diagnosis=primary_diagnosis,
            diagnosis_codes=[primary_diagnosis],  # Simplified to primary diagnosis
            severity_score=severity_score,
            mortality_risk=mortality_risk,
            treatment_complexity=treatment_complexity,
            symptoms=symptoms,
            current_treatments=current_treatments,
            treatment_effectiveness=treatment_effectiveness,
            estimated_treatment_cost=estimated_cost,
            insurance_coverage=insurance_coverage,
            length_of_stay_prediction=los_prediction,
            urgency_level=urgency_level
        )
        
        return patient_condition
    
    def _generate_realistic_age(self) -> int:
        """Generate realistic age distribution"""
        # Use normal distribution to generate age, limited to reasonable range
        age = np.random.normal(
            self.patient_age_dist['mean'], 
            self.patient_age_dist['std']
        )
        
        # Limit age range
        age = max(18, min(int(age), 100))
        return age
    
    def _select_realistic_diagnosis(self) -> str:
        """Select diagnosis based on real frequency"""
        # Use frequency-weighted random selection
        diagnosis_idx = np.random.choice(
            len(self.diagnoses_mapping),
            p=self.diagnosis_probabilities
        )
        
        return self.diagnoses_mapping.iloc[diagnosis_idx]['icd9_code']
    
    def _generate_symptoms_from_diagnosis(self, diagnosis: str, severity: float, age: int) -> Dict[str, float]:
        """Generate symptom presentation based on diagnosis"""
        # Get diagnosis description
        diagnosis_info = self.diagnoses_mapping[
            self.diagnoses_mapping['icd9_code'] == diagnosis
        ]
        
        if len(diagnosis_info) == 0:
            diagnosis_title = "Unknown"
        else:
            diagnosis_title = diagnosis_info.iloc[0]['short_title'].lower()
        
        # Base symptom intensity
        base_intensity = min(severity / 5.0, 1.0)
        
        # Age adjustment (elderly may have more pronounced symptoms)
        age_factor = 1.0 + (age - 65) * 0.01 if age > 65 else 1.0
        
        symptoms = {}
        
        # Generate specific symptoms based on diagnosis type
        if any(keyword in diagnosis_title for keyword in ['hypertension', 'blood pressure']):
            symptoms = {
                'headache': base_intensity * 0.6 * age_factor,
                'dizziness': base_intensity * 0.5 * age_factor,
                'chest_pain': base_intensity * 0.3,
                'fatigue': base_intensity * 0.4 * age_factor,
                'shortness_of_breath': base_intensity * 0.3
            }
        
        elif any(keyword in diagnosis_title for keyword in ['pneumonia', 'respiratory']):
            symptoms = {
                'cough': base_intensity * 0.9,
                'fever': base_intensity * 0.7,
                'shortness_of_breath': base_intensity * 0.8,
                'chest_pain': base_intensity * 0.6,
                'fatigue': base_intensity * 0.8 * age_factor
            }
        
        elif any(keyword in diagnosis_title for keyword in ['sepsis', 'infection']):
            symptoms = {
                'fever': base_intensity * 0.9,
                'chills': base_intensity * 0.7,
                'rapid_heart_rate': base_intensity * 0.8,
                'confusion': base_intensity * 0.5 * age_factor,
                'weakness': base_intensity * 0.9 * age_factor
            }
        
        elif any(keyword in diagnosis_title for keyword in ['kidney', 'renal']):
            symptoms = {
                'nausea': base_intensity * 0.6,
                'vomiting': base_intensity * 0.4,
                'swelling': base_intensity * 0.7,
                'fatigue': base_intensity * 0.8 * age_factor,
                'urination_changes': base_intensity * 0.9
            }
        
        elif any(keyword in diagnosis_title for keyword in ['diabetes']):
            symptoms = {
                'excessive_thirst': base_intensity * 0.7,
                'frequent_urination': base_intensity * 0.8,
                'blurred_vision': base_intensity * 0.5,
                'fatigue': base_intensity * 0.9 * age_factor,
                'slow_healing': base_intensity * 0.6
            }
        
        else:
            # General symptoms
            symptoms = {
                'pain': base_intensity * 0.6,
                'fatigue': base_intensity * 0.5 * age_factor,
                'discomfort': base_intensity * 0.4,
                'weakness': base_intensity * 0.3 * age_factor,
                'general_malaise': base_intensity * 0.4
            }
        
        # Add random variation
        for symptom in symptoms:
            noise = np.random.normal(0, 0.1)
            symptoms[symptom] = max(0, min(symptoms[symptom] + noise, 1.0))
        
        return symptoms
    
    def _select_realistic_treatments(self, diagnosis: str, severity: float) -> List[str]:
        """Select realistic treatment plan based on diagnosis"""
        treatments = []
        
        # Determine number of treatments based on severity
        num_treatments = max(1, int(severity / 2) + np.random.randint(0, 3))
        
        # Get diagnosis description for treatment selection
        diagnosis_info = self.diagnoses_mapping[
            self.diagnoses_mapping['icd9_code'] == diagnosis
        ]
        
        if len(diagnosis_info) == 0:
            diagnosis_title = "Unknown"
        else:
            diagnosis_title = diagnosis_info.iloc[0]['short_title'].lower()
        
        # Select relevant drugs based on diagnosis type
        relevant_drugs = []
        
        if any(keyword in diagnosis_title for keyword in ['hypertension']):
            relevant_drugs = self.drugs_mapping[
                self.drugs_mapping['drug_type'].isin(['diuretic', 'other'])
            ]['drug_name'].tolist()
        
        elif any(keyword in diagnosis_title for keyword in ['infection', 'pneumonia', 'sepsis']):
            relevant_drugs = self.drugs_mapping[
                self.drugs_mapping['drug_type'] == 'antibiotic'
            ]['drug_name'].tolist()
        
        elif any(keyword in diagnosis_title for keyword in ['diabetes']):
            relevant_drugs = self.drugs_mapping[
                self.drugs_mapping['drug_type'] == 'diabetes'
            ]['drug_name'].tolist()
        
        else:
            # Use most common drugs
            relevant_drugs = self.drugs_mapping.nlargest(20, 'frequency')['drug_name'].tolist()
        
        # If no relevant drugs found, use most common ones
        if not relevant_drugs:
            relevant_drugs = self.drugs_mapping.nlargest(10, 'frequency')['drug_name'].tolist()
        
        # Randomly select treatment plan
        selected_treatments = np.random.choice(
            relevant_drugs, 
            size=min(num_treatments, len(relevant_drugs)), 
            replace=False
        ).tolist()
        
        return selected_treatments
    
    def _calculate_treatment_effectiveness(self, treatments: List[str], severity: float, age: int) -> float:
        """Calculate treatment effectiveness"""
        if not treatments:
            return 0.0
        
        base_effectiveness = 0.7
        
        # Get drug effectiveness scores
        total_effectiveness = 0
        for treatment in treatments:
            drug_info = self.drugs_mapping[
                self.drugs_mapping['drug_name'] == treatment
            ]
            
            if len(drug_info) > 0:
                drug_effectiveness = drug_info.iloc[0]['effectiveness_score']
                total_effectiveness += drug_effectiveness
            else:
                total_effectiveness += base_effectiveness
        
        avg_effectiveness = total_effectiveness / len(treatments)
        
        # Severity adjustment (more severe diseases may have worse treatment outcomes)
        severity_penalty = severity * 0.1
        avg_effectiveness -= severity_penalty
        
        # Age adjustment (older patients may have worse treatment outcomes)
        age_penalty = max(0, (age - 65) * 0.005)
        avg_effectiveness -= age_penalty
        
        # Add random variation
        noise = np.random.normal(0, 0.1)
        final_effectiveness = avg_effectiveness + noise
        
        return max(0.1, min(final_effectiveness, 1.0))
    
    def _estimate_realistic_costs(self, diagnosis: str, complexity: float, treatments: List[str]) -> Tuple[float, float]:
        """Estimate realistic costs and insurance coverage"""
        base_cost = 5000.0
        
        # Adjust cost based on treatment complexity
        complexity_multiplier = 1.0 + complexity * 0.5
        base_cost *= complexity_multiplier
        
        # Adjust based on drug costs
        drug_costs = 0
        for treatment in treatments:
            drug_info = self.drugs_mapping[
                self.drugs_mapping['drug_name'] == treatment
            ]
            
            if len(drug_info) > 0:
                drug_costs += drug_info.iloc[0]['estimated_cost']
            else:
                drug_costs += 50  # Default drug cost
        
        total_cost = base_cost + drug_costs
        
        # Get real insurance coverage from DRG data
        avg_coverage = self.cost_mapping['insurance_coverage'].mean()
        coverage_noise = np.random.normal(0, 0.1)
        insurance_coverage = max(0.5, min(avg_coverage + coverage_noise, 0.95))
        
        return total_cost, insurance_coverage
    
    def _predict_length_of_stay(self, severity: float, complexity: float, age: int) -> float:
        """Predict length of stay"""
        # Average length of stay based on real data
        base_los = self.episodes_df['length_of_stay'].mean()
        
        # Severity and complexity adjustments
        severity_factor = 1.0 + severity * 0.3
        complexity_factor = 1.0 + complexity * 0.2
        
        # Age adjustment
        age_factor = 1.0 + max(0, (age - 65) * 0.02)
        
        predicted_los = base_los * severity_factor * complexity_factor * age_factor
        
        # Add random variation
        noise = np.random.normal(0, 2)
        predicted_los += noise
        
        return max(1, predicted_los)
    
    def _determine_urgency_level(self, severity: float, mortality_risk: float) -> int:
        """Determine urgency level"""
        urgency_score = severity * 0.6 + mortality_risk * 0.4
        
        if urgency_score >= 4.0:
            return 5  # Critical
        elif urgency_score >= 3.0:
            return 4  # Urgent
        elif urgency_score >= 2.0:
            return 3  # Moderate
        elif urgency_score >= 1.0:
            return 2  # Low
        else:
            return 1  # Lowest
    
    def generate_patient_batch(self, batch_size: int = 10) -> List[PatientCondition]:
        """Generate batch of patients"""
        print(f"\n👥 Generating {batch_size} realistic patient models...")
        
        patients = []
        for i in range(batch_size):
            patient = self.generate_realistic_patient(seed=i)
            patients.append(patient)
        
        print(f"   ✅ Successfully generated {len(patients)} patients")
        
        # Statistics
        ages = [p.age for p in patients]
        severities = [p.severity_score for p in patients]
        costs = [p.estimated_treatment_cost for p in patients]
        
        print(f"   📊 Statistics:")
        print(f"     - Age range: {min(ages)}-{max(ages)}, average: {np.mean(ages):.1f}")
        print(f"     - Severity: {min(severities):.2f}-{max(severities):.2f}, average: {np.mean(severities):.2f}")
        print(f"     - Estimated cost: ${min(costs):.0f}-${max(costs):.0f}, average: ${np.mean(costs):.0f}")
        
        return patients
    
    def save_patient_models(self, patients: List[PatientCondition], output_file: str = "patient_models.json"):
        """Save patient models to file"""
        output_path = self.data_path / output_file
        
        patients_data = []
        for patient in patients:
            patient_dict = {
                'subject_id': int(patient.subject_id),
                'age': int(patient.age),
                'gender': int(patient.gender),
                'primary_diagnosis': str(patient.primary_diagnosis),
                'diagnosis_codes': [str(code) for code in patient.diagnosis_codes],
                'severity_score': float(patient.severity_score),
                'mortality_risk': float(patient.mortality_risk),
                'treatment_complexity': float(patient.treatment_complexity),
                'symptoms': {k: float(v) for k, v in patient.symptoms.items()},
                'current_treatments': [str(t) for t in patient.current_treatments],
                'treatment_effectiveness': float(patient.treatment_effectiveness),
                'estimated_treatment_cost': float(patient.estimated_treatment_cost),
                'insurance_coverage': float(patient.insurance_coverage),
                'length_of_stay_prediction': float(patient.length_of_stay_prediction),
                'urgency_level': int(patient.urgency_level)
            }
            patients_data.append(patient_dict)
        
        with open(output_path, 'w') as f:
            json.dump(patients_data, f, indent=2)
        
        print(f"💾 Patient models saved to: {output_path}")

def main():
    """Main function"""
    print("🚀 Starting Real Patient Model Generation")
    print("=" * 60)
    
    # Create patient model generator
    generator = RealPatientModelGenerator()
    
    # Generate patient batch
    patients = generator.generate_patient_batch(batch_size=20)
    
    # Save patient models
    generator.save_patient_models(patients)
    
    # Display example patient
    print(f"\n👤 Example Patient Information:")
    example_patient = patients[0]
    print(f"   ID: {example_patient.subject_id}")
    print(f"   Age/Gender: {example_patient.age} years old, {'Male' if example_patient.gender else 'Female'}")
    print(f"   Primary diagnosis: {example_patient.primary_diagnosis}")
    print(f"   Severity: {example_patient.severity_score:.2f}")
    print(f"   Mortality risk: {example_patient.mortality_risk:.2%}")
    print(f"   Current treatments: {example_patient.current_treatments[:3]}")  # Show first 3 treatments
    print(f"   Treatment effectiveness: {example_patient.treatment_effectiveness:.2%}")
    print(f"   Estimated cost: ${example_patient.estimated_treatment_cost:.0f}")
    print(f"   Insurance coverage: {example_patient.insurance_coverage:.2%}")
    print(f"   Urgency level: {example_patient.urgency_level}/5")
    
    print("\n🎉 Real patient model generation completed!")
    print("\n🔄 Next Steps:")
    print("   1. Update multi-agent environment to use real patient models")
    print("   2. Optimize reward function based on real data")
    print("   3. Retrain agent models")

if __name__ == "__main__":
    main() 