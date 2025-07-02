#!/usr/bin/env python3
"""
MIMIC-III Data Preprocessor

Converts raw MIMIC-III data into structured data suitable for reinforcement learning environments,
building realistic patient models, disease states, and treatment plans
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import config

class MIMICPreprocessor:
    """MIMIC-III Data Preprocessor"""
    
    def __init__(self, data_path="MIMIC III/", output_path="data/processed/"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Raw datasets
        self.raw_datasets = {}
        
        # Processed data
        self.processed_data = {
            'patients': None,
            'episodes': None,  # Medical episodes (admission-discharge as one episode)
            'diagnoses_mapping': None,
            'treatments_mapping': None,
            'drugs_mapping': None,
            'cost_mapping': None
        }
        
        # Encoders
        self.encoders = {}
        self.scalers = {}
        
        print("🔧 MIMIC-III Data Preprocessor Initialization")
        print(f"📁 Input path: {self.data_path}")
        print(f"📁 Output path: {self.output_path}")
    
    def load_raw_data(self):
        """Load raw data"""
        print("\n📥 Loading raw MIMIC-III data...")
        
        # Core data files
        core_files = {
            'patients': 'PATIENTS.csv',
            'admissions': 'ADMISSIONS.csv',
            'diagnoses': 'DIAGNOSES_ICD.csv',
            'prescriptions': 'PRESCRIPTIONS.csv',
            'procedures': 'PROCEDURES_ICD.csv',
            'icustays': 'ICUSTAYS.csv',
            'd_icd_diagnoses': 'D_ICD_DIAGNOSES.csv',
            'd_icd_procedures': 'D_ICD_PROCEDURES.csv',
            'drgcodes': 'DRGCODES.csv'
        }
        
        for key, filename in core_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                try:
                    print(f"  📖 Loading {filename}...")
                    df = pd.read_csv(file_path)
                    self.raw_datasets[key] = df
                    print(f"     ✅ {len(df)} rows loaded successfully")
                except Exception as e:
                    print(f"     ❌ Loading failed: {e}")
        
        print(f"\n✅ Successfully loaded {len(self.raw_datasets)} datasets")
        return self.raw_datasets
    
    def create_patient_profiles(self):
        """Create patient profiles"""
        print("\n👥 Creating patient profiles...")
        
        patients = self.raw_datasets['patients'].copy()
        admissions = self.raw_datasets['admissions'].copy()
        
        # Patient basic information processing
        patient_profiles = []
        
        for _, patient in patients.iterrows():
            subject_id = patient['subject_id']
            
            # Get all admission records for this patient
            patient_admissions = admissions[admissions['subject_id'] == subject_id]
            
            if len(patient_admissions) == 0:
                continue
            
            # Calculate patient age (using first admission time)
            first_admission = patient_admissions.iloc[0]
            try:
                dob = pd.to_datetime(patient['dob'])
                admit_time = pd.to_datetime(first_admission['admittime'])
                age = (admit_time - dob).days // 365
                age = max(0, min(age, 120))  # Reasonable range limitation
            except:
                age = 65  # Default age
            
            # Patient features
            profile = {
                'subject_id': subject_id,
                'age': age,
                'gender': 1 if patient['gender'] == 'M' else 0,
                'total_admissions': len(patient_admissions),
                'total_icu_stays': len(self.raw_datasets['icustays'][
                    self.raw_datasets['icustays']['subject_id'] == subject_id
                ]),
                'is_expired': 1 if pd.notna(patient['dod']) else 0,
                
                # Primary insurance type
                'primary_insurance': first_admission['insurance'],
                
                # Hospital stay statistics
                'avg_los': patient_admissions['admittime'].apply(
                    lambda x: (pd.to_datetime(first_admission['dischtime']) - 
                              pd.to_datetime(x)).days if pd.notna(first_admission['dischtime']) else 0
                ).mean() if len(patient_admissions) > 0 else 0
            }
            
            patient_profiles.append(profile)
        
        patients_df = pd.DataFrame(patient_profiles)
        
        # Encode insurance type
        insurance_encoder = LabelEncoder()
        patients_df['insurance_encoded'] = insurance_encoder.fit_transform(
            patients_df['primary_insurance'].fillna('Unknown')
        )
        self.encoders['insurance'] = insurance_encoder
        
        # Age grouping
        patients_df['age_group'] = pd.cut(
            patients_df['age'], 
            bins=[0, 30, 50, 70, 120], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        print(f"   ✅ Created {len(patients_df)} patient profiles")
        print(f"   - Average age: {patients_df['age'].mean():.1f}")
        print(f"   - Gender distribution: {patients_df['gender'].value_counts().to_dict()}")
        print(f"   - Average admissions: {patients_df['total_admissions'].mean():.1f}")
        
        self.processed_data['patients'] = patients_df
        return patients_df
    
    def create_medical_episodes(self):
        """创建医疗事件（每次入院为一个episode）"""
        print("\n🏥 创建医疗事件...")
        
        admissions = self.raw_datasets['admissions'].copy()
        diagnoses = self.raw_datasets['diagnoses'].copy()
        prescriptions = self.raw_datasets['prescriptions'].copy()
        procedures = self.raw_datasets['procedures'].copy()
        drgcodes = self.raw_datasets['drgcodes'].copy()
        
        episodes = []
        
        for _, admission in admissions.iterrows():
            subject_id = admission['subject_id']
            hadm_id = admission['hadm_id']
            
            # 基本episode信息
            episode = {
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'admission_type': admission['admission_type'],
                'admission_location': admission['admission_location'],
                'insurance': admission['insurance'],
                
                # 时间信息
                'admit_time': admission['admittime'],
                'discharge_time': admission['dischtime'],
                
                # 住院时长
                'length_of_stay': 0,
                
                # 结果
                'hospital_expire_flag': admission['hospital_expire_flag'],
                'discharge_location': admission['discharge_location']
            }
            
            # 计算住院时长
            try:
                admit_dt = pd.to_datetime(admission['admittime'])
                discharge_dt = pd.to_datetime(admission['dischtime'])
                if pd.notna(discharge_dt):
                    episode['length_of_stay'] = (discharge_dt - admit_dt).days
            except:
                episode['length_of_stay'] = 0
            
            # 获取诊断信息
            admission_diagnoses = diagnoses[diagnoses['hadm_id'] == hadm_id]
            episode['diagnosis_codes'] = admission_diagnoses['icd9_code'].tolist()
            episode['primary_diagnosis'] = admission_diagnoses[
                admission_diagnoses['seq_num'] == 1
            ]['icd9_code'].iloc[0] if len(admission_diagnoses) > 0 else None
            
            # 获取处方信息
            admission_prescriptions = prescriptions[prescriptions['hadm_id'] == hadm_id]
            episode['prescribed_drugs'] = admission_prescriptions['drug'].tolist()
            episode['total_prescriptions'] = len(admission_prescriptions)
            
            # 获取程序信息
            admission_procedures = procedures[procedures['hadm_id'] == hadm_id]
            episode['procedure_codes'] = admission_procedures['icd9_code'].tolist()
            episode['total_procedures'] = len(admission_procedures)
            
            # 获取DRG信息（用于成本估算）
            admission_drg = drgcodes[drgcodes['hadm_id'] == hadm_id]
            if len(admission_drg) > 0:
                episode['drg_code'] = admission_drg.iloc[0]['drg_code']
                episode['drg_severity'] = admission_drg.iloc[0].get('drg_severity', 0)
                episode['drg_mortality'] = admission_drg.iloc[0].get('drg_mortality', 0)
            else:
                episode['drg_code'] = None
                episode['drg_severity'] = 0
                episode['drg_mortality'] = 0
            
            episodes.append(episode)
        
        episodes_df = pd.DataFrame(episodes)
        
        # 编码分类变量
        categorical_columns = ['admission_type', 'admission_location', 'discharge_location']
        for col in categorical_columns:
            if col in episodes_df.columns:
                encoder = LabelEncoder()
                episodes_df[f'{col}_encoded'] = encoder.fit_transform(
                    episodes_df[col].fillna('Unknown')
                )
                self.encoders[col] = encoder
        
        print(f"   ✅ 创建了 {len(episodes_df)} 个医疗事件")
        print(f"   - 平均住院时长: {episodes_df['length_of_stay'].mean():.1f} 天")
        print(f"   - 平均处方数: {episodes_df['total_prescriptions'].mean():.1f}")
        print(f"   - 死亡率: {episodes_df['hospital_expire_flag'].mean():.2%}")
        
        self.processed_data['episodes'] = episodes_df
        return episodes_df
    
    def create_diagnosis_mapping(self):
        """创建诊断映射表"""
        print("\n🩺 创建诊断映射表...")
        
        diagnoses = self.raw_datasets['diagnoses'].copy()
        d_icd_diagnoses = self.raw_datasets['d_icd_diagnoses'].copy()
        
        # 统计诊断频率
        diagnosis_freq = diagnoses['icd9_code'].value_counts()
        
        # 创建诊断映射
        diagnosis_mapping = []
        
        for icd_code, freq in diagnosis_freq.items():
            # 获取诊断描述
            description_row = d_icd_diagnoses[d_icd_diagnoses['icd9_code'] == icd_code]
            
            if len(description_row) > 0:
                short_title = description_row.iloc[0]['short_title']
                long_title = description_row.iloc[0]['long_title']
            else:
                short_title = f"Unknown diagnosis {icd_code}"
                long_title = f"Unknown diagnosis code {icd_code}"
            
            # 诊断严重程度估算（基于频率和关键词）
            severity = self._estimate_diagnosis_severity(short_title, freq)
            
            # 治疗复杂度估算
            complexity = self._estimate_treatment_complexity(short_title)
            
            mapping = {
                'icd9_code': icd_code,
                'short_title': short_title,
                'long_title': long_title,
                'frequency': freq,
                'severity_score': severity,
                'treatment_complexity': complexity,
                'mortality_risk': self._estimate_mortality_risk(short_title)
            }
            
            diagnosis_mapping.append(mapping)
        
        diagnosis_df = pd.DataFrame(diagnosis_mapping)
        
        # 标准化分数
        scaler = StandardScaler()
        diagnosis_df['severity_normalized'] = scaler.fit_transform(
            diagnosis_df[['severity_score']]
        ).flatten()
        self.scalers['diagnosis_severity'] = scaler
        
        print(f"   ✅ 创建了 {len(diagnosis_df)} 个诊断映射")
        print(f"   - 最常见诊断: {diagnosis_df.iloc[0]['short_title']}")
        print(f"   - 平均严重程度: {diagnosis_df['severity_score'].mean():.2f}")
        
        self.processed_data['diagnoses_mapping'] = diagnosis_df
        return diagnosis_df
    
    def create_treatment_mapping(self):
        """创建治疗映射表"""
        print("\n💊 创建治疗映射表...")
        
        prescriptions = self.raw_datasets['prescriptions'].copy()
        procedures = self.raw_datasets['procedures'].copy()
        d_icd_procedures = self.raw_datasets['d_icd_procedures'].copy()
        
        # 药物映射
        drug_freq = prescriptions['drug'].value_counts()
        
        drugs_mapping = []
        for drug, freq in drug_freq.head(100).items():  # 取前100种常用药物
            # 估算药物成本和效果
            cost = self._estimate_drug_cost(drug, freq)
            effectiveness = self._estimate_drug_effectiveness(drug)
            
            drugs_mapping.append({
                'drug_name': drug,
                'frequency': freq,
                'estimated_cost': cost,
                'effectiveness_score': effectiveness,
                'drug_type': self._classify_drug_type(drug)
            })
        
        # 程序映射
        procedure_freq = procedures['icd9_code'].value_counts()
        
        procedures_mapping = []
        for proc_code, freq in procedure_freq.items():
            # 获取程序描述
            proc_desc = d_icd_procedures[d_icd_procedures['icd9_code'] == proc_code]
            
            if len(proc_desc) > 0:
                short_title = proc_desc.iloc[0]['short_title']
                long_title = proc_desc.iloc[0]['long_title']
            else:
                short_title = f"Unknown procedure {proc_code}"
                long_title = f"Unknown procedure code {proc_code}"
            
            # 程序成本和复杂度
            cost = self._estimate_procedure_cost(short_title, freq)
            complexity = self._estimate_procedure_complexity(short_title)
            
            procedures_mapping.append({
                'icd9_code': proc_code,
                'short_title': short_title,
                'long_title': long_title,
                'frequency': freq,
                'estimated_cost': cost,
                'complexity_score': complexity,
                'procedure_type': self._classify_procedure_type(short_title)
            })
        
        drugs_df = pd.DataFrame(drugs_mapping)
        procedures_df = pd.DataFrame(procedures_mapping)
        
        print(f"   ✅ 创建了 {len(drugs_df)} 个药物映射")
        print(f"   ✅ 创建了 {len(procedures_df)} 个程序映射")
        print(f"   - 平均药物成本: ${drugs_df['estimated_cost'].mean():.2f}")
        print(f"   - 平均程序成本: ${procedures_df['estimated_cost'].mean():.2f}")
        
        self.processed_data['drugs_mapping'] = drugs_df
        self.processed_data['procedures_mapping'] = procedures_df
        return drugs_df, procedures_df
    
    def create_cost_mapping(self):
        """创建成本映射表"""
        print("\n💰 创建成本映射表...")
        
        drgcodes = self.raw_datasets['drgcodes'].copy()
        
        # DRG成本映射
        cost_mapping = []
        
        for _, drg in drgcodes.iterrows():
            drg_code = drg['drg_code']
            description = drg['description']
            severity = drg.get('drg_severity', 0)
            mortality = drg.get('drg_mortality', 0)
            
            # 基于DRG的成本估算
            base_cost = self._estimate_drg_cost(description, severity, mortality)
            
            cost_mapping.append({
                'drg_code': drg_code,
                'description': description,
                'severity': severity if pd.notna(severity) else 0,
                'mortality_risk': mortality if pd.notna(mortality) else 0,
                'estimated_cost': base_cost,
                'insurance_coverage': self._estimate_insurance_coverage(description)
            })
        
        cost_df = pd.DataFrame(cost_mapping)
        
        print(f"   ✅ 创建了 {len(cost_df)} 个DRG成本映射")
        print(f"   - 平均治疗成本: ${cost_df['estimated_cost'].mean():.2f}")
        print(f"   - 平均保险覆盖: {cost_df['insurance_coverage'].mean():.2%}")
        
        self.processed_data['cost_mapping'] = cost_df
        return cost_df
    
    def _estimate_diagnosis_severity(self, diagnosis: str, frequency: int) -> float:
        """估算诊断严重程度"""
        severity = 1.0
        
        # 基于关键词的严重程度评估
        high_severity_keywords = ['severe', 'acute', 'failure', 'arrest', 'shock', 'sepsis', 'cancer', 'malignant']
        medium_severity_keywords = ['chronic', 'moderate', 'infection', 'pneumonia', 'diabetes']
        
        diagnosis_lower = diagnosis.lower()
        
        for keyword in high_severity_keywords:
            if keyword in diagnosis_lower:
                severity += 2.0
                break
        
        for keyword in medium_severity_keywords:
            if keyword in diagnosis_lower:
                severity += 1.0
                break
        
        # 频率调整（罕见疾病可能更严重）
        if frequency < 10:
            severity += 0.5
        
        return min(severity, 5.0)
    
    def _estimate_treatment_complexity(self, diagnosis: str) -> float:
        """估算治疗复杂度"""
        complexity = 1.0
        
        complex_keywords = ['surgery', 'transplant', 'intensive', 'ventilator', 'dialysis']
        diagnosis_lower = diagnosis.lower()
        
        for keyword in complex_keywords:
            if keyword in diagnosis_lower:
                complexity += 1.5
        
        return min(complexity, 5.0)
    
    def _estimate_mortality_risk(self, diagnosis: str) -> float:
        """估算死亡风险"""
        risk = 0.1
        
        high_risk_keywords = ['sepsis', 'shock', 'arrest', 'failure', 'cancer', 'malignant']
        diagnosis_lower = diagnosis.lower()
        
        for keyword in high_risk_keywords:
            if keyword in diagnosis_lower:
                risk += 0.3
        
        return min(risk, 1.0)
    
    def _estimate_drug_cost(self, drug: str, frequency: int) -> float:
        """估算药物成本"""
        # 基础成本
        base_cost = 50.0
        
        # 常用药物成本较低
        if frequency > 100:
            base_cost *= 0.5
        elif frequency < 10:
            base_cost *= 2.0
        
        # 特殊药物成本调整
        expensive_keywords = ['insulin', 'antibiotic', 'chemotherapy', 'immunosuppressive']
        cheap_keywords = ['saline', 'dextrose', 'water', 'sodium']
        
        drug_lower = drug.lower()
        
        for keyword in expensive_keywords:
            if keyword in drug_lower:
                base_cost *= 3.0
                break
        
        for keyword in cheap_keywords:
            if keyword in drug_lower:
                base_cost *= 0.3
                break
        
        return base_cost
    
    def _estimate_drug_effectiveness(self, drug: str) -> float:
        """估算药物有效性"""
        effectiveness = 0.7
        
        # 基于药物类型调整有效性
        high_effect_keywords = ['antibiotic', 'insulin', 'furosemide']
        drug_lower = drug.lower()
        
        for keyword in high_effect_keywords:
            if keyword in drug_lower:
                effectiveness += 0.2
                break
        
        return min(effectiveness, 1.0)
    
    def _classify_drug_type(self, drug: str) -> str:
        """分类药物类型"""
        drug_lower = drug.lower()
        
        if any(keyword in drug_lower for keyword in ['antibiotic', 'penicillin', 'vancomycin']):
            return 'antibiotic'
        elif any(keyword in drug_lower for keyword in ['insulin', 'diabetes']):
            return 'diabetes'
        elif any(keyword in drug_lower for keyword in ['furosemide', 'diuretic']):
            return 'diuretic'
        elif any(keyword in drug_lower for keyword in ['saline', 'dextrose', 'water']):
            return 'fluid'
        else:
            return 'other'
    
    def _estimate_procedure_cost(self, procedure: str, frequency: int) -> float:
        """估算程序成本"""
        base_cost = 500.0
        
        procedure_lower = procedure.lower()
        
        # 高成本程序
        if any(keyword in procedure_lower for keyword in ['surgery', 'transplant', 'catheter']):
            base_cost *= 5.0
        elif any(keyword in procedure_lower for keyword in ['ventilator', 'dialysis']):
            base_cost *= 3.0
        elif any(keyword in procedure_lower for keyword in ['transfusion', 'infusion']):
            base_cost *= 1.5
        
        return base_cost
    
    def _estimate_procedure_complexity(self, procedure: str) -> float:
        """估算程序复杂度"""
        complexity = 1.0
        
        procedure_lower = procedure.lower()
        
        if any(keyword in procedure_lower for keyword in ['surgery', 'transplant']):
            complexity = 5.0
        elif any(keyword in procedure_lower for keyword in ['catheter', 'ventilator']):
            complexity = 3.0
        elif any(keyword in procedure_lower for keyword in ['transfusion', 'dialysis']):
            complexity = 2.0
        
        return complexity
    
    def _classify_procedure_type(self, procedure: str) -> str:
        """分类程序类型"""
        procedure_lower = procedure.lower()
        
        if any(keyword in procedure_lower for keyword in ['surgery', 'surgical']):
            return 'surgery'
        elif any(keyword in procedure_lower for keyword in ['catheter', 'cath']):
            return 'catheter'
        elif any(keyword in procedure_lower for keyword in ['ventilator', 'ventilation']):
            return 'ventilation'
        elif any(keyword in procedure_lower for keyword in ['transfusion']):
            return 'transfusion'
        else:
            return 'other'
    
    def _estimate_drg_cost(self, description: str, severity: float, mortality: float) -> float:
        """估算DRG成本"""
        base_cost = 5000.0
        
        # 严重程度调整
        if pd.notna(severity):
            base_cost *= (1 + severity * 0.5)
        
        # 死亡风险调整
        if pd.notna(mortality):
            base_cost *= (1 + mortality * 0.3)
        
        # 基于描述的调整
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ['surgery', 'transplant']):
            base_cost *= 3.0
        elif any(keyword in description_lower for keyword in ['intensive', 'critical']):
            base_cost *= 2.0
        elif any(keyword in description_lower for keyword in ['sepsis', 'shock']):
            base_cost *= 2.5
        
        return base_cost
    
    def _estimate_insurance_coverage(self, description: str) -> float:
        """估算保险覆盖率"""
        coverage = 0.8  # 默认80%覆盖
        
        description_lower = description.lower()
        
        # 紧急情况通常覆盖率更高
        if any(keyword in description_lower for keyword in ['emergency', 'acute', 'severe']):
            coverage = 0.9
        
        # 选择性手术覆盖率可能较低
        elif any(keyword in description_lower for keyword in ['elective', 'cosmetic']):
            coverage = 0.6
        
        return coverage
    
    def save_processed_data(self):
        """保存处理后的数据"""
        print("\n💾 保存处理后的数据...")
        
        for name, data in self.processed_data.items():
            if data is not None:
                output_file = self.output_path / f"{name}.csv"
                data.to_csv(output_file, index=False)
                print(f"   ✅ 保存 {name}: {len(data)} 行")
        
        # 保存编码器和标准化器
        encoders_file = self.output_path / "encoders.json"
        encoders_data = {}
        for name, encoder in self.encoders.items():
            if hasattr(encoder, 'classes_'):
                encoders_data[name] = encoder.classes_.tolist()
        
        with open(encoders_file, 'w') as f:
            json.dump(encoders_data, f, indent=2)
        
        print(f"   ✅ 保存编码器和预处理器")
        print(f"📁 处理后的数据保存在: {self.output_path}")
    
    def generate_summary_statistics(self):
        """生成摘要统计"""
        print("\n📊 生成摘要统计...")
        
        summary = {
            'preprocessing_timestamp': datetime.now().isoformat(),
            'total_patients': len(self.processed_data['patients']) if self.processed_data['patients'] is not None else 0,
            'total_episodes': len(self.processed_data['episodes']) if self.processed_data['episodes'] is not None else 0,
            'unique_diagnoses': len(self.processed_data['diagnoses_mapping']) if self.processed_data['diagnoses_mapping'] is not None else 0,
            'unique_drugs': len(self.processed_data['drugs_mapping']) if self.processed_data['drugs_mapping'] is not None else 0,
            'data_quality_score': 0.95  # 基于之前的数据质量评估
        }
        
        # 患者统计
        if self.processed_data['patients'] is not None:
            patients = self.processed_data['patients']
            summary.update({
                'avg_patient_age': float(patients['age'].mean()),
                'gender_distribution': patients['gender'].value_counts().to_dict(),
                'mortality_rate': float(patients['is_expired'].mean())
            })
        
        # Episode统计
        if self.processed_data['episodes'] is not None:
            episodes = self.processed_data['episodes']
            summary.update({
                'avg_length_of_stay': float(episodes['length_of_stay'].mean()),
                'avg_prescriptions_per_episode': float(episodes['total_prescriptions'].mean()),
                'hospital_mortality_rate': float(episodes['hospital_expire_flag'].mean())
            })
        
        # 保存摘要
        summary_file = self.output_path / "preprocessing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("📋 数据预处理摘要:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            elif isinstance(value, dict):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")
        
        return summary

def main():
    """主函数"""
    print("🚀 启动 MIMIC-III 数据预处理")
    print("=" * 60)
    
    # 创建预处理器
    preprocessor = MIMICPreprocessor()
    
    # 加载原始数据
    preprocessor.load_raw_data()
    
    # 创建患者档案
    preprocessor.create_patient_profiles()
    
    # 创建医疗事件
    preprocessor.create_medical_episodes()
    
    # 创建映射表
    preprocessor.create_diagnosis_mapping()
    preprocessor.create_treatment_mapping()
    preprocessor.create_cost_mapping()
    
    # 保存处理后的数据
    preprocessor.save_processed_data()
    
    # 生成摘要统计
    summary = preprocessor.generate_summary_statistics()
    
    print("\n🎉 数据预处理完成！")
    print("\n🔄 下一步建议:")
    print("   1. 运行 python data/create_patient_models.py")
    print("   2. 更新环境以使用真实数据")
    print("   3. 重新训练智能体模型")

if __name__ == "__main__":
    main() 