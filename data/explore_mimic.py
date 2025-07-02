#!/usr/bin/env python3
"""
MIMIC-III Data Exploration Script

Analyzes the structure, quality, and distribution of the MIMIC-III dataset
to provide data foundation for building realistic patient models
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import config

class MIMICExplorer:
    """MIMIC-III Data Explorer"""
    
    def __init__(self, data_path="MIMIC III/"):
        self.data_path = Path(data_path)
        self.datasets = {}
        self.summary_stats = {}
        
        print("🔍 MIMIC-III Data Explorer Initialization")
        print(f"📁 Data Path: {self.data_path}")
        
    def scan_data_files(self):
        """Scan available data files"""
        print("\n📋 Scanning data files...")
        
        csv_files = list(self.data_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files:")
        
        file_info = []
        for file_path in csv_files:
            file_size = file_path.stat().st_size
            file_info.append({
                'filename': file_path.name,
                'size_mb': file_size / (1024 * 1024),
                'path': file_path
            })
        
        # Sort by file size
        file_info.sort(key=lambda x: x['size_mb'], reverse=True)
        
        for info in file_info:
            print(f"  📄 {info['filename']:<25} ({info['size_mb']:.1f} MB)")
            
        return file_info
    
    def load_core_datasets(self):
        """Load core datasets"""
        print("\n📥 Loading core datasets...")
        
        # Define core datasets and their importance
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
                    self.datasets[key] = df
                    print(f"     ✅ {len(df)} rows x {len(df.columns)} columns")
                except Exception as e:
                    print(f"     ❌ Loading failed: {e}")
            else:
                print(f"     ⚠️  File not found: {filename}")
        
        print(f"\n✅ Successfully loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def analyze_data_structure(self):
        """Analyze data structure"""
        print("\n🏗️  Data Structure Analysis...")
        
        for name, df in self.datasets.items():
            print(f"\n📊 {name.upper()}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check key ID columns
            id_columns = [col for col in df.columns if 'id' in col.lower()]
            if id_columns:
                print(f"   ID columns: {id_columns}")
                
            # Check missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"   Missing values: {missing[missing > 0].to_dict()}")
            
            # Data types
            print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
            
            print("   " + "-" * 50)
    
    def analyze_patient_demographics(self):
        """Analyze patient demographic characteristics"""
        print("\n👥 Patient Demographics Analysis...")
        
        if 'patients' not in self.datasets:
            print("❌ Patient data not available")
            return
        
        patients = self.datasets['patients']
        
        # Gender distribution
        gender_col = 'gender' if 'gender' in patients.columns else 'GENDER'
        if gender_col in patients.columns:
            gender_dist = patients[gender_col].value_counts()
            print(f"📊 Gender Distribution:")
            for gender, count in gender_dist.items():
                print(f"   {gender}: {count} ({count/len(patients)*100:.1f}%)")
        
        # Age analysis (based on DOB and DOD)
        dob_col = 'dob' if 'dob' in patients.columns else 'DOB'
        if dob_col in patients.columns:
            # Simplified age calculation
            print("📊 Age Analysis:")
            print("   (Age information needs to be calculated with admission time)")
        
        # Mortality rate
        dod_col = 'dod' if 'dod' in patients.columns else 'DOD'
        if dod_col in patients.columns:
            death_rate = patients[dod_col].notna().sum()
            print(f"📊 Total Deaths: {death_rate} ({death_rate/len(patients)*100:.1f}%)")
        
        self.summary_stats['patients'] = {
            'total_patients': len(patients),
            'gender_distribution': patients[gender_col].value_counts().to_dict() if gender_col in patients.columns else {},
            'death_rate': patients[dod_col].notna().sum() / len(patients) if dod_col in patients.columns else 0
        }
    
    def analyze_admissions_patterns(self):
        """Analyze admission patterns"""
        print("\n🏥 Admission Patterns Analysis...")
        
        if 'admissions' not in self.datasets:
            print("❌ Admission data not available")
            return
            
        admissions = self.datasets['admissions']
        
        # Admission type distribution
        admission_type_col = 'admission_type' if 'admission_type' in admissions.columns else 'ADMISSION_TYPE'
        if admission_type_col in admissions.columns:
            admission_types = admissions[admission_type_col].value_counts()
            print("📊 Admission Type Distribution:")
            for type_name, count in admission_types.head().items():
                print(f"   {type_name}: {count}")
        
        # Admission location
        admission_location_col = 'admission_location' if 'admission_location' in admissions.columns else 'ADMISSION_LOCATION'
        if admission_location_col in admissions.columns:
            locations = admissions[admission_location_col].value_counts()
            print("📊 Admission Sources (Top 5):")
            for location, count in locations.head().items():
                print(f"   {location}: {count}")
        
        # Insurance type
        insurance_col = 'insurance' if 'insurance' in admissions.columns else 'INSURANCE'
        if insurance_col in admissions.columns:
            insurance = admissions[insurance_col].value_counts()
            print("📊 Insurance Type Distribution:")
            for ins_type, count in insurance.items():
                print(f"   {ins_type}: {count}")
        
        # Length of stay analysis
        admittime_col = 'admittime' if 'admittime' in admissions.columns else 'ADMITTIME'
        dischtime_col = 'dischtime' if 'dischtime' in admissions.columns else 'DISCHTIME'
        
        if admittime_col in admissions.columns and dischtime_col in admissions.columns:
            try:
                admit_time = pd.to_datetime(admissions[admittime_col])
                discharge_time = pd.to_datetime(admissions[dischtime_col])
                length_of_stay = (discharge_time - admit_time).dt.days
                
                print("📊 Length of Stay Statistics:")
                print(f"   Average: {length_of_stay.mean():.1f} days")
                print(f"   Median: {length_of_stay.median():.1f} days")
                print(f"   Maximum: {length_of_stay.max()} days")
                
            except Exception as e:
                print(f"   Length of stay calculation failed: {e}")
        
        self.summary_stats['admissions'] = {
            'total_admissions': len(admissions),
            'admission_types': admissions[admission_type_col].value_counts().to_dict() if admission_type_col in admissions.columns else {},
            'insurance_types': admissions[insurance_col].value_counts().to_dict() if insurance_col in admissions.columns else {}
        }
    
    def analyze_diagnoses_distribution(self):
        """Analyze diagnosis distribution"""
        print("\n🩺 Diagnosis Distribution Analysis...")
        
        if 'diagnoses' not in self.datasets:
            print("❌ Diagnosis data not available")
            return
        
        diagnoses = self.datasets['diagnoses']
        
        # ICD diagnosis code distribution
        icd_code_col = 'icd9_code' if 'icd9_code' in diagnoses.columns else 'ICD9_CODE'
        if icd_code_col in diagnoses.columns:
            icd_dist = diagnoses[icd_code_col].value_counts()
            print("📊 Most Common Diagnoses (Top 10):")
            
            # If diagnosis dictionary is available, show descriptions
            if 'd_icd_diagnoses' in self.datasets:
                # Check column names in dictionary
                dict_df = self.datasets['d_icd_diagnoses']
                dict_icd_col = 'icd9_code' if 'icd9_code' in dict_df.columns else 'ICD9_CODE'
                dict_title_col = 'short_title' if 'short_title' in dict_df.columns else 'SHORT_TITLE'
                
                icd_dict = dict_df.set_index(dict_icd_col)[dict_title_col].to_dict()
                
                for icd_code, count in icd_dist.head(10).items():
                    description = icd_dict.get(icd_code, "Unknown diagnosis")
                    print(f"   {icd_code}: {description} ({count} times)")
            else:
                for icd_code, count in icd_dist.head(10).items():
                    print(f"   {icd_code}: {count} times")
        
        # Diagnosis sequence analysis
        seq_num_col = 'seq_num' if 'seq_num' in diagnoses.columns else 'SEQ_NUM'
        if seq_num_col in diagnoses.columns:
            seq_dist = diagnoses[seq_num_col].value_counts().sort_index()
            print("📊 Diagnosis Sequence Distribution:")
            for seq, count in seq_dist.head().items():
                print(f"   Sequence {seq}: {count} diagnoses")
        
        self.summary_stats['diagnoses'] = {
            'total_diagnoses': len(diagnoses),
            'unique_icd_codes': diagnoses[icd_code_col].nunique() if icd_code_col in diagnoses.columns else 0,
            'top_diagnoses': icd_dist.head(10).to_dict() if icd_code_col in diagnoses.columns else {}
        }
    
    def analyze_treatments_and_costs(self):
        """Analyze treatments and costs"""
        print("\n💊 Treatment and Cost Analysis...")
        
        # Prescription analysis
        if 'prescriptions' in self.datasets:
            prescriptions = self.datasets['prescriptions']
            print("📊 Prescription Drug Analysis:")
            
            drug_col = 'drug' if 'drug' in prescriptions.columns else 'DRUG'
            if drug_col in prescriptions.columns:
                drug_dist = prescriptions[drug_col].value_counts()
                print("   Most Common Drugs (Top 10):")
                for drug, count in drug_dist.head(10).items():
                    print(f"     {drug}: {count} times")
            
            dose_col = 'dose_val_rx' if 'dose_val_rx' in prescriptions.columns else 'DOSE_VAL_RX'
            if dose_col in prescriptions.columns:
                # Only analyze numeric dose data
                numeric_doses = pd.to_numeric(prescriptions[dose_col], errors='coerce').dropna()
                if len(numeric_doses) > 0:
                    dose_stats = numeric_doses.describe()
                    print("📊 Dosage Statistics:")
                    print(f"   Average dose: {dose_stats['mean']:.2f}")
                    print(f"   Dose range: {dose_stats['min']:.2f} - {dose_stats['max']:.2f}")
                    print(f"   Valid dose records: {len(numeric_doses)}/{len(prescriptions)} ({len(numeric_doses)/len(prescriptions)*100:.1f}%)")
                else:
                    print("📊 Dosage Statistics: No valid numeric data")
        
        # DRG cost analysis
        if 'drgcodes' in self.datasets:
            drg = self.datasets['drgcodes']
            print("📊 DRG Diagnosis Related Group Analysis:")
            
            desc_col = 'description' if 'description' in drg.columns else 'DESCRIPTION'
            if desc_col in drg.columns:
                drg_dist = drg[desc_col].value_counts()
                print("   Most Common DRGs (Top 5):")
                for desc, count in drg_dist.head(5).items():
                    print(f"     {desc}: {count} times")
        
        # Procedure analysis
        if 'procedures' in self.datasets:
            procedures = self.datasets['procedures']
            print("📊 Medical Procedure Analysis:")
            print(f"   Total procedures: {len(procedures)}")
            
            proc_icd_col = 'icd9_code' if 'icd9_code' in procedures.columns else 'ICD9_CODE'
            if proc_icd_col in procedures.columns:
                proc_dist = procedures[proc_icd_col].value_counts()
                print("   Most Common Procedures (Top 5):")
                
                # If procedure dictionary is available
                if 'd_icd_procedures' in self.datasets:
                    proc_dict_df = self.datasets['d_icd_procedures']
                    proc_dict_icd_col = 'icd9_code' if 'icd9_code' in proc_dict_df.columns else 'ICD9_CODE'
                    proc_dict_title_col = 'short_title' if 'short_title' in proc_dict_df.columns else 'SHORT_TITLE'
                    
                    proc_dict = proc_dict_df.set_index(proc_dict_icd_col)[proc_dict_title_col].to_dict()
                    
                    for proc_code, count in proc_dist.head(5).items():
                        description = proc_dict.get(proc_code, "Unknown procedure")
                        print(f"     {proc_code}: {description} ({count} times)")
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n📋 Data Summary Report")
        print("=" * 60)
        
        # Dataset overview
        print("📊 Dataset Overview:")
        for name, df in self.datasets.items():
            print(f"   {name:20} {len(df):6,} rows x {len(df.columns):2} columns")
        
        # Key statistics
        if self.summary_stats:
            print("\n📈 Key Statistics:")
            
            if 'patients' in self.summary_stats:
                stats = self.summary_stats['patients']
                print(f"   Total patients: {stats['total_patients']:,}")
                print(f"   Mortality rate: {stats['death_rate']:.2%}")
            
            if 'admissions' in self.summary_stats:
                stats = self.summary_stats['admissions']
                print(f"   Total admissions: {stats['total_admissions']:,}")
            
            if 'diagnoses' in self.summary_stats:
                stats = self.summary_stats['diagnoses']
                print(f"   Total diagnoses: {stats['total_diagnoses']:,}")
                print(f"   Unique ICD codes: {stats['unique_icd_codes']:,}")
        
        print("\n🎯 Data Quality Assessment:")
        data_quality = self.assess_data_quality()
        for aspect, score in data_quality.items():
            print(f"   {aspect}: {score}")
        
        print("\n🚀 Modeling Recommendations:")
        recommendations = self.generate_modeling_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def assess_data_quality(self):
        """Assess data quality"""
        quality_scores = {}
        
        # Completeness score
        total_completeness = 0
        count = 0
        
        for name, df in self.datasets.items():
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            quality_scores[f"{name}_completeness"] = f"{completeness:.1f}%"
            total_completeness += completeness
            count += 1
        
        if count > 0:
            quality_scores["overall_completeness"] = f"{total_completeness/count:.1f}%"
        
        # Consistency check
        consistency_issues = 0
        
        # Check ID consistency
        if 'patients' in self.datasets and 'admissions' in self.datasets:
            patients_df = self.datasets['patients']
            admissions_df = self.datasets['admissions']
            
            # Check column names (lowercase or uppercase)
            patient_id_col_patients = 'subject_id' if 'subject_id' in patients_df.columns else 'SUBJECT_ID'
            patient_id_col_admissions = 'subject_id' if 'subject_id' in admissions_df.columns else 'SUBJECT_ID'
            
            if patient_id_col_patients in patients_df.columns and patient_id_col_admissions in admissions_df.columns:
                patient_ids_in_patients = set(patients_df[patient_id_col_patients])
                patient_ids_in_admissions = set(admissions_df[patient_id_col_admissions])
                
                if not patient_ids_in_admissions.issubset(patient_ids_in_patients):
                    consistency_issues += 1
        
        quality_scores["consistency"] = "Good" if consistency_issues == 0 else f"{consistency_issues} issues"
        
        return quality_scores
    
    def generate_modeling_recommendations(self):
        """Generate modeling recommendations"""
        recommendations = []
        
        # Generate recommendations based on data analysis
        if 'patients' in self.datasets:
            recommendations.append("Use real patient demographic features (age, gender) as patient states")
        
        if 'diagnoses' in self.datasets:
            recommendations.append("Build realistic disease models based on ICD-9 diagnosis codes")
            recommendations.append("Use diagnosis frequency as prior probabilities for disease occurrence")
        
        if 'prescriptions' in self.datasets:
            recommendations.append("Build treatment option space based on real prescription data")
            recommendations.append("Use drug usage frequency to optimize treatment selection rewards")
        
        if 'drgcodes' in self.datasets:
            recommendations.append("Use DRG codes for cost modeling and insurance review logic")
        
        if 'admissions' in self.datasets:
            recommendations.append("Optimize treatment effectiveness evaluation based on length of stay data")
            recommendations.append("Improve insurance agent decisions using insurance type data")
        
        recommendations.append("Implement reward functions based on real medical outcomes")
        recommendations.append("Add time series modeling for disease progression")
        
        return recommendations
    
    def save_exploration_results(self, output_dir="data/exploration_results"):
        """Save exploration results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving exploration results to {output_path}")
        
        # Save summary statistics
        import json
        with open(output_path / "summary_stats.json", "w", encoding="utf-8") as f:
            json.dump(self.summary_stats, f, indent=2, ensure_ascii=False)
        
        # Save dataset information
        dataset_info = {}
        for name, df in self.datasets.items():
            dataset_info[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
        
        with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print("   ✅ Exploration results saved")

def main():
    """Main function"""
    print("🚀 Starting MIMIC-III Data Exploration")
    print("=" * 60)
    
    # Create explorer
    explorer = MIMICExplorer()
    
    # Scan files
    file_info = explorer.scan_data_files()
    
    # Load core datasets
    datasets = explorer.load_core_datasets()
    
    if not datasets:
        print("❌ No datasets available, please check data path")
        return
    
    # Data structure analysis
    explorer.analyze_data_structure()
    
    # Detailed analysis
    explorer.analyze_patient_demographics()
    explorer.analyze_admissions_patterns()
    explorer.analyze_diagnoses_distribution()
    explorer.analyze_treatments_and_costs()
    
    # Generate report
    explorer.generate_summary_report()
    
    # Save results
    explorer.save_exploration_results()
    
    print("\n🎉 Data exploration completed!")
    print("\n🔄 Next Steps:")
    print("   1. Run python data/create_patient_models.py")
    print("   2. Optimize reward function based on exploration results")
    print("   3. Retrain agent models")

if __name__ == "__main__":
    main() 