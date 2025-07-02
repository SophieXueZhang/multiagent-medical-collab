#!/usr/bin/env python3
"""
Multi-Agent Healthcare Collaboration System Web Demo

Interactive demonstration interface based on real MIMIC-III data
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import random

# Flask related imports
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit

# Add project root directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Project component imports
from env.enhanced_multi_agent_env import EnhancedMultiAgentHealthcareEnv
from data.patient_models import RealPatientModelGenerator
from env.reward_optimizer import RealDataRewardOptimizer

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multi_agent_healthcare_demo_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
current_env = None
current_simulation = {
    'active': False,
    'step': 0,
    'max_steps': 20,
    'patient': None,
    'history': [],
    'performance': {}
}

def initialize_components():
    """Initialize system components"""
    global current_env
    try:
        # Create enhanced environment
        current_env = EnhancedMultiAgentHealthcareEnv(
            render_mode="rgb_array",
            use_real_data=True
        )
        print("✅ Multi-agent environment initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    current_agent = None
    if current_env and current_simulation['active']:
        current_agent = current_env.agent_selection
    
    return jsonify({
        'environment_ready': current_env is not None,
        'simulation_active': current_simulation['active'],
        'current_step': current_simulation['step'],
        'max_steps': current_simulation['max_steps'],
        'current_agent': current_agent
    })

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    global current_simulation
    
    if not current_env:
        return jsonify({'success': False, 'error': 'Environment not initialized'})
    
    try:
        # Reset environment
        observations, infos = current_env.reset(seed=np.random.randint(0, 1000))
        
        # Initialize simulation state
        current_simulation = {
            'active': True,
            'step': 0,
            'max_steps': 20,
            'patient': format_patient_data(current_env.current_patient),
            'history': [],
            'performance': {
                'treatment_effectiveness': 0.0,
                'cost_efficiency': 0.0,
                'patient_satisfaction': 0.0,
                'communication_quality': 0.0
            }
        }
        
        # Broadcast new simulation start
        socketio.emit('simulation_started', current_simulation)
        
        return jsonify({
            'success': True,
            'simulation': current_simulation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/execute_action', methods=['POST'])
def execute_action():
    """Execute intelligent agent action"""
    global current_simulation
    
    if not current_simulation['active']:
        return jsonify({'success': False, 'error': 'Simulation is not active'})
    
    try:
        data = request.json
        agent = data.get('agent')
        action = data.get('action', [0, 0, 0, 0])
        
        # Generate intelligent action
        if agent == 'auto':
            # Auto run one step
            agent = current_env.agent_selection
            action = generate_intelligent_action(agent, current_env.current_patient, current_simulation['step'])
        
        # Execute action
        current_env.step(action)
        
        # Get reward and new state
        reward = current_env.rewards.get(agent, 0)
        
        # Record history
        action_record = {
            'step': current_simulation['step'],
            'agent': agent,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'description': get_action_description(agent, action)
        }
        
        current_simulation['history'].append(action_record)
        current_simulation['step'] += 1
        
        # Update performance metrics
        current_simulation['performance'] = {
            'treatment_effectiveness': current_env.episode_stats['treatment_effectiveness'],
            'cost_efficiency': current_env.episode_stats['cost_efficiency'],
            'patient_satisfaction': current_env.episode_stats['patient_satisfaction'],
            'communication_quality': current_env.episode_stats['communication_quality']
        }
        
        # Check if finished
        if (current_simulation['step'] >= current_simulation['max_steps'] or 
            all(current_env.terminations.values())):
            current_simulation['active'] = False
            socketio.emit('simulation_completed', current_simulation)
        else:
            socketio.emit('action_executed', action_record)
        
        return jsonify({
            'success': True,
            'action_record': action_record,
            'performance': current_simulation['performance'],
            'active': current_simulation['active'],
            'current_agent': current_env.agent_selection if current_simulation['active'] else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simulation_data')
def get_simulation_data():
    """Get current simulation data"""
    return jsonify(current_simulation)

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation environment"""
    global current_simulation
    
    try:
        if current_env:
            # Reset environment
            observations, infos = current_env.reset(seed=np.random.randint(0, 1000))
        
        # Reset simulation state
        current_simulation = {
            'active': False,
            'step': 0,
            'max_steps': 20,
            'patient': None,
            'history': [],
            'performance': {
                'treatment_effectiveness': 0.0,
                'cost_efficiency': 0.0,
                'patient_satisfaction': 0.0,
                'communication_quality': 0.0
            }
        }
        
        # Broadcast reset message
        socketio.emit('simulation_reset', current_simulation)
        
        return jsonify({'success': True, 'message': 'Environment has been reset'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    emit('connected', {'data': 'Connection successful'})

@socketio.on('request_auto_step')
def handle_auto_step():
    """Handle auto step request"""
    if current_simulation['active']:
        # Execute auto action
        result = execute_action_internal('auto')
        emit('auto_step_result', result)

def execute_action_internal(agent_type):
    """Internal action execution function"""
    global current_simulation
    
    if not current_simulation['active']:
        return {'success': False, 'error': 'Simulation is not active'}
    
    try:
        agent = current_env.agent_selection
        action = generate_intelligent_action(agent, current_env.current_patient, current_simulation['step'])
        
        # Execute action
        current_env.step(action)
        reward = current_env.rewards.get(agent, 0)
        
        # Record history
        action_record = {
            'step': current_simulation['step'],
            'agent': agent,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'description': get_action_description(agent, action)
        }
        
        current_simulation['history'].append(action_record)
        current_simulation['step'] += 1
        
        # Update performance metrics
        current_simulation['performance'] = {
            'treatment_effectiveness': current_env.episode_stats['treatment_effectiveness'],
            'cost_efficiency': current_env.episode_stats['cost_efficiency'],
            'patient_satisfaction': current_env.episode_stats['patient_satisfaction'],
            'communication_quality': current_env.episode_stats['communication_quality']
        }
        
        # Check if finished
        if (current_simulation['step'] >= current_simulation['max_steps'] or 
            all(current_env.terminations.values())):
            current_simulation['active'] = False
        
        return {
            'success': True,
            'action_record': action_record,
            'performance': current_simulation['performance'],
            'active': current_simulation['active']
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def format_patient_data(patient):
    """Format patient data for display"""
    if not patient:
        return None
    
    return {
        'id': patient.subject_id,
        'age': patient.age,
        'gender': 'Male' if patient.gender else 'Female',
        'diagnosis': patient.primary_diagnosis,
        'severity': round(patient.severity_score, 2),
        'mortality_risk': f"{patient.mortality_risk:.1%}",
        'estimated_cost': f"${patient.estimated_treatment_cost:,.0f}",
        'insurance_coverage': f"{patient.insurance_coverage:.1%}",
        'urgency': patient.urgency_level,
        'symptoms': patient.symptoms,
        'treatments': len(patient.current_treatments)
    }

def generate_intelligent_action(agent: str, patient, step: int):
    """Generate intelligent action"""
    if agent == "doctor":
        severity = patient.severity_score if patient else 2.0
        urgency = patient.urgency_level if patient else 2
        
        # Add random disturbance and stage-based change
        if step < 3:  # Initial diagnosis stage
            return [
                min(14, int(severity * 1.5) + random.randint(0, 2)),
                min(19, int(severity) + random.randint(0, 3)),
                min(9, step + random.randint(1, 3)),
                min(14, urgency + random.randint(0, 2))
            ]
        elif step < 8:  # Deep diagnosis stage
            return [
                min(14, int(severity * 2.5) + random.randint(-1, 2)),
                min(19, int(severity * 1.5) + urgency + random.randint(0, 2)),
                min(9, int(severity * 1.5) + random.randint(0, 2)),
                min(14, urgency * 2 + random.randint(-1, 1))
            ]
        elif step < 15:  # Treatment stage
            return [
                min(14, int(severity * 2) + random.randint(0, 3)),
                min(19, int(severity * 3) + urgency + random.randint(-1, 2)),
                min(9, int(severity * 2) + random.randint(0, 2)),
                min(14, urgency * 2 + step % 3 + random.randint(0, 1))
            ]
        else:  # Monitoring observation stage
            return [
                min(14, int(severity) + random.randint(0, 2)),
                min(19, int(severity * 1.5) + random.randint(0, 2)),
                min(9, int(severity) + random.randint(1, 3)),
                min(14, urgency + step % 4 + random.randint(0, 1))
            ]
    
    elif agent == "patient":
        severity = patient.severity_score if patient else 2.0
        pain_level = min(5, int(severity) + random.randint(0, 2))
        anxiety = min(4, 1 + step // 3 + random.randint(0, 1))
        
        # Patient status changes with time and treatment effect
        if step < 3:  # Initial anxiety
            return [
                min(9, pain_level + random.randint(0, 2)),
                min(7, 2 + random.randint(0, 2)),
                min(4, 1 + random.randint(0, 1)),
                min(9, anxiety + random.randint(1, 2))
            ]
        elif step < 8:  # Gradually adapt and cooperate
            return [
                min(9, int(severity * 1.5) + random.randint(0, 1)),
                min(7, 3 + step % 3 + random.randint(0, 1)),
                min(4, 2 + step // 2 + random.randint(0, 1)),
                min(9, int(severity) + random.randint(0, 2))
            ]
        elif step < 15:  # Actively cooperate with treatment
            return [
                min(9, int(severity * 2) + random.randint(-1, 1)),
                min(7, 4 + step % 2 + random.randint(0, 1)),
                min(4, 3 + random.randint(0, 1)),
                min(9, int(severity * 1.2) + random.randint(0, 1))
            ]
        else:  # Recovery period feedback
            return [
                min(9, int(severity * 1.2) + random.randint(0, 1)),
                min(7, 5 + random.randint(0, 1)),
                min(4, 3 + random.randint(0, 1)),
                min(9, max(1, int(severity) - step // 5) + random.randint(0, 1))
            ]
    
    elif agent == "insurance":
        cost = patient.estimated_treatment_cost if patient else 15000
        coverage = patient.insurance_coverage if patient else 0.8
        
        # Strictness and depth of insurance approval
        if cost < 5000:
            base_approval = 4
            investigation_level = 1
        elif cost < 15000:
            base_approval = 3
            investigation_level = 2
        elif cost < 30000:
            base_approval = 2
            investigation_level = 3
        else:
            base_approval = 1
            investigation_level = 4
        
        # Different stages of approval strategy
        if step < 4:  # Initial review
            return [
                min(4, base_approval + random.randint(0, 1)),
                min(7, investigation_level + random.randint(0, 1)),
                min(5, int((1 - coverage) * 3) + random.randint(0, 1)),
                min(11, 1 + step + random.randint(0, 1))
            ]
        elif step < 10:  # Detailed assessment
            return [
                min(4, base_approval + random.randint(-1, 1)),
                min(7, investigation_level + step // 2 + random.randint(0, 1)),
                min(5, int((1 - coverage) * 4) + random.randint(0, 1)),
                min(11, 3 + step % 3 + random.randint(0, 1))
            ]
        else:  # Final decision
            return [
                min(4, base_approval + random.randint(0, 1)),
                min(7, investigation_level + random.randint(0, 2)),
                min(5, int((1 - coverage) * 3) + random.randint(0, 1)),
                min(11, 2 + step % 4 + random.randint(0, 1))
            ]
    
    return [0, 0, 0, 0]

def get_action_description(agent: str, action: List[int]) -> str:
    """Get action description"""
    descriptions = {
        'doctor': {
            'action_types': ['Diagnosis', 'Treatment', 'Examination', 'Communication'],
            'diagnostic_actions': [
                "Initial inquiry of condition", "Detailed symptom history", "Physical examination", "Lab result analysis",
                "Review medical records", "Assess severity", "Develop diagnostic plan", "Professional diagnosis",
                "Consultation discussion", "Confirm diagnosis", "Update diagnostic record", "Adjust diagnostic plan",
                "In-depth diagnostic analysis", "Comprehensive assessment", "Expert-level diagnosis"
            ],
            'treatment_actions': [
                "Monitor condition changes", "Prescribe basic medication", "Adjust medication plan", "Conservative treatment",
                "Physical therapy", "Arrange rehabilitation", "Develop treatment plan", "Standard treatment",
                "Adjust treatment intensity", "Professional treatment", "Comprehensive treatment", "Optimize treatment plan",
                "Advanced treatment", "Precision treatment", "Active treatment", "Innovative treatment",
                "Emergency treatment", "Urgent intervention", "Expert treatment", "Optimal treatment"
            ],
            'examination_actions': [
                "Basic vital signs check", "Routine blood test", "Detailed assessment",
                "Imaging examination", "Special function test", "Comprehensive physical exam",
                "In-depth examination", "Advanced test", "Precision instrument exam", "Expert-level examination"
            ],
            'communication_actions': [
                "Simple condition communication", "Understand patient feelings", "Explain treatment plan", "Detailed condition discussion",
                "Comfort and encourage patient", "Health education", "In-depth discussion", "Comprehensive coordination",
                "Professional guidance", "Patiently answer questions", "Explain risks in detail", "Deep psychological counseling",
                "Comprehensive communication", "Expert consultation", "All-round communication"
            ]
        },
        'patient': {
            'symptom_actions': [
                "Briefly describe discomfort", "State main symptoms", "Describe condition in detail", "Comprehensive symptom description",
                "Add symptom details", "Update symptom changes", "In-depth feeling description", "Complete symptom report",
                "Accurate symptom description", "Full symptom report"
            ],
            'feedback_actions': [
                "Express basic feelings", "General feedback", "Detailed feedback", "Actively express opinions",
                "Proactively provide feedback", "Very cooperative", "Deep feedback exchange", "Comprehensive feedback evaluation"
            ],
            'cooperation_actions': [
                "Basic cooperation", "Partial compliance", "Good cooperation", "Full cooperation", "Proactive cooperation"
            ],
            'communication_actions': [
                "Simple response to doctor", "Basic communication", "Detailed condition discussion", "In-depth discussion",
                "Proactively ask questions", "Actively participate", "Comprehensive communication", "Deep interaction",
                "Professional-level communication", "High-quality communication"
            ]
        },
        'insurance': {
            'approval_actions': [
                "Initial application review", "Basic approval process", "Standard approval procedure", "Detailed approval assessment", "Cautious approval decision"
            ],
            'investigation_actions': [
                "Basic information verification", "Routine investigation", "Detailed background check", "In-depth information verification",
                "Comprehensive investigation", "Professional assessment", "Comprehensive research", "Thorough verification"
            ],
            'negotiation_actions': [
                "Initial negotiation", "Simple negotiation", "Normal negotiation", "In-depth negotiation",
                "Complex negotiation", "Professional negotiation"
            ],
            'communication_actions': [
                "Basic information communication", "Routine business exchange", "Detailed coordination", "In-depth discussion",
                "Professional negotiation", "Comprehensive coordination", "Advanced negotiation", "Expert communication",
                "Comprehensive coordination", "Strategic communication planning", "All-round communication", "Deep cooperation"
            ]
        }
    }
    
    if agent == 'doctor':
        diag_desc = descriptions[agent]['diagnostic_actions'][min(action[0], len(descriptions[agent]['diagnostic_actions'])-1)]
        treat_desc = descriptions[agent]['treatment_actions'][min(action[1], len(descriptions[agent]['treatment_actions'])-1)]
        exam_desc = descriptions[agent]['examination_actions'][min(action[2], len(descriptions[agent]['examination_actions'])-1)]
        comm_desc = descriptions[agent]['communication_actions'][min(action[3], len(descriptions[agent]['communication_actions'])-1)]
        return f"{diag_desc}, {treat_desc}, {exam_desc}, {comm_desc}"
    
    elif agent == 'patient':
        symp_desc = descriptions[agent]['symptom_actions'][min(action[0], len(descriptions[agent]['symptom_actions'])-1)]
        feed_desc = descriptions[agent]['feedback_actions'][min(action[1], len(descriptions[agent]['feedback_actions'])-1)]
        coop_desc = descriptions[agent]['cooperation_actions'][min(action[2], len(descriptions[agent]['cooperation_actions'])-1)]
        comm_desc = descriptions[agent]['communication_actions'][min(action[3], len(descriptions[agent]['communication_actions'])-1)]
        return f"{symp_desc}, {feed_desc}, {coop_desc}, {comm_desc}"
    
    elif agent == 'insurance':
        appr_desc = descriptions[agent]['approval_actions'][min(action[0], len(descriptions[agent]['approval_actions'])-1)]
        inv_desc = descriptions[agent]['investigation_actions'][min(action[1], len(descriptions[agent]['investigation_actions'])-1)]
        neg_desc = descriptions[agent]['negotiation_actions'][min(action[2], len(descriptions[agent]['negotiation_actions'])-1)]
        comm_desc = descriptions[agent]['communication_actions'][min(action[3], len(descriptions[agent]['communication_actions'])-1)]
        return f"{appr_desc}, {inv_desc}, {neg_desc}, {comm_desc}"
    
    return f"Action executed: {action}"

if __name__ == '__main__':
    print("🚀 Starting Multi-Agent Healthcare Collaboration System Web Demo")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing system components...")
    if initialize_components():
        print("✅ System initialized successfully")
        print("🌐 Starting web server...")
        print("📱 Access at: http://localhost:8080")
        print("=" * 60)
        
        # Start Flask app
        import os
        port = int(os.environ.get('PORT', 8080))
        socketio.run(app, debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ System initialization failed, unable to start web service") 