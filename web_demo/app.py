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
    return jsonify({
        'environment_ready': current_env is not None,
        'simulation_active': current_simulation['active'],
        'current_step': current_simulation['step'],
        'max_steps': current_simulation['max_steps']
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
    """执行智能体动作"""
    global current_simulation
    
    if not current_simulation['active']:
        return jsonify({'success': False, 'error': '模拟未激活'})
    
    try:
        data = request.json
        agent = data.get('agent')
        action = data.get('action', [0, 0, 0, 0])
        
        # 生成智能动作
        if agent == 'auto':
            # 自动运行一步
            agent = current_env.agent_selection
            action = generate_intelligent_action(agent, current_env.current_patient, current_simulation['step'])
        
        # 执行动作
        current_env.step(action)
        
        # 获取奖励和新状态
        reward = current_env.rewards.get(agent, 0)
        
        # 记录历史
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
        
        # 更新性能指标
        current_simulation['performance'] = {
            'treatment_effectiveness': current_env.episode_stats['treatment_effectiveness'],
            'cost_efficiency': current_env.episode_stats['cost_efficiency'],
            'patient_satisfaction': current_env.episode_stats['patient_satisfaction'],
            'communication_quality': current_env.episode_stats['communication_quality']
        }
        
        # 检查是否完成
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
            'active': current_simulation['active']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simulation_data')
def get_simulation_data():
    """获取当前模拟数据"""
    return jsonify(current_simulation)

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """重置模拟环境"""
    global current_simulation
    
    try:
        if current_env:
            # 重置环境
            observations, infos = current_env.reset(seed=np.random.randint(0, 1000))
        
        # 重置模拟状态
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
        
        # 广播重置消息
        socketio.emit('simulation_reset', current_simulation)
        
        return jsonify({'success': True, 'message': '环境已重置'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    emit('connected', {'data': '连接成功'})

@socketio.on('request_auto_step')
def handle_auto_step():
    """处理自动步进请求"""
    if current_simulation['active']:
        # 执行自动动作
        result = execute_action_internal('auto')
        emit('auto_step_result', result)

def execute_action_internal(agent_type):
    """内部动作执行函数"""
    global current_simulation
    
    if not current_simulation['active']:
        return {'success': False, 'error': '模拟未激活'}
    
    try:
        agent = current_env.agent_selection
        action = generate_intelligent_action(agent, current_env.current_patient, current_simulation['step'])
        
        # 执行动作
        current_env.step(action)
        reward = current_env.rewards.get(agent, 0)
        
        # 记录历史
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
        
        # 更新性能指标
        current_simulation['performance'] = {
            'treatment_effectiveness': current_env.episode_stats['treatment_effectiveness'],
            'cost_efficiency': current_env.episode_stats['cost_efficiency'],
            'patient_satisfaction': current_env.episode_stats['patient_satisfaction'],
            'communication_quality': current_env.episode_stats['communication_quality']
        }
        
        # 检查是否完成
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
    """格式化患者数据用于显示"""
    if not patient:
        return None
    
    return {
        'id': patient.subject_id,
        'age': patient.age,
        'gender': '男性' if patient.gender else '女性',
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
    """生成智能动作"""
    if agent == "doctor":
        severity = patient.severity_score if patient else 2.0
        urgency = patient.urgency_level if patient else 2
        
        # 增加随机扰动和阶段性变化
        if step < 3:  # 初期诊断阶段
            return [
                min(14, int(severity * 1.5) + random.randint(0, 2)),
                min(19, int(severity) + random.randint(0, 3)),
                min(9, step + random.randint(1, 3)),
                min(14, urgency + random.randint(0, 2))
            ]
        elif step < 8:  # 深度诊断阶段
            return [
                min(14, int(severity * 2.5) + random.randint(-1, 2)),
                min(19, int(severity * 1.5) + urgency + random.randint(0, 2)),
                min(9, int(severity * 1.5) + random.randint(0, 2)),
                min(14, urgency * 2 + random.randint(-1, 1))
            ]
        elif step < 15:  # 治疗阶段
            return [
                min(14, int(severity * 2) + random.randint(0, 3)),
                min(19, int(severity * 3) + urgency + random.randint(-1, 2)),
                min(9, int(severity * 2) + random.randint(0, 2)),
                min(14, urgency * 2 + step % 3 + random.randint(0, 1))
            ]
        else:  # 监护观察阶段
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
        
        # 患者状态会随时间和治疗效果变化
        if step < 3:  # 初期焦虑不安
            return [
                min(9, pain_level + random.randint(0, 2)),
                min(7, 2 + random.randint(0, 2)),
                min(4, 1 + random.randint(0, 1)),
                min(9, anxiety + random.randint(1, 2))
            ]
        elif step < 8:  # 逐渐适应配合
            return [
                min(9, int(severity * 1.5) + random.randint(0, 1)),
                min(7, 3 + step % 3 + random.randint(0, 1)),
                min(4, 2 + step // 2 + random.randint(0, 1)),
                min(9, int(severity) + random.randint(0, 2))
            ]
        elif step < 15:  # 积极配合治疗
            return [
                min(9, int(severity * 2) + random.randint(-1, 1)),
                min(7, 4 + step % 2 + random.randint(0, 1)),
                min(4, 3 + random.randint(0, 1)),
                min(9, int(severity * 1.2) + random.randint(0, 1))
            ]
        else:  # 恢复期反馈
            return [
                min(9, int(severity * 1.2) + random.randint(0, 1)),
                min(7, 5 + random.randint(0, 1)),
                min(4, 3 + random.randint(0, 1)),
                min(9, max(1, int(severity) - step // 5) + random.randint(0, 1))
            ]
    
    elif agent == "insurance":
        cost = patient.estimated_treatment_cost if patient else 15000
        coverage = patient.insurance_coverage if patient else 0.8
        
        # 保险审批的严格程度和调查深度
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
        
        # 不同阶段的审批策略
        if step < 4:  # 初步审核
            return [
                min(4, base_approval + random.randint(0, 1)),
                min(7, investigation_level + random.randint(0, 1)),
                min(5, int((1 - coverage) * 3) + random.randint(0, 1)),
                min(11, 1 + step + random.randint(0, 1))
            ]
        elif step < 10:  # 详细评估
            return [
                min(4, base_approval + random.randint(-1, 1)),
                min(7, investigation_level + step // 2 + random.randint(0, 1)),
                min(5, int((1 - coverage) * 4) + random.randint(0, 1)),
                min(11, 3 + step % 3 + random.randint(0, 1))
            ]
        else:  # 最终决策
            return [
                min(4, base_approval + random.randint(0, 1)),
                min(7, investigation_level + random.randint(0, 2)),
                min(5, int((1 - coverage) * 3) + random.randint(0, 1)),
                min(11, 2 + step % 4 + random.randint(0, 1))
            ]
    
    return [0, 0, 0, 0]

def get_action_description(agent: str, action: List[int]) -> str:
    """获取动作描述"""
    descriptions = {
        'doctor': {
            'action_types': ['诊断', '治疗', '检查', '沟通'],
            'diagnostic_actions': [
                "初步问诊了解病情", "详细询问症状历史", "进行体格检查", "分析化验结果",
                "查阅病历资料", "评估病情严重程度", "制定诊断计划", "进行专业诊断",
                "会诊讨论病情", "确认诊断结果", "更新诊断记录", "调整诊断方案",
                "深度诊断分析", "综合评估病情", "专家级诊断"
            ],
            'treatment_actions': [
                "观察病情变化", "开具基础药物", "调整用药方案", "实施保守治疗",
                "进行物理治疗", "安排康复训练", "制定治疗计划", "实施标准治疗",
                "调整治疗强度", "进行专业治疗", "实施综合治疗", "优化治疗方案",
                "进行高级治疗", "实施精准治疗", "执行积极治疗", "进行创新治疗",
                "实施抢救治疗", "进行紧急处置", "执行专家治疗", "实施最优治疗"
            ],
            'examination_actions': [
                "基础生命体征检查", "常规血液检查", "详细检查评估",
                "影像学检查", "专项功能检查", "全面体检评估",
                "深度检查分析", "高级检查项目", "精密仪器检查", "专家级检查"
            ],
            'communication_actions': [
                "简单交流病情", "了解患者感受", "解释治疗方案", "详细病情沟通",
                "安慰鼓励患者", "教育健康知识", "深入交流讨论", "全面沟通协调",
                "专业指导建议", "耐心解答疑问", "详细解释风险", "深度心理疏导",
                "综合沟通协调", "专家级咨询", "全方位交流"
            ]
        },
        'patient': {
            'symptom_actions': [
                "简单描述不适", "说明主要症状", "详细叙述病情", "全面描述症状",
                "补充症状细节", "更新症状变化", "深入描述感受", "完整症状报告",
                "精确描述症状", "全面症状汇报"
            ],
            'feedback_actions': [
                "表达基本感受", "一般性反馈", "详细反馈感受", "积极表达意见",
                "主动提供反馈", "非常积极配合", "深度反馈交流", "全面反馈评价"
            ],
            'cooperation_actions': [
                "基本配合治疗", "部分配合医嘱", "良好配合治疗", "完全配合治疗", "积极主动配合"
            ],
            'communication_actions': [
                "简单回应医生", "基本交流沟通", "详细交流病情", "深入沟通讨论",
                "主动询问疑问", "积极参与交流", "全面沟通协调", "深度互动交流",
                "专业水平交流", "高质量沟通"
            ]
        },
        'insurance': {
            'approval_actions': [
                "初步审核申请", "基础审批流程", "标准审批程序", "详细审批评估", "谨慎审批决策"
            ],
            'investigation_actions': [
                "基础信息核实", "常规调查验证", "详细背景调查", "深度信息核查",
                "全面调查分析", "专业调查评估", "综合调查研究", "精细调查核实"
            ],
            'negotiation_actions': [
                "初步协商讨论", "简单协商沟通", "正常协商谈判", "深度协商讨论",
                "复杂协商处理", "专业协商谈判"
            ],
            'communication_actions': [
                "基础信息沟通", "常规业务交流", "详细沟通协调", "深入交流讨论",
                "专业沟通协商", "全面沟通协调", "高级沟通谈判", "专家级沟通",
                "综合沟通协调", "战略沟通规划", "全方位沟通", "深度沟通合作"
            ]
        }
    }
    
    if agent == 'doctor':
        diag_desc = descriptions[agent]['diagnostic_actions'][min(action[0], len(descriptions[agent]['diagnostic_actions'])-1)]
        treat_desc = descriptions[agent]['treatment_actions'][min(action[1], len(descriptions[agent]['treatment_actions'])-1)]
        exam_desc = descriptions[agent]['examination_actions'][min(action[2], len(descriptions[agent]['examination_actions'])-1)]
        comm_desc = descriptions[agent]['communication_actions'][min(action[3], len(descriptions[agent]['communication_actions'])-1)]
        return f"{diag_desc}，{treat_desc}，{exam_desc}，{comm_desc}"
    
    elif agent == 'patient':
        symp_desc = descriptions[agent]['symptom_actions'][min(action[0], len(descriptions[agent]['symptom_actions'])-1)]
        feed_desc = descriptions[agent]['feedback_actions'][min(action[1], len(descriptions[agent]['feedback_actions'])-1)]
        coop_desc = descriptions[agent]['cooperation_actions'][min(action[2], len(descriptions[agent]['cooperation_actions'])-1)]
        comm_desc = descriptions[agent]['communication_actions'][min(action[3], len(descriptions[agent]['communication_actions'])-1)]
        return f"{symp_desc}，{feed_desc}，{coop_desc}，{comm_desc}"
    
    elif agent == 'insurance':
        appr_desc = descriptions[agent]['approval_actions'][min(action[0], len(descriptions[agent]['approval_actions'])-1)]
        inv_desc = descriptions[agent]['investigation_actions'][min(action[1], len(descriptions[agent]['investigation_actions'])-1)]
        neg_desc = descriptions[agent]['negotiation_actions'][min(action[2], len(descriptions[agent]['negotiation_actions'])-1)]
        comm_desc = descriptions[agent]['communication_actions'][min(action[3], len(descriptions[agent]['communication_actions'])-1)]
        return f"{appr_desc}，{inv_desc}，{neg_desc}，{comm_desc}"
    
    return f"执行动作: {action}"

if __name__ == '__main__':
    print("🚀 启动多智能体医疗协作系统网页演示")
    print("=" * 60)
    
    # 初始化组件
    print("🔧 初始化系统组件...")
    if initialize_components():
        print("✅ 系统初始化完成")
        print("🌐 启动Web服务器...")
        print("📱 访问地址: http://localhost:8080")
        print("=" * 60)
        
        # 启动Flask应用
        socketio.run(app, debug=True, host='0.0.0.0', port=8080)
    else:
        print("❌ 系统初始化失败，无法启动Web服务") 