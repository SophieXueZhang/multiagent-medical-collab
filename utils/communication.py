"""
智能体间通信协议模块

定义医生、患者、保险审核员之间的通信规范和消息格式
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import time
import json
import numpy as np

class AgentRole(Enum):
    """智能体角色枚举"""
    DOCTOR = "doctor"
    PATIENT = "patient"
    INSURANCE = "insurance"

class MessageType(Enum):
    """消息类型枚举"""
    # 患者 -> 医生
    SYMPTOM_REPORT = "symptom_report"
    FEEDBACK = "feedback"
    PAIN_LEVEL = "pain_level"
    
    # 医生 -> 患者
    DIAGNOSIS = "diagnosis"
    TREATMENT_PLAN = "treatment_plan"
    QUESTION = "question"
    
    # 医生 -> 保险
    TREATMENT_REQUEST = "treatment_request"
    COST_ESTIMATE = "cost_estimate"
    URGENCY_CLAIM = "urgency_claim"
    
    # 保险 -> 医生
    APPROVAL = "approval"
    REJECTION = "rejection"
    REQUEST_INFO = "request_info"
    
    # 系统消息
    SYSTEM_ALERT = "system_alert"
    EPISODE_END = "episode_end"

class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Message:
    """通信消息类"""
    sender: AgentRole
    receiver: AgentRole
    message_type: MessageType
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = None
    message_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.message_id is None:
            self.message_id = f"{self.sender.value}_{self.receiver.value}_{int(self.timestamp * 1000)}"
    
    def to_vector(self, max_content_size: int = 20) -> np.ndarray:
        """将消息转换为向量表示，用于智能体观察"""
        vector = []
        
        # 发送者编码 (one-hot)
        sender_encoding = [0, 0, 0]
        sender_encoding[list(AgentRole).index(self.sender)] = 1
        vector.extend(sender_encoding)
        
        # 接收者编码 (one-hot)
        receiver_encoding = [0, 0, 0]
        receiver_encoding[list(AgentRole).index(self.receiver)] = 1
        vector.extend(receiver_encoding)
        
        # 消息类型编码 (one-hot)
        msg_type_encoding = [0] * len(MessageType)
        msg_type_encoding[list(MessageType).index(self.message_type)] = 1
        vector.extend(msg_type_encoding)
        
        # 优先级编码
        vector.append(self.priority.value / 4.0)  # 归一化到[0,1]
        
        # 内容向量化 (简化实现)
        content_vector = self._vectorize_content(max_content_size)
        vector.extend(content_vector)
        
        return np.array(vector, dtype=np.float32)
    
    def _vectorize_content(self, max_size: int) -> List[float]:
        """将消息内容向量化"""
        # 简化实现：将内容转换为固定长度的数值向量
        content_str = json.dumps(self.content, sort_keys=True)
        
        # 使用哈希值生成伪随机向量 (实际项目中应该用更复杂的编码)
        hash_value = hash(content_str)
        np.random.seed(abs(hash_value) % (2**31))
        vector = np.random.rand(max_size).tolist()
        
        return vector

class CommunicationProtocol:
    """通信协议管理器"""
    
    def __init__(self):
        self.message_history: List[Message] = []
        self.agent_inboxes: Dict[AgentRole, List[Message]] = {
            role: [] for role in AgentRole
        }
        self.communication_rules = self._setup_communication_rules()
        self.max_message_per_step = 3  # 每步最多发送3条消息
        self.message_cost = 0.1  # 发送消息的成本
        
    def _setup_communication_rules(self) -> Dict[str, Dict]:
        """设置通信规则"""
        return {
            # 患者可以向医生发送的消息类型
            f"{AgentRole.PATIENT.value}_{AgentRole.DOCTOR.value}": {
                "allowed_types": [
                    MessageType.SYMPTOM_REPORT,
                    MessageType.FEEDBACK,
                    MessageType.PAIN_LEVEL
                ],
                "max_per_step": 2,
                "cost": 0.05
            },
            
            # 医生可以向患者发送的消息类型
            f"{AgentRole.DOCTOR.value}_{AgentRole.PATIENT.value}": {
                "allowed_types": [
                    MessageType.DIAGNOSIS,
                    MessageType.TREATMENT_PLAN,
                    MessageType.QUESTION
                ],
                "max_per_step": 2,
                "cost": 0.1
            },
            
            # 医生可以向保险发送的消息类型
            f"{AgentRole.DOCTOR.value}_{AgentRole.INSURANCE.value}": {
                "allowed_types": [
                    MessageType.TREATMENT_REQUEST,
                    MessageType.COST_ESTIMATE,
                    MessageType.URGENCY_CLAIM
                ],
                "max_per_step": 1,
                "cost": 0.2
            },
            
            # 保险可以向医生发送的消息类型
            f"{AgentRole.INSURANCE.value}_{AgentRole.DOCTOR.value}": {
                "allowed_types": [
                    MessageType.APPROVAL,
                    MessageType.REJECTION,
                    MessageType.REQUEST_INFO
                ],
                "max_per_step": 1,
                "cost": 0.15
            }
        }
    
    def can_send_message(self, sender: AgentRole, receiver: AgentRole, 
                        message_type: MessageType, step_messages: int = 0) -> bool:
        """检查是否允许发送消息"""
        rule_key = f"{sender.value}_{receiver.value}"
        
        if rule_key not in self.communication_rules:
            return False
            
        rules = self.communication_rules[rule_key]
        
        # 检查消息类型是否允许
        if message_type not in rules["allowed_types"]:
            return False
            
        # 检查步骤内消息数量限制
        if step_messages >= rules["max_per_step"]:
            return False
            
        return True
    
    def send_message(self, message: Message) -> bool:
        """发送消息"""
        # 验证消息合法性
        if not self.can_send_message(message.sender, message.receiver, message.message_type):
            return False
        
        # 添加到历史记录
        self.message_history.append(message)
        
        # 添加到接收者收件箱
        self.agent_inboxes[message.receiver].append(message)
        
        return True
    
    def get_messages(self, receiver: AgentRole, clear_inbox: bool = True) -> List[Message]:
        """获取指定智能体的消息"""
        messages = self.agent_inboxes[receiver].copy()
        
        if clear_inbox:
            self.agent_inboxes[receiver].clear()
            
        return messages
    
    def get_communication_cost(self, sender: AgentRole, receiver: AgentRole) -> float:
        """获取通信成本"""
        rule_key = f"{sender.value}_{receiver.value}"
        
        if rule_key in self.communication_rules:
            return self.communication_rules[rule_key]["cost"]
        
        return self.message_cost
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """获取最近的消息"""
        return self.message_history[-count:] if self.message_history else []
    
    def clear_history(self):
        """清空消息历史"""
        self.message_history.clear()
        for inbox in self.agent_inboxes.values():
            inbox.clear()

class MessageBuilder:
    """消息构建器 - 简化消息创建过程"""
    
    @staticmethod
    def symptom_report(symptoms: Dict[str, float], severity: float) -> Dict[str, Any]:
        """构建症状报告消息内容"""
        return {
            "symptoms": symptoms,
            "severity": severity,
            "description": "Patient symptom report"
        }
    
    @staticmethod
    def diagnosis(diagnosis_code: str, confidence: float, description: str) -> Dict[str, Any]:
        """构建诊断消息内容"""
        return {
            "diagnosis_code": diagnosis_code,
            "confidence": confidence,
            "description": description,
            "timestamp": time.time()
        }
    
    @staticmethod
    def treatment_request(treatment_code: str, cost: float, urgency: int) -> Dict[str, Any]:
        """构建治疗请求消息内容"""
        return {
            "treatment_code": treatment_code,
            "estimated_cost": cost,
            "urgency_level": urgency,
            "justification": "Medical necessity"
        }
    
    @staticmethod
    def approval_response(approved: bool, coverage_percent: float, 
                         conditions: List[str] = None) -> Dict[str, Any]:
        """构建审批回复消息内容"""
        return {
            "approved": approved,
            "coverage_percent": coverage_percent,
            "conditions": conditions or [],
            "processing_time": time.time()
        }

# 工厂函数
def create_communication_protocol() -> CommunicationProtocol:
    """创建通信协议实例"""
    return CommunicationProtocol()

# 常用消息模板
COMMON_MESSAGES = {
    "emergency": {
        "type": MessageType.URGENCY_CLAIM,
        "priority": MessagePriority.URGENT,
        "template": {"urgency_level": 4, "requires_immediate_attention": True}
    },
    "routine_checkup": {
        "type": MessageType.TREATMENT_REQUEST,
        "priority": MessagePriority.NORMAL,
        "template": {"treatment_type": "routine", "urgency_level": 1}
    },
    "insurance_denial": {
        "type": MessageType.REJECTION,
        "priority": MessagePriority.HIGH,
        "template": {"approved": False, "reason": "insufficient_documentation"}
    }
}

print("✅ 通信协议模块创建完成") 