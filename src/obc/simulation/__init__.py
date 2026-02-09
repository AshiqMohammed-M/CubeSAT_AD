"""
OBC System Simulation Package
System Testing and Validation
"""

from .obc_simulate_incoming_data import create_sample_mcu_message, simulate_critical_anomaly, test_obc_system
from .obc_realtime_fusion_test import RealAI_FusionTest

__all__ = [
    'create_sample_mcu_message',
    'simulate_critical_anomaly', 
    'test_obc_system',
    'RealAI_FusionTest'
]
__version__ = '1.0.0'