import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agents.base_agent import AgentValidator

@pytest.mark.asyncio
async def test_byzantine_agent_detection():
    validator = AgentValidator()

    # Good output
    good_output = {'metrics': {'performance': 0.8, 'time': 0.5}}
    validation = validator.validate_output('agent_1', good_output, {})
    assert validation['valid'] == True

    # Malicious output (impossible performance)
    malicious_output = {'metrics': {'performance': 0.999, 'time': 0.0001}}
    validation = validator.validate_output('agent_1', malicious_output, {})
    assert validation['valid'] == False
    assert validation['reason'] == 'impossible_performance_time_combination'
