"""
Test suite for the Red Team Adversarial Testing Framework.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from red_team_framework import RedTeamOrchestrator

# Mock system for testing purposes
class MockSystem:
    def __init__(self):
        from intelligence.curriculum_manager import CurriculumManager
        from intelligence.memory.memory_store import MemoryStore
        from core.resources.resource_manager import ResourceManager

        self.curriculum_manager = CurriculumManager()
        self.memory = MemoryStore()
        # Provide default values for the mock resource manager
        self.resource_manager = ResourceManager(max_cpu=100.0, max_memory_mb=4096.0)
        self.agents = {}

    async def register_agent(self, agent):
        self.agents[agent.agent_id] = agent

    async def solve_task(self, task):
        # Simulate task solving
        if "cascade" in task.task_id:
            # Fail for the cascade failure test
            return {'status': 'failed', 'error': 'Intentional failure'}
        return {'status': 'success', 'performance': 0.9}

    async def get_system_status(self):
        return {
            'status': 'healthy',
            'agents_registered': len(self.agents),
            'curriculum': self.curriculum_manager.get_statistics(),
            'memory': self.memory.get_statistics(),
            'tasks_completed': 0,
            'average_performance': 0.0
        }

@pytest.mark.asyncio
async def test_red_team_campaign():
    """
    Test that the full red team campaign runs without crashing and generates a report.
    """
    system = MockSystem()
    red_team = RedTeamOrchestrator()

    # Run the campaign
    report = await red_team.run_full_campaign(system)

    # Assert that a report was generated
    assert report is not None
    assert 'total_attacks' in report
    assert 'overall_security_score' in report
    assert isinstance(report['overall_security_score'], float)
    assert report['total_attacks'] == len(red_team.attack_suite)
