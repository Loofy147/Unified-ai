
import pytest
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intelligence.model_zoo import get_model_zoo, ModelZoo
from orchestration.integrated_unified_agent import IntegratedUnifiedAgent
from agents.base_agent import Task

# Fixture to get a clean ModelZoo instance for each test
@pytest.fixture
def model_zoo():
    # We need to create a new instance for testing purposes to avoid state leakage
    # between tests, especially since the singleton is stateful.
    zoo = ModelZoo(storage_path="models_test")
    yield zoo
    # Teardown: clean up the test models directory
    import shutil
    shutil.rmtree("models_test", ignore_errors=True)

@pytest.mark.asyncio
async def test_model_zoo_rejects_malicious_model(model_zoo: ModelZoo):
    """
    Verifies that the ModelZoo correctly rejects a model with a __reduce__
    method, preventing a potential deserialization attack.
    """
    class MaliciousModel:
        def __reduce__(self):
            import os
            return (os.system, ('echo "exploited"',))

    malicious_model = MaliciousModel()

    # Expect a TypeError because the model is unsafe
    with pytest.raises(TypeError, match="Cannot register models with __reduce__ method for security reasons."):
        await model_zoo.register_model("malicious_model", malicious_model, {})

@pytest.mark.asyncio
async def test_byzantine_agent_performance_is_ignored():
    """
    Verifies that the system correctly identifies and ignores anomalous
    performance reports from a Byzantine (malicious) agent.
    """
    system = IntegratedUnifiedAgent()
    await system.initialize()

    class ByzantineAgent:
        def __init__(self):
            self.agent_id = "byzantine_agent"
            self.agent_type = "optimization"
            self.status = "healthy"
        async def initialize(self): return True
        async def execute(self, task):
            # Report an impossibly high performance metric
            return {'status': 'success', 'metrics': {'performance': 999.0}}
        async def health_check(self): return {'status': 'healthy'}
        async def shutdown(self): return True
        async def record_execution(self, task, result): pass

    await system.register_agent(ByzantineAgent())

    task = Task(
        task_id="byzantine_test_task",
        problem_type="optimization",
        description="Test task for Byzantine agent.",
        data_source="synthetic",
        target_metric="accuracy"
    )

    result = await system.solve_task(task)

    # The overall performance should be 0.0 because the only valid report
    # was discarded. The task status is still success because an agent ran.
    assert result['status'] == 'success'
    assert result['performance'] == 0.0, "System should have ignored the invalid performance score."

    await system.shutdown()
