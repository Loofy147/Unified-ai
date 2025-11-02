
import pytest
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intelligence.model_zoo import get_model_zoo
from orchestration.integrated_unified_agent import IntegratedUnifiedAgent
from agents.base_agent import Task
from core.config import settings

@pytest.fixture(autouse=True)
def isolated_singletons(monkeypatch, tmp_path):
    """
    Fixture to ensure that singleton instances are re-created for each test
    with a temporary, isolated configuration.
    """
    # Use a temporary directory for the model zoo storage
    test_model_path = tmp_path / "models"
    monkeypatch.setattr(settings.model_zoo, 'storage_path', str(test_model_path))

    # Force the singletons to be re-created on their next call
    get_model_zoo(force_reload=True)

    yield

    # Teardown is handled automatically by pytest's tmp_path and monkeypatch

@pytest.mark.asyncio
async def test_model_zoo_rejects_malicious_model():
    """
    Verifies that the ModelZoo correctly rejects a model with a __reduce__
    method, preventing a potential deserialization attack.
    """
    model_zoo = get_model_zoo()

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

class SimpleTestModel:
    def __init__(self, value):
        self.value = value
    def predict(self, x):
        return self.value * x

@pytest.mark.asyncio
async def test_model_zoo_can_store_and_retrieve_complex_object():
    """
    Verifies that the ModelZoo can correctly serialize and deserialize a
    non-trivial Python object using dill.
    """
    model_zoo = get_model_zoo()

    # 1. Create and register a complex object
    original_model = SimpleTestModel(value=42)
    model_id = "complex_test_model"
    version = await model_zoo.register_model(model_id, original_model, {"type": "test"})

    assert version == 1

    # 2. Retrieve the model
    # To ensure it's loaded from disk, we'll force-reload the singleton
    retrieved_model = await get_model_zoo(force_reload=True).get_model(model_id, version)

    # 3. Verify the retrieved object
    assert retrieved_model is not None
    assert isinstance(retrieved_model, SimpleTestModel)
    assert retrieved_model.value == 42
    assert retrieved_model.predict(2) == 84
    assert retrieved_model is not original_model # Should be a new instance

@pytest.mark.asyncio
async def test_model_zoo_delete_model_removes_files():
    """
    Verifies that deleting a model version also removes its corresponding
    .dill file from the disk.
    """
    model_zoo = get_model_zoo()
    model_id = "deletable_model"

    # 1. Register a model, which should create files on disk
    model = SimpleTestModel(100)
    version = await model_zoo.register_model(model_id, model, {})

    model_path = Path(settings.model_zoo.storage_path) / model_id
    model_file = model_path / f"v{version}.dill"
    metadata_file = model_path / f"v{version}_metadata.json"

    assert model_file.exists()
    assert metadata_file.exists()

    # 2. Delete the model
    await model_zoo.delete_model(model_id, version)

    # 3. Verify that the files have been removed
    assert not model_file.exists()
    assert not metadata_file.exists()
