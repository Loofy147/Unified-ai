
import pytest
import asyncio
from pathlib import Path
import sys

# Add project root to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.integrated_unified_agent import IntegratedUnifiedAgent
from agents.base_agent import Task
from core.monitoring.monitoring_service import monitoring_service

@pytest.fixture(autouse=True)
def reset_monitoring_data():
    """Ensure the monitoring service is clean before each test."""
    monitoring_service.reset()

@pytest.mark.asyncio
async def test_observability_integration_on_task_solve():
    """
    Tests that solving a task correctly updates the MonitoringService.
    """
    # 1. Setup
    agent = IntegratedUnifiedAgent(system_name="TestObservabilityAgent")
    await agent.initialize()

    # Create a simple task
    task = Task(
        task_id="obs_test_task_01",
        problem_type="optimization",
        description="A test task for observability.",
        data_source="synthetic",
        target_metric="accuracy"
    )

    # 2. Pre-execution check
    initial_summary = agent.get_metrics_summary()
    assert "tasks.received" not in initial_summary["counters"]
    assert "tasks.succeeded" not in initial_summary["counters"]
    assert "task.solve.latency_seconds" not in initial_summary["latencies"]

    # 3. Execution
    result = await agent.solve_task(task)
    assert result['status'] == 'success'

    # 4. Post-execution verification
    final_summary = agent.get_metrics_summary()

    # Verify counters
    assert "counters" in final_summary
    assert final_summary["counters"].get("tasks.received") == 1.0
    assert final_summary["counters"].get("tasks.succeeded") == 1.0
    assert "tasks.failed" not in final_summary["counters"] # Should not be present

    # Verify latencies
    assert "latencies" in final_summary
    latency_data = final_summary["latencies"].get("task.solve.latency_seconds")
    assert latency_data is not None
    assert latency_data["count"] == 1
    assert latency_data["avg_seconds"] > 0
    assert latency_data["avg_seconds"] == pytest.approx(result['elapsed_time'], abs=1e-2)

    # 5. Shutdown
    await agent.shutdown()
