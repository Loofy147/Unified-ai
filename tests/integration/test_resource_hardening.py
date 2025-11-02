import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pytest
from core.resources.resource_manager import ResourceManager

@pytest.mark.asyncio
async def test_resource_over_allocation_prevented():
    rm = ResourceManager()

    # Try to allocate 10x available resources
    results = []
    for i in range(10):
        result = await rm.allocate(f"test_{i}", {'cpu': 20.0})
        results.append(result)

    # Should only succeed for 5 (100/20 = 5)
    assert sum(results) == 5
    assert rm.resources['cpu']['available'] == 0
