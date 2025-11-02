import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from intelligence.curriculum_manager import CurriculumManager

@pytest.mark.asyncio
async def test_curriculum_gaming_prevented():
    cm = CurriculumManager()

    initial_level = cm.current_level

    # Try to game with fake high performance
    for i in range(20):
        result = await cm.evaluate_performance(0.99, {})

    # Should NOT advance due to validation failures
    assert cm.current_level == initial_level
    assert 'validation_failed' in str(result)
