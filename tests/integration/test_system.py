"""
Tests d'Intégration - Système Complet
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestration.integrated_unified_agent import IntegratedUnifiedAgent
from agents.base_agent import BaseAgent, Task, ComponentStatus

# ============================================================================
# MOCK AGENTS POUR TESTS
# ============================================================================

class MockOptimizationAgent(BaseAgent):
    """Agent de test pour l'optimisation"""

    def __init__(self):
        super().__init__("mock_optimization_agent", "optimization")

    async def initialize(self) -> bool:
        self.status = ComponentStatus.HEALTHY
        return True

    async def execute(self, task: Task) -> dict:
        await asyncio.sleep(0.1)
        return {
            'status': 'success',
            'agent_id': self.agent_id,
            'metrics': {'performance': 0.85, 'iterations': 100}
        }

    async def shutdown(self) -> bool:
        return True

class MockRLAgent(BaseAgent):
    """Agent de test pour RL"""

    def __init__(self):
        super().__init__("mock_rl_agent", "rl_control")

    async def initialize(self) -> bool:
        self.status = ComponentStatus.HEALTHY
        return True

    async def execute(self, task: Task) -> dict:
        await asyncio.sleep(0.15)
        return {
            'status': 'success',
            'agent_id': self.agent_id,
            'metrics': {'performance': 0.92, 'episodes': 1000}
        }

    async def shutdown(self) -> bool:
        return True

# ============================================================================
# TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_system_initialization():
    """Test de l'initialisation du système"""
    system = IntegratedUnifiedAgent("TestSystem")

    success = await system.initialize()
    assert success == True
    assert system.status == ComponentStatus.HEALTHY

    await system.shutdown()

@pytest.mark.asyncio
async def test_agent_registration():
    """Test de l'enregistrement d'agents"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    # Enregistrer des agents
    opt_agent = MockOptimizationAgent()
    rl_agent = MockRLAgent()

    success1 = await system.register_agent(opt_agent)
    success2 = await system.register_agent(rl_agent)

    assert success1 == True
    assert success2 == True
    assert len(system.execution_agents) == 2

    await system.shutdown()

@pytest.mark.asyncio
async def test_task_execution():
    """Test de l'exécution d'une tâche"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    # Enregistrer un agent
    opt_agent = MockOptimizationAgent()
    await system.register_agent(opt_agent)

    # Créer une tâche
    task = Task(
        task_id="test_001",
        problem_type="optimization",
        description="Test optimization task",
        data_source="test_data",
        target_metric="accuracy"
    )

    # Exécuter
    result = await system.solve_task(task)

    assert result['status'] == 'success'
    assert 'performance' in result
    assert result['performance'] > 0.0

    await system.shutdown()

@pytest.mark.asyncio
async def test_curriculum_progression():
    """Test de la progression du curriculum"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    opt_agent = MockOptimizationAgent()
    await system.register_agent(opt_agent)

    initial_level = system.curriculum_manager.current_level

    # Exécuter plusieurs tâches avec bonnes performances
    for i in range(15):
        task = Task(
            task_id=f"test_{i:03d}",
            problem_type="optimization",
            description=f"Test task {i}",
            data_source="test_data",
            target_metric="accuracy"
        )

        result = await system.solve_task(task)
        assert result['status'] == 'success'

    # Vérifier progression
    final_level = system.curriculum_manager.current_level
    assert final_level >= initial_level

    await system.shutdown()

@pytest.mark.asyncio
async def test_resource_management():
    """Test de la gestion des ressources"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    opt_agent = MockOptimizationAgent()
    await system.register_agent(opt_agent)

    # Statut initial des ressources
    initial_status = system.resource_manager.get_status()
    assert initial_status['cpu']['utilization'] == 0.0

    # Exécuter une tâche
    task = Task(
        task_id="test_resource",
        problem_type="optimization",
        description="Test resource task",
        data_source="test_data",
        target_metric="accuracy",
        resources_required={'cpu': 20.0, 'memory': 2000.0}
    )

    result = await system.solve_task(task)

    # Les ressources devraient être libérées après exécution
    final_status = system.resource_manager.get_status()
    assert final_status['cpu']['utilization'] == 0.0

    await system.shutdown()

@pytest.mark.asyncio
async def test_memory_storage():
    """Test du stockage en mémoire"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    opt_agent = MockOptimizationAgent()
    await system.register_agent(opt_agent)

    initial_memory = system.memory.get_statistics()

    # Exécuter une tâche
    task = Task(
        task_id="test_memory",
        problem_type="optimization",
        description="Test memory task",
        data_source="test_data",
        target_metric="accuracy"
    )

    await system.solve_task(task)

    # Vérifier que l'expérience est stockée
    final_memory = system.memory.get_statistics()
    assert final_memory['total_experiences'] > initial_memory['total_experiences']

    await system.shutdown()

@pytest.mark.asyncio
async def test_system_status():
    """Test du statut système"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    opt_agent = MockOptimizationAgent()
    await system.register_agent(opt_agent)

    status = await system.get_system_status()

    assert 'agent_id' in status
    assert 'resources' in status
    assert 'curriculum' in status
    assert 'memory' in status
    assert 'model_zoo' in status
    assert 'agents_registered' in status
    assert status['agents_registered'] == 1

    await system.shutdown()

@pytest.mark.asyncio
async def test_system_optimization():
    """Test de l'optimisation système"""
    system = IntegratedUnifiedAgent("TestSystem")
    await system.initialize()

    opt_agent = MockOptimizationAgent()
    await system.register_agent(opt_agent)

    # Exécuter quelques tâches
    for i in range(5):
        task = Task(
            task_id=f"opt_test_{i}",
            problem_type="optimization",
            description=f"Optimization test {i}",
            data_source="test_data",
            target_metric="accuracy"
        )
        await system.solve_task(task)

    # Optimiser
    optimization_result = await system.optimize_system()

    assert 'timestamp' in optimization_result
    assert 'actions' in optimization_result
    assert isinstance(optimization_result['actions'], list)

    await system.shutdown()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])
