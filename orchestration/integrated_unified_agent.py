"""
Integrated Unified Agent - Version complète sans mocks
Intègre tous les composants réels du système
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Imports réels des composants
from agents.base_agent import BaseAgent, Task, ComponentStatus
from core.resources.resource_manager import get_resource_manager
from intelligence.model_zoo import get_model_zoo
from intelligence.curriculum_manager import get_curriculum_manager
from intelligence.scenario_generator import get_scenario_generator
from intelligence.memory.memory_store import get_memory_store

logger = logging.getLogger(__name__)

class IntegratedUnifiedAgent:
    """Agent unifié intégré avec tous les composants réels"""

    def __init__(self, system_name: str = "UnifiedAI"):
        self.system_name = system_name
        self.agent_id = "integrated_unified_agent"
        self.status = ComponentStatus.INITIALIZING

        # Composants du système (singletons)
        self.resource_manager = get_resource_manager()
        self.model_zoo = get_model_zoo()
        self.curriculum_manager = get_curriculum_manager()
        self.scenario_generator = get_scenario_generator()
        self.memory = get_memory_store()

        # Agents enregistrés
        self.execution_agents: Dict[str, BaseAgent] = {}

        # Historique
        self.task_history = []
        self.performance_log = []

        logger.info(f"IntegratedUnifiedAgent '{system_name}' created")

    async def initialize(self) -> bool:
        """Initialise le système complet"""
        logger.info("Initializing IntegratedUnifiedAgent...")

        try:
            # Démarrer le ResourceManager
            await self.resource_manager.start()

            self.status = ComponentStatus.HEALTHY
            logger.info("✓ IntegratedUnifiedAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            self.status = ComponentStatus.FAILED
            return False

    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Enregistre un agent d'exécution

        Args:
            agent: Instance de BaseAgent

        Returns:
            True si succès
        """
        try:
            # Initialiser l'agent
            success = await agent.initialize()

            if success:
                self.execution_agents[agent.agent_id] = agent
                logger.info(f"✓ Agent registered: {agent.agent_id} ({agent.agent_type})")
                return True
            else:
                logger.error(f"Failed to initialize agent: {agent.agent_id}")
                return False

        except Exception as e:
            logger.error(f"Error registering agent {agent.agent_id}: {e}")
            return False

    async def solve_task(self, task: Task) -> Dict[str, Any]:
        """
        Résout une tâche de bout en bout

        Args:
            task: Tâche à résoudre

        Returns:
            Résultat de l'exécution
        """
        start_time = datetime.now()
        logger.info(f"Starting task: {task.task_id} (type={task.problem_type})")

        try:
            # 1. Obtenir le curriculum actuel
            curriculum = self.curriculum_manager.get_current_curriculum()
            complexity = curriculum['complexity']

            # 2. Générer un scénario adapté
            scenario = await self.scenario_generator.generate_scenario(
                task.problem_type,
                complexity
            )

            # 3. Sélectionner le meilleur modèle
            best_model_info = await self.model_zoo.get_best_model(task.problem_type)

            # 4. Allouer les ressources
            resources_needed = task.resources_required or {'cpu': 10.0, 'memory': 1000.0}
            resources_allocated = await self.resource_manager.allocate(
                task.task_id,
                resources_needed
            )

            if not resources_allocated:
                return {
                    'status': 'failed',
                    'reason': 'insufficient_resources',
                    'task_id': task.task_id
                }

            # 5. Sélectionner et exécuter les agents appropriés
            execution_results = []

            for agent_id, agent in self.execution_agents.items():
                if agent.agent_type == task.problem_type:
                    logger.info(f"Executing with agent: {agent_id}")
                    result = await agent.execute(task)
                    execution_results.append(result)

                    # Enregistrer l'exécution
                    await agent.record_execution(task, result)

            if not execution_results:
                logger.warning(f"No suitable agent found for task type: {task.problem_type}")
                execution_results.append({
                    'status': 'no_agent',
                    'message': f'No agent available for {task.problem_type}'
                })

            # 6. Évaluer la performance
            performance = self._evaluate_performance(execution_results)

            # 7. Mettre à jour le curriculum
            curriculum_decision = await self.curriculum_manager.evaluate_performance(
                performance,
                {'task_id': task.task_id, 'task_type': task.problem_type}
            )

            # 8. Stocker l'expérience en mémoire
            experience = {
                'task_id': task.task_id,
                'task_type': task.problem_type,
                'scenario': scenario,
                'performance': performance,
                'curriculum_level': curriculum['level'],
                'execution_results': execution_results,
                'curriculum_decision': curriculum_decision,
                'timestamp': datetime.now().isoformat()
            }
            await self.memory.store_experience(experience)

            # 9. Libérer les ressources
            await self.resource_manager.release(task.task_id)

            # 10. Enregistrer dans l'historique
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.task_history.append({
                'task': task.to_dict(),
                'performance': performance,
                'elapsed_time': elapsed_time,
                'scenario_id': scenario['scenario_id']
            })
            self.performance_log.append(performance)

            logger.info(f"✓ Task completed: {task.task_id} "
                       f"(performance={performance:.2f}, time={elapsed_time:.2f}s)")

            return {
                'status': 'success',
                'task_id': task.task_id,
                'performance': performance,
                'elapsed_time': elapsed_time,
                'scenario_level': scenario.get('complexity', 0),
                'curriculum_level': curriculum['level'],
                'curriculum_decision': curriculum_decision,
                'execution_results': execution_results
            }

        except Exception as e:
            logger.error(f"Error solving task {task.task_id}: {e}")

            # Libérer les ressources en cas d'erreur
            try:
                await self.resource_manager.release(task.task_id)
            except:
                pass

            return {
                'status': 'error',
                'task_id': task.task_id,
                'error': str(e)
            }

    def _evaluate_performance(self, results: List[Dict[str, Any]]) -> float:
        """Évalue la performance globale"""
        if not results:
            return 0.0

        # Compter les succès
        successes = sum(1 for r in results if r.get('status') == 'success')
        success_rate = successes / len(results)

        # Prendre en compte les métriques de performance si disponibles
        performances = [
            r.get('metrics', {}).get('performance', success_rate)
            for r in results
        ]

        return sum(performances) / len(performances) if performances else success_rate

    async def get_system_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du système"""
        # Statut des agents
        agent_statuses = {}
        for agent_id, agent in self.execution_agents.items():
            metrics = await agent.health_check()
            agent_statuses[agent_id] = metrics.to_dict()

        return {
            'agent_id': self.agent_id,
            'system_name': self.system_name,
            'status': self.status.value,
            'resources': self.resource_manager.get_status(),
            'curriculum': self.curriculum_manager.get_statistics(),
            'memory': self.memory.get_statistics(),
            'model_zoo': self.model_zoo.get_statistics(),
            'scenario_generator': self.scenario_generator.get_statistics(),
            'tasks_completed': len(self.task_history),
            'average_performance': (sum(self.performance_log) / len(self.performance_log))
                                   if self.performance_log else 0.0,
            'agents_registered': len(self.execution_agents),
            'agent_statuses': agent_statuses
        }

    async def optimize_system(self) -> Dict[str, Any]:
        """Optimise le système globalement"""
        logger.info("Running system optimization...")

        optimizations = {
            'timestamp': datetime.now().isoformat(),
            'actions': []
        }

        # 1. Optimiser les allocations de ressources
        resource_recommendations = await self.resource_manager.optimize_allocations()
        if resource_recommendations:
            optimizations['actions'].extend([
                {'source': 'resource_manager', 'recommendation': r}
                for r in resource_recommendations
            ])

        # 2. Analyser les performances récentes
        if len(self.performance_log) >= 10:
            recent_avg = sum(self.performance_log[-10:]) / 10

            if recent_avg < 0.6:
                optimizations['actions'].append({
                    'source': 'performance_analyzer',
                    'type': 'performance_boost',
                    'action': 'Recent performance low - consider adjusting curriculum or agents',
                    'recent_avg': recent_avg
                })

            if recent_avg > 0.9:
                optimizations['actions'].append({
                    'source': 'performance_analyzer',
                    'type': 'efficiency',
                    'action': 'High performance detected - system running optimally',
                    'recent_avg': recent_avg
                })

        # 3. Vérifier la santé des agents
        unhealthy_agents = []
        for agent_id, agent in self.execution_agents.items():
            if agent.status != ComponentStatus.HEALTHY:
                unhealthy_agents.append(agent_id)

        if unhealthy_agents:
            optimizations['actions'].append({
                'source': 'agent_health_monitor',
                'type': 'agent_health',
                'action': f'Unhealthy agents detected: {unhealthy_agents}',
                'agents': unhealthy_agents
            })

        # 4. Nettoyer les caches si nécessaire
        memory_stats = self.memory.get_statistics()
        if memory_stats['total_experiences'] > 50000:
            optimizations['actions'].append({
                'source': 'memory_manager',
                'type': 'cache_cleanup',
                'action': 'Memory size large - consider cleanup',
                'size': memory_stats['total_experiences']
            })

        logger.info(f"System optimization complete: {len(optimizations['actions'])} recommendations")
        return optimizations

    async def shutdown(self) -> bool:
        """Arrête le système proprement"""
        logger.info("Shutting down IntegratedUnifiedAgent...")

        try:
            # Arrêter tous les agents
            for agent_id, agent in self.execution_agents.items():
                await agent.shutdown()
                logger.info(f"  Agent {agent_id} shut down")

            # Arrêter le ResourceManager
            await self.resource_manager.stop()

            self.status = ComponentStatus.SHUTDOWN
            logger.info("✓ IntegratedUnifiedAgent shut down successfully")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
