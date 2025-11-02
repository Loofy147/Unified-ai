"""
Base Agent - Classe abstraite pour tous les agents
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    """États des composants"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"

@dataclass
class ComponentMetrics:
    """Métriques standardisées pour tous les composants"""
    component_id: str
    status: ComponentStatus
    performance_score: float
    latency_ms: float
    error_count: int
    success_rate: float
    resource_usage: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'status': self.status.value,
            'performance_score': self.performance_score,
            'latency_ms': self.latency_ms,
            'error_count': self.error_count,
            'success_rate': self.success_rate,
            'resource_usage': self.resource_usage,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

@dataclass
class Task:
    """Représentation unifiée d'une tâche"""
    task_id: str
    problem_type: str
    description: str
    data_source: str
    target_metric: str
    priority: int = 1
    deadline: Optional[str] = None
    resources_required: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'problem_type': self.problem_type,
            'description': self.description,
            'data_source': self.data_source,
            'target_metric': self.target_metric,
            'priority': self.priority,
            'deadline': self.deadline,
            'resources_required': self.resources_required,
            'context': self.context,
            'status': self.status
        }

class BaseAgent(ABC):
    """Classe de base abstraite pour tous les agents"""

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = ComponentStatus.INITIALIZING

        self.metrics = ComponentMetrics(
            component_id=agent_id,
            status=self.status,
            performance_score=0.0,
            latency_ms=0.0,
            error_count=0,
            success_rate=0.0,
            resource_usage={},
            timestamp=datetime.now().isoformat()
        )

        self.execution_history = []
        self.total_executions = 0
        self.successful_executions = 0

        logger.info(f"BaseAgent {agent_id} ({agent_type}) created")

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialise l'agent - À implémenter par les sous-classes"""
        pass

    @abstractmethod
    async def execute(self, task: Task) -> Dict[str, Any]:
        """Exécute une tâche - À implémenter par les sous-classes"""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Arrête l'agent - À implémenter par les sous-classes"""
        pass

    async def health_check(self) -> ComponentMetrics:
        """Vérifie la santé de l'agent"""
        self.metrics.timestamp = datetime.now().isoformat()
        self.metrics.status = self.status

        # Calculer le taux de succès
        if self.total_executions > 0:
            self.metrics.success_rate = self.successful_executions / self.total_executions

        return self.metrics

    async def update_metrics(self, success: bool, latency: float,
                            performance: Optional[float] = None):
        """
        Met à jour les métriques après exécution

        Args:
            success: Si l'exécution a réussi
            latency: Temps d'exécution en ms
            performance: Score de performance (optionnel)
        """
        self.total_executions += 1

        if success:
            self.successful_executions += 1
        else:
            self.metrics.error_count += 1

        # Moyenne mobile exponentielle pour la latence
        alpha = 0.1
        self.metrics.latency_ms = alpha * latency + (1 - alpha) * self.metrics.latency_ms

        # Mise à jour du score de performance
        if performance is not None:
            alpha_perf = 0.2
            self.metrics.performance_score = (alpha_perf * performance +
                                             (1 - alpha_perf) * self.metrics.performance_score)

        # Calcul du taux de succès
        self.metrics.success_rate = self.successful_executions / self.total_executions

        # Déterminer le statut
        if self.metrics.success_rate > 0.9:
            self.status = ComponentStatus.HEALTHY
        elif self.metrics.success_rate > 0.7:
            self.status = ComponentStatus.DEGRADED
        else:
            self.status = ComponentStatus.FAILED

        self.metrics.status = self.status

    async def record_execution(self, task: Task, result: Dict[str, Any]):
        """Enregistre une exécution dans l'historique"""
        execution_record = {
            'task_id': task.task_id,
            'task_type': task.problem_type,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id
        }

        self.execution_history.append(execution_record)

        # Garder seulement les 1000 dernières
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'agent"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': self.metrics.success_rate,
            'avg_latency_ms': self.metrics.latency_ms,
            'performance_score': self.metrics.performance_score,
            'error_count': self.metrics.error_count,
            'history_size': len(self.execution_history)
        }

    async def reset_metrics(self):
        """Réinitialise les métriques"""
        self.metrics.error_count = 0
        self.metrics.latency_ms = 0.0
        self.metrics.performance_score = 0.0
        self.total_executions = 0
        self.successful_executions = 0
        self.metrics.success_rate = 0.0

        logger.info(f"Metrics reset for agent {self.agent_id}")

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} id={self.agent_id} "
                f"type={self.agent_type} status={self.status.value}>")
