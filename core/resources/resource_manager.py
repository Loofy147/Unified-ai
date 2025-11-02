"""
Resource Manager - Gestion complète des ressources système
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResourceAllocation:
    """Représente une allocation de ressources"""
    agent_id: str
    resources: Dict[str, float]
    allocated_at: str
    priority: int = 1

class ResourceMonitor:
    """Moniteur de ressources en temps réel"""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.history = []
        self.running = False

    async def start_monitoring(self):
        """Démarre le monitoring"""
        self.running = True
        while self.running:
            snapshot = self._capture_snapshot()
            self.history.append(snapshot)

            # Garder seulement les 1000 derniers
            if len(self.history) > 1000:
                self.history = self.history[-1000:]

            await asyncio.sleep(self.update_interval)

    def _capture_snapshot(self) -> Dict[str, Any]:
        """Capture l'état actuel des ressources"""
        import psutil

        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }

    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.running = False

    def get_statistics(self, window: int = 100) -> Dict[str, Any]:
        """Retourne les statistiques récentes"""
        if not self.history:
            return {}

        recent = self.history[-window:]

        return {
            'cpu': {
                'mean': np.mean([s['cpu_percent'] for s in recent]),
                'max': np.max([s['cpu_percent'] for s in recent]),
                'min': np.min([s['cpu_percent'] for s in recent])
            },
            'memory': {
                'mean': np.mean([s['memory_percent'] for s in recent]),
                'max': np.max([s['memory_percent'] for s in recent]),
                'min': np.min([s['memory_percent'] for s in recent])
            }
        }

class ResourceManager:
    """Gestionnaire de ressources complet avec monitoring"""

    def __init__(self):
        self.resources = {
            'cpu': {'total': 100.0, 'available': 100.0, 'unit': 'percent'},
            'memory': {'total': 16000.0, 'available': 16000.0, 'unit': 'MB'},
            'gpu': {'total': 1.0, 'available': 1.0, 'unit': 'device'}
        }
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.lock = asyncio.Lock()
        self.monitor = ResourceMonitor()
        self.allocation_history = []

        logger.info("ResourceManager initialized")

    async def start(self):
        """Démarre le gestionnaire"""
        asyncio.create_task(self.monitor.start_monitoring())
        logger.info("ResourceManager started with monitoring")

    async def stop(self):
        """Arrête le gestionnaire"""
        self.monitor.stop_monitoring()
        logger.info("ResourceManager stopped")

    async def allocate(self, agent_id: str, requirements: Dict[str, float],
                      priority: int = 1) -> bool:
        """
        Alloue des ressources à un agent

        Args:
            agent_id: ID de l'agent
            requirements: Ressources demandées {'cpu': X, 'memory': Y}
            priority: Priorité de l'allocation (1-10)

        Returns:
            True si allocation réussie
        """
        async with self.lock:
            # Vérifier disponibilité
            for resource, amount in requirements.items():
                if resource not in self.resources:
                    logger.warning(f"Unknown resource type: {resource}")
                    continue

                if self.resources[resource]['available'] < amount:
                    logger.warning(
                        f"Insufficient {resource}: "
                        f"requested={amount}, available={self.resources[resource]['available']}"
                    )
                    return False

            # Allouer
            allocation = ResourceAllocation(
                agent_id=agent_id,
                resources=requirements.copy(),
                allocated_at=datetime.now().isoformat(),
                priority=priority
            )

            for resource, amount in requirements.items():
                if resource in self.resources:
                    self.resources[resource]['available'] -= amount

            self.allocations[agent_id] = allocation
            self.allocation_history.append(allocation)

            logger.info(f"Resources allocated to {agent_id}: {requirements}")
            return True

    async def release(self, agent_id: str) -> bool:
        """
        Libère les ressources d'un agent

        Args:
            agent_id: ID de l'agent

        Returns:
            True si libération réussie
        """
        async with self.lock:
            if agent_id not in self.allocations:
                logger.warning(f"No allocation found for {agent_id}")
                return False

            allocation = self.allocations[agent_id]

            for resource, amount in allocation.resources.items():
                if resource in self.resources:
                    self.resources[resource]['available'] += amount

            del self.allocations[agent_id]
            logger.info(f"Resources released from {agent_id}")
            return True

    async def reallocate(self, agent_id: str, new_requirements: Dict[str, float]) -> bool:
        """
        Réalloue les ressources d'un agent

        Args:
            agent_id: ID de l'agent
            new_requirements: Nouvelles ressources demandées

        Returns:
            True si réallocation réussie
        """
        success = await self.release(agent_id)
        if not success:
            return False

        return await self.allocate(agent_id, new_requirements)

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état des ressources"""
        return {
            res_name: {
                'total': res['total'],
                'available': res['available'],
                'allocated': res['total'] - res['available'],
                'utilization': (res['total'] - res['available']) / res['total'],
                'unit': res['unit']
            }
            for res_name, res in self.resources.items()
        }

    def get_allocations(self) -> List[Dict[str, Any]]:
        """Retourne toutes les allocations actives"""
        return [
            {
                'agent_id': alloc.agent_id,
                'resources': alloc.resources,
                'allocated_at': alloc.allocated_at,
                'priority': alloc.priority
            }
            for alloc in self.allocations.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes"""
        return {
            'status': self.get_status(),
            'allocations': self.get_allocations(),
            'monitor': self.monitor.get_statistics(),
            'total_allocations': len(self.allocation_history)
        }

    async def optimize_allocations(self) -> List[str]:
        """
        Optimise les allocations actuelles

        Returns:
            Liste de recommandations
        """
        recommendations = []
        status = self.get_status()

        # Détecter surallocation
        for resource, info in status.items():
            if info['utilization'] > 0.9:
                recommendations.append(
                    f"HIGH_UTILIZATION: {resource} at {info['utilization']:.1%}"
                )
            elif info['utilization'] < 0.3:
                recommendations.append(
                    f"LOW_UTILIZATION: {resource} at {info['utilization']:.1%} - "
                    f"Consider reducing allocations"
                )

        # Détecter allocations obsolètes
        current_time = datetime.now()
        for alloc in self.allocations.values():
            allocated_time = datetime.fromisoformat(alloc.allocated_at)
            duration = (current_time - allocated_time).total_seconds()

            if duration > 3600:  # Plus d'1 heure
                recommendations.append(
                    f"LONG_ALLOCATION: {alloc.agent_id} allocated for "
                    f"{duration/3600:.1f} hours"
                )

        return recommendations

# Singleton instance
_resource_manager_instance = None

def get_resource_manager() -> ResourceManager:
    """Retourne l'instance singleton du ResourceManager"""
    global _resource_manager_instance
    if _resource_manager_instance is None:
        _resource_manager_instance = ResourceManager()
    return _resource_manager_instance
