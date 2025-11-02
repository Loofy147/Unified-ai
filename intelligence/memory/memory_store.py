"""
Memory Store - Système de mémoire hiérarchique avec consolidation
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class MemoryConsolidation:
    """Gère la consolidation de la mémoire"""

    def __init__(self, importance_threshold: float = 0.7):
        self.importance_threshold = importance_threshold

    def calculate_importance(self, experience: Dict[str, Any]) -> float:
        """
        Calcule l'importance d'une expérience

        Critères:
        - Reward élevé: +0.4
        - Erreur: +0.3
        - Rare: +0.2
        - Récent: +0.1
        """
        score = 0.0

        # Reward élevé
        if experience.get('reward', 0) > 0.7:
            score += 0.4

        # Présence d'erreur
        if experience.get('error', False):
            score += 0.3

        # Nouveauté (à implémenter)
        score += 0.1

        return min(score, 1.0)

    def should_consolidate(self, experience: Dict[str, Any]) -> bool:
        """Détermine si une expérience doit être consolidée"""
        return self.calculate_importance(experience) >= self.importance_threshold

    def compress_experiences(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compresse plusieurs expériences similaires"""
        if not experiences:
            return {}

        # Agréger les métriques
        avg_reward = np.mean([e.get('reward', 0) for e in experiences])
        total_count = len(experiences)

        return {
            'type': 'compressed',
            'count': total_count,
            'avg_reward': avg_reward,
            'first_timestamp': experiences[0].get('timestamp'),
            'last_timestamp': experiences[-1].get('timestamp'),
            'sample': experiences[0]  # Garder un exemple
        }

class MemoryStore:
    """Système de mémoire hiérarchique avancé"""

    def __init__(self, max_short_term: int = 1000, max_long_term: int = 10000):
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term

        # Différents types de mémoire
        self.short_term = deque(maxlen=max_short_term)
        self.long_term = {}  # Indexé par task_id
        self.episodic = []  # Séquences d'actions
        self.semantic = {}  # Concepts appris

        self.consolidation = MemoryConsolidation()
        self.access_counts = {}  # Track accesses pour LRU

        logger.info(f"MemoryStore initialized (short={max_short_term}, long={max_long_term})")

    async def store_experience(self, experience: Dict[str, Any]):
        """
        Stocke une expérience dans la mémoire

        Args:
            experience: Dict contenant l'expérience à stocker
        """
        # Ajouter timestamp si absent
        if 'timestamp' not in experience:
            experience['timestamp'] = datetime.now().isoformat()

        # Ajouter à la mémoire à court terme
        self.short_term.append(experience)

        # Vérifier si consolidation nécessaire
        if len(self.short_term) >= self.max_short_term * 0.9:
            await self._consolidate()

        logger.debug(f"Experience stored (short_term size: {len(self.short_term)})")

    async def _consolidate(self):
        """Consolide la mémoire à court terme vers long terme"""
        logger.info("Starting memory consolidation...")

        consolidated_count = 0

        # Copier pour éviter modification pendant itération
        experiences = list(self.short_term)

        for exp in experiences:
            if self.consolidation.should_consolidate(exp):
                task_id = exp.get('task_id', 'general')

                if task_id not in self.long_term:
                    self.long_term[task_id] = []

                # Limiter la taille de long term
                if len(self.long_term[task_id]) >= self.max_long_term:
                    # Supprimer les plus anciennes (FIFO)
                    self.long_term[task_id] = self.long_term[task_id][-self.max_long_term:]

                self.long_term[task_id].append(exp)
                consolidated_count += 1

        # Garder seulement les N dernières expériences en short term
        keep_recent = int(self.max_short_term * 0.5)
        self.short_term = deque(list(self.short_term)[-keep_recent:],
                               maxlen=self.max_short_term)

        logger.info(f"Consolidated {consolidated_count} experiences to long-term memory")

    async def retrieve(self, query: Dict[str, Any], limit: int = 10,
                      memory_type: str = 'all') -> List[Dict[str, Any]]:
        """
        Récupère des expériences pertinentes

        Args:
            query: Critères de recherche
            limit: Nombre max de résultats
            memory_type: 'short', 'long', ou 'all'

        Returns:
            Liste d'expériences correspondantes
        """
        results = []

        # Recherche dans short term
        if memory_type in ['short', 'all']:
            for exp in reversed(self.short_term):
                if self._matches_query(exp, query):
                    results.append(exp)
                    self._update_access(exp)
                    if len(results) >= limit:
                        break

        # Recherche dans long term
        if memory_type in ['long', 'all'] and len(results) < limit:
            task_id = query.get('task_id')

            if task_id and task_id in self.long_term:
                for exp in reversed(self.long_term[task_id]):
                    if self._matches_query(exp, query):
                        results.append(exp)
                        self._update_access(exp)
                        if len(results) >= limit:
                            break
            elif not task_id:
                # Chercher dans toutes les tâches
                for task_experiences in self.long_term.values():
                    for exp in reversed(task_experiences):
                        if self._matches_query(exp, query):
                            results.append(exp)
                            self._update_access(exp)
                            if len(results) >= limit:
                                break
                    if len(results) >= limit:
                        break

        logger.debug(f"Retrieved {len(results)} experiences for query: {query}")
        return results[:limit]

    def _matches_query(self, experience: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Vérifie si une expérience correspond aux critères"""
        for key, value in query.items():
            if key not in experience:
                return False

            exp_value = experience[key]

            # Comparaison selon le type
            if isinstance(value, (int, float)):
                # Pour les nombres, tolérance de 10%
                if abs(exp_value - value) > abs(value * 0.1):
                    return False
            elif exp_value != value:
                return False

        return True

    def _update_access(self, experience: Dict[str, Any]):
        """Met à jour le compteur d'accès"""
        exp_id = id(experience)
        self.access_counts[exp_id] = self.access_counts.get(exp_id, 0) + 1

    async def store_episode(self, episode: List[Dict[str, Any]]):
        """Stocke une séquence d'expériences (épisode)"""
        episode_data = {
            'episode_id': f"ep_{datetime.now().timestamp()}",
            'length': len(episode),
            'experiences': episode,
            'timestamp': datetime.now().isoformat()
        }

        self.episodic.append(episode_data)
        logger.info(f"Episode stored (length={len(episode)})")

    async def learn_concept(self, concept_name: str, concept_data: Dict[str, Any]):
        """Apprend un concept (mémoire sémantique)"""
        self.semantic[concept_name] = {
            'data': concept_data,
            'learned_at': datetime.now().isoformat(),
            'access_count': 0
        }
        logger.info(f"Concept learned: {concept_name}")

    def get_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Récupère un concept appris"""
        if concept_name in self.semantic:
            self.semantic[concept_name]['access_count'] += 1
            return self.semantic[concept_name]['data']
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de la mémoire"""
        long_term_total = sum(len(exps) for exps in self.long_term.values())

        return {
            'short_term_size': len(self.short_term),
            'long_term_keys': len(self.long_term),
            'long_term_total': long_term_total,
            'episodic_count': len(self.episodic),
            'semantic_concepts': len(self.semantic),
            'total_experiences': len(self.short_term) + long_term_total,
            'most_accessed': self._get_most_accessed()
        }

    def _get_most_accessed(self) -> Optional[str]:
        """Retourne l'expérience la plus accédée"""
        if not self.access_counts:
            return None
        return max(self.access_counts.items(), key=lambda x: x[1])[0]

    async def clear_old_data(self, days_old: int = 30):
        """Nettoie les données anciennes"""
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_str = cutoff_date.isoformat()

        # Nettoyer short term
        self.short_term = deque(
            [exp for exp in self.short_term if exp.get('timestamp', '') >= cutoff_str],
            maxlen=self.max_short_term
        )

        # Nettoyer long term
        for task_id in list(self.long_term.keys()):
            self.long_term[task_id] = [
                exp for exp in self.long_term[task_id]
                if exp.get('timestamp', '') >= cutoff_str
            ]

            # Supprimer les tâches vides
            if not self.long_term[task_id]:
                del self.long_term[task_id]

        logger.info(f"Cleaned data older than {days_old} days")

# Singleton instance
_memory_store_instance = None

def get_memory_store() -> MemoryStore:
    """Retourne l'instance singleton du MemoryStore"""
    global _memory_store_instance
    if _memory_store_instance is None:
        _memory_store_instance = MemoryStore()
    return _memory_store_instance
