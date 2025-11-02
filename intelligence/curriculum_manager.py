"""
Curriculum Manager - Gestion de l'apprentissage progressif
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DifficultyLevel:
    """Représente un niveau de difficulté"""

    def __init__(self, level: int, name: str, complexity: float,
                 samples: int, success_threshold: float = 0.8):
        self.level = level
        self.name = name
        self.complexity = complexity
        self.samples = samples
        self.success_threshold = success_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'name': self.name,
            'complexity': self.complexity,
            'samples': self.samples,
            'success_threshold': self.success_threshold
        }

class CurriculumManager:
    """Gère l'apprentissage progressif par curriculum"""

    def __init__(self, initial_level: int = 1, max_level: int = 10):
        self.current_level = initial_level
        self.max_level = max_level
        self.performance_history = []
        self.level_history = []

        self.levels = self._initialize_levels()
        self.advancement_log = []

        logger.info(f"CurriculumManager initialized (level={initial_level}/{max_level})")

    def _initialize_levels(self) -> Dict[int, DifficultyLevel]:
        """Initialise les niveaux de difficulté"""
        return {
            1: DifficultyLevel(1, 'Novice', 0.1, 100, 0.8),
            2: DifficultyLevel(2, 'Beginner', 0.2, 200, 0.8),
            3: DifficultyLevel(3, 'Elementary', 0.3, 300, 0.8),
            4: DifficultyLevel(4, 'Intermediate', 0.5, 500, 0.8),
            5: DifficultyLevel(5, 'Advanced', 0.7, 700, 0.85),
            6: DifficultyLevel(6, 'Expert', 0.85, 1000, 0.85),
            7: DifficultyLevel(7, 'Master', 0.95, 1500, 0.9),
            8: DifficultyLevel(8, 'Grandmaster', 1.0, 2000, 0.9),
            9: DifficultyLevel(9, 'Legend', 1.2, 3000, 0.92),
            10: DifficultyLevel(10, 'Mythic', 1.5, 5000, 0.95)
        }

    async def evaluate_performance(self, performance: float,
                                   task_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Évalue la performance et décide de la progression

        Args:
            performance: Score de performance (0.0 à 1.0)
            task_context: Contexte de la tâche (optionnel)

        Returns:
            Dictionnaire avec décision et détails
        """
        self.performance_history.append({
            'performance': performance,
            'level': self.current_level,
            'timestamp': datetime.now().isoformat(),
            'context': task_context or {}
        })

        current_level_obj = self.levels[self.current_level]

        # Calculer la performance récente
        window = 10
        if len(self.performance_history) >= window:
            recent_performances = [
                p['performance'] for p in self.performance_history[-window:]
                if p['level'] == self.current_level
            ]

            if len(recent_performances) >= window:
                recent_avg = np.mean(recent_performances)
                recent_std = np.std(recent_performances)

                decision = {
                    'current_level': self.current_level,
                    'performance': performance,
                    'recent_average': recent_avg,
                    'recent_std': recent_std,
                    'threshold': current_level_obj.success_threshold,
                    'action': 'continue'
                }

                # Décision d'avancement
                if (recent_avg >= current_level_obj.success_threshold and
                    recent_std < 0.1 and
                    self.current_level < self.max_level):

                    await self.advance_level()
                    decision['action'] = 'advanced'
                    decision['new_level'] = self.current_level

                # Décision de régression (si performance très faible)
                elif recent_avg < 0.5 and self.current_level > 1:
                    await self.regress_level()
                    decision['action'] = 'regressed'
                    decision['new_level'] = self.current_level

                return decision

        return {
            'current_level': self.current_level,
            'performance': performance,
            'action': 'collecting_data',
            'samples_needed': window - len(self.performance_history)
        }

    async def advance_level(self):
        """Passe au niveau suivant"""
        if self.current_level < self.max_level:
            old_level = self.current_level
            self.current_level += 1

            self.level_history.append({
                'from_level': old_level,
                'to_level': self.current_level,
                'direction': 'advance',
                'timestamp': datetime.now().isoformat()
            })

            self.advancement_log.append({
                'level': self.current_level,
                'timestamp': datetime.now().isoformat(),
                'performances': self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            })

            # Reset l'historique de performance pour le nouveau niveau
            self.performance_history = []

            logger.info(f"Advanced to level {self.current_level}: {self.levels[self.current_level].name}")

    async def regress_level(self):
        """Régresse au niveau précédent"""
        if self.current_level > 1:
            old_level = self.current_level
            self.current_level -= 1

            self.level_history.append({
                'from_level': old_level,
                'to_level': self.current_level,
                'direction': 'regress',
                'timestamp': datetime.now().isoformat()
            })

            self.performance_history = []

            logger.warning(f"Regressed to level {self.current_level}: {self.levels[self.current_level].name}")

    def get_current_curriculum(self) -> Dict[str, Any]:
        """Retourne le curriculum actuel"""
        level_obj = self.levels[self.current_level]

        return {
            'level': self.current_level,
            'name': level_obj.name,
            'complexity': level_obj.complexity,
            'samples': level_obj.samples,
            'success_threshold': level_obj.success_threshold,
            'progress': len(self.performance_history) / 10,  # Sur 10 échantillons
            'can_advance': self.current_level < self.max_level
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques du curriculum"""
        recent_perf = None
        if self.performance_history:
            recent = [p['performance'] for p in self.performance_history[-10:]]
            recent_perf = np.mean(recent) if recent else 0.0

        return {
            'current_level': self.current_level,
            'max_level': self.max_level,
            'progress_percentage': (self.current_level / self.max_level) * 100,
            'recent_performance': recent_perf,
            'total_evaluations': len(self.performance_history),
            'level_changes': len(self.level_history),
            'advancements': len([h for h in self.level_history if h['direction'] == 'advance']),
            'regressions': len([h for h in self.level_history if h['direction'] == 'regress'])
        }

    def get_level_config(self, level: Optional[int] = None) -> Dict[str, Any]:
        """Retourne la configuration d'un niveau"""
        if level is None:
            level = self.current_level

        if level in self.levels:
            return self.levels[level].to_dict()
        return {}

    def reset(self, level: int = 1):
        """Réinitialise le curriculum"""
        self.current_level = level
        self.performance_history = []
        self.level_history = []
        self.advancement_log = []
        logger.info(f"Curriculum reset to level {level}")

# Singleton instance
_curriculum_manager_instance = None

def get_curriculum_manager() -> CurriculumManager:
    """Retourne l'instance singleton du CurriculumManager"""
    global _curriculum_manager_instance
    if _curriculum_manager_instance is None:
        _curriculum_manager_instance = CurriculumManager()
    return _curriculum_manager_instance
