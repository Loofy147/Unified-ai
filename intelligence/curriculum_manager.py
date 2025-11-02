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
        self.validation_enabled = True  # NEW
        self.min_samples_for_advancement = 20  # NEW (was 10)
        self.consistency_threshold = 0.15  # NEW

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

    async def evaluate_performance(self, performance, task_context):
        """Evaluate with validation"""

        # Store with metadata for validation
        self.performance_history.append({
            'performance': performance,
            'level': self.current_level,
            'timestamp': datetime.now().isoformat(),
            'context': task_context,
            'validated': False  # NEW
        })

        # Need enough samples
        if len(self.performance_history) < self.min_samples_for_advancement:
            return {'action': 'collecting_data'}

        recent = self.performance_history[-self.min_samples_for_advancement:]
        recent_perf = [p['performance'] for p in recent]

        # CRITICAL: Validation checks
        if self.validation_enabled:
            # 1. Check for suspicious consistency (gaming indicator)
            std_dev = np.std(recent_perf)
            if std_dev < 0.02:  # Too consistent
                logger.warning(
                    f"Suspiciously consistent performance: std={std_dev:.4f}"
                )
                return {
                    'action': 'validation_failed',
                    'reason': 'performance_too_consistent'
                }

            # 2. Check for impossible values
            if any(p > 0.98 for p in recent_perf):
                logger.warning("Suspiciously high performance detected")
                # Require external validation
                if not await self._external_validation(recent):
                    return {
                        'action': 'validation_failed',
                        'reason': 'external_validation_required'
                    }

            # 3. Check for monotonic improvement (gaming indicator)
            if len(recent_perf) >= 10:
                is_monotonic = all(
                    recent_perf[i] <= recent_perf[i+1]
                    for i in range(len(recent_perf)-1)
                )
                if is_monotonic:
                    logger.warning("Monotonic improvement detected (suspicious)")
                    return {
                        'action': 'validation_failed',
                        'reason': 'monotonic_improvement_suspicious'
                    }

            # 4. Check consistency across metrics
            # If we have multiple metrics, they should correlate
            if not await self._check_metric_correlation(recent):
                return {
                    'action': 'validation_failed',
                    'reason': 'metric_inconsistency'
                }

        # Validation passed, proceed with normal logic
        avg_performance = np.mean(recent_perf)
        current_threshold = self.levels[self.current_level].success_threshold

        if avg_performance >= current_threshold:
            # Mark as validated before advancing
            for p in recent:
                p['validated'] = True

            await self.advance_level()
            return {'action': 'advanced', 'new_level': self.current_level}

        return {'action': 'continue'}

    async def _external_validation(self, recent_experiences):
        """Require external validation for suspicious performance"""
        # In production: run agent on held-out validation set
        # For now: placeholder that would connect to validation system
        logger.info("External validation required")
        return False  # Deny by default until validated

    async def _check_metric_correlation(self, recent_experiences):
        """Check if multiple metrics correlate appropriately"""
        # If agent reports high accuracy but low reward, that's suspicious
        # Implementation depends on available metrics
        return True  # Placeholder

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
