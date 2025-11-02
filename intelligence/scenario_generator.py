"""
Scenario Generator - Génère des scénarios d'entraînement adaptés
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ScenarioGenerator:
    """Génère des scénarios d'entraînement basés sur le curriculum"""

    def __init__(self):
        self.scenario_cache = {}
        self.generation_history = []
        self.scenario_templates = self._initialize_templates()

        logger.info("ScenarioGenerator initialized")

    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les templates de scénarios"""
        return {
            'optimization': {
                'base_params': {
                    'dimensions': 10,
                    'constraints': 5,
                    'noise_level': 0.1,
                    'max_iterations': 1000
                },
                'scaling_factors': {
                    'dimensions': 1.5,
                    'constraints': 1.3,
                    'noise_level': 1.2
                }
            },
            'rl_control': {
                'base_params': {
                    'state_dim': 20,
                    'action_dim': 5,
                    'episode_length': 200,
                    'reward_sparsity': 0.1
                },
                'scaling_factors': {
                    'state_dim': 1.4,
                    'action_dim': 1.2,
                    'episode_length': 1.5,
                    'reward_sparsity': 1.3
                }
            },
            'analytical': {
                'base_params': {
                    'matrix_size': 10,
                    'condition_number': 10,
                    'sparsity': 0.5
                },
                'scaling_factors': {
                    'matrix_size': 1.6,
                    'condition_number': 2.0
                }
            }
        }

    async def generate_scenario(self, task_type: str, complexity: float,
                               custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Génère un scénario adapté

        Args:
            task_type: Type de tâche (optimization, rl_control, analytical)
            complexity: Niveau de complexité (0.1 à 1.5)
            custom_params: Paramètres personnalisés (optionnel)

        Returns:
            Scénario généré
        """
        if task_type not in self.scenario_templates:
            logger.warning(f"Unknown task type: {task_type}, using default")
            task_type = 'optimization'

        template = self.scenario_templates[task_type]
        base_params = template['base_params'].copy()
        scaling_factors = template['scaling_factors']

        # Appliquer la complexité
        scenario_params = {}
        for param, base_value in base_params.items():
            if param in scaling_factors:
                scaling = scaling_factors[param]
                scaled_value = base_value * (scaling ** complexity)

                # Arrondir si entier
                if isinstance(base_value, int):
                    scaled_value = int(scaled_value)

                scenario_params[param] = scaled_value
            else:
                scenario_params[param] = base_value

        # Appliquer paramètres personnalisés
        if custom_params:
            scenario_params.update(custom_params)

        # Créer le scénario
        scenario = {
            'scenario_id': f"scenario_{task_type}_{datetime.now().timestamp()}",
            'task_type': task_type,
            'complexity': complexity,
            'parameters': scenario_params,
            'expected_performance': self._calculate_expected_performance(complexity),
            'estimated_duration': self._estimate_duration(task_type, scenario_params),
            'difficulty_description': self._get_difficulty_description(complexity),
            'generated_at': datetime.now().isoformat()
        }

        # Cache et historique
        self.scenario_cache[scenario['scenario_id']] = scenario
        self.generation_history.append({
            'scenario_id': scenario['scenario_id'],
            'task_type': task_type,
            'complexity': complexity,
            'timestamp': scenario['generated_at']
        })

        logger.info(f"Generated scenario: {scenario['scenario_id']} "
                   f"(type={task_type}, complexity={complexity:.2f})")

        return scenario

    def _calculate_expected_performance(self, complexity: float) -> float:
        """Calcule la performance attendue selon la complexité"""
        # Performance diminue avec la complexité
        base_performance = 0.95
        complexity_penalty = 0.3 * complexity

        expected = max(0.3, base_performance - complexity_penalty)

        # Ajouter du bruit
        noise = np.random.normal(0, 0.05)
        return np.clip(expected + noise, 0.0, 1.0)

    def _estimate_duration(self, task_type: str, params: Dict[str, Any]) -> float:
        """Estime la durée d'exécution en secondes"""
        if task_type == 'optimization':
            return params.get('max_iterations', 1000) * 0.001
        elif task_type == 'rl_control':
            return params.get('episode_length', 200) * 0.01
        elif task_type == 'analytical':
            matrix_size = params.get('matrix_size', 10)
            return (matrix_size ** 2) * 0.0001
        return 1.0

    def _get_difficulty_description(self, complexity: float) -> str:
        """Retourne une description de la difficulté"""
        if complexity < 0.3:
            return "Easy - Suitable for beginners"
        elif complexity < 0.6:
            return "Medium - Moderate challenge"
        elif complexity < 0.9:
            return "Hard - Significant challenge"
        elif complexity < 1.2:
            return "Very Hard - Expert level"
        else:
            return "Extreme - Master level"

    async def generate_batch(self, task_type: str, complexity: float,
                            batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Génère un batch de scénarios similaires

        Args:
            task_type: Type de tâche
            complexity: Niveau de complexité
            batch_size: Nombre de scénarios

        Returns:
            Liste de scénarios
        """
        scenarios = []

        for i in range(batch_size):
            # Ajouter une variation de ±10% à la complexité
            variation = np.random.uniform(-0.1, 0.1)
            varied_complexity = max(0.1, complexity + variation)

            scenario = await self.generate_scenario(task_type, varied_complexity)
            scenarios.append(scenario)

        logger.info(f"Generated batch of {batch_size} scenarios for {task_type}")
        return scenarios

    async def generate_progressive_batch(self, task_type: str,
                                        start_complexity: float,
                                        end_complexity: float,
                                        steps: int = 5) -> List[Dict[str, Any]]:
        """
        Génère un batch progressif de scénarios

        Args:
            task_type: Type de tâche
            start_complexity: Complexité initiale
            end_complexity: Complexité finale
            steps: Nombre d'étapes

        Returns:
            Liste de scénarios progressifs
        """
        scenarios = []
        complexity_range = np.linspace(start_complexity, end_complexity, steps)

        for complexity in complexity_range:
            scenario = await self.generate_scenario(task_type, complexity)
            scenarios.append(scenario)

        logger.info(f"Generated progressive batch: {steps} scenarios from "
                   f"{start_complexity:.2f} to {end_complexity:.2f}")

        return scenarios

    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un scénario du cache"""
        return self.scenario_cache.get(scenario_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques du générateur"""
        scenarios_by_type = {}
        for history in self.generation_history:
            task_type = history['task_type']
            scenarios_by_type[task_type] = scenarios_by_type.get(task_type, 0) + 1

        complexities = [h['complexity'] for h in self.generation_history]

        return {
            'total_scenarios': len(self.scenario_cache),
            'scenarios_by_type': scenarios_by_type,
            'avg_complexity': np.mean(complexities) if complexities else 0.0,
            'complexity_range': (min(complexities), max(complexities)) if complexities else (0, 0),
            'cache_size': len(self.scenario_cache)
        }

    def clear_cache(self, keep_recent: int = 100):
        """Nettoie le cache en gardant les N plus récents"""
        if len(self.scenario_cache) > keep_recent:
            recent_ids = [h['scenario_id'] for h in self.generation_history[-keep_recent:]]
            self.scenario_cache = {
                sid: scenario for sid, scenario in self.scenario_cache.items()
                if sid in recent_ids
            }
            logger.info(f"Cache cleared, kept {len(self.scenario_cache)} recent scenarios")

# Singleton instance
_scenario_generator_instance = None

def get_scenario_generator() -> ScenarioGenerator:
    """Retourne l'instance singleton du ScenarioGenerator"""
    global _scenario_generator_instance
    if _scenario_generator_instance is None:
        _scenario_generator_instance = ScenarioGenerator()
    return _scenario_generator_instance
