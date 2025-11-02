#!/usr/bin/env python3
"""
Point d'entrée principal du Système Unifié d'IA
"""

import asyncio
import sys
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from orchestration.integrated_unified_agent import IntegratedUnifiedAgent
from agents.base_agent import Task
from agents.optimization_agent import OptimizationAgent
from agents.rl_agent import RLAgent

async def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("SYSTÈME UNIFIÉ D'IA - VERSION 2.0")
    print("="*70 + "\n")

    # 1. Initialiser le système
    print("1. Initialisation du système...")
    system = IntegratedUnifiedAgent("UnifiedAI_v2")
    success = await system.initialize()

    if not success:
        print("✗ Échec de l'initialisation")
        return

    print("✓ Système initialisé\n")

    # 2. Enregistrer les agents
    print("2. Enregistrement des agents...")
    await system.register_agent(OptimizationAgent())
    await system.register_agent(RLAgent())
    print("✓ Agents enregistrés\n")

    # 3. Afficher le statut initial
    print("3. Statut initial du système:")
    status = await system.get_system_status()
    print(f"   - Agents: {status['agents_registered']}")
    print(f"   - Curriculum Level: {status['curriculum']['current_level']}")
    print(f"   - Mémoire: {status['memory']['total_experiences']} expériences\n")

    # 4. Démonstration avec une tâche exemple
    print("4. Exécution d'une tâche d'optimisation...")

    opt_task = Task(
        task_id="opt_demo_001",
        problem_type="optimization",
        description="Tâche de démonstration - Optimisation de hyperparamètres",
        data_source="demo_dataset",
        target_metric="accuracy",
        priority=1
    )

    result = await system.solve_task(opt_task)
    print(f"   Statut: {result['status']}")
    print(f"   Performance: {result.get('performance', 'N/A'):.2f}\n")

    print("5. Exécution d'une tâche d'apprentissage par renforcement...")

    rl_task = Task(
        task_id="rl_demo_001",
        problem_type="rl_control",
        description="Tâche de démonstration - Apprentissage par renforcement",
        data_source="demo_dataset",
        target_metric="reward",
        priority=1
    )

    result = await system.solve_task(rl_task)
    print(f"   Statut: {result['status']}")
    print(f"   Performance: {result.get('performance', 'N/A'):.2f}\n")

    # 6. Afficher le statut final
    print("6. Statut final du système:")
    final_status = await system.get_system_status()
    print(f"   - Tâches complétées: {final_status['tasks_completed']}")
    print(f"   - Performance moyenne: {final_status['average_performance']:.2f}")
    print(f"   - Curriculum Level: {final_status['curriculum']['current_level']}\n")

    # 7. Arrêt du système
    print("7. Arrêt du système...")
    await system.shutdown()
    print("✓ Système arrêté proprement\n")

    # 8. Lancer la campagne de Red Team
    print("8. Lancement de la campagne de test Red Team...")
    from red_team_framework import RedTeamOrchestrator
    red_team = RedTeamOrchestrator()
    report = await red_team.run_full_campaign(system)

    print("\n" + "="*70)
    print("RAPPORT DE CAMPAGNE RED TEAM")
    print("="*70)
    print(f"Score de sécurité global: {report['overall_security_score']:.1f}/100")
    print(f"Attaques critiques: {report['severity_breakdown']['critical']}")
    print("="*70 + "\n")


    print("="*70)
    print("DÉMONSTRATION TERMINÉE")
    print("="*70)
    print("""
PROCHAINES ÉTAPES:
- Implémenter les agents spécialisés dans agents/
- Créer des environnements dans environments/
- Lancer les tests: pytest tests/ -v
- Consulter la documentation dans docs/
    """)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
