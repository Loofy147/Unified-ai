"""
Model Zoo - Dépôt centralisé de modèles avec versioning
"""

import json
import dill
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import logging

from core.config import settings

logger = logging.getLogger(__name__)

class ModelVersion:
    """Représente une version d'un modèle"""

    def __init__(self, model_id: str, version: int, model: Any, metadata: Dict[str, Any]):
        self.model_id = model_id
        self.version = version
        self.model = model
        self.metadata = metadata
        self.metadata['version'] = version
        self.metadata['created_at'] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'metadata': self.metadata
        }

class ModelZoo:
    """Dépôt centralisé de modèles avec versioning et persistence"""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Dict[int, ModelVersion]] = defaultdict(dict)
        self.metadata_index = {}
        self.latest_versions = {}

        self._load_index()
        logger.info(f"ModelZoo initialized at {storage_path}")

    def _load_index(self):
        """Charge l'index des modèles"""
        index_file = self.storage_path / "index.json"

        if index_file.exists():
            with open(index_file, 'r') as f:
                data = json.load(f)
                self.metadata_index = data.get('metadata', {})
                self.latest_versions = data.get('latest_versions', {})
            logger.info(f"Loaded {len(self.metadata_index)} models from index")

    def _save_index(self):
        """Sauvegarde l'index des modèles"""
        index_file = self.storage_path / "index.json"

        with open(index_file, 'w') as f:
            json.dump({
                'metadata': self.metadata_index,
                'latest_versions': self.latest_versions,
                'updated_at': datetime.now().isoformat()
            }, f, indent=2)

    async def register_model(self, model_id: str, model: Any,
                           metadata: Dict[str, Any], save_to_disk: bool = True) -> int:
        """
        Enregistre un nouveau modèle ou une nouvelle version

        Args:
            model_id: Identifiant unique du modèle
            model: Le modèle lui-même
            metadata: Métadonnées du modèle
            save_to_disk: Sauvegarder sur disque

        Returns:
            Version number
        """
        # Security check: prevent registering models that have an overridden __reduce__ method,
        # which is a common vector for pickle-based attacks.
        if '__reduce__' in model.__class__.__dict__:
            raise TypeError("Cannot register models with __reduce__ method for security reasons.")

        # Déterminer le numéro de version
        if model_id in self.models:
            version = max(self.models[model_id].keys()) + 1
        else:
            version = 1

        # Créer la version
        model_version = ModelVersion(model_id, version, model, metadata)

        # Stocker en mémoire
        self.models[model_id][version] = model_version
        self.latest_versions[model_id] = version
        self.metadata_index[f"{model_id}_v{version}"] = model_version.to_dict()

        # Sauvegarder sur disque
        if save_to_disk:
            self._save_model_to_disk(model_id, version, model, metadata)

        self._save_index()

        logger.info(f"Model registered: {model_id} v{version}")
        return version

    def _save_model_to_disk(self, model_id: str, version: int,
                           model: Any, metadata: Dict[str, Any]):
        """Sauvegarde un modèle sur disque"""
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)

        # Sauvegarder le modèle avec dill
        model_file = model_dir / f"v{version}.dill"
        with open(model_file, 'wb') as f:
            dill.dump(model, f)

        # Sauvegarder les métadonnées
        metadata_file = model_dir / f"v{version}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Model saved to disk: {model_file}")

    async def get_model(self, model_id: str, version: Optional[int] = None) -> Optional[Any]:
        """
        Récupère un modèle

        Args:
            model_id: ID du modèle
            version: Version spécifique (None = dernière version)

        Returns:
            Le modèle ou None
        """
        if model_id not in self.models:
            # Essayer de charger depuis le disque
            loaded = self._load_model_from_disk(model_id, version)
            if loaded:
                return loaded
            return None

        if version is None:
            version = self.latest_versions.get(model_id)

        if version in self.models[model_id]:
            model_version = self.models[model_id][version]
            model_version.metadata['access_count'] = model_version.metadata.get('access_count', 0) + 1
            return model_version.model

        return None

    def _load_model_from_disk(self, model_id: str, version: Optional[int] = None) -> Optional[Any]:
        """Charge un modèle depuis le disque"""
        model_dir = self.storage_path / model_id

        if not model_dir.exists():
            return None

        # Déterminer la version
        if version is None:
            versions = [int(f.stem[1:]) for f in model_dir.glob("v*.dill")]
            if not versions:
                return None
            version = max(versions)

        model_file = model_dir / f"v{version}.dill"
        metadata_file = model_dir / f"v{version}_metadata.json"

        if not model_file.exists():
            return None

        # Charger le modèle avec dill
        with open(model_file, 'rb') as f:
            model = dill.load(f)

        # Charger les métadonnées
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Stocker en mémoire
        model_version = ModelVersion(model_id, version, model, metadata)
        self.models[model_id][version] = model_version

        logger.info(f"Model loaded from disk: {model_id} v{version}")
        return model

    async def get_best_model(self, task_type: str,
                           metric: str = 'performance') -> Optional[Tuple[str, int, Any]]:
        """
        Retourne le meilleur modèle pour un type de tâche

        Args:
            task_type: Type de tâche
            metric: Métrique à optimiser

        Returns:
            (model_id, version, model) ou None
        """
        candidates = []

        for model_id, versions in self.models.items():
            for version, model_version in versions.items():
                if model_version.metadata.get('task_type') == task_type:
                    score = model_version.metadata.get(metric, 0)
                    candidates.append((score, model_id, version, model_version.model))

        if not candidates:
            return None

        # Trier par score décroissant
        candidates.sort(reverse=True, key=lambda x: x[0])
        best = candidates[0]

        logger.info(f"Best model for {task_type}: {best[1]} v{best[2]} (score={best[0]:.3f})")
        return best[1], best[2], best[3]

    async def compare_versions(self, model_id: str,
                              versions: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compare différentes versions d'un modèle

        Args:
            model_id: ID du modèle
            versions: Versions à comparer (None = toutes)

        Returns:
            Dictionnaire de comparaison
        """
        if model_id not in self.models:
            return {}

        if versions is None:
            versions = list(self.models[model_id].keys())

        comparison = {
            'model_id': model_id,
            'versions': {}
        }

        for version in versions:
            if version in self.models[model_id]:
                mv = self.models[model_id][version]
                comparison['versions'][version] = mv.metadata

        return comparison

    async def delete_model(self, model_id: str, version: Optional[int] = None) -> bool:
        """
        Supprime un modèle ou une version

        Args:
            model_id: ID du modèle
            version: Version spécifique (None = toutes les versions)

        Returns:
            True si succès
        """
        if model_id not in self.models:
            return False

        if version is None:
            # Supprimer toutes les versions
            del self.models[model_id]
            if model_id in self.latest_versions:
                del self.latest_versions[model_id]

            # Supprimer du disque
            model_dir = self.storage_path / model_id
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)

            logger.info(f"Model deleted: {model_id} (all versions)")
        else:
            # Supprimer version spécifique
            if version in self.models[model_id]:
                del self.models[model_id][version]

                # Mettre à jour latest_version
                if self.models[model_id]:
                    self.latest_versions[model_id] = max(self.models[model_id].keys())
                else:
                    del self.latest_versions[model_id]
                    del self.models[model_id]

                # Supprimer du disque
                model_file = self.storage_path / model_id / f"v{version}.dill"
                metadata_file = self.storage_path / model_id / f"v{version}_metadata.json"

                if model_file.exists():
                    model_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()

                logger.info(f"Model version deleted: {model_id} v{version}")

        self._save_index()
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du ModelZoo"""
        total_models = len(self.models)
        total_versions = sum(len(versions) for versions in self.models.values())

        models_by_type = defaultdict(int)
        for versions in self.models.values():
            for mv in versions.values():
                task_type = mv.metadata.get('task_type', 'unknown')
                models_by_type[task_type] += 1

        most_accessed = None
        max_access = 0
        for model_id, versions in self.models.items():
            for version, mv in versions.items():
                access_count = mv.metadata.get('access_count', 0)
                if access_count > max_access:
                    max_access = access_count
                    most_accessed = f"{model_id}_v{version}"

        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'models_by_type': dict(models_by_type),
            'most_accessed': most_accessed,
            'storage_path': str(self.storage_path)
        }

    def list_models(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Liste tous les modèles

        Args:
            task_type: Filtrer par type de tâche

        Returns:
            Liste de métadonnées de modèles
        """
        models_list = []

        for model_id, versions in self.models.items():
            for version, mv in versions.items():
                if task_type is None or mv.metadata.get('task_type') == task_type:
                    models_list.append({
                        'model_id': model_id,
                        'version': version,
                        'is_latest': version == self.latest_versions.get(model_id),
                        'metadata': mv.metadata
                    })

        return models_list

# Singleton instance
_model_zoo_instance = None

def get_model_zoo(force_reload: bool = False) -> ModelZoo:
    """
    Retourne l'instance singleton du ModelZoo.

    Args:
        force_reload: Si True, force la recréation de l'instance. Utile pour les tests.
    """
    global _model_zoo_instance
    if _model_zoo_instance is None or force_reload:
        from core.config import settings
        _model_zoo_instance = ModelZoo(storage_path=settings.model_zoo.storage_path)
    return _model_zoo_instance
