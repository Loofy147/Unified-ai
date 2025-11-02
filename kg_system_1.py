"""
Knowledge Graph System for Unified AI
Provides persistent memory and knowledge management for the entire system
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    entity_type: str  # 'agent', 'task', 'outcome', 'concept'
    properties: Dict[str, Any]
    created_at: str
    updated_at: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source_id: str
    target_id: str
    relationship_type: str  # 'performs', 'learns_from', 'depends_on', etc.
    strength: float  # 0.0 to 1.0
    properties: Dict[str, Any]
    created_at: str
    updated_at: str

@dataclass
class Outcome:
    """Represents an outcome of an action"""
    entity_id: str
    action: str
    result: str  # 'success', 'failure', 'partial'
    metrics: Dict[str, float]
    timestamp: str
    context: Dict[str, Any]

class KnowledgeGraphDB:
    """SQLite-based Knowledge Graph Database"""
    
    def __init__(self, db_path: str = '/home/ubuntu/unified_ai_system/knowledge_graph/kg.db'):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    properties TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )
            ''')
            
            # Outcomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT NOT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities(id)
                )
            ''')
            
            # Performance history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities(id)
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationship_type ON relationships(relationship_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_outcomes ON outcomes(entity_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_entity ON performance_history(entity_id)')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with thread safety"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def add_entity(self, entity: Entity) -> bool:
        """Add or update an entity"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO entities 
                    (id, entity_type, properties, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    entity.id,
                    entity.entity_type,
                    json.dumps(entity.properties),
                    entity.created_at,
                    entity.updated_at
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding entity: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM entities WHERE id = ?', (entity_id,))
                row = cursor.fetchone()
                
                if row:
                    return Entity(
                        id=row['id'],
                        entity_type=row['entity_type'],
                        properties=json.loads(row['properties']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
        except Exception as e:
            print(f"Error retrieving entity: {e}")
        
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM entities WHERE entity_type = ?', (entity_type,))
                rows = cursor.fetchall()
                
                return [
                    Entity(
                        id=row['id'],
                        entity_type=row['entity_type'],
                        properties=json.loads(row['properties']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"Error retrieving entities: {e}")
        
        return []
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship between entities"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO relationships 
                    (source_id, target_id, relationship_type, strength, properties, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    relationship.source_id,
                    relationship.target_id,
                    relationship.relationship_type,
                    relationship.strength,
                    json.dumps(relationship.properties),
                    relationship.created_at,
                    relationship.updated_at
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding relationship: {e}")
            return False
    
    def get_relationships(self, source_id: str, relationship_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships from a source entity"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if relationship_type:
                    cursor.execute('''
                        SELECT * FROM relationships 
                        WHERE source_id = ? AND relationship_type = ?
                    ''', (source_id, relationship_type))
                else:
                    cursor.execute('''
                        SELECT * FROM relationships WHERE source_id = ?
                    ''', (source_id,))
                
                rows = cursor.fetchall()
                
                return [
                    Relationship(
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        relationship_type=row['relationship_type'],
                        strength=row['strength'],
                        properties=json.loads(row['properties']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"Error retrieving relationships: {e}")
        
        return []
    
    def record_outcome(self, outcome: Outcome) -> bool:
        """Record an outcome of an action"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO outcomes 
                    (entity_id, action, result, metrics, timestamp, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    outcome.entity_id,
                    outcome.action,
                    outcome.result,
                    json.dumps(outcome.metrics),
                    outcome.timestamp,
                    json.dumps(outcome.context)
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error recording outcome: {e}")
            return False
    
    def get_outcomes(self, entity_id: str, limit: int = 100) -> List[Outcome]:
        """Get recent outcomes for an entity"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM outcomes 
                    WHERE entity_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (entity_id, limit))
                
                rows = cursor.fetchall()
                
                return [
                    Outcome(
                        entity_id=row['entity_id'],
                        action=row['action'],
                        result=row['result'],
                        metrics=json.loads(row['metrics']),
                        timestamp=row['timestamp'],
                        context=json.loads(row['context'])
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"Error retrieving outcomes: {e}")
        
        return []
    
    def record_performance(self, entity_id: str, metric_name: str, metric_value: float) -> bool:
        """Record a performance metric"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_history 
                    (entity_id, metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    entity_id,
                    metric_name,
                    metric_value,
                    datetime.now().isoformat()
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error recording performance: {e}")
            return False
    
    def get_performance_history(self, entity_id: str, metric_name: str, limit: int = 100) -> List[Tuple[str, float]]:
        """Get performance history for an entity and metric"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, metric_value FROM performance_history 
                    WHERE entity_id = ? AND metric_name = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (entity_id, metric_name, limit))
                
                rows = cursor.fetchall()
                return [(row['timestamp'], row['metric_value']) for row in rows]
        except Exception as e:
            print(f"Error retrieving performance history: {e}")
        
        return []
    
    def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an entity"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get entity
                cursor.execute('SELECT * FROM entities WHERE id = ?', (entity_id,))
                entity_row = cursor.fetchone()
                
                if not entity_row:
                    return {}
                
                # Get outcomes
                cursor.execute('SELECT result FROM outcomes WHERE entity_id = ?', (entity_id,))
                outcomes = cursor.fetchall()
                
                # Get relationships
                cursor.execute('SELECT COUNT(*) as count FROM relationships WHERE source_id = ?', (entity_id,))
                rel_count = cursor.fetchone()['count']
                
                # Calculate success rate
                if outcomes:
                    success_count = sum(1 for o in outcomes if o['result'] == 'success')
                    success_rate = success_count / len(outcomes)
                else:
                    success_rate = 0.0
                
                return {
                    'entity_id': entity_id,
                    'entity_type': entity_row['entity_type'],
                    'total_outcomes': len(outcomes),
                    'success_rate': success_rate,
                    'total_relationships': rel_count,
                    'created_at': entity_row['created_at'],
                    'updated_at': entity_row['updated_at']
                }
        except Exception as e:
            print(f"Error getting entity statistics: {e}")
        
        return {}
    
    def clear_old_data(self, days_old: int = 30) -> bool:
        """Clear outcomes older than specified days"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM outcomes WHERE timestamp < ?', (cutoff_date,))
                cursor.execute('DELETE FROM performance_history WHERE timestamp < ?', (cutoff_date,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing old data: {e}")
            return False


class KnowledgeGraphManager:
    """High-level manager for knowledge graph operations"""
    
    def __init__(self, db_path: str = '/home/ubuntu/unified_ai_system/knowledge_graph/kg.db'):
        self.db = KnowledgeGraphDB(db_path)
    
    def register_agent(self, agent_id: str, agent_type: str, properties: Dict[str, Any] = None) -> bool:
        """Register an agent in the knowledge graph"""
        entity = Entity(
            id=agent_id,
            entity_type='agent',
            properties=properties or {'type': agent_type},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        return self.db.add_entity(entity)
    
    def record_agent_action(self, agent_id: str, action: str, result: str, metrics: Dict[str, float], context: Dict[str, Any] = None) -> bool:
        """Record an agent's action and outcome"""
        outcome = Outcome(
            entity_id=agent_id,
            action=action,
            result=result,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            context=context or {}
        )
        return self.db.record_outcome(outcome)
    
    def link_agents(self, source_agent_id: str, target_agent_id: str, relationship_type: str, strength: float = 0.5) -> bool:
        """Create a relationship between two agents"""
        relationship = Relationship(
            source_id=source_agent_id,
            target_id=target_agent_id,
            relationship_type=relationship_type,
            strength=strength,
            properties={},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        return self.db.add_relationship(relationship)
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive performance data for an agent"""
        return self.db.get_entity_statistics(agent_id)
    
    def get_agent_memory(self, agent_id: str, limit: int = 50) -> List[Outcome]:
        """Get agent's action history (memory)"""
        return self.db.get_outcomes(agent_id, limit)
    
    def learn_from_peers(self, agent_id: str) -> Dict[str, Any]:
        """Analyze peer agents' outcomes for learning"""
        # Get relationships to peer agents
        relationships = self.db.get_relationships(agent_id, 'learns_from')
        
        learning_data = {}
        for rel in relationships:
            peer_outcomes = self.db.get_outcomes(rel.target_id, limit=10)
            peer_stats = self.db.get_entity_statistics(rel.target_id)
            
            learning_data[rel.target_id] = {
                'outcomes': peer_outcomes,
                'statistics': peer_stats,
                'relationship_strength': rel.strength
            }
        
        return learning_data


# Test and demonstration
if __name__ == '__main__':
    # Initialize knowledge graph
    kg_manager = KnowledgeGraphManager()
    
    # Register agents
    kg_manager.register_agent('agent_1', 'DataPreprocessing', {'version': '1.0'})
    kg_manager.register_agent('agent_2', 'ModelTraining', {'version': '1.0'})
    
    # Record actions
    kg_manager.record_agent_action(
        'agent_1',
        'preprocess_data',
        'success',
        {'time': 2.5, 'samples_processed': 1000},
        {'dataset': 'mnist'}
    )
    
    # Link agents
    kg_manager.link_agents('agent_1', 'agent_2', 'feeds_data_to', 0.8)
    
    # Get performance
    perf = kg_manager.get_agent_performance('agent_1')
    print(f"Agent 1 Performance: {perf}")
    
    # Get memory
    memory = kg_manager.get_agent_memory('agent_1')
    print(f"Agent 1 Memory: {memory}")
    
    print("\n✓ Knowledge Graph System initialized successfully!")
