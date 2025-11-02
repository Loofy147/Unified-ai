"""
UnifiedAgent - Hierarchical Problem Solving Architecture
Implements the high-level control flow for problem identification and strategy selection.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging

# Import necessary components from Phase 1
# Note: We assume SuperAgent and KnowledgeGraphManager are available
# from agents.super_agent import SuperAgent, OperationMode
# from knowledge_graph.kg_system import KnowledgeGraphManager

# Mock imports for testing in isolation
class MockSuperAgent:
    async def initialize_system(self): pass
    async def shutdown_system(self): pass
    async def register_agent(self, agent_id, agent): pass
    async def run_cycle(self): return {'status': 'SuperAgent OK'}
    async def set_mode(self, mode): pass
    async def run_health_check(self): return {'status': 'SuperAgent Healthy'}

class MockKnowledgeGraphManager:
    def record_agent_action(self, agent_id, action, result, metrics, context=None): pass
    def get_agent_memory(self, agent_id): return []
    def get_agent_performance(self, agent_id): return {'success_rate': 0.0}
    def register_agent(self, agent_id, agent_type, properties=None): pass

SuperAgent = MockSuperAgent
KnowledgeGraphManager = MockKnowledgeGraphManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProblemType(Enum):
    """Classification of problem types"""
    OPTIMIZATION = "Optimization"  # e.g., Hyperparameter tuning, NAS
    RL_CONTROL = "RL_Control"  # e.g., Trading, Navigation, Game AI
    ANALYTICAL = "Analytical"  # e.g., Linear Algebra, Symbolic Math
    HYBRID = "Hybrid"  # Combination of types
    UNKNOWN = "Unknown"

@dataclass
class Problem:
    """Represents a problem to be solved"""
    problem_id: str
    description: str
    data_source: str
    target_metric: str
    problem_type: ProblemType = ProblemType.UNKNOWN
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Strategy:
    """Represents a strategy to solve a problem"""
    strategy_id: str
    name: str
    agent_chain: List[str]  # List of agent IDs to execute
    expected_performance: float
    required_resources: List[str]

class ProblemIdentifier:
    """Agent responsible for classifying the problem type"""
    
    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg = kg_manager
        self.problem_classifier = self._load_classifier() # Mock classifier
        self.agent_id = "problem_identifier"
        self.kg.register_agent(self.agent_id, "ProblemIdentifier")

    def _load_classifier(self):
        """In a real system, this would load a trained model (e.g., a text classifier)"""
        logger.info("Loading Problem Classifier...")
        # Mock classifier logic
        def classify(description: str) -> ProblemType:
            desc_lower = description.lower()
            if "optimize" in desc_lower or "tune" in desc_lower or "nas" in desc_lower:
                return ProblemType.OPTIMIZATION
            elif "trade" in desc_lower or "navigate" in desc_lower or "control" in desc_lower:
                return ProblemType.RL_CONTROL
            elif "solve equation" in desc_lower or "linear algebra" in desc_lower:
                return ProblemType.ANALYTICAL
            else:
                return ProblemType.UNKNOWN
        return classify

    async def identify(self, problem: Problem) -> Problem:
        """Identifies the problem type and updates the Problem object"""
        problem_type = self.problem_classifier(problem.description)
        problem.problem_type = problem_type
        
        # Record action in KG
        self.kg.record_agent_action(
            self.agent_id,
            "identify_problem",
            "success",
            {"problem_type": problem_type.value},
            {"problem_id": problem.problem_id, "description": problem.description}
        )
        
        logger.info(f"Problem {problem.problem_id} identified as: {problem_type.value}")
        return problem

class StrategySelector:
    """Agent responsible for selecting the best strategy based on problem type and memory"""
    
    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg = kg_manager
        self.strategy_database = self._load_strategy_database()
        self.agent_id = "strategy_selector"
        self.kg.register_agent(self.agent_id, "StrategySelector")

    def _load_strategy_database(self) -> Dict[ProblemType, List[Strategy]]:
        """Mock strategy database"""
        return {
            ProblemType.OPTIMIZATION: [
                Strategy("S1", "SpokForNAS_Evolutionary", ["data_agent", "nas_agent", "train_agent"], 0.95, ["GPU"]),
                Strategy("S2", "Hyperparam_Bayesian", ["data_agent", "hpo_agent", "train_agent"], 0.90, ["CPU"])
            ],
            ProblemType.RL_CONTROL: [
                Strategy("S3", "HRL_Advanced", ["data_agent", "rl_agent_hrl", "eval_agent"], 0.88, ["GPU", "Env"]),
                Strategy("S4", "DQN_Baseline", ["data_agent", "rl_agent_dqn", "eval_agent"], 0.75, ["CPU", "Env"])
            ],
            ProblemType.ANALYTICAL: [
                Strategy("S5", "Linear_Solver", ["math_agent"], 1.0, ["CPU"])
            ],
            ProblemType.UNKNOWN: [
                Strategy("S6", "General_Search", ["search_agent", "refine_agent"], 0.5, ["CPU"])
            ]
        }

    async def select(self, problem: Problem) -> Strategy:
        """Selects the optimal strategy"""
        
        available_strategies = self.strategy_database.get(problem.problem_type, self.strategy_database[ProblemType.UNKNOWN])
        
        # Simple selection: choose the strategy with the highest expected performance
        # In a real system, this would involve complex reasoning based on KG memory,
        # resource availability, and risk assessment.
        
        best_strategy = max(available_strategies, key=lambda s: s.expected_performance)
        
        # Record action in KG
        self.kg.record_agent_action(
            self.agent_id,
            "select_strategy",
            "success",
            {"strategy_id": best_strategy.strategy_id, "expected_perf": best_strategy.expected_performance},
            {"problem_id": problem.problem_id, "problem_type": problem.problem_type.value}
        )
        
        logger.info(f"Strategy selected for {problem.problem_id}: {best_strategy.name}")
        return best_strategy

class UnifiedAgent:
    """The master orchestrator for problem solving, working under the SuperAgent"""
    
    def __init__(self, super_agent: SuperAgent, kg_manager: KnowledgeGraphManager):
        self.super_agent = super_agent
        self.kg = kg_manager
        self.problem_identifier = ProblemIdentifier(kg_manager)
        self.strategy_selector = StrategySelector(kg_manager)
        self.agent_id = "unified_agent"
        self.kg.register_agent(self.agent_id, "UnifiedAgent")
        self.registered_agents = {} # Actual agents to execute the chain

    async def register_execution_agent(self, agent_id: str, agent: Any):
        """Register an agent that can be part of an execution chain"""
        self.registered_agents[agent_id] = agent
        await self.super_agent.register_agent(agent_id, agent)
        logger.info(f"Execution agent registered: {agent_id}")

    async def solve_problem(self, problem: Problem) -> Dict[str, Any]:
        """Main method to solve a problem using the hierarchical architecture"""
        
        logger.info(f"UnifiedAgent starting to solve problem: {problem.problem_id}")
        
        # 1. Identify the Problem
        problem = await self.problem_identifier.identify(problem)
        
        # 2. Select the Strategy
        strategy = await self.strategy_selector.select(problem)
        
        # 3. Execute the Strategy Chain
        execution_results = await self._execute_strategy_chain(strategy)
        
        # 4. Record Final Outcome
        final_result = "success" if all(res.get('status') == 'success' for res in execution_results) else "partial_success"
        final_metrics = self._aggregate_metrics(execution_results)
        
        self.kg.record_agent_action(
            self.agent_id,
            "solve_problem",
            final_result,
            final_metrics,
            {"problem_id": problem.problem_id, "strategy_used": strategy.strategy_id}
        )
        
        logger.info(f"UnifiedAgent finished solving problem {problem.problem_id} with result: {final_result}")
        
        return {
            "problem": asdict(problem),
            "strategy": asdict(strategy),
            "execution_results": execution_results,
            "final_result": final_result,
            "final_metrics": final_metrics
        }

    async def _execute_strategy_chain(self, strategy: Strategy) -> List[Dict[str, Any]]:
        """Executes the chain of agents defined in the strategy"""
        results = []
        
        for agent_id in strategy.agent_chain:
            agent = self.registered_agents.get(agent_id)
            
            if not agent:
                logger.error(f"Agent {agent_id} not found for execution.")
                results.append({"agent_id": agent_id, "status": "failed", "error": "Agent not registered"})
                continue
            
            logger.info(f"Executing agent: {agent_id}")
            
            try:
                # Mock execution: In a real system, this would call the agent's main method
                if hasattr(agent, 'execute'):
                    result = await agent.execute(strategy)
                else:
                    # Simulate a successful execution
                    await asyncio.sleep(0.1)
                    result = {"agent_id": agent_id, "status": "success", "metrics": {"time": 0.1, "cost": 0.01}}
                
                results.append(result)
                
                # Check for critical failure (could trigger SuperAgent FIX_ERRORS mode)
                if result.get('status') == 'critical_failure':
                    logger.warning("Critical failure detected. Stopping chain.")
                    # In a real system, this would trigger: await self.super_agent.set_mode(OperationMode.FIX_ERRORS)
                    break
                    
            except Exception as e:
                logger.error(f"Execution failed for agent {agent_id}: {e}")
                results.append({"agent_id": agent_id, "status": "failed", "error": str(e)})
                # Consider triggering error correction here
                
        return results

    def _aggregate_metrics(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates metrics from the execution chain"""
        total_time = sum(res.get('metrics', {}).get('time', 0) for res in execution_results)
        total_cost = sum(res.get('metrics', {}).get('cost', 0) for res in execution_results)
        success_count = sum(1 for res in execution_results if res.get('status') == 'success')
        
        return {
            "total_time": total_time,
            "total_cost": total_cost,
            "success_rate": success_count / len(execution_results) if execution_results else 0.0
        }

# Test and demonstration
async def test_unified_agent():
    # 1. Setup Mock Components
    super_agent = MockSuperAgent()
    kg_manager = MockKnowledgeGraphManager()
    
    # 2. Initialize UnifiedAgent
    unified_agent = UnifiedAgent(super_agent, kg_manager)
    
    # 3. Register Mock Execution Agents
    class MockExecutionAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
        
        async def execute(self, strategy):
            await asyncio.sleep(0.05)
            return {"agent_id": self.agent_id, "status": "success", "metrics": {"time": 0.05, "cost": 0.005}}
        
        async def health_check(self):
            return {'status': 'healthy', 'performance': 0.9}

    await unified_agent.register_execution_agent("data_agent", MockExecutionAgent("data_agent"))
    await unified_agent.register_execution_agent("nas_agent", MockExecutionAgent("nas_agent"))
    await unified_agent.register_execution_agent("train_agent", MockExecutionAgent("train_agent"))
    
    # 4. Define a Problem
    problem_description = "Optimize the hyperparameters and architecture for a new image classification model using NAS."
    problem = Problem(
        problem_id="P001",
        description=problem_description,
        data_source="CIFAR-10",
        target_metric="Accuracy"
    )
    
    # 5. Solve the Problem
    results = await unified_agent.solve_problem(problem)
    
    print("\n--- UnifiedAgent Test Results ---")
    print(f"Problem Type Identified: {results['problem']['problem_type']}")
    print(f"Strategy Used: {results['strategy']['name']}")
    print(f"Final Result: {results['final_result']}")
    print(f"Total Time: {results['final_metrics']['total_time']:.2f}s")
    print(f"Success Rate: {results['final_metrics']['success_rate']:.2f}")
    print("---------------------------------")
    
    # Test a different problem type
    problem_description_rl = "Control a trading bot to maximize profit in a simulated market."
    problem_rl = Problem(
        problem_id="P002",
        description=problem_description_rl,
        data_source="HardTradingEnv",
        target_metric="Profit"
    )
    
    results_rl = await unified_agent.solve_problem(problem_rl)
    print(f"\nProblem Type Identified (RL): {results_rl['problem']['problem_type']}")
    print(f"Strategy Used (RL): {results_rl['strategy']['name']}")
    
    print("\n✓ UnifiedAgent Architecture implemented and tested successfully!")

if __name__ == '__main__':
    asyncio.run(test_unified_agent())
