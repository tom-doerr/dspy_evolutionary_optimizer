"""Core tests for the evolutionary prompt optimizer."""

import pytest
import dspy
from typing import Callable, Any
from evoprompt.chromosome import Chromosome
from evoprompt import FullyEvolutionaryPromptOptimizer

@pytest.fixture
def mock_metric() -> Callable[[Any, Any], float]:
    def metric(pred: Any, example: Any) -> float:
        return 1.0
    return metric

@pytest.fixture
def basic_optimizer(mock_metric: Callable[[Any, Any], float) -> FullyEvolutionaryPromptOptimizer:
    return FullyEvolutionaryPromptOptimizer(mock_metric)

@pytest.fixture
def mock_signature() -> dspy.Signature:
    signature = dspy.Signature("text -> label")
    signature.__doc__ = "Given text, generate a label"
    return signature

def test_optimizer_initialization(mock_metric: Callable[[Any, Any], float]) -> None:
    """Test optimizer initialization with various parameters."""
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=mock_metric,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True
    )
    
    # Verify configuration
    assert optimizer.config.metric == mock_metric
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.growth_rate == 0.3
    assert optimizer.config.max_population == 20
    assert optimizer.config.debug is True
    
    # Verify state
    assert optimizer.state.inference_count == 0
    assert optimizer.state.population is None
    assert optimizer.state.history is None

def test_parallel_initialization(mock_metric: Callable[[Any, Any], float) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=4)
    assert optimizer.max_workers == 4

def test_population_initialization(basic_optimizer: FullyEvolutionaryPromptOptimizer) -> None:
    population = basic_optimizer._initialize_population()
    assert len(population) == 1
    assert isinstance(population[0]["chromosome"], Chromosome)
    assert population[0]["score"] is None

def test_optimizer_with_mock_mode(mock_metric: Callable[[Any, Any], float) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
    assert optimizer.use_mock is True

def test_parameter_validation(mock_metric: Callable[[Any, Any], float) -> None:
    # Test invalid metric
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer("not_a_function")

    # Test invalid generations
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, generations=-1)

    # Test invalid mutation rate
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, mutation_rate=-0.1)

    # Test invalid max workers
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=-1)

def test_population_handling(mock_metric: Callable[[Any, Any], float) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    # Test empty population
    with pytest.raises(ValueError):
        optimizer._update_population([], iteration=1, recent_scores=[])

    # Test population update
    population = optimizer._initialize_population()
    updated = optimizer._update_population(population, iteration=1, recent_scores=[0.9, 0.8])
    assert len(updated) <= 100  # Default max population size

def test_mutation_logic(mock_metric: Callable[[Any, Any], float) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    original = "Given {{input}}, generate {{output}}"
    mutated = optimizer._mutate(original)
    assert "{{input}}" in mutated
    assert "{{output}}" in mutated
    assert len(mutated) > len(original)

def test_crossover_logic(mock_metric: Callable[[Any, Any], float) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    p1 = "Given {{input}}, generate {{output}}"
    p2 = "Analyze {{input}} and produce {{output}}"
    crossed = optimizer._crossover(p1, p2)
    assert "{{input}}" in crossed
    assert "{{output}}" in crossed
    assert len(crossed) > min(len(p1), len(p2))

def test_prompt_validation(mock_metric: Callable[[Any, Any], float) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    with pytest.raises(TypeError):
        optimizer._ensure_placeholders(None)
    with pytest.raises(TypeError):
        optimizer._ensure_placeholders(123)
    with pytest.raises(ValueError):
        optimizer._ensure_placeholders("No placeholders here")
