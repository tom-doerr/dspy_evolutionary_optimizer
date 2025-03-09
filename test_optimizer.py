"""Test suite for the evolutionary prompt optimizer."""

import pytest
from evoprompt.optimizer import FullyEvolutionaryPromptOptimizer


@pytest.fixture
def mock_metric():
    """Fixture providing a mock metric function that always returns 1.0."""
    def metric(pred, example):  # pylint: disable=unused-argument
        return 1.0
    return metric

def test_optimizer_initialization(mock_metric):  # pylint: disable=redefined-outer-name
    """Test basic optimizer initialization."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    # Test public interface only
    assert optimizer.use_mock is False

def test_parallel_initialization(mock_metric):  # pylint: disable=redefined-outer-name
    """Test optimizer initialization with parallel workers."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=4)
    assert optimizer.max_workers == 4

def test_initial_population(mock_metric):  # pylint: disable=redefined-outer-name
    """Test that initial population is created correctly."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    population = optimizer._initialize_population()  # pylint: disable=protected-access
    
    assert len(population) == 1
    assert population[0]["prompt"] == "{{input}} {{output}}"
    assert population[0]["score"] is None
    assert population[0]["last_used"] == 0

def test_optimizer_with_mock_mode(mock_metric):  # pylint: disable=redefined-outer-name
    """Test optimizer with mock mode enabled."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
    assert optimizer.use_mock is True

def test_invalid_metric_function(mock_metric):  # pylint: disable=redefined-outer-name,unused-argument
    """Test handling of invalid metric functions."""
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer("not_a_function")

def test_invalid_generations(mock_metric):  # pylint: disable=redefined-outer-name
    """Test validation of generations parameter."""
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, generations=-1)

def test_invalid_mutation_rate(mock_metric):  # pylint: disable=redefined-outer-name
    """Test validation of mutation rate parameter."""
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, mutation_rate=-0.1)

def test_invalid_max_workers(mock_metric):  # pylint: disable=redefined-outer-name
    """Test validation of max workers parameter."""
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=-1)

def test_empty_population_handling(mock_metric):  # pylint: disable=redefined-outer-name
    """Test handling of empty population scenarios."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    with pytest.raises(ValueError):
        optimizer._update_population([], iteration=1, recent_scores=[])  # pylint: disable=protected-access
