import pytest

from evoprompt.optimizer import FullyEvolutionaryPromptOptimizer


@pytest.fixture
def mock_metric():
    def metric(pred, example):
        return 1.0
    return metric

def test_optimizer_initialization(mock_metric):
    """Test basic optimizer initialization."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    assert optimizer.metric == mock_metric
    assert optimizer.generations == 10
    assert optimizer.mutation_rate == 0.5
    assert optimizer.growth_rate == 0.3
    assert optimizer.max_population == 100
    assert optimizer.max_inference_calls == 100
    assert optimizer.debug is False
    assert optimizer.use_mock is False
    assert optimizer.max_workers == 1

def test_parallel_initialization(mock_metric):
    """Test optimizer initialization with parallel workers."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=4)
    assert optimizer.max_workers == 4

def test_initial_population(mock_metric):
    """Test that initial population is created correctly."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    population = optimizer._initialize_population()
    
    assert len(population) == 1
    assert population[0]["prompt"] == "{{input}} {{output}}"
    assert population[0]["score"] is None
    assert population[0]["last_used"] == 0

def test_optimizer_with_mock_mode(mock_metric):
    """Test optimizer with mock mode enabled."""
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
    assert optimizer.use_mock is True
