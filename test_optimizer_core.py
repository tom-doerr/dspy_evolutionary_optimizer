"""Core tests for the evolutionary prompt optimizer."""

import pytest
import dspy

from evoprompt.chromosome import Chromosome
from evoprompt import FullyEvolutionaryPromptOptimizer

@pytest.fixture
def mock_metric():
    """Fixture providing a mock metric function."""
    def metric(pred, example):  # pylint: disable=unused-argument
        return 1.0
    return metric

@pytest.fixture
def basic_optimizer(mock_metric):
    """Fixture providing a basic optimizer instance."""
    return FullyEvolutionaryPromptOptimizer(mock_metric)

@pytest.fixture
def mock_signature():
    """Fixture providing a mock DSPy signature."""
    signature = dspy.Signature("text -> label")
    signature.__doc__ = "Given text, generate a label"
    return signature

@pytest.fixture
def configured_optimizer(mock_metric):
    """Fixture providing a configured optimizer instance."""
    return FullyEvolutionaryPromptOptimizer(
        metric=mock_metric,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True
    )

@pytest.fixture
def parallel_optimizer(mock_metric):
    """Fixture providing an optimizer configured for parallel execution."""
    return FullyEvolutionaryPromptOptimizer(
        metric=mock_metric,
        max_workers=2
    )

@pytest.fixture
def mock_optimizer(mock_metric):
    """Fixture providing an optimizer configured for mock mode."""
    return FullyEvolutionaryPromptOptimizer(
        metric=mock_metric,
        use_mock=True
    )


def test_optimizer_initialization(configured_optimizer):
    """Test basic optimizer initialization."""
    assert configured_optimizer.use_mock is False

def test_population_initialization(basic_optimizer):
    """Test that population is initialized correctly."""
    population = basic_optimizer._initialize_population()
    
    assert len(population) == 1
    assert isinstance(population[0]["chromosome"], Chromosome)
    assert population[0]["score"] is None

def test_prompt_selection():
    """Test prompt selection using Pareto distribution."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    # Create test population with varying scores
    population = [
        {"prompt": "prompt1", "score": 0.9, "last_used": 0},
        {"prompt": "prompt2", "score": 0.8, "last_used": 0},
        {"prompt": "prompt3", "score": 0.7, "last_used": 0},
        {"prompt": "prompt4", "score": 0.6, "last_used": 0},
    ]
    
    # Test selection favors higher scores
    selected = optimizer._select_prompt(population)  # pylint: disable=protected-access
    assert selected["score"] >= 0.6

def test_population_update():
    """Test population update logic."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    population = [
        {"prompt": "prompt1", "score": 0.9, "last_used": 0},
        {"prompt": "prompt2", "score": 0.8, "last_used": 0},
        {"prompt": "prompt3", "score": 0.7, "last_used": 0},
        {"prompt": "prompt4", "score": 0.6, "last_used": 0},
    ]
    
    # Test population size is maintained
    updated = optimizer._update_population(population, iteration=1, recent_scores=[0.9, 0.8])  # pylint: disable=protected-access
    assert len(updated) <= 100  # Default max population size

def test_mutation_logic():
    """Test prompt mutation functionality."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    original = "Given {{input}}, generate {{output}}"
    mutated = optimizer._mutate(original)  # pylint: disable=protected-access
    
    # Basic mutation checks
    assert "{{input}}" in mutated
    assert "{{output}}" in mutated
    assert len(mutated) > len(original)

def test_crossover_logic():
    """Test prompt crossover functionality."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    p1 = "Given {{input}}, generate {{output}}"
    p2 = "Analyze {{input}} and produce {{output}}"
    
    crossed = optimizer._crossover(p1, p2)  # pylint: disable=protected-access
    
    # Basic crossover checks
    assert "{{input}}" in crossed
    assert "{{output}}" in crossed
    assert len(crossed) > min(len(p1), len(p2))

def test_parallel_evaluation():
    """Test parallel evaluation functionality."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = optimizer(max_workers=2)
    
    # Create mock program and examples
    signature = dspy.Signature("text -> label")
    signature.__doc__ = "Given text, generate a label"
    program = dspy.Predict(signature)
    
    examples = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
    ]
    
    # Test parallel evaluation
    score = optimizer._evaluate(program, Chromosome(), examples)  # pylint: disable=protected-access
    assert 0 <= score <= 1.0

def test_mock_prediction():
    """Test mock prediction generation."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = optimizer(use_mock=True)
    
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Great product!", label="positive")
    
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)  # pylint: disable=protected-access
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

def test_evolution_history():
    """Test evolution history tracking."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    # Simulate some evolution
    population = optimizer._initialize_population()  # pylint: disable=protected-access
    optimizer._log_progress(1, population)  # pylint: disable=protected-access
    
    history = optimizer.get_history()
    assert len(history) == 1
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]

def test_mock_prediction_validation():
    """Test mock prediction validation."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
    
    # Test with invalid signature
    with pytest.raises(ValueError):
        optimizer._create_mock_prediction(None, {}, None)

    # Test with empty input kwargs
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    with pytest.raises(ValueError):
        optimizer._create_mock_prediction(signature, {}, example)

def test_parallel_execution_edge_cases():
    """Test edge cases in parallel execution."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = optimizer(max_workers=2)
    
    # Test with empty examples
    signature = dspy.Signature("text -> label")
    program = dspy.Predict(signature)
    with pytest.raises(ValueError):
        optimizer._evaluate(program, Chromosome(), [])

    # Test with invalid program
    with pytest.raises(TypeError):
        optimizer._evaluate(None, Chromosome(), [dspy.Example(text="test", label="test")])

def test_prompt_validation():
    """Test prompt validation logic."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    # Test invalid prompt types
    with pytest.raises(TypeError):
        optimizer._ensure_placeholders(None)
    with pytest.raises(TypeError):
        optimizer._ensure_placeholders(123)

    # Test missing placeholders
    with pytest.raises(ValueError):
        optimizer._ensure_placeholders("No placeholders here")
