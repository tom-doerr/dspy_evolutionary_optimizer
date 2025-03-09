"""Core tests for the evolutionary prompt optimizer."""

import pytest
from evoprompt.optimizer import FullyEvolutionaryPromptOptimizer
from evoprompt.chromosome import Chromosome
import dspy

def test_optimizer_initialization():
    """Test basic optimizer initialization."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=mock_metric,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True
    )
    
    assert optimizer.generations == 5
    assert optimizer.mutation_rate == 0.5
    assert optimizer.growth_rate == 0.3
    assert optimizer.max_population == 20
    assert optimizer.debug is True

def test_population_initialization():
    """Test that population is initialized correctly."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    population = optimizer._initialize_population()
    
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
    selected = optimizer._select_prompt(population)
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
    updated = optimizer._update_population(population, iteration=1, recent_scores=[0.9, 0.8])
    assert len(updated) <= optimizer.max_population

def test_mutation_logic():
    """Test prompt mutation functionality."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    original = "Given {{input}}, generate {{output}}"
    mutated = optimizer._mutate(original)
    
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
    
    crossed = optimizer._crossover(p1, p2)
    
    # Basic crossover checks
    assert "{{input}}" in crossed
    assert "{{output}}" in crossed
    assert len(crossed) > min(len(p1), len(p2))

def test_parallel_evaluation():
    """Test parallel evaluation functionality."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=2)
    
    # Create mock program and examples
    signature = dspy.Signature("text -> label")
    program = dspy.Predict(signature)
    
    examples = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
    ]
    
    # Test parallel evaluation
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0

def test_mock_prediction():
    """Test mock prediction generation."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
    
    signature = dspy.Signature("text -> label")
    example = dspy.Example(text="Great product!", label="positive")
    
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

def test_evolution_history():
    """Test evolution history tracking."""
    def mock_metric(pred, example):
        return 1.0 if pred.label == example.label else 0.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    
    # Simulate some evolution
    population = optimizer._initialize_population()
    optimizer._log_progress(1, population)
    
    history = optimizer.get_history()
    assert len(history) == 1
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]
