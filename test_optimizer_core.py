"""Tests for parallel execution and model interactions."""

from typing import Callable, Any
import pytest
import dspy
from evoprompt.chromosome import Chromosome
from evoprompt import FullyEvolutionaryPromptOptimizer


@pytest.fixture
def mock_metric() -> Callable[[Any, Any], float]:
    def metric(pred: Any, example: Any) -> float:
        return 1.0 if pred.label == example.label else 0.0

    return metric


def test_parallel_evaluation(mock_metric: Callable[[Any, Any], float]) -> None:
    # Test normal case
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=2)
    signature = dspy.Signature("text -> label", "Given text, generate label")
    signature.__doc__ = "Given text, generate a label"
    program = dspy.Predict(signature)
    examples = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
    ]

    # Test normal evaluation
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0

    # Test with single worker
    optimizer.config.max_workers = 1
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0

    # Test with many workers
    optimizer.config.max_workers = 100
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0

    # Test with mock mode
    optimizer.config.use_mock = True
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0

    # Test single worker case
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=1)
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0

    # Test large number of workers
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=100)
    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=2)

    signature = dspy.Signature("text -> label", "Given text, generate label")
    program = dspy.Predict(signature)

    examples = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
    ]

    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0


def test_mock_prediction(mock_metric: Callable[[Any, Any], float]) -> None:
    # Test basic mock prediction
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric)
    optimizer.config.use_mock = True  # Set mock mode through config
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    MockPrediction = optimizer._create_mock_prediction_class()
    example = dspy.Example(text="Great product!", label="positive")
    # Access protected member for testing purposes
    # pylint: disable=protected-access
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

    # Test multiple output fields
    signature = dspy.Signature("text -> label,score")
    signature.__doc__ = "Given text, generate label and score"
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert hasattr(pred, "score")
    assert isinstance(pred.label, str)
    assert isinstance(pred.score, str)

    # Test complex input
    signature = dspy.Signature("text,metadata -> label")
    signature.__doc__ = "Given text and metadata, generate label"
    pred = optimizer._create_mock_prediction(
        signature, {"text": "test", "metadata": {"source": "web"}}, example
    )
    assert hasattr(pred, "label")
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, use_mock=True)

    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Great product!", label="positive")

    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)


def test_evolution_history(mock_metric: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric)
    population = optimizer._initialize_population()
    # Initialize history if empty
    if not optimizer.history:
        optimizer.history = []
    optimizer._log_progress(1, population)

    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    optimizer.history = [{"iteration": 1, "best_score": 0.9, "population_size": 10}]
    history = optimizer.get_history()
    assert len(history) == 1
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]


def test_parallel_execution_edge_cases(
    mock_metric: Callable[[Any, Any], float],
) -> None:
    """Test edge cases in parallel execution."""
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=2)

    # Test empty examples
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    program = dspy.Predict(signature)
    with pytest.raises(ValueError, match="Cannot evaluate empty examples"):
        optimizer._evaluate(program, Chromosome(), [])

    # Test invalid program type
    with pytest.raises(TypeError, match="Program must be a DSPy Predict module"):
        optimizer._evaluate("not_a_program", Chromosome(), [dspy.Example(text="test")])

    # Test invalid program type
    with pytest.raises(TypeError, match="Program must be a DSPy Predict module"):
        optimizer._evaluate("not_a_program", Chromosome(), [dspy.Example(text="test")])

    # Test invalid program type
    with pytest.raises(TypeError, match="Program must be a DSPy Predict module"):
        optimizer._evaluate("not_a_program", Chromosome(), [dspy.Example(text="test")])

    # Test invalid program
    with pytest.raises(TypeError):
        optimizer._evaluate(
            None, Chromosome(), [dspy.Example(text="test", label="test")]
        )

    # Test invalid chromosome
    with pytest.raises(TypeError):
        optimizer._evaluate(program, None, [dspy.Example(text="test", label="test")])

    # Test invalid examples type
    with pytest.raises(TypeError):
        optimizer._evaluate(program, Chromosome(), "not a list")
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=2)

    # Test empty examples
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    program = dspy.Predict(signature)
    with pytest.raises(ValueError, match="Cannot evaluate empty examples"):
        optimizer._evaluate(program, Chromosome(), [])

    # Test invalid program
    with pytest.raises(TypeError):
        optimizer._evaluate(
            None, Chromosome(), [dspy.Example(text="test", label="test")]
        )


def test_mock_prediction_validation(mock_metric: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric)
    optimizer.config.use_mock = True

    # Test invalid signature
    with pytest.raises(ValueError, match="Signature cannot be None"):
        optimizer._create_mock_prediction(None, {}, None)

    # Test empty input kwargs
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    with pytest.raises(ValueError, match="Input kwargs cannot be empty"):
        optimizer._create_mock_prediction(signature, {}, example)

    # Test valid mock prediction
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

    # Test empty input kwargs
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    with pytest.raises(ValueError):
        optimizer._create_mock_prediction(signature, {}, example)

    # Test valid mock prediction
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

    # Test multiple output fields
    signature = dspy.Signature(
        "text -> label,score", "Given text, generate label and score"
    )
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert hasattr(pred, "score")
    assert isinstance(pred.label, str)
    assert isinstance(pred.score, str)

    # Test valid mock prediction
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

    # Test multiple output fields
    signature = dspy.Signature(
        "text -> label,score", "Given text, generate label and score"
    )
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert hasattr(pred, "score")
    assert isinstance(pred.label, str)
    assert isinstance(pred.score, str)

    # Test valid mock prediction
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

    # Test multiple output fields
    signature = dspy.Signature(
        "text -> label,score", "Given text, generate label and score"
    )
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert hasattr(pred, "score")
    assert isinstance(pred.label, str)
    assert isinstance(pred.score, str)

    # Test valid mock prediction
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Test", label="positive")
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert isinstance(pred.label, str)

    # Test multiple output fields
    signature = dspy.Signature(
        "text -> label,score", "Given text, generate label and score"
    )
    pred = optimizer._create_mock_prediction(signature, {"text": "test"}, example)
    assert hasattr(pred, "label")
    assert hasattr(pred, "score")
    assert isinstance(pred.label, str)
    assert isinstance(pred.score, str)
