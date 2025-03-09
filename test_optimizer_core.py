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


def test_parallel_evaluation(_mock_metric: Callable[[Any, Any], float]) -> None:
    # Test normal case
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric, max_workers=2)
    signature = dspy.Signature("text -> label")
    signature.__doc__ = "Given text, generate a label"
    program = dspy.Predict(signature)
    examples = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
    ]
    # Access protected member for testing purposes
    # pylint: disable=protected-access
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

    signature = dspy.Signature("text -> label", "Given text, generate a label")
    program = dspy.Predict(signature)

    examples = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
    ]

    score = optimizer._evaluate(program, Chromosome(), examples)
    assert 0 <= score <= 1.0


def test_mock_prediction(_mock_metric: Callable[[Any, Any], float]) -> None:
    # Test basic mock prediction
    optimizer = FullyEvolutionaryPromptOptimizer(metric=mock_metric)
    optimizer.config.use_mock = True  # Set mock mode through config
    signature = dspy.Signature("text -> label", "Given text, generate a label")
    example = dspy.Example(text="Great product!", label="positive")
    # Access protected member for testing purposes
    # pylint: disable=protected-access
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

    # Test complex input
    signature = dspy.Signature(
        "text,metadata -> label", "Given text and metadata, generate label"
    )
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
    optimizer._log_progress(1, population)

    # Initialize history if empty
    if not optimizer.history:
        optimizer.history = []

    history = optimizer.get_history()
    assert len(history) == 1
    assert "iteration" in history[0]
    assert "best_score" in history[0]
    assert "population_size" in history[0]


def test_parallel_execution_edge_cases(
    mock_metric: Callable[[Any, Any], float],
) -> None:
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
    with pytest.raises(ValueError):
        optimizer._create_mock_prediction(None, {}, None)

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
