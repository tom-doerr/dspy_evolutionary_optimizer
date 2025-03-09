"""Core tests for the evolutionary prompt optimizer."""

from typing import Callable, Any
import pytest
import dspy
from evoprompt.chromosome import Chromosome
from evoprompt import FullyEvolutionaryPromptOptimizer


@pytest.fixture(name="mock_metric")
def _mock_metric() -> Callable[[Any, Any], float]:
    def metric(_pred: Any, _example: Any) -> float:
        return 1.0

    return metric


@pytest.fixture(name="metric_fixture")
def _metric_fixture() -> Callable[[Any, Any], float]:
    def metric(_pred: Any, _example: Any) -> float:
        return 1.0

    return metric


@pytest.fixture(name="basic_optimizer_fixture")
def basic_optimizer_fixture(
    mock_metric: Callable[[Any, Any], float],
) -> FullyEvolutionaryPromptOptimizer:
    return FullyEvolutionaryPromptOptimizer(mock_metric)


@pytest.fixture
def mock_signature() -> dspy.Signature:
    signature = dspy.Signature("text -> label")
    signature.__doc__ = "Given text, generate a label"
    return signature


def test_optimizer_initialization(_mock_metric: Callable[[Any, Any], float]) -> None:
    """Test optimizer initialization with various parameters."""
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_mock_metric,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
    )

    # Verify configuration
    assert optimizer.config.metric == _mock_metric
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.growth_rate == 0.3
    assert optimizer.config.max_population == 20
    assert optimizer.config.debug is True

    # Verify state
    assert optimizer.state.inference_count == 0
    assert optimizer.state.population is None
    assert optimizer.state.history is None


def test_parallel_initialization(_metric_fixture: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture, max_workers=4)
    assert optimizer.config.max_workers == 4


def test_population_initialization(
    _basic_optimizer_fixture: FullyEvolutionaryPromptOptimizer,
) -> None:
    # Access protected member for testing purposes
    # pylint: disable=protected-access
    population = basic_optimizer_fixture._initialize_population()
    assert len(population) == 1
    assert isinstance(population[0]["chromosome"], Chromosome)
    assert population[0]["score"] is None


def test_optimizer_with_mock_mode(_metric_fixture: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture, use_mock=True)
    assert optimizer.config.use_mock is True


def _test_parameter_validation_cases(
    optimizer: FullyEvolutionaryPromptOptimizer,
) -> None:
    """Test various parameter validation cases."""
    # Test invalid generations
    with pytest.raises(ValueError):
        optimizer("not_a_function", generations=0)
    with pytest.raises(ValueError):
        optimizer("not_a_function", generations=-1)

    # Test invalid mutation rate
    with pytest.raises(ValueError):
        optimizer("not_a_function", mutation_rate=1.1)
    with pytest.raises(ValueError):
        optimizer("not_a_function", mutation_rate=-0.1)

    # Test invalid max workers
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_workers=0)
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_workers=-1)

    # Test invalid growth rate
    with pytest.raises(ValueError):
        optimizer("not_a_function", growth_rate=1.1)
    with pytest.raises(ValueError):
        optimizer("not_a_function", growth_rate=-0.1)

    # Test invalid max population
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_population=0)
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_population=-1)

    # Test invalid debug type
    with pytest.raises(TypeError):
        optimizer("not_a_function", debug="not_a_boolean")

    # Test invalid use_mock type
    with pytest.raises(TypeError):
        optimizer("not_a_function", use_mock="not_a_boolean")


def _test_parameter_validation_cases(
    optimizer: FullyEvolutionaryPromptOptimizer,
) -> None:
    """Test various parameter validation cases."""
    # Test invalid generations
    with pytest.raises(ValueError):
        optimizer("not_a_function", generations=0)
    with pytest.raises(ValueError):
        optimizer("not_a_function", generations=-1)

    # Test invalid mutation rate
    with pytest.raises(ValueError):
        optimizer("not_a_function", mutation_rate=1.1)
    with pytest.raises(ValueError):
        optimizer("not_a_function", mutation_rate=-0.1)

    # Test invalid max workers
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_workers=0)
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_workers=-1)

    # Test invalid growth rate
    with pytest.raises(ValueError):
        optimizer("not_a_function", growth_rate=1.1)
    with pytest.raises(ValueError):
        optimizer("not_a_function", growth_rate=-0.1)

    # Test invalid max population
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_population=0)
    with pytest.raises(ValueError):
        optimizer("not_a_function", max_population=-1)

    # Test invalid debug type
    with pytest.raises(TypeError):
        optimizer("not_a_function", debug="not_a_boolean")

    # Test invalid use_mock type
    with pytest.raises(TypeError):
        optimizer("not_a_function", use_mock="not_a_boolean")


def test_parameter_validation(metric_fixture: Callable[[Any, Any], float]) -> None:
    # Remove duplicate docstring
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0

    # Test invalid parameters
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_population=0)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric="not_a_function")
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_workers=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, growth_rate=-0.1)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, debug="not_a_boolean")
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(
            metric=metric_fixture, use_mock="not_a_boolean"
        )
    """Test validation of optimizer parameters."""
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0

    # Test invalid parameters
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_population=0)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric="not_a_function")
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_workers=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, growth_rate=-0.1)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, debug="not_a_boolean")
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(
            metric=metric_fixture, use_mock="not_a_boolean"
        )
    """Test validation of optimizer parameters."""
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0

    # Test invalid parameters
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_population=0)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric="not_a_function")
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_workers=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, growth_rate=-0.1)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, debug="not_a_boolean")
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(
            metric=metric_fixture, use_mock="not_a_boolean"
        )
    """Test validation of optimizer parameters."""
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0

    # Test invalid parameters
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_population=0)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric="not_a_function")
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_workers=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, growth_rate=-0.1)
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, debug="not_a_boolean")
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(
            metric=metric_fixture, use_mock="not_a_boolean"
        )
    """Test validation of optimizer parameters."""
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0

    # Test invalid parameters
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=metric_fixture, max_population=0)
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0

    # Test invalid parameters
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=_metric_fixture, generations=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=_metric_fixture, mutation_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(metric=_metric_fixture, max_population=0)
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_metric_fixture,
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=_metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0
    # Test valid parameters
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,  # pylint: disable=undefined-variable
        generations=5,
        mutation_rate=0.5,
        growth_rate=0.3,
        max_population=20,
        debug=True,
        use_mock=True,
    )
    assert optimizer.config.generations == 5
    assert optimizer.config.mutation_rate == 0.5
    assert optimizer.config.use_mock is True

    # Test edge cases
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=metric_fixture,
        generations=1,
        mutation_rate=0.0,
        growth_rate=0.0,
        max_population=1,
        debug=False,
        use_mock=False,
    )
    assert optimizer.config.generations == 1
    assert optimizer.config.mutation_rate == 0.0
    # Test invalid metric
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer("not_a_function")

    # Test invalid generations
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(
            metric=_mock_metric, generations=0
        )  # pylint: disable=undefined-variable
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

    # Test invalid growth rate
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, growth_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, growth_rate=-0.1)

    # Test invalid max population
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_population=0)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_population=-1)

    # Test invalid debug type
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(mock_metric, debug="not_a_boolean")

    # Test invalid use_mock type
    with pytest.raises(TypeError):
        FullyEvolutionaryPromptOptimizer(mock_metric, use_mock="not_a_boolean")
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

    # Test invalid growth rate
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, growth_rate=1.1)
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, growth_rate=-0.1)

    # Test invalid max population
    with pytest.raises(ValueError):
        FullyEvolutionaryPromptOptimizer(mock_metric, max_population=0)
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


def test_population_handling(_metric_fixture: Callable[[Any, Any], float]) -> None:
    """Test population handling and evolution logic."""
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture)

    # Test empty population
    with pytest.raises(ValueError):
        optimizer._update_population([], iteration=1, recent_scores=[])

    # Test population initialization
    population = optimizer._initialize_population()
    assert len(population) == 1
    assert isinstance(population[0]["chromosome"], Chromosome)
    assert population[0]["score"] is None

    # Test population update
    updated = optimizer._update_population(
        population, iteration=1, recent_scores=[0.9, 0.8]
    )
    assert len(updated) <= 100  # Default max population size

    # Test population scoring
    scored_population = optimizer._select_using_pareto(population)
    assert len(scored_population) > 0
    assert all("score" in item for item in scored_population)

    # Test population selection
    selected = optimizer._select_prompt(population)
    assert selected is not None
    assert isinstance(selected, dict)
    assert "chromosome" in selected

    # Test edge cases
    with pytest.raises(TypeError):
        optimizer._update_population(None, iteration=1, recent_scores=[0.9])
    with pytest.raises(TypeError):
        optimizer._update_population([{"invalid": "data"}], iteration=1, recent_scores=[0.9])
    """Test population handling and evolution logic."""
    optimizer = FullyEvolutionaryPromptOptimizer(metric=metric_fixture)

    # Test empty population
    with pytest.raises(ValueError):
        optimizer._update_population([], iteration=1, recent_scores=[])

    # Test population initialization
    population = optimizer._initialize_population()
    assert len(population) == 1
    assert isinstance(population[0]["chromosome"], Chromosome)
    assert population[0]["score"] is None

    # Test population update
    updated = optimizer._update_population(
        population, iteration=1, recent_scores=[0.9, 0.8]
    )
    assert len(updated) <= 100  # Default max population size

    # Test population scoring
    scored_population = optimizer._select_using_pareto(population)
    assert len(scored_population) > 0
    assert all("score" in item for item in scored_population)

    # Test population selection
    selected = optimizer._select_prompt(population)
    assert selected is not None
    assert isinstance(selected, dict)
    assert "chromosome" in selected

    # Test edge cases
    with pytest.raises(TypeError):
        optimizer._update_population(None, iteration=1, recent_scores=[0.9])
    with pytest.raises(TypeError):
        optimizer._update_population(
            [{"invalid": "data"}], iteration=1, recent_scores=[0.9]
        )
    """Test population handling and evolution logic."""
    optimizer = FullyEvolutionaryPromptOptimizer(metric=metric_fixture)

    # Test empty population
    with pytest.raises(ValueError):
        optimizer._update_population([], iteration=1, recent_scores=[])

    # Test population initialization
    population = optimizer._initialize_population()
    assert len(population) == 1
    assert isinstance(population[0]["chromosome"], Chromosome)
    assert population[0]["score"] is None

    # Test population update
    updated = optimizer._update_population(
        population, iteration=1, recent_scores=[0.9, 0.8]
    )
    assert len(updated) <= 100  # Default max population size

    # Test population scoring
    scored_population = optimizer._select_using_pareto(population)
    assert len(scored_population) > 0
    assert all("score" in item for item in scored_population)

    # Test population selection
    selected = optimizer._select_prompt(population)
    assert selected is not None
    assert isinstance(selected, dict)
    assert "chromosome" in selected
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture)

    # Test empty population
    with pytest.raises(ValueError):
        optimizer._update_population([], iteration=1, recent_scores=[])

    # Test population update
    population = optimizer._initialize_population()
    updated = optimizer._update_population(
        population, iteration=1, recent_scores=[0.9, 0.8]
    )
    assert len(updated) <= 100  # Default max population size

    # Test population scoring
    scored_population = optimizer._select_using_pareto(population)
    assert len(scored_population) > 0
    assert all("score" in item for item in scored_population)

    # Test population selection
    selected = optimizer._select_prompt(population)
    assert selected is not None
    assert isinstance(selected, dict)
    assert "chromosome" in selected


def test_mutation_logic(_metric_fixture: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture)

    # Test basic mutation
    original = "Given {{input}}, generate {{output}}"
    mutated = optimizer._mutate(original)
    assert "{{input}}" in mutated
    assert "{{output}}" in mutated
    assert len(mutated) > len(original)

    # Test mutation rate extremes
    optimizer.config.mutation_rate = 0.0
    no_mutation = optimizer._mutate(original)
    assert no_mutation == original

    optimizer.config.mutation_rate = 1.0
    highly_mutated = optimizer._mutate(original)
    assert len(highly_mutated) > len(original) * 1.5  # Significant growth

    # Test invalid mutation rate handling
    with pytest.raises(ValueError):
        optimizer._mutate("")
    with pytest.raises(TypeError):
        optimizer._mutate(None)


def test_crossover_logic(_metric_fixture: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture)
    p1 = "Given {{input}}, generate {{output}}"
    p2 = "Analyze {{input}} and produce {{output}}"
    crossed = optimizer._crossover(p1, p2)
    assert "{{input}}" in crossed
    assert "{{output}}" in crossed
    assert len(crossed) > min(len(p1), len(p2))


def test_prompt_validation(_metric_fixture: Callable[[Any, Any], float]) -> None:
    optimizer = FullyEvolutionaryPromptOptimizer(metric=_metric_fixture)
    with pytest.raises(TypeError):
        optimizer._ensure_placeholders(None)
    with pytest.raises(TypeError):
        optimizer._ensure_placeholders(123)
    with pytest.raises(ValueError):
        optimizer._ensure_placeholders("No placeholders here")
