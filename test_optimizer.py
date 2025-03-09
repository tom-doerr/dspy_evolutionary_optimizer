"""Test suite for the evolutionary prompt optimizer."""

# ruff: noqa: S101  # Allow assert statements in tests
import time

import dspy
import pytest

from evoprompt.optimizer import FullyEvolutionaryPromptOptimizer


class TestFullyEvolutionaryPromptOptimizer:
    """Test cases for the FullyEvolutionaryPromptOptimizer class."""

    @pytest.fixture
    def simple_program(self):
        """Fixture for a simple DSPy program."""
        return dspy.Predict(dspy.Signature("input -> output"))

    @pytest.fixture
    def simple_trainset(self):
        """Fixture for a simple training set."""
        return [
            dspy.Example(input="test1", output="result1"),
            dspy.Example(input="test2", output="result2")
        ]

    @pytest.fixture
    def mock_metric(self):
        """Fixture for a mock metric function."""
        def metric(prediction, example):
            return 1.0 if prediction.output == example.output else 0.0
        return metric

    def test_initialization(self, mock_metric):
        """Test optimizer initialization with default parameters."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        assert optimizer.metric == mock_metric
        assert optimizer.generations == 10
        assert optimizer.mutation_rate == 0.5
        assert optimizer.growth_rate == 0.3
        assert optimizer.max_population == 100
        assert optimizer.max_inference_calls == 100
        assert optimizer.debug is False
        assert optimizer.use_mock is False

    def test_mock_mode_initialization(self, mock_metric):
        """Test optimizer initialization with mock mode enabled."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
        assert optimizer.use_mock is True

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_compile_basic_functionality(self, simple_program, simple_trainset, mock_metric):
        """Test basic compilation functionality."""
        # Configure mock LM
        lm = dspy.LM('mock', cache=False)
        dspy.configure(lm=lm)
        
        optimizer = FullyEvolutionaryPromptOptimizer(
            mock_metric,
            max_inference_calls=10,
            use_mock=True,
            debug=True
        )
        
        start_time = time.time()
        optimized_program = optimizer.compile(simple_program, simple_trainset)
        elapsed = time.time() - start_time
        
        assert optimized_program is not None
        assert hasattr(optimized_program, 'signature')
        assert hasattr(optimized_program, 'predict')
        assert elapsed < 30, f"Test took too long: {elapsed:.1f}s"

    @pytest.mark.integration
    @pytest.mark.timeout(60)  # 60 second timeout for real API calls
    def test_real_api_call(self):
        """Test a single real API call to verify connectivity."""
        # Use a minimal configuration
        lm = dspy.LM('openrouter/google/gemini-2.0-flash-001', cache=False)
        dspy.configure(lm=lm)
        
        # Create a simple program
        program = dspy.Predict(dspy.Signature("input -> output"))
        
        # Create optimizer with minimal settings
        optimizer = FullyEvolutionaryPromptOptimizer(
            metric=lambda pred, ex: 1.0,  # Simple metric
            generations=1,  # Only 1 generation
            max_inference_calls=1,  # Only 1 call
            use_mock=False,
            debug=True
        )
        
        # Minimal trainset
        trainset = [dspy.Example(input="test", output="result")]
        
        start_time = time.time()
        result = optimizer.compile(program, trainset)
        elapsed = time.time() - start_time
        
        # Basic validation
        assert result is not None
        assert hasattr(result, 'signature')
        assert elapsed < 60, f"Test took too long: {elapsed:.1f}s"

    def test_population_management(self, simple_program, simple_trainset, mock_metric):
        """Test population management and evolution."""
        # Configure mock LM
        lm = dspy.LM('mock', cache=False)
        dspy.configure(lm=lm)
        
        optimizer = FullyEvolutionaryPromptOptimizer(
            mock_metric,
            max_population=5,
            max_inference_calls=10,
            use_mock=True
        )
        
        # Run compilation and check history
        _ = optimizer.compile(simple_program, simple_trainset)
        
        history = optimizer.get_history()
        assert len(history) > 0
        assert all(isinstance(entry, dict) for entry in history)
        assert all('iteration' in entry for entry in history)
        assert all('best_score' in entry for entry in history)

    def test_pareto_selection(self, mock_metric):
        """Test Pareto distribution selection favors top performers."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        # Create test population with varied scores
        population = [
            {"prompt": "prompt1", "score": 0.9, "last_used": 1},
            {"prompt": "prompt2", "score": 0.8, "last_used": 2},
            {"prompt": "prompt3", "score": 0.7, "last_used": 3},
            {"prompt": "prompt4", "score": 0.6, "last_used": 4},
            {"prompt": "prompt5", "score": 0.5, "last_used": 5},
        ]
        
        # Run selection 100 times and count selections
        selections = {p["prompt"]: 0 for p in population}
        for _ in range(100):
            selected = optimizer._select_prompt(population)
            selections[selected["prompt"]] += 1
            
        # Top performer should be selected much more often
        assert selections["prompt1"] > selections["prompt5"] * 2

    def test_crossover_operations(self, mock_metric):
        """Test crossover produces valid combinations."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        parent1 = "This is the first prompt"
        parent2 = "Another different prompt here"
        
        # Test multiple crossovers
        for _ in range(10):
            child = optimizer._crossover(parent1, parent2)
            assert isinstance(child, str)
            assert len(child.split()) >= min(len(parent1.split()), len(parent2.split()))
            assert any(word in child for word in parent1.split())
            assert any(word in child for word in parent2.split())

    def test_mutation_operations(self, mock_metric):
        """Test mutation produces valid variations."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        original = "This is a test prompt"
        
        # Test multiple mutations
        for _ in range(10):
            mutated = optimizer._mutate(original)
            assert isinstance(mutated, str)
            assert len(mutated) >= len(original)
            assert "{{input}}" in mutated
            assert "{{output}}" in mutated

    def test_mock_predictions(self, simple_program, mock_metric):
        """Test mock prediction generation."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
        
        example = dspy.Example(input="test", output="result")
        prediction = optimizer._create_mock_prediction(
            simple_program.signature,
            {"input": "test"},
            example
        )
        
        assert hasattr(prediction, 'output')
        assert prediction.output == "result"

    def test_population_limits(self, mock_metric):
        """Test population size enforcement."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_population=3)
        
        population = [
            {"prompt": "p1", "score": 0.9, "last_used": 1},
            {"prompt": "p2", "score": 0.8, "last_used": 2},
            {"prompt": "p3", "score": 0.7, "last_used": 3},
            {"prompt": "p4", "score": 0.6, "last_used": 4},
        ]
        
        updated = optimizer._update_population(population, iteration=5, recent_scores=[0.9, 0.8])
        assert len(updated) == 3
        assert all(p["score"] is not None for p in updated)

    def test_error_handling(self, simple_program, simple_trainset, mock_metric):
        """Test error handling during evaluation."""
        # Create optimizer with mock that will fail
        optimizer = FullyEvolutionaryPromptOptimizer(
            lambda pred, ex: 1/0,  # Will raise ZeroDivisionError
            use_mock=True
        )
        
        # Should handle errors gracefully
        score = optimizer._evaluate(simple_program, "test prompt", simple_trainset)
        assert score == 0.0

    def test_crossover(self, mock_metric):
        """Test crossover functionality."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        prompt1 = "This is a test prompt"
        prompt2 = "Another prompt for testing"
        result = optimizer._crossover(prompt1, prompt2)
        
        assert isinstance(result, str)
        assert len(result.split()) >= min(len(prompt1.split()), len(prompt2.split()))

    def test_mutation(self, mock_metric):
        """Test mutation functionality."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        original_prompt = "{{input}} {{output}}"
        mutated = optimizer._mutate(original_prompt)
        
        assert isinstance(mutated, str)
        assert "{{input}}" in mutated
        assert "{{output}}" in mutated
        assert len(mutated) >= len(original_prompt)

    def test_mock_prediction(self, simple_program, mock_metric):
        """Test mock prediction creation."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, use_mock=True)
        
        example = dspy.Example(input="test", output="result")
        prediction = optimizer._create_mock_prediction(
            simple_program.signature,
            {"input": "test"},
            example
        )
        
        assert hasattr(prediction, 'output')
        assert prediction.output == "result"

    def test_evaluation(self, simple_program, simple_trainset, mock_metric):
        """Test prompt evaluation functionality."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
        
        prompt = "{{input}} {{output}}"
        score = optimizer._evaluate(simple_program, prompt, simple_trainset)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_population_update(self, mock_metric):
        """Test population update logic."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_population=5)
        
        population = [
            {"prompt": "prompt1", "score": 0.8, "last_used": 1},
            {"prompt": "prompt2", "score": 0.9, "last_used": 2},
            {"prompt": "prompt3", "score": None, "last_used": 3},
        ]
        
        updated_population = optimizer._update_population(population, iteration=5, recent_scores=[0.8, 0.9])
        
        assert len(updated_population) <= 5
        assert all(isinstance(p, dict) for p in updated_population)
        assert all("prompt" in p for p in updated_population)

    def test_progress_logging(self, mock_metric, capsys):
        """Test progress logging functionality."""
        # Configure mock LM
        lm = dspy.LM('mock', cache=False)
        dspy.configure(lm=lm)
        
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, debug=True, use_mock=True)
        
        population = [
            {"prompt": "prompt1", "score": 0.8, "last_used": 1},
            {"prompt": "prompt2", "score": 0.9, "last_used": 2},
        ]
        
        # Test normal progress logging
        optimizer._log_progress(iteration=1, population=population)
        captured = capsys.readouterr()
        assert "Iteration" in captured.out
        assert "Best Score" in captured.out
        assert "Population" in captured.out
        assert "0.8" in captured.out
        assert "0.9" in captured.out
        
        # Test error handling
        optimizer.max_inference_calls = None  # Invalid value
        optimizer._log_progress(iteration=1, population=population)
        captured = capsys.readouterr()
        assert "Error creating progress bar" in captured.out
