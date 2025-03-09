import dspy
import pytest

from evoprompt.optimizer import FullyEvolutionaryPromptOptimizer


class TestFullyEvolutionaryPromptOptimizer:
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

    def test_compile_basic_functionality(self, simple_program, simple_trainset, mock_metric):
        """Test basic compilation functionality."""
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_inference_calls=10)
        optimized_program = optimizer.compile(simple_program, simple_trainset)
        
        assert optimized_program is not None
        assert hasattr(optimized_program, 'signature')
        assert hasattr(optimized_program, 'predict')

    def test_population_management(self, simple_program, simple_trainset, mock_metric):
        """Test population management and evolution."""
        optimizer = FullyEvolutionaryPromptOptimizer(
            mock_metric,
            max_population=5,
            max_inference_calls=10
        )
        
        # Run compilation
        optimized_program = optimizer.compile(simple_program, simple_trainset)
        
        # Check history
        history = optimizer.get_history()
        assert len(history) > 0
        assert all(isinstance(entry, dict) for entry in history)
        assert all('iteration' in entry for entry in history)
        assert all('best_score' in entry for entry in history)

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
        optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, debug=True)
        
        population = [
            {"prompt": "prompt1", "score": 0.8, "last_used": 1},
            {"prompt": "prompt2", "score": 0.9, "last_used": 2},
        ]
        
        optimizer._log_progress(iteration=1, population=population)
        
        captured = capsys.readouterr()
        assert "Iteration" in captured.out
        assert "Best Score" in captured.out
        assert "Population" in captured.out
