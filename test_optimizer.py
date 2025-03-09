from evoprompt.optimizer import FullyEvolutionaryPromptOptimizer


def test_optimizer_initialization():
    """Test basic optimizer initialization."""
    def mock_metric(pred, example):
        return 1.0
        
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

def test_parallel_initialization():
    """Test optimizer initialization with parallel workers."""
    def mock_metric(pred, example):
        return 1.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric, max_workers=4)
    assert optimizer.max_workers == 4

def test_initial_population():
    """Test that initial population is created correctly."""
    def mock_metric(pred, example):
        return 1.0
        
    optimizer = FullyEvolutionaryPromptOptimizer(mock_metric)
    population = optimizer._initialize_population()
    
    assert len(population) == 1
    assert population[0]["prompt"] == "{{input}} {{output}}"
    assert population[0]["score"] is None
    assert population[0]["last_used"] == 0
