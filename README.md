# EvolutionaryPromptOptimizer (Work in Progress)

⚠️ This project is currently under active development and should be considered experimental. ⚠️

A package for evolving prompts with numeric feedback using evolutionary algorithms. The API and functionality may change significantly in future versions.

## Installation

```bash
pip install -e .
```

> **Note**: This package is still in development. We recommend installing in a virtual environment and pinning to specific versions as they become available.

## Features (Current and Planned)

✅ **Implemented Features**
- Evolutionary Optimization: Evolves prompts over generations to maximize a metric
- Numeric Feedback: Tracks and reports scores for each generation
- Visualization: Plot evolution history to see improvement over time
- Flexible Metrics: Use any custom metric function for optimization

🚧 **Planned Features**
- More sophisticated mutation strategies
- Better parallelization support
- Enhanced visualization capabilities
- Comprehensive documentation
- Additional examples and tutorials

## Usage

### Classification Example

```python
import dspy
from evoprompt import FullyEvolutionaryPromptOptimizer

# Define a simple classification task
signature = dspy.Signature("text -> label")
predictor = dspy.Predict(signature)

# Create a training set
trainset = [
    dspy.Example(text="Great product!", label="positive"),
    dspy.Example(text="Awful service.", label="negative"),
]

# Define a metric function
def accuracy_metric(prediction, example):
    return 1.0 if prediction.label == example.label else 0.0

# Create and run the optimizer
optimizer = FullyEvolutionaryPromptOptimizer(metric=accuracy_metric)
optimized_predictor = optimizer.compile(predictor, trainset)

# Test the optimized predictor
result = optimized_predictor(text="I love this!").label
print(f"Prediction: {result}")

# Visualize the evolution history
from evoprompt.visualization import plot_evolution_history
plot_evolution_history(optimizer.get_history())
```

### Question Answering Example

```python
import dspy
from evoprompt import FullyEvolutionaryPromptOptimizer

# Define a QA task
qa_signature = dspy.Signature("question, context -> answer")
qa_predictor = dspy.Predict(qa_signature)

# Create a training set
qa_trainset = [
    dspy.Example(question="What's the capital?", context="France", answer="Paris"),
    dspy.Example(question="Where's the president?", context="USA", answer="Washington DC"),
]

# Define a metric function
def qa_metric(prediction, example):
    return 1.0 if prediction.answer.lower() == example.answer.lower() else 0.0

# Create and run the optimizer
qa_optimizer = FullyEvolutionaryPromptOptimizer(metric=qa_metric)
optimized_qa_predictor = qa_optimizer.compile(qa_predictor, qa_trainset)

# Test the optimized predictor
result = optimized_qa_predictor(question="What's the capital?", context="Brazil").answer
print(f"Answer: {result}")
```

### Pattern Optimization Example

```python
import dspy
from evoprompt import FullyEvolutionaryPromptOptimizer

# Define a text generation task
signature = dspy.Signature("prompt -> text")
generator = dspy.Predict(signature)

# Create a training set
trainset = [
    dspy.Example(prompt="Write a short sentence about the beach.", text="The beach was peaceful."),
    dspy.Example(prompt="Describe a meal you enjoyed.", text="I ate a great meal yesterday."),
]

# Define a metric function that counts 'a' after 'e' in first 23 chars and penalizes length
def pattern_metric(prediction, example):
    text = prediction.text.lower()
    
    # Count 'a' after 'e' within first 23 chars
    count = 0
    for i in range(min(len(text)-1, 22)):
        if text[i] == 'e' and text[i+1] == 'a':
            count += 1
    
    # Penalty for each character beyond 23 chars (0.1 points per char)
    length_penalty = max(0, len(text) - 23) * 0.1
    
    # Final score: pattern count minus length penalty (can be negative)
    return count - length_penalty

# Create and run the optimizer
optimizer = FullyEvolutionaryPromptOptimizer(metric=pattern_metric)
optimized_generator = optimizer.compile(generator, trainset)

# Test the optimized generator
result = optimized_generator(prompt="Write a short sentence.").text
print(f"Result: {result}")
```

## How It Works (Current Implementation)

1. **Population Initialization**: Starts with a seed prompt
2. **Evaluation**: Scores each prompt using the provided metric
3. **Evolution**: Applies mutations and crossovers to create new prompts
4. **Selection**: Keeps the best prompts for the next generation
5. **Feedback**: Reports statistics for each generation

> **Note**: The current implementation is a proof-of-concept and may undergo significant changes as we refine the evolutionary algorithms and optimization strategies.

## Customization (Experimental)

You can customize the optimizer with these parameters:

- `generations`: Number of generations to evolve (default: 10)
- `mutation_rate`: Probability of mutating a prompt (default: 0.5)
- `growth_rate`: Base rate for spawning new variants (default: 0.3)

> **Warning**: These parameters are experimental and their effects may change in future versions. We recommend testing different configurations and providing feedback on what works best for your use case.
