# EvolutionaryPromptOptimizer

A package for evolving prompts with numeric feedback using evolutionary algorithms.

## Installation

```bash
pip install -e .
```

## Features

- **Evolutionary Optimization**: Evolves prompts over generations to maximize a metric
- **Numeric Feedback**: Tracks and reports scores for each generation
- **Visualization**: Plot evolution history to see improvement over time
- **Flexible Metrics**: Use any custom metric function for optimization

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

## How It Works

1. **Population Initialization**: Starts with a seed prompt
2. **Evaluation**: Scores each prompt using the provided metric
3. **Evolution**: Applies mutations and crossovers to create new prompts
4. **Selection**: Keeps the best prompts for the next generation
5. **Feedback**: Reports statistics for each generation

## Customization

You can customize the optimizer with these parameters:

- `generations`: Number of generations to evolve (default: 10)
- `mutation_rate`: Probability of mutating a prompt (default: 0.5)
- `growth_rate`: Base rate for spawning new variants (default: 0.3)
