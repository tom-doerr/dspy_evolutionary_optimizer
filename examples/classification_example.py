"""
Example of using the FullyEvolutionaryPromptOptimizer for a classification task.
"""

import dspy
from evoprompt import FullyEvolutionaryPromptOptimizer
from evoprompt.visualization import plot_evolution_history


def main():
    # Initialize the language model
    # Use a direct model configuration to ensure we're actually calling the API
    lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
    
    # Explicitly disable caching and configure tracing to see what's happening
    dspy.settings.configure(lm=lm, cache=False, trace=True)
    print("DSPy settings - Cache:", dspy.settings.cache, "Trace:", dspy.settings.trace)
    
    # Force a test call to the model to verify connectivity
    try:
        test_result = lm("This is a test call to verify the model is working.")
        print(f"Test LM call successful. Response length: {len(test_result)}")
        print(f"Response preview: {test_result[:50]}...")
    except Exception as e:
        print(f"ERROR: Test call to language model failed: {e}")
    # Define a simple classification task
    signature = dspy.Signature("text -> label")
    predictor = dspy.Predict(signature)
    
    # Create a small training set
    trainset = [
        dspy.Example(text="Great product!", label="positive"),
        dspy.Example(text="Awful service.", label="negative"),
        dspy.Example(text="It's okay.", label="neutral"),
        dspy.Example(text="I love this!", label="positive"),
        dspy.Example(text="Terrible experience.", label="negative"),
    ]
    
    # Define a metric function
    def accuracy_metric(prediction, example):
        return 1.0 if prediction.label == example.label else 0.0
    
    # Create and run the optimizer
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=accuracy_metric,
        generations=10,  # Reduced for faster testing
        mutation_rate=0.7,  # Increased for more exploration
        growth_rate=0.4,  # Increased for more population diversity
        max_population=10,  # Limit population size to prevent exponential growth
        debug=True  # Enable debug logging
    )
    
    optimized_predictor = optimizer.compile(predictor, trainset)
    
    # Test the optimized predictor
    test_examples = [
        "This is amazing!",
        "I hate it.",
        "It's not bad."
    ]
    
    print("\nTesting optimized predictor:")
    for text in test_examples:
        result = optimized_predictor(text=text).label
        print(f"Text: '{text}' -> Label: '{result}'")
    
    # Visualize the evolution history
    plot_evolution_history(optimizer.get_history())


if __name__ == "__main__":
    main()
