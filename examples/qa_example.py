"""Example of using the FullyEvolutionaryPromptOptimizer for a question answering task."""

import dspy

from evoprompt import FullyEvolutionaryPromptOptimizer
from evoprompt.visualization import plot_evolution_history


def main() -> None:
    """Run the question answering example using evolutionary prompt optimization."""
    # Initialize the language model
    # Use a direct model configuration to ensure we're actually calling the API
    lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
    
    # Explicitly disable caching but don't enable tracing (causing errors)
    dspy.settings.configure(lm=lm, cache=False)
    print("DSPy settings - Cache:", dspy.settings.cache)
    
    # Force a test call to the model to verify connectivity
    try:
        test_result = lm("This is a test call to verify the model is working.")
        print(f"Test LM call successful. Response length: {len(test_result)}")
        print(f"Response preview: {test_result[:50]}...")
    except Exception as e:
        print(f"ERROR: Test call to language model failed: {e}")
    # Define a simple QA task
    qa_signature = dspy.Signature("question, context -> answer")
    qa_predictor = dspy.Predict(qa_signature)
    
    # Create a small training set
    qa_trainset = [
        dspy.Example(question="What's the capital?", context="France is a country in Western Europe.",
                    answer="Paris"),
        dspy.Example(question="Where's the president?", context="The USA has its government in Washington DC.",
                    answer="Washington DC"),
        dspy.Example(question="What's the largest city?", context="Japan is an island nation with Tokyo as its largest city.",
                    answer="Tokyo"),
        dspy.Example(question="What's the main language?", context="Brazil is the largest country in South America where Portuguese is spoken.",
                    answer="Portuguese"),
    ]
    
    # Define a metric function
    def qa_metric(prediction, example):
        return 1.0 if prediction.answer.lower() == example.answer.lower() else 0.0
    
    # Create and run the optimizer
    qa_optimizer = FullyEvolutionaryPromptOptimizer(
        metric=qa_metric,
        generations=4,  # Reduced for faster testing
        mutation_rate=0.7,  # Increased for more exploration
        growth_rate=0.5,  # Increased for more population diversity
        max_population=30,  # Limit population size to prevent exponential growth
        max_workers=4,  # Use 4 parallel workers
        debug=True,  # Enable debug logging
        use_mock=True  # Use mock mode for testing
    )
    
    optimized_qa_predictor = qa_optimizer.compile(qa_predictor, qa_trainset)
    
    # Test the optimized predictor
    test_examples = [
        {"question": "What's the capital?", "context": "Germany is a country in Central Europe."},
        {"question": "What's the main export?", "context": "Saudi Arabia is known for its oil production."},
    ]
    
    print("\nTesting optimized QA predictor:")
    for example in test_examples:
        result = optimized_qa_predictor(**example).answer
        print(f"Question: '{example['question']}' Context: '{example['context']}' -> Answer: '{result}'")
    
    # Visualize the evolution history
    plot_evolution_history(qa_optimizer.get_history())


if __name__ == "__main__":
    main()
