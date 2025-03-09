"""Example of using the FullyEvolutionaryPromptOptimizer for a pattern optimization task.

This example optimizes for a specific pattern: maximizing occurrences of 'a' after 'e'
within the first 23 characters, with penalties for each character beyond that limit.
"""

import dspy

from evoprompt import FullyEvolutionaryPromptOptimizer
from evoprompt.visualization import plot_evolution_history


def _initialize_language_model() -> dspy.LM:
    """Initialize and configure the language model."""
    lm = dspy.LM('openrouter/google/gemini-2.0-flash-001', cache=False, max_tokens=30)
    dspy.configure(lm=lm)
    return lm

def _test_lm_connectivity(lm: dspy.LM) -> None:
    """Test the language model connectivity."""
    try:
        test_result = lm("This is a test call to verify the model is working.")
        print(f"Test LM call successful. Response length: {len(test_result)}")
        print(f"Response preview: {test_result[:50]}...")
    except Exception as e:
        print(f"ERROR: Test call to language model failed: {e}")

def _create_training_set() -> list[dspy.Example]:
    """Create a training set with prompts designed to elicit 'ea' patterns."""
    return [
        dspy.Example(prompt="Write a short sentence about the beach.", text="The beach was peaceful."),
        dspy.Example(prompt="Describe a meal you enjoyed.", text="I ate a great meal yesterday."),
        dspy.Example(prompt="Tell me about reading.", text="Reading creates a pleasant escape."),
        dspy.Example(prompt="Mention something about weather.", text="The weather is really nice today."),
    ]

def _pattern_metric(prediction: dspy.Prediction, example: dspy.Example) -> float:
    """Calculate pattern metric score for optimization."""
    text = prediction.text.lower()
    
    # Count 'a' after 'e' within first 23 chars
    count = sum(1 for i in range(min(len(text)-1, 22)) if text[i] == 'e' and text[i+1] == 'a')
    
    # Penalty for length beyond 23 chars (1 point per extra char)
    length_penalty = max(0, len(text) - 23)
    
    # Final score: pattern count minus length penalty
    score = count - length_penalty * 0.1
    
    print(f"Text: '{text[:30]}...' | Length: {len(text)} | 'e->a' count: {count} | Penalty: {length_penalty * 0.1:.1f} | Score: {score:.1f}")
    
    return score

def _create_optimizer() -> FullyEvolutionaryPromptOptimizer:
    """Create and configure the evolutionary prompt optimizer."""
    return FullyEvolutionaryPromptOptimizer(
        metric=_pattern_metric,
        generations=8,
        mutation_rate=0.8,
        growth_rate=0.5,
        max_population=100,
        max_inference_calls=200,
        max_workers=4,  # Use 4 parallel workers
        debug=True,
        use_mock=False
    )

def _test_optimized_generator(generator: dspy.Predict) -> None:
    """Test the optimized generator with various prompts."""
    test_prompts = [
        "Write a short sentence.",
        "Tell me something interesting.",
        "Give me a quick fact."
    ]
    
    print("\nTesting optimized generator:")
    for prompt in test_prompts:
        result = generator(prompt=prompt).text
        
        # Count 'a' after 'e' patterns in the result
        ea_count = sum(1 for i in range(len(result)-1) if result[i].lower() == 'e' and result[i+1].lower() == 'a')
        
        # Calculate the same score as our metric function
        ea_count_first_23 = sum(1 for i in range(min(len(result)-1, 22)) if result[i].lower() == 'e' and result[i+1].lower() == 'a')
        length_penalty = max(0, len(result) - 23) * 0.1
        final_score = ea_count_first_23 - length_penalty
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Result: '{result}'")
        print(f"Length: {len(result)} chars")
        print(f"'e->a' patterns total: {ea_count}")
        print(f"'e->a' patterns in first 23 chars: {ea_count_first_23}")
        print(f"Length penalty: {length_penalty:.1f}")
        print(f"Final score: {final_score:.1f}")

def main() -> None:
    """Run the pattern optimization example using evolutionary prompt optimization."""
    lm = _initialize_language_model()
    _test_lm_connectivity(lm)
    
    # Define a simple task that generates text
    generator = dspy.Predict(dspy.Signature("prompt -> text"))
    
    # Create training set and optimizer
    trainset = _create_training_set()
    optimizer = _create_optimizer()
    
    # Run optimization
    optimized_generator = optimizer.compile(generator, trainset)
    
    # Test optimized generator
    _test_optimized_generator(optimized_generator)
    
    # Visualize the evolution history
    plot_evolution_history(optimizer.get_history())


if __name__ == "__main__":
    main()
