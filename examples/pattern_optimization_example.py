"""
Example of using the FullyEvolutionaryPromptOptimizer for a pattern optimization task.

This example optimizes for a specific pattern: maximizing occurrences of 'a' after 'e'
within the first 23 characters, with penalties for longer responses.
"""

import dspy
from evoprompt import FullyEvolutionaryPromptOptimizer
from evoprompt.visualization import plot_evolution_history


def main():
    # Initialize the language model
    lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
    
    # Configure DSPy
    dspy.settings.configure(lm=lm, cache=False)
    print("DSPy settings - Cache:", dspy.settings.cache)
    
    # Test LM connectivity
    try:
        test_result = lm("This is a test call to verify the model is working.")
        print(f"Test LM call successful. Response length: {len(test_result)}")
        print(f"Response preview: {test_result[:50]}...")
    except Exception as e:
        print(f"ERROR: Test call to language model failed: {e}")
    
    # Define a simple task that generates text
    signature = dspy.Signature("prompt -> text")
    generator = dspy.Predict(signature)
    
    # Create a training set with prompts designed to elicit 'ea' patterns
    trainset = [
        dspy.Example(prompt="Write a short sentence about the beach.", text="The beach was peaceful."),
        dspy.Example(prompt="Describe a meal you enjoyed.", text="I ate a great meal yesterday."),
        dspy.Example(prompt="Tell me about reading.", text="Reading creates a pleasant escape."),
        dspy.Example(prompt="Mention something about weather.", text="The weather is really nice today."),
    ]
    
    # Define a metric function that counts 'ea' patterns in first 23 chars and penalizes length
    def pattern_metric(prediction, example):
        text = prediction.text.lower()
        
        # Count 'ea' patterns within first 23 chars
        count = 0
        for i in range(min(len(text)-1, 22)):  # -1 to avoid index error, 22 to stay within 23 chars
            if text[i] == 'e' and text[i+1] == 'a':
                count += 1
        
        # Penalty for length beyond 23 chars
        length_penalty = max(0, len(text) - 23)
        
        # Final score: pattern count minus length penalty
        score = max(0, count - length_penalty * 0.1)
        
        if count > 0:
            print(f"Found {count} 'ea' patterns in '{text[:23]}...' (len={len(text)}, penalty={length_penalty})")
        
        return score
    
    # Create and run the optimizer
    optimizer = FullyEvolutionaryPromptOptimizer(
        metric=pattern_metric,
        generations=8,
        mutation_rate=0.8,  # High mutation rate for more exploration
        growth_rate=0.5,    # Higher growth rate for more population diversity
        max_population=15,  # Reasonable population size
        debug=True,
        use_mock=False      # Use real LLM calls for this example
    )
    
    optimized_generator = optimizer.compile(generator, trainset)
    
    # Test the optimized generator with various prompts
    test_prompts = [
        "Write a short sentence.",
        "Tell me something interesting.",
        "Give me a quick fact."
    ]
    
    print("\nTesting optimized generator:")
    for prompt in test_prompts:
        result = optimized_generator(prompt=prompt).text
        
        # Count 'ea' patterns in the result
        ea_count = 0
        for i in range(len(result)-1):
            if result[i].lower() == 'e' and result[i+1].lower() == 'a':
                ea_count += 1
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Result: '{result}'")
        print(f"Length: {len(result)} chars")
        print(f"'ea' patterns: {ea_count}")
        print(f"'ea' patterns in first 23 chars: {sum(1 for i in range(min(len(result)-1, 22)) if result[i].lower() == 'e' and result[i+1].lower() == 'a')}")
    
    # Visualize the evolution history
    plot_evolution_history(optimizer.get_history())


if __name__ == "__main__":
    main()
