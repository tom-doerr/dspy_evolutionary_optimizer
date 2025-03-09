"""
Example of using the FullyEvolutionaryPromptOptimizer for a question answering task.
"""

import dspy
from evoprompt import FullyEvolutionaryPromptOptimizer
from evoprompt.visualization import plot_evolution_history


def main():
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
        generations=8,
        mutation_rate=0.6,
        growth_rate=0.4
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
