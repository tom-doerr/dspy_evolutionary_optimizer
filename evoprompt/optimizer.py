"""
Core implementation of the evolutionary prompt optimizer.
"""

import random
import string
from statistics import mean
import dspy


class FullyEvolutionaryPromptOptimizer:
    """
    An optimizer that evolves prompts over generations, providing numeric feedback.
    
    This optimizer uses evolutionary algorithms to improve prompts by:
    - Tracking numeric scores for each prompt
    - Reporting statistics per generation
    - Logging evolution history
    """
    
    def __init__(self, metric, generations=10, mutation_rate=0.5, growth_rate=0.3):
        """
        Initialize the optimizer.
        
        Args:
            metric: Function that evaluates a prediction against an example
            generations: Number of generations to evolve
            mutation_rate: Probability of mutating a prompt
            growth_rate: Base rate for spawning new variants (multiplied by score)
        """
        self.metric = metric
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.growth_rate = growth_rate
        self.history = []  # Store evolution stats per generation

    def compile(self, program, trainset):
        """
        Evolve prompts over multiple generations to optimize the given metric.
        
        Args:
            program: DSPy program to optimize
            trainset: Examples to use for evaluation
            
        Returns:
            A DSPy Predict module with the optimized prompt
        """
        # Start with a single seed prompt
        population = [{"prompt": "{{input}} {{output}}", "score": None}]
        
        # Evolve all prompts over generations
        for generation in range(self.generations):
            # Evaluate all prompts that need scoring
            for chromosome in population:
                if chromosome["score"] is None:
                    chromosome["score"] = self._evaluate(program, chromosome["prompt"], trainset)
            
            # Collect numeric feedback
            scores = [c["score"] for c in population]
            best_score = max(scores)
            avg_score = mean(scores)
            population_size = len(population)
            
            # Log stats for this generation
            self.history.append({
                "generation": generation + 1,
                "best_score": best_score,
                "avg_score": avg_score,
                "population_size": population_size,
                "best_prompt": max(population, key=lambda x: x["score"])["prompt"]
            })
            
            # Report progress
            print(f"Generation {generation + 1}: Best Score = {best_score:.3f}, "
                  f"Avg Score = {avg_score:.3f}, Population Size = {population_size}")
            
            # Evolve all prompts
            new_population = []
            for chromosome in population:
                new_population.append(chromosome)  # Keep original
                
                # Mutation: evolve this prompt
                if random.random() < self.mutation_rate:
                    mutated = self._mutate(chromosome["prompt"])
                    new_population.append({"prompt": mutated, "score": None})
                
                # Growth: spawn new variants based on performance
                if random.random() < self.growth_rate * chromosome["score"]:
                    crossed = self._crossover(chromosome["prompt"], random.choice(population)["prompt"])
                    new_population.append({"prompt": crossed, "score": None})
            
            population = new_population
        
        # Sort by score and return best predictor
        population.sort(key=lambda x: x["score"], reverse=True)
        best_prompt = population[0]["prompt"]
        print("\nEvolution History:")
        for entry in self.history:
            print(f"Gen {entry['generation']}: Best Score = {entry['best_score']:.3f}, "
                  f"Avg Score = {entry['avg_score']:.3f}, Size = {entry['population_size']}")
        
        print(f"\nBest Prompt: '{best_prompt}'")
        return dspy.Predict(program.signature, prompt=best_prompt)

    def _evaluate(self, program, prompt, trainset):
        """
        Evaluate a prompt's performance on the training set.
        
        Args:
            program: DSPy program to evaluate
            prompt: Prompt template to evaluate
            trainset: Examples to use for evaluation
            
        Returns:
            Average score across all examples
        """
        predictor = dspy.Predict(program.signature, prompt=prompt)
        predictions = [predictor(**{k: ex[k] for k in program.signature.input_fields}) 
                       for ex in trainset]
        return sum(self.metric(pred, ex) for pred, ex in zip(predictions, trainset)) / len(trainset)

    def _crossover(self, prompt1, prompt2):
        """
        Combine two prompts by splitting at a random point and joining.
        
        Args:
            prompt1: First parent prompt
            prompt2: Second parent prompt
            
        Returns:
            A new prompt created by combining parts of both parents
        """
        p1_parts = prompt1.split()
        p2_parts = prompt2.split()
        if not p1_parts or not p2_parts:
            return prompt1
        min_len = min(len(p1_parts), len(p2_parts))
        crossover_point = random.randint(0, min_len)
        return " ".join(p1_parts[:crossover_point] + p2_parts[crossover_point:])

    def _mutate(self, prompt):
        """
        Apply a random mutation to a prompt.
        
        Args:
            prompt: Prompt to mutate
            
        Returns:
            A mutated version of the prompt
        """
        mutations = [
            lambda p: p + " " + random.choice(["to", "with", "for", "->"]),
            lambda p: " ".join(w for w in p.split() if random.random() > 0.2),
            lambda p: p.replace("{{input}}", random.choice(["Input:", "{{input}} here", "Given {{input}}"])),
            lambda p: p.replace("{{output}}", random.choice(["-> {{output}}", "{{output}} result", "yields {{output}}"])),
            lambda p: "".join(c if random.random() > 0.05 else random.choice(string.ascii_letters) for c in p)
        ]
        return random.choice(mutations)(prompt)
    
    def get_history(self):
        """
        Get the evolution history.
        
        Returns:
            List of dictionaries with statistics for each generation
        """
        return self.history
