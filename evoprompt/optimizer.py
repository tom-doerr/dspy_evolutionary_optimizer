"""Core implementation of the evolutionary prompt optimizer."""

import os
import random
from statistics import mean
import time

import dspy
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from textual.widgets import ProgressBar


class FullyEvolutionaryPromptOptimizer:
    """An optimizer that evolves prompts over generations, providing numeric feedback.
    
    This optimizer uses evolutionary algorithms to improve prompts by:
    - Tracking numeric scores for each prompt
    - Reporting statistics per generation
    - Logging evolution history
    """

    def __init__(self, metric, generations=10, mutation_rate=0.5, growth_rate=0.3, max_population=100,
                 max_inference_calls=100, debug=False, use_mock=None, max_workers=1):
        """Initialize the optimizer.
        
        Args:
            metric: Function that evaluates a prediction against an example
            generations: Number of generations to evolve
            mutation_rate: Probability of mutating a prompt
            growth_rate: Base rate for spawning new variants (multiplied by score)
            max_population: Maximum number of prompts in the population
            max_inference_calls: Maximum number of LLM inference calls to make
            debug: Enable debug logging
            use_mock: Force mock mode (True/False) or auto-detect if None
            max_workers: Number of parallel workers for evaluation (1 = serial)

        """
        self.metric = metric
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.growth_rate = growth_rate
        self.max_population = max_population
        self.max_workers = max_workers
        self.history = []  # Store evolution stats per generation
        self.debug = debug
        self.inference_count = 0
        self.max_inference_calls = max_inference_calls

        # Determine if we should use mock mode
        if use_mock is None:
            # Auto-detect based on environment variable
            self.use_mock = os.environ.get('EVOPROMPT_MOCK', 'false').lower() == 'true'
        else:
            self.use_mock = use_mock

        if self.use_mock and self.debug:
            print("MOCK MODE ENABLED: Using simulated responses instead of real LLM calls")

    def _select_prompt(self, population):
        """Select a prompt using Pareto distribution to favor top performers."""
        scored_population = [p for p in population if p["score"] is not None]
        if not scored_population:
            return random.choice(population)
        
        return self._select_using_pareto(scored_population)

    def _select_using_pareto(self, scored_population):
        """Select a prompt using Pareto distribution weights."""
        # Sort population by score descending
        scored_population.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate Pareto weights (80/20 rule)
        weights = []
        alpha = 1.16  # Pareto shape parameter (80/20 rule)
        n = len(scored_population)
        for i in range(n):
            # Weight decreases exponentially based on rank
            weight = (n - i) ** (-alpha)
            weights.append(weight)
            
        # Normalize weights
        total = sum(weights)
        probs = [w / total for w in weights]
        
        # Select using weighted probabilities
        return random.choices(scored_population, weights=probs, k=1)[0]

    def _update_population(self, population, iteration, recent_scores):
        """Update population based on scores and iteration."""
        if recent_scores:
            recent_scores_sorted = sorted(recent_scores)
            _ = recent_scores_sorted[int(len(recent_scores_sorted) * 0.8)]  # Calculate but don't use percentile
            
        # Remove stale prompts
        population = [p for p in population
                     if iteration - p["last_used"] < 10
                     or p["score"] is None]
        
        # Enforce population size limit
        if len(population) > self.max_population:
            scored_population = [p for p in population if p["score"] is not None]
            if len(scored_population) > self.max_population:
                scored_population.sort(key=lambda x: x["score"])
                population = scored_population[-self.max_population:]
                
        return population

    def _get_population_stats(self, population):
        """Calculate population statistics."""
        scores = [p["score"] for p in population if p["score"] is not None]
        if not scores:
            return None, None, None
            
        best_score = max(scores)
        avg_score = mean(scores)
        best_prompt = max(population, key=lambda x: x["score"] if x["score"] is not None else -float('inf'))["prompt"]
        return best_score, avg_score, best_prompt

    def _create_progress_bar(self):
        """Create and return a progress bar with error handling."""
        try:
            return ProgressBar(
                total=self.max_inference_calls,
                progress=self.inference_count,
                width=50,
                style="green",
                complete_style="bold white on green",
                pulse_style="bold white on blue"
            )
        except Exception as e:
            if self.debug:
                print(f"Error creating progress bar: {e}")
            return f"[Progress: {self.inference_count}/{self.max_inference_calls}]"

    def _create_main_panel(self, iteration, best_score, avg_score, population):
        """Create the main stats panel."""
        panel = Table.grid(padding=(1, 2))
        panel.add_column(justify="left", style="cyan")
        panel.add_column(justify="right", style="magenta")
        
        panel.add_row("Iteration", f"[bold]{iteration}")
        panel.add_row("Best Score", f"[green]{best_score:.3f}")
        panel.add_row("Avg Score", f"[yellow]{avg_score:.3f}")
        panel.add_row("Population", f"[blue]{len(population)}")
        panel.add_row("Inference Calls", f"[cyan]{self.inference_count}/{self.max_inference_calls}")
        return panel

    def _create_history_table(self):
        """Create the recent history table."""
        table = Table(title="[bold]Recent History", show_header=True, header_style="bold magenta")
        table.add_column("Iteration", justify="right")
        table.add_column("Best Score", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Population", justify="right")

        for entry in self.history[-5:]:
            table.add_row(
                str(entry['iteration']),
                f"{entry['best_score']:.3f}",
                f"{entry['avg_score']:.3f}",
                str(entry['population_size'])
            )
        return table

    def _log_progress(self, iteration, population):
        """Log and display progress information."""
        best_score, avg_score, best_prompt = self._get_population_stats(population)
        if best_score is None:
            return
            
        self.history.append({
            "iteration": iteration,
            "best_score": best_score,
            "avg_score": avg_score,
            "population_size": len(population),
            "best_prompt": best_prompt
        })

        console = Console()
        progress = self._create_progress_bar()
        main_panel = self._create_main_panel(iteration, best_score, avg_score, population)
        history_table = self._create_history_table()

        prompt_panel = Panel(
            best_prompt,
            title="[bold]Best Prompt",
            border_style="blue",
            padding=(1, 2),
            width=80
        )

        layout_components = [main_panel, prompt_panel, history_table]
        if progress is not None:
            layout_components.insert(1, progress)
            
        try:
            console.print(Panel(
                Group(*layout_components),
                title=f"[bold]Evolution Progress - Generation {iteration}",
                border_style="green",
                padding=(1, 2),
                width=80
            ))
        except Exception as e:
            if self.debug:
                print(f"Error rendering progress display: {e}")
            console.print(f"[bold]Generation {iteration}")
            console.print(f"Best Score: {best_score:.3f}")
            console.print(f"Avg Score: {avg_score:.3f}")
            console.print(f"Population: {len(population)}")
            console.print(f"Inference Calls: {self.inference_count}/{self.max_inference_calls}")

        console.print()

    def _initialize_population(self):
        """Initialize the starting population with chromosomes."""
        base_task = ["Given {{input}},", "generate {{output}}"]
        base_mutation = ["Be concise.", "Be accurate."]
        
        return [{
            "chromosome": Chromosome(base_task, base_mutation),
            "score": None,
            "last_used": 0
        }]

    def _process_population(self, population, iteration, recent_scores, program, trainset):
        """Process one iteration of population evolution."""
        # Select a prompt probabilistically based on score
        selected = self._select_prompt(population)
        
        # Evaluate the selected prompt on a random example
        example = random.choice(trainset)
        selected["score"] = self._evaluate(program, selected["prompt"], [example])
        selected["last_used"] = iteration
        recent_scores.append(selected["score"])

        # Keep recent scores window (last 20 scores)
        if len(recent_scores) > 20:
            recent_scores.pop(0)

        # Calculate recent score thresholds
        if recent_scores:
            recent_scores_sorted = sorted(recent_scores)
            top_20_percentile = recent_scores_sorted[int(len(recent_scores_sorted) * 0.8)]
        else:
            top_20_percentile = 0

        # If this score is in top 20%, mate it with another high performer
        if selected["score"] >= top_20_percentile:
            self._mate_high_performers(population, selected, top_20_percentile, iteration)

        # Apply mutation to selected prompt
        if random.random() < self.mutation_rate:
            self._apply_mutation(population, selected, iteration)

        return population, recent_scores

    def _mate_high_performers(self, population, selected, top_20_percentile, iteration):
        """Mate high performing prompts using Pareto selection."""
        # Get all high performers (top 20%)
        high_performers = [p for p in population
                         if p["score"] is not None
                         and p["score"] >= top_20_percentile
                         and p != selected]

        if high_performers:
            # Sort high performers by score descending
            high_performers.sort(key=lambda x: x["score"], reverse=True)
            
            # Calculate Pareto weights for high performers
            weights = []
            alpha = 1.16  # Pareto shape parameter
            n = len(high_performers)
            for i in range(n):
                weight = (n - i) ** (-alpha)
                weights.append(weight)
                
            # Normalize weights
            total = sum(weights)
            probs = [w / total for w in weights]
            
            # Select mate using Pareto distribution
            mate = random.choices(high_performers, weights=probs, k=1)[0]
            
            # Create new prompt through crossover
            new_prompt = self._crossover(selected["prompt"], mate["prompt"])
            population.append({"prompt": new_prompt, "score": None, "last_used": iteration})

    def _apply_mutation(self, population, selected, iteration):
        """Apply mutation to a selected chromosome."""
        new_chromosome = copy.deepcopy(selected["chromosome"])
        new_chromosome.mutate(self.mutation_rate)
        population.append({
            "chromosome": new_chromosome,
            "score": None,
            "last_used": iteration
        })

    def _initialize_evolution(self):
        """Initialize evolution parameters and population."""
        return [{"prompt": "{{input}} {{output}}", "score": None, "last_used": 0}], [], 0

    def _select_prompt(self, population):
        """Select a prompt probabilistically based on score."""
        scored_population = [p for p in population if p["score"] is not None]
        if not scored_population:
            return random.choice(population)
        
        return self._select_using_normalized_scores(scored_population)

    def _select_using_normalized_scores(self, scored_population):
        """Select a prompt using normalized score weights."""
        scores = [p["score"] for p in scored_population]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return random.choice(scored_population)
            
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        total = sum(normalized_scores)
        probs = [s / total for s in normalized_scores]
        return random.choices(scored_population, weights=probs, k=1)[0]

    def _update_population(self, population, iteration, recent_scores):
        """Update population based on scores and iteration."""
        if recent_scores:
            recent_scores_sorted = sorted(recent_scores)
            _ = recent_scores_sorted[int(len(recent_scores_sorted) * 0.8)]  # Calculate but don't use percentile
            
        # Remove stale prompts
        population = [p for p in population
                     if iteration - p["last_used"] < 10
                     or p["score"] is None]
        
        # Enforce population size limit
        if len(population) > self.max_population:
            scored_population = [p for p in population if p["score"] is not None]
            if len(scored_population) > self.max_population:
                scored_population.sort(key=lambda x: x["score"])
                population = scored_population[-self.max_population:]
                
        return population

    def _process_generation(self, population, recent_scores, iteration, program, trainset):
        """Process one generation of evolution."""
        # Select a prompt probabilistically based on score
        selected = self._select_prompt(population)
        
        # Evaluate the selected prompt on a random example
        example = random.choice(trainset)
        selected["score"] = self._evaluate(program, selected["prompt"], [example])
        selected["last_used"] = iteration
        recent_scores.append(selected["score"])

        # Keep recent scores window (last 20 scores)
        if len(recent_scores) > 20:
            recent_scores.pop(0)

        # Calculate recent score thresholds
        if recent_scores:
            recent_scores_sorted = sorted(recent_scores)
            top_20_percentile = recent_scores_sorted[int(len(recent_scores_sorted) * 0.8)]
        else:
            top_20_percentile = 0

        # If this score is in top 20%, mate it with another high performer
        if selected["score"] >= top_20_percentile:
            self._mate_high_performers(population, selected, top_20_percentile, iteration)

        # Apply mutation to selected prompt
        if random.random() < self.mutation_rate:
            self._apply_mutation(population, selected, iteration)

        return population, recent_scores

    def _mate_high_performers(self, population, selected, top_20_percentile, iteration):
        """Mate high performing chromosomes."""
        high_performers = [p for p in population
                         if p["score"] is not None
                         and p["score"] >= top_20_percentile
                         and p != selected]

        if high_performers:
            mate = random.choice(high_performers)
            new_chromosome = selected["chromosome"].combine(mate["chromosome"])
            population.append({
                "chromosome": new_chromosome,
                "score": None,
                "last_used": iteration
            })

    def _apply_mutation(self, population, selected, iteration):
        """Apply mutation to a selected prompt."""
        mutated = self._mutate(selected["prompt"])
        population.append({"prompt": mutated, "score": None, "last_used": iteration})

    def compile(self, program, trainset):
        """Evolve prompts using continuous probabilistic selection and mating.
        
        Args:
            program: DSPy program to optimize
            trainset: Examples to use for evaluation
            
        Returns:
            A DSPy Predict module with the optimized prompt

        """
        population, recent_scores, iteration = self._initialize_evolution()

        while self.inference_count < self.max_inference_calls or self.max_inference_calls <= 0:
            iteration += 1
            population, recent_scores = self._process_generation(
                population, recent_scores, iteration, program, trainset
            )
            
            # Log progress periodically
            if iteration % 10 == 0:
                self._log_progress(iteration, population)

        # Finalize and return best prompt
        return self._finalize_evolution(program, population, iteration, trainset)

    def _finalize_evolution(self, program, population, iteration, trainset):
        """Finalize evolution and return best predictor."""
        # Evaluate any remaining unevaluated prompts
        for chromosome in population:
            if chromosome["score"] is None:
                chromosome["score"] = self._evaluate(program, chromosome["prompt"], trainset)

        # Sort by score and return best predictor
        population.sort(key=lambda x: x["score"], reverse=True)
        best_prompt = population[0]["prompt"]
        
        # Create final summary display
        console = Console()
        summary_panel = Table.grid(padding=(1, 2))
        summary_panel.add_column(justify="left", style="cyan")
        summary_panel.add_column(justify="right", style="magenta")

        summary_panel.add_row("[bold]Best Score", f"[green]{population[0]['score']:.3f}")
        summary_panel.add_row("Total Iterations", f"[bold]{iteration}")
        summary_panel.add_row("Population Size", f"[blue]{len(population)}")
        summary_panel.add_row("Inference Calls", f"[cyan]{self.inference_count}")

        # Best prompt panel
        prompt_panel = Panel(
            best_prompt,
            title="[bold]Optimized Prompt",
            border_style="green",
            padding=(1, 2),
            width=80
        )

        # Final layout
        console.print(Panel(
            Group(
                summary_panel,
                prompt_panel
            ),
            title="[bold]Evolution Results",
            border_style="blue",
            padding=(1, 2),
            width=80
        ))
        
        return dspy.Predict(program.signature, prompt=best_prompt)

    def _make_prediction(self, program, prompt, input_kwargs):
        """Make a prediction using the program and prompt."""
        if self.use_mock:
            return self._create_mock_prediction(program.signature, input_kwargs, None)
            
        if self.max_inference_calls > 0 and self.inference_count >= self.max_inference_calls:
            if self.debug:
                print("  Inference call limit reached, using mock prediction")
            return self._create_mock_prediction(program.signature, input_kwargs, None)
            
        predictor = dspy.Predict(program.signature, prompt=prompt)
        return predictor(**input_kwargs)

    def _calculate_scores(self, predictions, trainset):
        """Calculate scores for predictions against training examples."""
        scores = []
        for pred, ex in zip(predictions, trainset):
            try:
                score = self.metric(pred, ex)
                scores.append(score)
            except Exception as e:
                print(f"Error in metric calculation: {e}")
                scores.append(0.0)
        return scores

    def _handle_prediction_error(self, e):
        """Handle prediction errors and return appropriate score."""
        print(f"Error during prediction: {e}")
        return 0.0  # Return low score for failed predictions

    def _evaluate_predictions(self, predictions, trainset):
        """Calculate average score for predictions."""
        if not predictions:
            return 0.0

        scores = []
        for pred, ex in zip(predictions, trainset):
            try:
                score = self.metric(pred, ex)
                scores.append(score)
            except Exception as e:
                print(f"Error in metric calculation: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _get_input_kwargs(self, program, example):
        """Extract input kwargs from example based on program signature."""
        return {k: example[k] for k in program.signature.input_fields}

    def _make_mock_prediction(self, predictor, input_kwargs, example, start_time):
        """Handle mock prediction logic."""
        if self.debug and not self.use_mock:
            print("  Inference call limit reached, using mock prediction")
        
        # Simulate realistic API latency between 0.1-0.5 seconds
        latency = random.uniform(0.1, 0.5)
        time.sleep(latency)
        
        pred = self._create_mock_prediction(predictor.signature, input_kwargs, example)
        
        if self.debug:
            elapsed = time.time() - start_time
            print(f"  Mock prediction took {elapsed:.4f}s")
            print(f"  Prediction result: {pred}")
        
        return pred

    def _log_real_prediction(self, pred, start_time):
        """Log details of a real prediction."""
        elapsed = time.time() - start_time
        print(f"  Real prediction took {elapsed:.4f}s")
        if elapsed < 0.05:
            print("  WARNING: Prediction was extremely fast - verify LLM is being called")
        print(f"  Prediction result: {pred}")

    def _make_single_prediction(self, predictor, input_kwargs, example):
        """Make a single prediction handling mock mode and inference limits."""
        start_time = time.time()
        
        if self.use_mock or (self.max_inference_calls > 0 and self.inference_count >= self.max_inference_calls):
            return self._make_mock_prediction(predictor, input_kwargs, example, start_time)

        # Make real prediction
        pred = predictor(**input_kwargs)
        self.inference_count += 1
        
        if self.debug:
            self._log_real_prediction(pred, start_time)
            
        return pred

    def _make_predictions(self, program, prompt, trainset):
        """Make predictions for all training examples."""
        predictions = []
        predictor = dspy.Predict(program.signature, prompt=prompt)
        
        for ex in trainset:
            try:
                input_kwargs = self._get_input_kwargs(program, ex)
                pred = self._make_single_prediction(predictor, input_kwargs, ex)
                predictions.append(pred)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return None

        return predictions

    def _calculate_evaluation_scores(self, predictions, trainset):
        """Calculate scores for all predictions."""
        if not predictions:
            return 0.0

        scores = []
        for pred, ex in zip(predictions, trainset):
            try:
                score = self.metric(pred, ex)
                scores.append(score)
            except Exception as e:
                print(f"Error in metric calculation: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _make_prediction(self, predictor, input_kwargs, ex):
        """Make a single prediction, handling mock mode and inference limits."""
        if self.use_mock:
            pred = self._create_mock_prediction(predictor.signature, input_kwargs, ex)
            time.sleep(0.1)  # Simulate API latency
            return pred

        if self.max_inference_calls > 0 and self.inference_count >= self.max_inference_calls:
            if self.debug:
                print("  Inference call limit reached, using mock prediction")
            return self._create_mock_prediction(predictor.signature, input_kwargs, ex)

        pred = predictor(**input_kwargs)
        self.inference_count += 1
        return pred

    def _log_prediction_debug(self, input_kwargs, pred, elapsed):
        """Log prediction debug information if enabled."""
        if not self.debug:
            return
            
        print(f"  Prediction for '{input_kwargs}' took {elapsed:.4f}s")
        if elapsed < 0.05 and not self.use_mock:
            print("  WARNING: Prediction was extremely fast - likely not calling LLM")
        print(f"  Prediction result: {pred}")

    def _calculate_scores(self, predictions, trainset):
        """Calculate scores for predictions against training examples."""
        scores = []
        for pred, ex in zip(predictions, trainset):
            try:
                score = self.metric(pred, ex)
                scores.append(score)
            except Exception as e:
                print(f"Error in metric calculation: {e}")
                scores.append(0.0)
        return scores

    def _evaluate_predictions(self, predictions, trainset):
        """Calculate average score for predictions."""
        if not predictions:
            return 0.0

        scores = []
        for pred, ex in zip(predictions, trainset):
            try:
                score = self.metric(pred, ex)
                scores.append(score)
            except Exception as e:
                print(f"Error in metric calculation: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _evaluate(self, program, chromosome, trainset):
        """Evaluate a chromosome's performance on the training set.
        
        Args:
            program: DSPy program to evaluate
            chromosome: Chromosome containing task and mutation parts
            trainset: Examples to use for evaluation
            
        Returns:
            Average score across all examples

        """
        try:
            prompt = chromosome.to_prompt()
            if self.debug:
                print(f"\nEvaluating chromosome:\nTask: {chromosome.task_parts}\nMutation: {chromosome.mutation_parts}")

            if self.max_workers > 1:
                return self._parallel_evaluate(program, prompt, trainset)
            else:
                predictions = self._make_predictions(program, prompt, trainset)
                if not predictions:
                    return 0.0
                return self._evaluate_predictions(predictions, trainset)

        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0

    def _parallel_evaluate(self, program, prompt, trainset):
        """Evaluate prompt using parallel workers."""
        from concurrent.futures import as_completed, ThreadPoolExecutor
        
        predictor = dspy.Predict(program.signature, prompt=prompt)
        scores = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for ex in trainset:
                input_kwargs = self._get_input_kwargs(program, ex)
                futures.append(executor.submit(
                    self._make_single_prediction,
                    predictor,
                    input_kwargs,
                    ex
                ))
            
            for future in as_completed(futures):
                try:
                    pred = future.result()
                    score = self.metric(pred, ex)
                    scores.append(score)
                except Exception as e:
                    print(f"Parallel evaluation error: {e}")
                    scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _crossover(self, prompt1: str, prompt2: str) -> str:
        """Combine two prompts by splitting at a random point and joining.
        
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

    def _ensure_placeholders(self, prompt):
        """Ensure input and output placeholders are present."""
        if "{{input}}" not in prompt:
            prompt = prompt.replace("Input:", "{{input}}").replace("Given ", "Given {{input}}")
            if "{{input}}" not in prompt:
                prompt = "{{input}} " + prompt

        if "{{output}}" not in prompt:
            prompt = prompt.replace("-> ", "-> {{output}}").replace(" result", " {{output}} result")
            if "{{output}}" not in prompt:
                prompt = prompt + " {{output}}"
        return prompt

    def _get_mutations(self):
        """Return list of mutation functions with weights based on effectiveness."""
        return [
            # Add instructional phrases (weighted by usefulness)
            (0.3, lambda p: p + " " + random.choice([
                "to generate", "with details", "for classification", "-> answer",
                "analyze and respond", "consider carefully", "be concise", "keep it short"
            ])),

            # Remove some words (but preserve placeholders)
            (0.2, lambda p: " ".join(w for w in p.split()
                if "{{input}}" in w or "{{output}}" in w or random.random() > 0.2)),

            # Enhance input placeholder
            (0.15, lambda p: p.replace("{{input}}", random.choice([
                "Input: {{input}}", "{{input}} here", "Given {{input}}",
                "Consider {{input}}", "Analyze {{input}}", "From {{input}}"
            ]))),

            # Enhance output placeholder
            (0.15, lambda p: p.replace("{{output}}", random.choice([
                "-> {{output}}", "{{output}} result", "yields {{output}}",
                "produce {{output}}", "return {{output}}", "output: {{output}}"
            ]))),

            # Add task-specific instructions
            (0.1, lambda p: p + " " + random.choice([
                "Be concise.", "Explain reasoning.", "Be accurate.",
                "Consider all aspects.", "Focus on key points.",
                "Be specific.", "Be brief.", "Be clear.",
                "Provide details.", "Be precise."
            ])),

            # Add general quality instructions
            (0.05, lambda p: p + " " + random.choice([
                "Ensure high quality.", "Maximize accuracy.",
                "Optimize for clarity.", "Be comprehensive yet concise.",
                "Focus on relevance.", "Prioritize correctness.",
                "Maintain consistency.", "Aim for completeness."
            ])),

            # Character-level mutations (limited to avoid breaking placeholders)
            (0.05, lambda p: p.replace(" ", " " + random.choice([
                "", "", "", "really ", "carefully ", "properly ", "thoroughly "
            ])))
        ]

    def _mutate(self, prompt: str) -> str:
        """Apply weighted random mutations to a prompt.
        
        Args:
            prompt: Prompt to mutate
            
        Returns:
            A mutated version of the prompt with controlled intensity

        """
        prompt = self._ensure_placeholders(prompt)
        weighted_mutations = self._get_mutations()
        
        # Calculate mutation intensity based on current score
        current_score = self._get_prompt_score(prompt)
        intensity = 1.0 - current_score  # More intense mutations for lower scores
        
        # Apply mutations based on intensity
        num_mutations = max(1, min(3, int(3 * intensity)))
        mutations = [m for _, m in weighted_mutations]
        weights = [w for w, _ in weighted_mutations]
        
        for _ in range(num_mutations):
            # Select mutation based on weights
            mutation_fn = random.choices(mutations, weights=weights, k=1)[0]
            prompt = mutation_fn(prompt)
            
            # Ensure prompt doesn't grow too large
            if len(prompt.split()) > 50:
                prompt = " ".join(prompt.split()[:50])

        return prompt

    def _get_prompt_score(self, prompt: str) -> float:
        """Get the current score for a prompt if it exists in population."""
        for p in self.population:
            if p["prompt"] == prompt and p["score"] is not None:
                return p["score"]
        return 0.0  # Default score for new prompts

    def _create_mock_prediction_class(self):
        """Create the MockPrediction class."""
        class MockPrediction:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def __repr__(self):
                attrs = ', '.join(f"{k}='{v}'" for k, v in self.__dict__.items())
                return f"MockPrediction({attrs})"
        return MockPrediction

    def _get_output_values(self, signature, input_kwargs, example):
        """Generate output values for mock prediction."""
        output_values = {}
        for field in signature.output_fields:
            if hasattr(example, field):
                output_values[field] = getattr(example, field)
            else:
                output_values[field] = f"Mock {field} for {input_kwargs}"
        return output_values

    def _create_mock_prediction(self, signature, input_kwargs, example):
        """Create a mock prediction that matches the expected output format.
        
        Args:
            signature: DSPy signature defining input/output fields
            input_kwargs: Input values
            example: The example we're trying to match
            
        Returns:
            A mock prediction object that will pass the metric

        """
        MockPrediction = self._create_mock_prediction_class()
        
        # Generate more realistic mock responses based on input
        output_values = {}
        for field in signature.output_fields:
            if hasattr(example, field):
                # Use example value if available
                output_values[field] = getattr(example, field)
            else:
                # Generate context-aware mock response
                input_text = next(iter(input_kwargs.values()), "")
                output_values[field] = self._generate_mock_response(field, input_text)
                
        return MockPrediction(**output_values)

    def _generate_mock_response(self, field_name: str, input_text: str) -> str:
        """Generate realistic mock responses based on field name and input."""
        # Common response patterns for different field types
        response_templates = {
            "text": [
                "This is a sample response to '{input}'",
                "Based on the input '{input}', here is the output",
                "The analysis of '{input}' suggests this result",
                "Considering '{input}', the conclusion is"
            ],
            "answer": [
                "The answer is 42",
                "After careful consideration, the solution is clear",
                "The correct response would be",
                "Based on the evidence, the answer is"
            ],
            "summary": [
                "In summary, '{input}' can be described as",
                "The key points from '{input}' are",
                "To summarize '{input}', we can say",
                "A brief overview would be"
            ],
            "label": [
                "positive",
                "negative",
                "neutral"
            ]
        }
        
        # Get appropriate templates for field
        templates = response_templates.get(field_name, ["Mock {field} for {input}"])
        
        # Select random template and format with input
        template = random.choice(templates)
        return template.format(field=field_name, input=input_text)

    def get_history(self):
        """Get the evolution history.
        
        Returns:
            List of dictionaries with statistics for each generation

        """
        return self.history
