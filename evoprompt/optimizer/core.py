"""Core evolutionary optimization logic."""

from dataclasses import dataclass
from statistics import mean
from typing import List, Dict, Any, Callable
import random


@dataclass
class EvolutionParams:
    generations: int = 10
    mutation_rate: float = 0.5
    growth_rate: float = 0.3
    max_population: int = 100


@dataclass
class RuntimeParams:
    max_workers: int = 1
    debug: bool = False
    max_inference_calls: int = 100


@dataclass
class OptimizerConfig:
    metric: Callable
    evolution: EvolutionParams = EvolutionParams()
    runtime: RuntimeParams = RuntimeParams()


@dataclass
class OptimizerState:
    inference_count: int = 0
    population: List[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = None


class EvolutionaryCore:
    """Core evolutionary algorithm implementation."""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.state = OptimizerState()

    def get_population_stats(self, population: List[Dict[str, Any]]) -> tuple:
        """Public method to get population statistics."""
        return self._get_population_stats(population)

    def _select_prompt(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a prompt using Pareto distribution."""
        if not population:
            raise ValueError("Cannot select from empty population")

        scored_population = [p for p in population if p["score"] is not None]
        if not scored_population:
            return random.choice(population)

        return self._select_using_pareto(scored_population)

    def _select_using_pareto(
        self, scored_population: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select using Pareto distribution weights."""
        scored_population.sort(key=lambda x: x["score"], reverse=True)
        alpha = 1.16
        n = len(scored_population)
        weights = [(n - i) ** (-alpha) for i in range(n)]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(scored_population, weights=probs, k=1)[0]

    def _update_population(
        self,
        population: List[Dict[str, Any]],
        iteration: int,
        recent_scores: List[float],
    ) -> List[Dict[str, Any]]:
        """Update population based on scores and iteration."""
        if recent_scores:
            recent_scores_sorted = sorted(recent_scores)
            _ = recent_scores_sorted[int(len(recent_scores_sorted) * 0.8)]

        population = [
            p
            for p in population
            if iteration - p["last_used"] < 10 or p["score"] is None
        ]

        if len(population) > self.config.max_population:
            scored_population = [p for p in population if p["score"] is not None]
            if len(scored_population) > self.config.max_population:
                scored_population.sort(key=lambda x: x["score"])
                population = scored_population[-self.config.max_population :]

        return population

    def _get_population_stats(self, population: List[Dict[str, Any]]) -> tuple:
        """Calculate population statistics."""
        scores = [p["score"] for p in population if p["score"] is not None]
        if not scores:
            return None, None, None

        best_score = max(scores)
        avg_score = mean(scores)
        best_prompt = max(
            population,
            key=lambda x: x["score"] if x["score"] is not None else -float("inf"),
        )["prompt"]

        return best_score, avg_score, best_prompt
"""Core evolutionary optimization logic."""

from dataclasses import dataclass
from statistics import mean
from typing import List, Dict, Any, Callable
import random


@dataclass
class EvolutionParams:
    generations: int = 10
    mutation_rate: float = 0.5
    growth_rate: float = 0.3
    max_population: int = 100


@dataclass
class RuntimeParams:
    max_workers: int = 1
    debug: bool = False
    max_inference_calls: int = 100


@dataclass
class OptimizerConfig:
    metric: Callable
    evolution: EvolutionParams = EvolutionParams()
    runtime: RuntimeParams = RuntimeParams()


@dataclass
class OptimizerState:
    inference_count: int = 0
    population: List[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = None


class EvolutionaryCore:
    """Core evolutionary algorithm implementation."""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.state = OptimizerState()

    def get_population_stats(self, population: List[Dict[str, Any]]) -> tuple:
        """Public method to get population statistics."""
        return self._get_population_stats(population)

    def _select_prompt(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a prompt using Pareto distribution."""
        if not population:
            raise ValueError("Cannot select from empty population")

        scored_population = [p for p in population if p["score"] is not None]
        if not scored_population:
            return random.choice(population)

        return self._select_using_pareto(scored_population)

    def _select_using_pareto(
        self, scored_population: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select using Pareto distribution weights."""
        scored_population.sort(key=lambda x: x["score"], reverse=True)
        alpha = 1.16
        n = len(scored_population)
        weights = [(n - i) ** (-alpha) for i in range(n)]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(scored_population, weights=probs, k=1)[0]

    def _update_population(
        self,
        population: List[Dict[str, Any]],
        iteration: int,
        recent_scores: List[float],
    ) -> List[Dict[str, Any]]:
        """Update population based on scores and iteration."""
        if recent_scores:
            recent_scores_sorted = sorted(recent_scores)
            _ = recent_scores_sorted[int(len(recent_scores_sorted) * 0.8)]

        population = [
            p
            for p in population
            if iteration - p["last_used"] < 10 or p["score"] is None
        ]

        if len(population) > self.config.max_population:
            scored_population = [p for p in population if p["score"] is not None]
            if len(scored_population) > self.config.max_population:
                scored_population.sort(key=lambda x: x["score"])
                population = scored_population[-self.config.max_population :]

        return population

    def _get_population_stats(self, population: List[Dict[str, Any]]) -> tuple:
        """Calculate population statistics."""
        scores = [p["score"] for p in population if p["score"] is not None]
        if not scores:
            return None, None, None

        best_score = max(scores)
        avg_score = mean(scores)
        best_prompt = max(
            population,
            key=lambda x: x["score"] if x["score"] is not None else -float("inf"),
        )["prompt"]

        return best_score, avg_score, best_prompt
