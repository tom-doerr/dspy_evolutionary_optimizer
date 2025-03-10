from typing import List, Optional
import random
import logging

logger = logging.getLogger(__name__)


class Chromosome:
    """Represents a prompt chromosome with multiple functional parts."""

    def _validate_parts(self) -> None:
        """Validate that parts are valid strings."""
        if not all(
            isinstance(part, str) for part in 
            self.task_parts + self.mutation_parts + self.mating_parts
        ):
            raise TypeError("All parts must be strings")

    def __init__(
        self,
        task_parts: Optional[List[str]] = None,
        mutation_parts: Optional[List[str]] = None,
        mating_parts: Optional[List[str]] = None,
        chromosome_type: str = "task"  # task/mutation/mating
    ) -> None:
        """Initialize a Chromosome instance.

        Args:
            task_parts: List of task-related prompt parts
            mutation_parts: List of mutation-related prompt parts  
            mating_parts: List of mating-related prompt parts
            chromosome_type: Type of chromosome (task/mutation/mating)
        """
        self.task_parts = task_parts or []
        self.mutation_parts = mutation_parts or []
        self.mating_parts = mating_parts or []
        self.chromosome_type = chromosome_type
        self._validate_parts()

    def combine(self, other: "Chromosome") -> "Chromosome":
        """Combine two chromosomes by merging their parts.

        Args:
            other: Another Chromosome to combine with

        Returns:
            A new Chromosome with combined parts

        Raises:
            TypeError: If other is not a Chromosome
        """
        if not isinstance(other, Chromosome):
            raise TypeError(f"Can only combine with Chromosome, got {type(other)}")

        try:
            new_task = self._merge_parts(self.task_parts, other.task_parts)
            new_mutation = self._merge_parts(self.mutation_parts, other.mutation_parts)
            return Chromosome(new_task, new_mutation)
        except Exception as e:
            logger.error("Error combining chromosomes: %s", e)
            raise ValueError("Failed to combine chromosomes") from e

    def _merge_parts(self, parts1, parts2):
        """Merge two sets of parts using crossover"""
        if not parts1 or not parts2:
            return parts1 or parts2

        min_len = min(len(parts1), len(parts2))
        crossover_point = random.randint(0, min_len)
        return parts1[:crossover_point] + parts2[crossover_point:]

    def mutate(self, mutation_rate: float = 0.5) -> None:
        """Apply mutations to the chromosome.

        Args:
            mutation_rate: Probability of mutation (0.0 to 1.0)

        Raises:
            ValueError: If mutation_rate is not between 0 and 1
        """
        if not 0 <= mutation_rate <= 1:
            raise ValueError(
                f"mutation_rate must be between 0 and 1, got {mutation_rate}"
            )

        if random.random() < mutation_rate:
            try:
                self.mutation_parts = self._shuffle_parts(self.mutation_parts)
                self._validate_parts()
            except Exception as e:
                logger.error("Error during mutation: %s", e)
                raise ValueError("Failed to mutate chromosome") from e

    def _shuffle_parts(self, parts):
        """Randomly shuffle parts while maintaining structure"""
        if not parts:
            return parts

        blocks = self._split_into_blocks(parts)
        random.shuffle(blocks)
        return self._flatten_blocks(blocks)

    def _split_into_blocks(self, parts):
        """Split parts into instruction blocks"""
        blocks = []
        current_block = []
        for part in parts:
            current_block.append(part)
            if part.endswith("."):
                blocks.append(current_block)
                current_block = []
        if current_block:
            blocks.append(current_block)
        return blocks

    def _flatten_blocks(self, blocks):
        """Flatten blocks back into parts list"""
        return [part for block in blocks for part in block]

    def to_prompt(self):
        """Convert chromosome to executable prompt"""
        task = " ".join(self.task_parts)
        mutation = " ".join(self.mutation_parts)
        return f"{task} {mutation}".strip()
