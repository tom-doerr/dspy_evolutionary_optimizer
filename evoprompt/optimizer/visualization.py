"""Visualization utilities for evolutionary optimization."""

from typing import List, Dict, Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from textual.widgets import ProgressBar

class EvolutionVisualizer:
    """Handles visualization of evolution progress."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.console = Console()
        
    def _create_progress_bar(self, inference_count: int, max_inference_calls: int):
        """Create progress bar with error handling."""
        try:
            progress = ProgressBar(total=max_inference_calls)
            progress.advance(inference_count)
            progress.width = 50
            return progress
        except (ValueError, TypeError, AttributeError) as e:
            if self.debug:
                print(f"Error creating progress bar: {e}")
            return f"[Progress: {inference_count}/{max_inference_calls}]"

    def _create_main_panel(self, iteration: int, best_score: float, 
                         avg_score: float, population_size: int,
                         inference_count: int, max_inference_calls: int) -> Table:
        """Create main stats panel."""
        panel = Table.grid(padding=(1, 2))
        panel.add_column(justify="left", style="cyan")
        panel.add_column(justify="right", style="magenta")
        
        panel.add_row("Iteration", f"[bold]{iteration}")
        panel.add_row("Best Score", f"[green]{best_score:.3f}")
        panel.add_row("Avg Score", f"[yellow]{avg_score:.3f}")
        panel.add_row("Population", f"[blue]{population_size}")
        panel.add_row("Inference Calls", f"[cyan]{inference_count}/{max_inference_calls}")
        return panel

    def _create_history_table(self, history: List[Dict[str, Any]]) -> Table:
        """Create recent history table."""
        table = Table(title="[bold]Recent History", show_header=True, header_style="bold magenta")
        table.add_column("Iteration", justify="right")
        table.add_column("Best Score", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Population", justify="right")

        for entry in history[-5:]:
            table.add_row(
                str(entry['iteration']),
                f"{entry['best_score']:.3f}",
                f"{entry['avg_score']:.3f}",
                str(entry['population_size'])
            )
        return table

    def log_progress(self, *, iteration: int, population: List[Dict[str, Any]],
                    inference_count: int, max_inference_calls: int,
                    history: List[Dict[str, Any]], best_prompt: str) -> None:
        """Log and display progress information."""
        best_score, avg_score, _ = self._get_population_stats(population)
        if best_score is None:
            return
            
        progress = self._create_progress_bar(inference_count, max_inference_calls)
        main_panel = self._create_main_panel(iteration, best_score, avg_score,
                                           len(population), inference_count,
                                           max_inference_calls)
        history_table = self._create_history_table(history)

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
            self.console.print(Panel(
                Group(*layout_components),
                title=f"[bold]Evolution Progress - Generation {iteration}",
                border_style="green",
                padding=(1, 2),
                width=80
            ))
        except (ValueError, TypeError, AttributeError) as e:
            if self.debug:
                print(f"Error rendering progress display: {e}")
            self.console.print(f"[bold]Generation {iteration}")
            self.console.print(f"Best Score: {best_score:.3f}")
            self.console.print(f"Avg Score: {avg_score:.3f}")
            self.console.print(f"Population: {len(population)}")
            self.console.print(f"Inference Calls: {inference_count}/{max_inference_calls}")

        self.console.print()
