"""
Metrics Tracker: Real-time tracking of training and evaluation metrics.
Provides summary statistics, trend detection, and early stopping signals.
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks metrics throughout training with windowed statistics,
    plateau/regression detection, and checkpointing decisions.
    """

    def __init__(
        self,
        output_dir: str = "./outputs/metrics",
        patience: int = 5,
        min_delta: float = 0.001,
    ):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save metric logs.
            patience: Number of evaluations without improvement before signaling.
            min_delta: Minimum improvement to count as progress.
        """
        self.output_dir = output_dir
        self.patience = patience
        self.min_delta = min_delta
        
        self.metrics: Dict[str, List[Dict]] = defaultdict(list)
        self.best_scores: Dict[str, float] = {}
        self.no_improvement_count: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()

        os.makedirs(output_dir, exist_ok=True)

    def log(self, metric_name: str, value: float, step: int, **extra):
        """
        Log a metric value.

        Args:
            metric_name: Name of the metric (e.g., "gsm8k_accuracy", "reward_mean").
            value: Metric value.
            step: Training step or epoch.
            **extra: Additional metadata.
        """
        entry = {
            "value": value,
            "step": step,
            "timestamp": time.time() - self.start_time,
            **extra,
        }
        self.metrics[metric_name].append(entry)

        # Check for improvement
        is_best = False
        if metric_name not in self.best_scores or value > self.best_scores[metric_name] + self.min_delta:
            self.best_scores[metric_name] = value
            self.no_improvement_count[metric_name] = 0
            is_best = True
        else:
            self.no_improvement_count[metric_name] += 1

        return is_best

    def log_batch(self, metrics: Dict[str, float], step: int):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log(name, value, step)

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]["value"]
        return None

    def get_best(self, metric_name: str) -> Optional[float]:
        """Get the best recorded value for a metric."""
        return self.best_scores.get(metric_name)

    def get_trend(self, metric_name: str, window: int = 10) -> Optional[str]:
        """
        Detect trend for a metric over the last N entries.
        
        Returns:
            "improving", "plateauing", "degrading", or None.
        """
        entries = self.metrics.get(metric_name, [])
        if len(entries) < window:
            return None

        recent = [e["value"] for e in entries[-window:]]
        first_half = sum(recent[:window//2]) / (window//2)
        second_half = sum(recent[window//2:]) / (window - window//2)

        diff = second_half - first_half
        if diff > self.min_delta:
            return "improving"
        elif diff < -self.min_delta:
            return "degrading"
        else:
            return "plateauing"

    def should_early_stop(self, metric_name: str) -> bool:
        """Check if training should stop based on lack of improvement."""
        return self.no_improvement_count.get(metric_name, 0) >= self.patience

    def get_summary(self) -> Dict:
        """Get a summary of all tracked metrics."""
        summary = {}
        for name, entries in self.metrics.items():
            if not entries:
                continue
            values = [e["value"] for e in entries]
            summary[name] = {
                "latest": values[-1],
                "best": max(values),
                "worst": min(values),
                "mean": sum(values) / len(values),
                "num_entries": len(entries),
                "trend": self.get_trend(name),
                "no_improvement_for": self.no_improvement_count.get(name, 0),
            }
        return summary

    def get_improvement_over_baseline(
        self, metric_name: str, baseline: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate improvement over a baseline.
        
        Returns:
            Tuple of (absolute_improvement, relative_improvement_pct) or None.
        """
        best = self.get_best(metric_name)
        if best is None:
            return None
        
        abs_improvement = best - baseline
        rel_improvement = (abs_improvement / baseline * 100) if baseline != 0 else 0
        return abs_improvement, rel_improvement

    def save(self):
        """Save all metrics to disk."""
        filepath = os.path.join(self.output_dir, "metrics_log.json")
        data = {
            name: entries for name, entries in self.metrics.items()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save summary
        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(f"Metrics saved to {self.output_dir}")

    def print_report(self, baselines: Optional[Dict[str, float]] = None):
        """Print a formatted metrics report."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("📊 METRICS REPORT")
        print("=" * 70)
        
        for name, stats in summary.items():
            trend_emoji = {
                "improving": "📈",
                "degrading": "📉",
                "plateauing": "➡️",
            }.get(stats["trend"], "❓")
            
            print(f"\n  {name}:")
            print(f"    Latest: {stats['latest']:.4f}  |  Best: {stats['best']:.4f}")
            print(f"    Trend: {trend_emoji} {stats['trend'] or 'N/A'}")
            
            if baselines and name in baselines:
                improvement = self.get_improvement_over_baseline(name, baselines[name])
                if improvement:
                    abs_imp, rel_imp = improvement
                    emoji = "✅" if rel_imp >= 5.0 else "⚠️"
                    print(f"    vs Baseline ({baselines[name]:.4f}): {emoji} +{abs_imp:.4f} ({rel_imp:+.1f}%)")

        print("\n" + "=" * 70)
