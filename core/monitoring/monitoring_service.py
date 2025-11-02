"""
Core Monitoring Service for the Unified AI System.

This service provides a centralized a mechanism for collecting and exposing
observability data, including metrics like counters, gauges, and timers.
It is designed as a singleton to ensure a single source of truth for metrics
across the entire application.
"""
import time
from collections import defaultdict
from typing import Dict, Any, List


class MonitoringService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitoringService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self._initialized = True

    def increment_counter(self, name: str, value: float = 1.0):
        """Increments a counter metric."""
        self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Sets a gauge metric to a specific value."""
        self.gauges[name] = value

    def record_latency(self, name: str, duration_seconds: float):
        """Records a latency or timing measurement."""
        self.latencies[name].append(duration_seconds)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all collected metrics."""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "latencies": {},
        }
        for name, values in self.latencies.items():
            if values:
                summary["latencies"][name] = {
                    "count": len(values),
                    "avg_seconds": sum(values) / len(values),
                    "max_seconds": max(values),
                    "min_seconds": min(values),
                }
        return summary

    def reset(self):
        """Resets all metrics. Primarily for testing purposes."""
        self.counters.clear()
        self.gauges.clear()
        self.latencies.clear()

# Singleton instance
monitoring_service = MonitoringService()
