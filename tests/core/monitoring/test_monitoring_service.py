
import pytest
import time
from core.monitoring.monitoring_service import MonitoringService, monitoring_service

@pytest.fixture(autouse=True)
def reset_monitoring_service():
    """Fixture to reset the singleton monitoring service before each test."""
    monitoring_service.reset()

def test_singleton_instance():
    """Test that the MonitoringService is a singleton."""
    instance1 = MonitoringService()
    instance2 = MonitoringService()
    assert instance1 is instance2
    assert instance1 is monitoring_service

def test_increment_counter():
    """Test the increment_counter method."""
    assert monitoring_service.counters.get("test.counter") is None
    monitoring_service.increment_counter("test.counter")
    assert monitoring_service.counters["test.counter"] == 1.0
    monitoring_service.increment_counter("test.counter", 5.0)
    assert monitoring_service.counters["test.counter"] == 6.0

def test_set_gauge():
    """Test the set_gauge method."""
    assert monitoring_service.gauges.get("test.gauge") is None
    monitoring_service.set_gauge("test.gauge", 100.0)
    assert monitoring_service.gauges["test.gauge"] == 100.0
    monitoring_service.set_gauge("test.gauge", 50.0)
    assert monitoring_service.gauges["test.gauge"] == 50.0

def test_record_latency():
    """Test the record_latency method."""
    assert not monitoring_service.latencies.get("test.latency")
    monitoring_service.record_latency("test.latency", 0.5)
    monitoring_service.record_latency("test.latency", 1.5)
    assert monitoring_service.latencies["test.latency"] == [0.5, 1.5]

def test_get_summary():
    """Test the get_summary method."""
    monitoring_service.increment_counter("tasks.completed")
    monitoring_service.set_gauge("agents.active", 5)
    monitoring_service.record_latency("task.duration", 0.1)
    monitoring_service.record_latency("task.duration", 0.3)

    summary = monitoring_service.get_summary()

    # Verify counters
    assert "counters" in summary
    assert summary["counters"]["tasks.completed"] == 1.0

    # Verify gauges
    assert "gauges" in summary
    assert summary["gauges"]["agents.active"] == 5

    # Verify latencies
    assert "latencies" in summary
    latency_summary = summary["latencies"]["task.duration"]
    assert latency_summary["count"] == 2
    assert latency_summary["avg_seconds"] == pytest.approx(0.2)
    assert latency_summary["max_seconds"] == 0.3
    assert latency_summary["min_seconds"] == 0.1

def test_reset():
    """Test that the reset method clears all metrics."""
    monitoring_service.increment_counter("test.counter")
    monitoring_service.set_gauge("test.gauge", 10)
    monitoring_service.record_latency("test.latency", 0.1)

    # Verify data exists
    assert monitoring_service.counters
    assert monitoring_service.gauges
    assert monitoring_service.latencies

    # Reset and verify it's empty
    monitoring_service.reset()
    assert not monitoring_service.counters
    assert not monitoring_service.gauges
    assert not monitoring_service.latencies
    summary = monitoring_service.get_summary()
    assert not summary["counters"]
    assert not summary["gauges"]
    assert not summary["latencies"]
