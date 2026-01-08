#!/usr/bin/env python3
"""
Performance Monitoring & Benchmarking Module

Provides real-time performance monitoring, telemetry collection,
and benchmarking utilities for the package damage detection system.
"""

import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetric:
    """Single latency measurement."""
    timestamp: datetime
    camera_id: str
    phase: str  # capture, inference, fusion, evidence
    latency_ms: float


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    gpu_utilization: float  # 0-100
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    cpu_utilization: float  # 0-100
    queue_depth: int


@dataclass
class PerformanceReport:
    """Performance report for a time window."""
    window_start: datetime
    window_end: datetime
    inspection_count: int
    
    # Latency stats (ms)
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    
    # Per-camera stats
    camera_latencies: Dict[str, float]
    
    # System stats
    gpu_util_mean: float
    gpu_memory_mean: float
    
    # Health indicators
    frame_drops: int
    degraded_inspections: int
    alerts: List[str]


class PerformanceMonitor:
    """
    Real-time performance monitoring for edge deployment.
    
    Tracks:
    - Per-phase latencies
    - GPU/CPU utilization
    - Queue depth
    - Frame drops
    - Alerts and anomalies
    """
    
    # Alert thresholds
    THRESHOLDS = {
        "inference_warning_ms": 35,
        "inference_critical_ms": 50,
        "total_warning_ms": 1000,
        "total_critical_ms": 2000,
        "gpu_util_warning": 80,
        "gpu_util_critical": 95,
        "gpu_memory_warning": 75,
        "gpu_memory_critical": 90,
        "frame_drop_warning": 1,  # percent
        "frame_drop_critical": 5,
        "queue_warning": 2,
        "queue_critical": 5,
    }
    
    def __init__(
        self,
        history_size: int = 1000,
        system_sample_interval: float = 0.1
    ):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of latency records to keep
            system_sample_interval: Seconds between system metric samples
        """
        self.history_size = history_size
        self.system_sample_interval = system_sample_interval
        
        # Latency history
        self._latencies: deque = deque(maxlen=history_size)
        self._inspection_times: deque = deque(maxlen=history_size)
        
        # System metrics
        self._system_metrics: deque = deque(maxlen=int(300 / system_sample_interval))
        
        # Counters
        self._inspection_count = 0
        self._frame_drops = 0
        self._degraded_count = 0
        
        # Current inspection tracking
        self._current_inspection: Dict[str, float] = {}
        
        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
        
        # Monitoring thread
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def start(self):
        """Start background monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitor started")
    
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        logger.info("Performance monitor stopped")
    
    def _monitor_loop(self):
        """Background loop for system metrics."""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self._system_metrics.append(metrics)
                
                # Check for alerts
                self._check_system_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.system_sample_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        gpu_util = 0.0
        gpu_mem_used = 0.0
        gpu_mem_total = 8192.0  # Default 8GB
        
        # Try to get GPU stats
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_util = float(parts[0])
                gpu_mem_used = float(parts[1])
                gpu_mem_total = float(parts[2])
        except Exception:
            pass
        
        # CPU utilization
        cpu_util = 0.0
        try:
            import psutil
            cpu_util = psutil.cpu_percent()
        except Exception:
            pass
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            gpu_utilization=gpu_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
            cpu_utilization=cpu_util,
            queue_depth=0  # Updated by inspection service
        )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds."""
        alerts = []
        
        if metrics.gpu_utilization > self.THRESHOLDS["gpu_util_critical"]:
            alerts.append(f"CRITICAL: GPU utilization {metrics.gpu_utilization:.1f}%")
        elif metrics.gpu_utilization > self.THRESHOLDS["gpu_util_warning"]:
            alerts.append(f"WARNING: GPU utilization {metrics.gpu_utilization:.1f}%")
        
        gpu_mem_pct = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
        if gpu_mem_pct > self.THRESHOLDS["gpu_memory_critical"]:
            alerts.append(f"CRITICAL: GPU memory {gpu_mem_pct:.1f}%")
        
        for alert in alerts:
            self._fire_alert(alert)
    
    def _fire_alert(self, message: str):
        """Fire alert to all callbacks."""
        logger.warning(f"Performance alert: {message}")
        for callback in self._alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def on_alert(self, callback: Callable):
        """Register alert callback."""
        self._alert_callbacks.append(callback)
    
    # Inspection timing methods
    
    def start_inspection(self, inspection_id: str):
        """Mark start of an inspection."""
        with self._lock:
            self._current_inspection = {
                "inspection_id": inspection_id,
                "start": time.perf_counter(),
                "phases": {}
            }
    
    def record_phase(self, phase: str, latency_ms: float, camera_id: str = ""):
        """Record a phase latency."""
        metric = LatencyMetric(
            timestamp=datetime.utcnow(),
            camera_id=camera_id,
            phase=phase,
            latency_ms=latency_ms
        )
        
        with self._lock:
            self._latencies.append(metric)
            
            if self._current_inspection:
                key = f"{phase}_{camera_id}" if camera_id else phase
                self._current_inspection["phases"][key] = latency_ms
        
        # Check phase thresholds
        if phase == "inference" and latency_ms > self.THRESHOLDS["inference_critical_ms"]:
            self._fire_alert(f"CRITICAL: Inference latency {latency_ms:.1f}ms for {camera_id}")
    
    def end_inspection(self, success: bool = True, degraded: bool = False):
        """Mark end of an inspection."""
        with self._lock:
            if self._current_inspection:
                total_ms = (time.perf_counter() - self._current_inspection["start"]) * 1000
                self._inspection_times.append(total_ms)
                self._inspection_count += 1
                
                if degraded:
                    self._degraded_count += 1
                
                # Check total latency
                if total_ms > self.THRESHOLDS["total_critical_ms"]:
                    self._fire_alert(f"CRITICAL: Total latency {total_ms:.1f}ms")
                
                self._current_inspection = {}
    
    def record_frame_drop(self, camera_id: str):
        """Record a frame drop."""
        with self._lock:
            self._frame_drops += 1
        logger.warning(f"Frame drop on {camera_id}")
    
    def update_queue_depth(self, depth: int):
        """Update current queue depth."""
        if depth > self.THRESHOLDS["queue_critical"]:
            self._fire_alert(f"CRITICAL: Queue depth {depth}")
        elif depth > self.THRESHOLDS["queue_warning"]:
            self._fire_alert(f"WARNING: Queue depth {depth}")
    
    # Reporting methods
    
    def get_current_stats(self) -> Dict:
        """Get current performance statistics."""
        with self._lock:
            latencies = [m.latency_ms for m in self._latencies if m.phase == "inference"]
            inspection_times = list(self._inspection_times)
        
        stats = {
            "inspection_count": self._inspection_count,
            "frame_drops": self._frame_drops,
            "degraded_count": self._degraded_count,
        }
        
        if latencies:
            stats["inference_latency"] = {
                "mean": statistics.mean(latencies),
                "p50": statistics.median(latencies),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99),
                "max": max(latencies),
            }
        
        if inspection_times:
            stats["total_latency"] = {
                "mean": statistics.mean(inspection_times),
                "p50": statistics.median(inspection_times),
                "p95": self._percentile(inspection_times, 95),
                "max": max(inspection_times),
            }
        
        # Latest system metrics
        if self._system_metrics:
            latest = self._system_metrics[-1]
            stats["system"] = {
                "gpu_utilization": latest.gpu_utilization,
                "gpu_memory_pct": (latest.gpu_memory_used_mb / latest.gpu_memory_total_mb) * 100,
                "cpu_utilization": latest.cpu_utilization,
            }
        
        return stats
    
    def generate_report(self, window_minutes: int = 60) -> PerformanceReport:
        """Generate performance report for time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        with self._lock:
            latencies = [
                m.latency_ms for m in self._latencies
                if m.phase == "inference" and m.timestamp > cutoff
            ]
            
            # Per-camera latencies
            camera_latencies = {}
            for m in self._latencies:
                if m.phase == "inference" and m.timestamp > cutoff and m.camera_id:
                    if m.camera_id not in camera_latencies:
                        camera_latencies[m.camera_id] = []
                    camera_latencies[m.camera_id].append(m.latency_ms)
            
            camera_means = {
                cam: statistics.mean(lats) for cam, lats in camera_latencies.items()
            }
            
            # System metrics
            system_metrics = [
                m for m in self._system_metrics if m.timestamp > cutoff
            ]
        
        gpu_utils = [m.gpu_utilization for m in system_metrics] if system_metrics else [0]
        gpu_mems = [
            (m.gpu_memory_used_mb / m.gpu_memory_total_mb) * 100
            for m in system_metrics
        ] if system_metrics else [0]
        
        return PerformanceReport(
            window_start=cutoff,
            window_end=datetime.utcnow(),
            inspection_count=self._inspection_count,
            latency_mean=statistics.mean(latencies) if latencies else 0,
            latency_p50=statistics.median(latencies) if latencies else 0,
            latency_p95=self._percentile(latencies, 95) if latencies else 0,
            latency_p99=self._percentile(latencies, 99) if latencies else 0,
            latency_max=max(latencies) if latencies else 0,
            camera_latencies=camera_means,
            gpu_util_mean=statistics.mean(gpu_utils),
            gpu_memory_mean=statistics.mean(gpu_mems),
            frame_drops=self._frame_drops,
            degraded_inspections=self._degraded_count,
            alerts=[]
        )
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]


class Benchmarker:
    """
    Benchmarking utility for performance validation.
    """
    
    def __init__(self, inspection_service):
        """
        Initialize benchmarker.
        
        Args:
            inspection_service: InspectionService instance to benchmark
        """
        self.service = inspection_service
        self.results = []
    
    def run_benchmark(
        self,
        num_inspections: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Run performance benchmark.
        
        Args:
            num_inspections: Number of inspections to run
            warmup: Warmup iterations (not counted)
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Starting benchmark: {warmup} warmup + {num_inspections} measured")
        
        # Warmup
        for i in range(warmup):
            self.service.inspect_package(f"WARMUP-{i}")
        
        # Measured runs
        latencies = []
        camera_latencies = {}
        
        for i in range(num_inspections):
            package_id = f"BENCH-{i:04d}"
            
            start = time.perf_counter()
            result = self.service.inspect_package(package_id)
            total_ms = (time.perf_counter() - start) * 1000
            
            latencies.append(total_ms)
            
            # Collect per-camera times
            for cam_result in result.inference_results:
                cam_id = cam_result.camera_id
                if cam_id not in camera_latencies:
                    camera_latencies[cam_id] = []
                camera_latencies[cam_id].append(cam_result.inference_time_ms)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Benchmark progress: {i + 1}/{num_inspections}")
        
        # Calculate statistics
        results = {
            "num_inspections": num_inspections,
            "total_latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "p99": sorted(latencies)[int(len(latencies) * 0.99)],
                "min": min(latencies),
                "max": max(latencies),
            },
            "camera_latencies": {
                cam: {
                    "mean": statistics.mean(lats),
                    "max": max(lats),
                }
                for cam, lats in camera_latencies.items()
            },
            "throughput": {
                "inspections_per_second": 1000 / statistics.mean(latencies),
                "inspections_per_minute": 60000 / statistics.mean(latencies),
            }
        }
        
        # Pass/fail check
        targets = {
            "total_mean": 500,
            "total_p95": 1000,
            "inference_mean": 35,
        }
        
        results["pass"] = (
            results["total_latency"]["mean"] < targets["total_mean"] and
            results["total_latency"]["p95"] < targets["total_p95"]
        )
        
        logger.info(f"Benchmark complete: {'PASS' if results['pass'] else 'FAIL'}")
        logger.info(f"  Mean latency: {results['total_latency']['mean']:.1f}ms")
        logger.info(f"  P95 latency: {results['total_latency']['p95']:.1f}ms")
        logger.info(f"  Throughput: {results['throughput']['inspections_per_minute']:.1f}/min")
        
        return results


# Singleton monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def start_monitoring():
    """Start global performance monitoring."""
    get_monitor().start()


def stop_monitoring():
    """Stop global performance monitoring."""
    if _monitor:
        _monitor.stop()
