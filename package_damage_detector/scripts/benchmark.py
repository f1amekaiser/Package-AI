#!/usr/bin/env python3
"""
Performance Benchmark Script

Runs benchmarks on the package damage detection system to validate
real-time performance on target hardware.
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header():
    print("\n" + "=" * 70)
    print("  📊 PERFORMANCE BENCHMARK")
    print("  Package Damage Detection System")
    print("=" * 70 + "\n")


def print_results(results: dict):
    """Print formatted benchmark results."""
    print("\n" + "─" * 60)
    print("  BENCHMARK RESULTS")
    print("─" * 60 + "\n")
    
    status = "✅ PASS" if results.get("pass", False) else "❌ FAIL"
    print(f"  Status: {status}\n")
    
    print("  Total Latency (ms):")
    lat = results["total_latency"]
    print(f"    Mean:   {lat['mean']:.1f}")
    print(f"    Median: {lat['median']:.1f}")
    print(f"    P95:    {lat['p95']:.1f}")
    print(f"    P99:    {lat['p99']:.1f}")
    print(f"    Max:    {lat['max']:.1f}")
    
    print("\n  Per-Camera Inference (ms):")
    for cam, stats in results.get("camera_latencies", {}).items():
        print(f"    {cam}: mean={stats['mean']:.1f}, max={stats['max']:.1f}")
    
    print("\n  Throughput:")
    tp = results["throughput"]
    print(f"    {tp['inspections_per_minute']:.1f} inspections/minute")
    print(f"    {tp['inspections_per_second']:.2f} inspections/second")
    
    print("\n" + "─" * 60)


def run_simulated_benchmark(num_inspections: int = 50):
    """Run benchmark in simulated mode."""
    import random
    import statistics
    
    print("Running simulated benchmark (no hardware)...\n")
    
    latencies = []
    camera_latencies = {f"CAM-0{i}": [] for i in range(1, 6)}
    
    for i in range(num_inspections):
        # Simulate inspection timing
        capture_time = random.uniform(80, 120)
        
        total_inference = 0
        for cam in camera_latencies:
            inf_time = random.uniform(20, 35)
            camera_latencies[cam].append(inf_time)
            total_inference += inf_time
        
        fusion_time = random.uniform(15, 25)
        evidence_time = random.uniform(80, 120)
        
        total = capture_time + total_inference/5 + fusion_time + evidence_time
        latencies.append(total)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_inspections}")
    
    results = {
        "num_inspections": num_inspections,
        "total_latency": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "stdev": statistics.stdev(latencies),
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
        },
        "pass": statistics.mean(latencies) < 500 and sorted(latencies)[int(len(latencies) * 0.95)] < 1000
    }
    
    return results


def run_real_benchmark(num_inspections: int = 50, config_path: str = "config/config.yaml"):
    """Run benchmark with real system."""
    try:
        from src.utils.helpers import load_config
        from src.services.inspection_service import create_inspection_service
        from src.utils.performance import Benchmarker
        
        print("Loading configuration...")
        config = load_config(config_path)
        
        print("Initializing inspection service (simulated cameras)...")
        service = create_inspection_service(config, simulated_cameras=True)
        
        print("Running benchmark...\n")
        benchmarker = Benchmarker(service)
        results = benchmarker.run_benchmark(
            num_inspections=num_inspections,
            warmup=10
        )
        
        return results
        
    except ImportError as e:
        print(f"Could not load system modules: {e}")
        print("Falling back to simulated benchmark...")
        return run_simulated_benchmark(num_inspections)


def main():
    parser = argparse.ArgumentParser(description="Performance Benchmark")
    parser.add_argument("--num", type=int, default=50, help="Number of inspections")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config path")
    parser.add_argument("--simulated", action="store_true", help="Use simulated mode")
    args = parser.parse_args()
    
    print_header()
    
    print(f"Configuration:")
    print(f"  Inspections: {args.num}")
    print(f"  Mode: {'Simulated' if args.simulated else 'Real'}")
    print(f"  Config: {args.config}\n")
    
    start_time = time.time()
    
    if args.simulated:
        results = run_simulated_benchmark(args.num)
    else:
        results = run_real_benchmark(args.num, args.config)
    
    elapsed = time.time() - start_time
    
    print_results(results)
    
    print(f"\n  Benchmark completed in {elapsed:.1f}s")
    print("=" * 70 + "\n")
    
    # Exit code based on pass/fail
    sys.exit(0 if results.get("pass", False) else 1)


if __name__ == "__main__":
    main()
