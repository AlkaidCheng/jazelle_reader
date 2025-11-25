import time
import statistics
import argparse
import os
import sys
import json
from functools import wraps
import tempfile

class BenchmarkResult:
    def __init__(self, name, times, n_events):
        self.name = name
        self.times = times
        self.mean = statistics.mean(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        self.n_events = n_events
        self.rate_khz = (n_events / self.mean) / 1000.0 if self.mean > 0 else 0.0

    def to_row(self):
        """Format data as strings for the table."""
        return [
            self.name, 
            f"{self.mean:.4f}s", 
            f"± {self.stdev:.4f}s", 
            f"{self.rate_khz:.2f} kHz"
        ]

    def to_dict(self):
        """Convert result to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "mean_seconds": self.mean,
            "stdev_seconds": self.stdev,
            "rate_khz": self.rate_khz,
            "n_events": self.n_events,
            "raw_times": self.times
        }

def benchmark(name, n_runs=5, warmup=1):
    """
    Decorator to benchmark a function.
    
    Parameters
    ----------
    name : str
        Display name for the benchmark.
    n_runs : int
        Default number of runs (can be overridden at call time via 'runs=X').
    warmup : int
        Number of warmup runs.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # INTERCEPT 'runs': Check for runtime override, otherwise use decorator default.
            # We use .pop() to remove it from kwargs so it isn't passed to the decorated function.
            actual_runs = kwargs.pop('runs', n_runs)
            
            times = []
            print(f"Running benchmark: {name} (Warmup: {warmup}, Runs: {actual_runs})...", end='', flush=True)
            
            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Actual runs
            for _ in range(actual_runs):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
                print(".", end='', flush=True)
            
            print(" Done.")
            
            # Attempt to determine N events for rate calculation
            n_events = kwargs.get('count', -1)
            if n_events == -1 and len(args) > 0 and hasattr(args[0], '__len__'):
                n_events = len(args[0])
            
            return BenchmarkResult(name, times, n_events)
        return wrapper
    return decorator

def get_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_file', help="Path to .jazelle file")
    parser.add_argument('-n', '--count', type=int, default=-1, help="Number of events to process (-1 for all)")
    parser.add_argument('-r', '--runs', type=int, default=5, help="Number of benchmark runs")
    parser.add_argument('-o', '--output', type=str, default=None, help="Path to save results as JSON")
    return parser

def print_results(results, title):
    """
    Manually constructs a pretty ASCII table without external dependencies.
    """
    print(f"\n=== {title} ===")
    
    headers = ["Benchmark", "Time (Mean)", "Std Dev", "Rate"]
    rows = [r.to_row() for r in results]
    
    # 1. Calculate column widths
    widths = [len(h) for h in headers]
    
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    
    # 2. Define a format string
    # Column 0 (Name): Left aligned (<)
    # Columns 1-3 (Numbers): Right aligned (>)
    fmt = (
        f"{{:<{widths[0]}}}  "  # Name
        f"{{:>{widths[1]}}}  "  # Time
        f"{{:>{widths[2]}}}  "  # Std Dev
        f"{{:>{widths[3]}}}"    # Rate
    )
    
    # 3. Print Table
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 6)) 
    
    for row in rows:
        print(fmt.format(*row))
    print("\n")

def save_results_json(results, filename, title=None, meta=None):
    """Save benchmark results to a JSON file."""
    if not filename:
        return

    output_data = {
        "title": title,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": meta or {},
        "results": [r.to_dict() for r in results]
    }

    try:
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to: {filename}")
    except IOError as e:
        print(f"Error saving results to {filename}: {e}", file=sys.stderr)

class TempFileManager:
    """Context manager for temporary filenames that ensures cleanup."""
    def __init__(self, suffix):
        self.suffix = suffix
        self.filename = None

    def __enter__(self):
        fd, self.filename = tempfile.mkstemp(suffix=self.suffix)
        os.close(fd)
        return self.filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filename and os.path.exists(self.filename):
            os.remove(self.filename)