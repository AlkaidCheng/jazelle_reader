import time
import statistics
import argparse
import os
import sys
import json
import tempfile
import jazelle

class BenchmarkResult:
    def __init__(self, name, times, n_events, meta=None):
        self.name = name
        self.times = times
        self.mean = statistics.mean(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        self.n_events = n_events
        self.rate_khz = (n_events / self.mean) / 1000.0 if self.mean > 0 else 0.0
        self.meta = meta or {}

    def to_row(self):
        return [
            self.name, 
            f"{self.mean:.4f}s", 
            f"± {self.stdev:.4f}s", 
            f"{self.rate_khz:.2f} kHz"
        ]

    def to_dict(self):
        return {
            "name": self.name,
            "mean_seconds": self.mean,
            "stdev_seconds": self.stdev,
            "rate_khz": self.rate_khz,
            "n_events": self.n_events,
            "raw_times": self.times,
            "meta": self.meta
        }

class JazelleBenchmark:
    """
    Base class for all Jazelle benchmarks.
    Handles file initialization, timing loops, and reporting.
    """
    def __init__(self, filepath, count=-1, runs=5, output=None):
        self.filepath = filepath
        self.target_count = count
        self.runs = runs
        self.output = output
        self.results = []
        
        # Initialize the Jazelle file once
        try:
            self.f = jazelle.open(self.filepath)
            self.total_file_events = len(self.f)
            
            # Determine actual processing count
            if self.target_count == -1:
                self.actual_count = self.total_file_events
            else:
                self.actual_count = min(self.target_count, self.total_file_events)
                
            print(f"Initialized {self.__class__.__name__}")
            print(f"File: {filepath}")
            print(f"Events: {self.total_file_events} | Processing: {self.actual_count}")
            
        except Exception as e:
            print(f"Failed to initialize JazelleFile: {e}", file=sys.stderr)
            sys.exit(1)

    def close(self):
        """Cleanup resources."""
        if hasattr(self, 'f') and self.f:
            self.f.close()

    def measure(self, name, func, **kwargs):
        """
        Core timing method.
        
        Parameters
        ----------
        name : str
            Display name for the result.
        func : callable
            The function to benchmark. Must accept (file_obj, count, **kwargs).
        **kwargs : dict
            Arguments passed to func.
        """
        print(f"Running: {name:<40}", end='', flush=True)
        times = []
        
        # Warmup (1 run)
        try:
            self.f.rewind()
            func(self.f, self.actual_count, **kwargs)
        except Exception as e:
            print(f"\nFailed during warmup: {e}")
            return

        # Measurement runs
        for _ in range(self.runs):
            self.f.rewind()
            
            start = time.perf_counter()
            func(self.f, self.actual_count, **kwargs)
            end = time.perf_counter()
            
            times.append(end - start)
            print(".", end='', flush=True)
        
        print(" Done.")
        
        # Store result
        res = BenchmarkResult(name, times, self.actual_count, meta=kwargs)
        self.results.append(res)

    def run(self):
        """
        Main entry point. Subclasses should override this 
        to call self.measure() for their specific tests.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def report(self, title=None):
        """Prints table and saves JSON."""
        if not title:
            title = self.__class__.__name__
            
        self._print_table(title)
        if self.output:
            self._save_json(title)

    def _print_table(self, title):
        print(f"\n=== {title} ===")
        if not self.results:
            print("No results.")
            return

        headers = ["Benchmark", "Time (Mean)", "Std Dev", "Rate"]
        rows = [r.to_row() for r in self.results]
        
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        
        fmt = f"{{:<{widths[0]}}}  {{:>{widths[1]}}}  {{:>{widths[2]}}}  {{:>{widths[3]}}}"
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 6)) 
        for row in rows:
            print(fmt.format(*row))
        print("\n")

    def _save_json(self, title):
        data = {
            "title": title,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file": self.filepath,
            "results": [r.to_dict() for r in self.results]
        }
        try:
            with open(self.output, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Results saved to: {self.output}")
        except IOError as e:
            print(f"Error saving JSON: {e}", file=sys.stderr)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to .jazelle file")
    parser.add_argument('-n', '--count', type=int, default=-1, help="Events to process")
    parser.add_argument('-r', '--runs', type=int, default=5, help="Benchmark runs")
    parser.add_argument('-o', '--output', type=str, default=None, help="JSON output path")
    return parser

class TempFileManager:
    def __init__(self, suffix):
        self.suffix = suffix; self.filename = None
    def __enter__(self):
        fd, self.filename = tempfile.mkstemp(suffix=self.suffix)
        os.close(fd)
        return self.filename
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filename and os.path.exists(self.filename): os.remove(self.filename)