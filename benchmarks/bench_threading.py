import multiprocessing
from utils import JazelleBenchmark, get_parser

class ThreadingBenchmark(JazelleBenchmark):
    def run(self):
        max_cores = multiprocessing.cpu_count()
        threads = [1]
        curr = 2
        while curr <= max_cores:
            threads.append(curr)
            curr *= 2
            
        # Fixed batch size for fair comparison
        fixed_batch = 1000
        
        print(f"Scanning thread counts: {threads}")
        print(f"Fixed Batch Size: {fixed_batch}")
        print("-" * 40)

        for t in threads:
            name = f"Threads: {t:2d}"
            self.measure(name, self._bench_func, num_threads=t, batch_size=fixed_batch)
            
        self.report(f"Threading Scaling (Batch={fixed_batch})")

    def _bench_func(self, f, count, num_threads, batch_size):
        # to_arrays exercises the parallel reader + data conversion
        f.to_arrays(count=count, num_threads=num_threads, batch_size=batch_size)

if __name__ == "__main__":
    args = get_parser().parse_args()
    bench = ThreadingBenchmark(args.input_file, args.count, args.runs, args.output)
    try:
        bench.run()
    finally:
        bench.close()