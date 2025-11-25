from utils import JazelleBenchmark, get_parser

class BatchingBenchmark(JazelleBenchmark):
    def __init__(self, filepath, count=-1, runs=5, output=None, num_threads=0):
        super().__init__(filepath, count, runs, output)
        self.num_threads = num_threads

    def run(self):
        # Define batch sizes to test
        batch_sizes = [1, 100, 500, 1000, 5000, 10000, 50000]
        
        # Filter sizes larger than total count to avoid redundant tests
        valid_sizes = [b for b in batch_sizes if b <= self.actual_count]
        if not valid_sizes: 
            valid_sizes = [self.actual_count]
        
        # Ensure largest possible batch is tested if not already included
        if self.actual_count not in valid_sizes and self.actual_count < 100000:
             valid_sizes.append(self.actual_count)
             
        valid_sizes.sort()

        print(f"Scanning batch sizes: {valid_sizes}")
        thread_str = "Auto" if self.num_threads == 0 else str(self.num_threads)
        print(f"Threads: {thread_str}")
        print("-" * 40)

        for b in valid_sizes:
            name = f"Batch Size: {b:6d}"
            self.measure(name, self._bench_func, batch_size=b)
            
        self.report(f"Batch Size Scaling (Threads={thread_str})")

    def _bench_func(self, f, count, batch_size):\
        return f.to_arrays(count=count, batch_size=batch_size, num_threads=self.num_threads)

if __name__ == "__main__":
    # Get base parser and extend it
    parser = get_parser()
    parser.add_argument('-t', '--threads', type=int, default=0, help="Number of threads (0=Auto)")
    args = parser.parse_args()

    bench = BatchingBenchmark(
        args.input_file, 
        args.count, 
        args.runs, 
        args.output,
        num_threads=args.threads
    )
    
    try:
        bench.run()
    finally:
        bench.close()