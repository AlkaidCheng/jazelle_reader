import jazelle
import sys
from utils import benchmark, get_parser, print_results, save_results_json

def main():
    parser = get_parser("Benchmark batch size scaling")
    args = parser.parse_args()
    
    batch_sizes = [1, 100, 500, 1000, 5000, 10000]
    THREADS = 8
    results = []
    
    try:
        with jazelle.open(args.input_file) as f:
            total = len(f)
            actual_count = total if args.count == -1 else min(args.count, total)
            threads_str = "auto" if THREADS == 0 else str(THREADS)
            print(f"File: {args.input_file} | Processing: {actual_count} | Threads: {threads_str}")

            for b in batch_sizes:
                if b > actual_count and b != batch_sizes[0]: 
                    continue

                @benchmark(f"Batch Size: {b:5d}", n_runs=args.runs)
                def run_batch(file_obj, cnt, b_size):
                    _ = file_obj.to_arrays(count=cnt, batch_size=b_size, num_threads=THREADS)
                
                res = run_batch(f, args.count, b)
                results.append(res)

        title = "Batch Size Benchmarks"
        print_results(results, title)

        if args.output:
            meta = {"input_file": args.input_file, "processed_events": actual_count, "threads": THREADS}
            save_results_json(results, args.output, title=title, meta=meta)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()