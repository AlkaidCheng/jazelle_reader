import jazelle
import multiprocessing
import sys
from utils import benchmark, get_parser, print_results, save_results_json

def main():
    parser = get_parser("Benchmark multi-threading scaling")
    args = parser.parse_args()
    
    max_cores = multiprocessing.cpu_count()
    thread_counts = [1]
    curr = 2
    while curr <= max_cores:
        thread_counts.append(curr)
        curr *= 2
    
    BATCH_SIZE = 1000 
    results = []
    
    try:
        with jazelle.open(args.input_file) as f:
            total = len(f)
            actual_count = total if args.count == -1 else min(args.count, total)
            print(f"File: {args.input_file} | Processing: {actual_count} | Batch Size: {BATCH_SIZE}")

            for t in thread_counts:
                @benchmark(f"Threads: {t:2d}", n_runs=args.runs)
                def run_threads(file_obj, cnt, threads):
                    _ = file_obj.to_arrays(count=cnt, batch_size=BATCH_SIZE, num_threads=threads)
                
                res = run_threads(f, args.count, t)
                results.append(res)

        title = f"Threading Benchmarks (Fixed Batch={BATCH_SIZE})"
        print_results(results, title)

        if args.output:
            meta = {"input_file": args.input_file, "batch_size": BATCH_SIZE, "processed_events": actual_count}
            save_results_json(results, args.output, title=title, meta=meta)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()