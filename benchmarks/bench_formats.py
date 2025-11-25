import jazelle
import sys
from utils import benchmark, get_parser, print_results, save_results_json, TempFileManager

@benchmark("1. Iteration (Sequential)")
def bench_iterate(f, count):
    # iterate() does not accept a 'count' argument, so we must break manually.
    i = 0
    # We use batch_size=1 to enforce sequential single-event yielding
    for _ in f.iterate(batch_size=1):
        i += 1
        if count != -1 and i >= count:
            break

@benchmark("2. To Dict (Columnar)")
def bench_to_dict(f, count):
    # to_dict accepts count natively
    _ = f.to_dict(layout='columnar', count=count)

@benchmark("3. To Awkward Array")
def bench_to_awkward(f, count):
    # to_arrays accepts count natively
    _ = f.to_arrays(count=count)

@benchmark("4. Save Parquet")
def bench_parquet(f, count):
    with TempFileManager(".parquet") as tmp:
        f.to_parquet(tmp, count=count)

@benchmark("5. Save Feather")
def bench_feather(f, count):
    with TempFileManager(".feather") as tmp:
        f.to_feather(tmp, count=count)

@benchmark("6. Save HDF5")
def bench_hdf5(f, count):
    with TempFileManager(".h5") as tmp:
        f.to_hdf5(tmp, count=count)

@benchmark("7. Save JSON")
def bench_json(f, count):
    with TempFileManager(".json") as tmp:
        f.to_json(tmp, count=count)

def main():
    args = get_parser("Benchmark data conversion formats").parse_args()
    results = []
    
    try:
        with jazelle.open(args.input_file) as f:
            total = len(f)
            actual_count = total if args.count == -1 else min(args.count, total)
            print(f"File: {args.input_file} | Events: {total} | Processing: {actual_count}")

            results.append(bench_iterate(f, count=args.count, runs=args.runs))
            results.append(bench_to_dict(f, count=args.count, runs=args.runs))
            results.append(bench_to_awkward(f, count=args.count, runs=args.runs))
            results.append(bench_parquet(f, count=args.count, runs=args.runs))
            results.append(bench_feather(f, count=args.count, runs=args.runs))
            results.append(bench_hdf5(f, count=args.count, runs=args.runs))
            results.append(bench_json(f, count=args.count, runs=args.runs))

        title = "Format Conversion Benchmarks"
        print_results(results, title)
        
        if args.output:
            meta = {"input_file": args.input_file, "total_events": total, "processed_events": actual_count}
            save_results_json(results, args.output, title=title, meta=meta)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Print traceback to help debug other potential issues
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()