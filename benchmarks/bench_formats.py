from utils import JazelleBenchmark, get_parser, TempFileManager

class FormatBenchmark(JazelleBenchmark):
    def run(self):
        # 1. Iteration
        self.measure("Iteration (Sequential)", self._bench_iterate_sequential)

        # 2. Batch Iteration
        self.measure("Iteration (Batched)", self._bench_iterate_batch)
        
        # 3. In-Memory Conversion
        self.measure("To Dict (Columnar)", self._bench_dict)
        self.measure("To Awkward Array", self._bench_awkward)
        
        # 4. IO Streamers
        self.measure("Save Parquet", self._bench_parquet)
        self.measure("Save Feather", self._bench_feather)
        self.measure("Save HDF5", self._bench_hdf5)
        self.measure("Save JSON", self._bench_json)
        
        self.report("Format Conversion Benchmarks")

    # --- Implementation Methods ---
    
    def _bench_iterate_sequential(self, f, count):
        i = 0
        # batch_size=1 forces sequential single-event yield
        for _ in f.iterate(batch_size=1):
            i += 1
            if count != -1 and i >= count:
                break

    def _bench_iterate_batch(self, f, count):
        i = 0
        for _ in f.iterate(batch_size=1000):
            i += 1
            if count != -1 and i >= count:
                break                

    def _bench_dict(self, f, count):
        f.to_dict(layout='columnar', count=count)

    def _bench_awkward(self, f, count):
        f.to_arrays(count=count)

    def _bench_parquet(self, f, count):
        with TempFileManager(".parquet") as tmp:
            f.to_parquet(tmp, count=count)

    def _bench_feather(self, f, count):
        with TempFileManager(".feather") as tmp:
            f.to_feather(tmp, count=count)

    def _bench_hdf5(self, f, count):
        with TempFileManager(".h5") as tmp:
            f.to_hdf5(tmp, count=count)

    def _bench_json(self, f, count):
        with TempFileManager(".json") as tmp:
            f.to_json(tmp, count=count)

if __name__ == "__main__":
    args = get_parser().parse_args()
    bench = FormatBenchmark(args.input_file, args.count, args.runs, args.output)
    try:
        bench.run()
    finally:
        bench.close()