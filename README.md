# Jazelle Reader

[![Publish to PyPI](https://github.com/AlkaidCheng/jazelle_reader/actions/workflows/build_and_publish.yml/badge.svg)](https://github.com/AlkaidCheng/jazelle_reader/actions/workflows/build_and_publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/jazelle.svg)](https://pypi.org/project/jazelle/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jazelle.svg)](https://pypi.org/project/jazelle/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/jazelle.svg)](https://pypi.org/project/jazelle/)
[![GitHub license](https://img.shields.io/github/license/AlkaidCheng/jazelle_reader.svg)](https://github.com/AlkaidCheng/jazelle_reader/blob/main/LICENSE)

> **A modern, high-performance Python reader for SLD Jazelle files with multi-threaded parallel processing and conversion to modern data formats.**

Jazelle reader resurrects legacy particle physics data from the Stanford Linear Collider Detector (SLD) experiment by translating the original java-based Jazelle format reader into a modern, efficient C++20 implementation with seamless Python integration. Built for performance and usability, it enables researchers to work with decades-old experimental data using contemporary analysis tools and workflows.

## 🎯 Motivation

The Stanford Linear Collider Detector (SLD) at SLAC produced invaluable particle physics data in a custom binary format called "Jazelle" during its operational years. As part of an effort to **resurrect old experiments via LLM agents**, we need to make this historical data accessible in modern formats like HDF5, Parquet, and Awkward Arrays for contemporary analysis pipelines.

The original Jazelle reader was written in Fortran (later translated to Java, see [repository](https://github.com/tony-johnson/Jazelle) from Tony Johnson), posing challenges for integration with modern Python-based physics analysis ecosystems. This project bridges that gap by:

- **Modernizing the codebase**: Complete rewrite in C++20 with modern best practices
- **Maximizing performance**: Multi-threaded parallel processing with lock-free queues
- **Enabling interoperability**: Native Python bindings via Cython for seamless integration
- **Supporting modern formats**: Direct conversion to HDF5, Parquet, Feather, and more
- **Preserving data integrity**: Faithful implementation of the original Jazelle format specification
- **Production-ready infrastructure**: CI/CD pipeline with automated testing, benchmarking, and multi-platform wheel building

## ✨ Key Features

### 🚀 High Performance
- **Modern C++20 core** with optimized binary I/O and VAX floating-point conversion
- **Multi-threaded parallel reading** with configurable thread pools
- **Efficient batching** system achieving high throughput at moderate batch sizes
- **Memory-efficient streaming** for processing files larger than available RAM
- **Lock-free queues** for thread-safe parallel event processing

### 🔄 Format Conversion
- **HDF5**: Industry-standard hierarchical data format with compression
- **Parquet**: Columnar storage for efficient analytics
- **Feather**: Fast binary format for data frames
- **JSON**: Human-readable interchange format
- **NumPy arrays**: Direct memory mapping for analysis
- **Awkward Arrays**: Jagged array support for variable-length data

### 🐍 Pythonic API
- Clean, intuitive interface following Python best practices
- Rich display system with HTML rendering in Jupyter notebooks
- Iterator protocol support for memory-efficient sequential processing
- Random access with intuitive indexing: `file[42]`
- Context managers for automatic resource cleanup
- Type hints for better IDE support

### 🛠️ Command-Line Interface
Powerful CLI tool accessible via `jazelle` command:
- **inspect**: View file metadata and statistics
- **read**: Examine specific events with wildcard filtering
- **convert**: Multi-threaded export to modern formats

### 🔧 Modern Development Infrastructure
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Multi-platform Wheels**: Pre-built binaries for Linux, macOS, and Windows
- **Comprehensive Testing**: Unit tests for both C++ and Python codebases
- **Performance Benchmarks**: Automated benchmarking suite for regression testing
- **Type Safety**: Full type hints for Python API
- **Documentation**: Inline docstrings and tutorial notebooks

### 📊 Advanced Features
- **Comprehensive display system**: HTML and ASCII rendering for events and families
- **Event slicing**: Extract subsets with `start` and `count` parameters
- **Family filtering**: Select specific detector banks for analysis
- **Flexible compression**: Customizable compression levels for output formats

## 📦 Installation

### Quick Install from PyPI

```bash
pip install jazelle
```

### With Optional Dependencies

```bash
# For HDF5 support
pip install jazelle h5py

# For Parquet support
pip install jazelle pyarrow

# For Awkward array support
pip install jazelle awkward

# For all format support
pip install jazelle h5py pyarrow awkward
```

### From Source

**Note**: Building from source requires a C++20 compatible compiler (GCC ≥ 10, Clang ≥ 11, MSVC ≥ 19.29).

```bash
git clone https://github.com/AlkaidCheng/jazelle_reader.git
cd jazelle_reader
pip install -e .
```

### Requirements

- **Python**: ≥ 3.8
- **NumPy**: ≥ 1.20
- **Optional**: h5py (HDF5), pyarrow (Parquet), awkward (analysis)

**For source builds only**:
- **C++ Compiler**: C++20 compatible (GCC ≥ 10, Clang ≥ 11, MSVC ≥ 19.29)

## 🚀 Quick Start

### Basic Reading

```python
import jazelle

# Open a Jazelle file
with jazelle.open(filepath) as f:
    # Get file information
    print(f"Total events: {len(f)}")
    
    # Read first event
    event = f.read()
    print(f"Run: {event.ieventh.run}, Event: {event.ieventh.event}")
    
    # Access physics summary
    if event.phpsum.size > 0:
        print(f"  Particle charge: {event.phpsum[0].charge}")
        print(f"  Particle x-position: {event.phpsum[0].x:.3f}")
        print(f"  Particle momentum: {event.phpsum[0].getPTot()}")
```

### Parallel Processing

```python
# Read all events with multi-threading
with jazelle.open('data.jazelle') as f:
    events = f.read_batch(num_threads=8)
    print(f"Read {len(events)} events")

# Process in batches (memory efficient for large files)
with jazelle.open('data.jazelle') as f:
    for batch in f.iterate(batch_size=1000, num_threads=8):
        # Analyze batch
        total_charged = sum(len(evt.phchrg) for evt in batch)
        total_clusters = sum(len(evt.phklus) for evt in batch)
        print(f"Batch: {total_charged} charged tracks, {total_clusters} clusters")
```

### Convert to Modern Formats

```python
# Convert to HDF5
with jazelle.open('input.jazelle') as f:
    f.to_hdf5('output.h5', num_threads=8, compression='gzip', compression_opts=4)

# Convert to Parquet (fastest for analytics)
with jazelle.open('input.jazelle') as f:
    f.to_parquet('output.parquet', num_threads=8)

# Convert to Feather (fast binary format)
with jazelle.open('input.jazelle') as f:
    f.to_feather('output.feather', num_threads=8)

# Convert specific event range
with jazelle.open('input.jazelle') as f:
    f.to_hdf5('output.h5', start=0, count=1000, num_threads=16)
```

### NumPy and Awkward Arrays

```python
# Get data as flat NumPy arrays
with jazelle.open('data.jazelle') as f:
    data = f.to_dict()
    
    # Access PHCHRG (particle tracks) data as structured arrays
    charges = data['PHCHRG']['charge']  # 1D array
    charged_particles = data['PHCHRG']['hlxpar']       # 2D array (N, 6)
    pT  = 1 / data['PHCHRG']['hlxpar'][:, 1]
    nhits = data['PHCHRG']['nhit']      # Number of hits per track
    
    print(f"Total charged particles: {len(charges)}")
    print(f"Mean momentum magnitude: {pT.mean():.2f} GeV")
    pT_split_events = np.split(pT, data['PHCHRG']['_offsets'][1:-1])
    pT_mean_event = np.mean([np.sum(pT_event) for pT_event in pT_split_events])
    print(f"Mean of total momentum per event: {pT_mean_event:.2f} GeV")

# Get data as Awkward Arrays (for jagged data)
with jazelle.open('data.jazelle') as f:
    arrays = f.to_arrays()
    
    # Awkward arrays handle variable-length data naturally
    n_charged = ak.num(arrays['PHCHRG']['charge'])
    print(f"Events with >10 charged tracks: {ak.sum(n_charged > 10)}")
```

### Random Access and Slicing

```python
with jazelle.open('data.jazelle') as f:
    # Access specific event
    event = f[42]
    
    # Slice notation
    first_100 = f.read_parallel(start=0, count=100)
    
    # Last 50 events
    last_50 = f.read_parallel(start=len(f)-50, count=50)
```

## 🖥️ Command-Line Interface

### Inspect File Metadata

```bash
# Show file summary with first 10 events
jazelle inspect data.jazelle

# Show first 20 events
jazelle inspect data.jazelle --lines 20

# Show last 5 events
jazelle inspect data.jazelle --lines 5 --tail

# Show specific bank counts in table
jazelle inspect data.jazelle --banks PHPSUM PHCHRG PHKLUS

# Use wildcard patterns for banks
jazelle inspect data.jazelle --banks "PH*"
```

### Read and Display Events

```bash
# Read first event (index 0)
jazelle read data.jazelle

# Read specific event by index
jazelle read data.jazelle --index 42

# Read last event (negative indexing)
jazelle read data.jazelle --index -1

# Show specific bank families
jazelle read data.jazelle --index 0 --banks PHPSUM PHCHRG

# Use wildcard patterns
jazelle read data.jazelle --index 0 --banks "PH*"

# Limit output lines
jazelle read data.jazelle --index 0 --limit 100 --banks "*"

# Customize display
jazelle read data.jazelle --index 0  --banks "*" --display-options "max_rows=20,float_precision=2"
```

### Convert Formats

```bash
# Convert to HDF5 (format inferred from extension)
jazelle convert -i data.jazelle -o output.h5 --threads 8

# Convert to Parquet
jazelle convert -i data.jazelle -o output.parquet --threads 8

# Convert to JSON
jazelle convert -i data.jazelle -o output.json

# Convert specific event range
jazelle convert -i data.jazelle -o output.h5 --start 100 --count 1000

# Optimize batch size for memory usage
jazelle convert -i data.jazelle -o output.h5 --batch-size 2000 --threads 16

# Convert all events (default)
jazelle convert -i data.jazelle -o output.parquet --threads 8
```

## 📊 Performance Benchmarks

Benchmarks performed on a 16-core workstation with a 9,994-event Jazelle file (~130 MB).

### Batch Size Impact (8 threads)

| Batch Size | Throughput (kHz) | Speedup vs. Single |
|------------|------------------|-------------------|
| 1          | 0.49            | 1.0×              |
| 100        | 31.26           | 63.3×             |
| 500        | 48.06           | 97.4×             |
| **1000**   | **49.13**       | **99.6×**         |
| 5000       | 40.02           | 81.1×             |
| 9994 (all) | 52.59           | 106.6×            |

**Optimal batch size: 1000 events** provides the best balance of throughput and memory usage.

### Threading Scalability (batch size 1000)

| Threads | Throughput (kHz) | Speedup vs. Single Thread |
|---------|------------------|---------------------------|
| 1       | 26.92            | 1.0×                      |
| 2       | 42.14            | 1.57×                     |
| 4       | 57.76            | 2.15×                     |
| **8**   | **67.43**        | **2.50×**                 |
| 16      | 45.79            | 1.70×                     |
| 32+     | ~40-45           | ~1.5-1.7×                 |

**Optimal thread count: 8 threads** on this system. Performance degrades beyond this due to overhead and resource contention.

### Format Conversion Performance

| Format | Throughput (kHz) | Mean Time (s) | Use Case |
|--------|------------------|---------------|----------|
| **Iteration (Batched)** | **74.98** | **0.133** | Internal processing |
| Iteration (Sequential) | 63.75 | 0.157 | Simple loops |
| Awkward Array | 38.84 | 0.257 | Jagged data analysis |
| To Dict (NumPy) | 29.29 | 0.341 | Array-based analysis |
| Feather | 29.57 | 0.338 | Fast binary I/O |
| **Parquet** | **10.02** | **0.998** | Columnar analytics |
| HDF5 | 3.65 | 2.736 | Hierarchical data |
| JSON | 0.28 | 35.24 | Human-readable export |

**Key Insight**: Parquet provides the best balance of performance and analytics capability, while JSON should only be used for small datasets or human inspection.

## 🏗️ Architecture

### Design Philosophy

Jazelle_reader follows a **layered architecture** that separates concerns and maximizes performance:

```
┌─────────────────────────────────────────────────────────────┐
│                      Python User API                        │
│  (JazelleFile, events, families, display, streaming)        │
├─────────────────────────────────────────────────────────────┤
│                      Cython Bindings                        │
│  (Type conversion, memory management, GIL handling)         │
├─────────────────────────────────────────────────────────────┤
│                  C++20 Core Engine                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  JazelleFile: File management & parallel reading    │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  JazelleEvent: Event container & bank access        │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  Family<T>: Type-safe bank management               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  Banks: IEVENTH, MCPART, PHCHRG, ...               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  JazelleStream: Binary I/O & VAX conversion         │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  Threading: Lock-free queues, thread pools          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **C++ Core Engine**
- **JazelleFile**: Main interface for file operations, handles multi-threading
- **JazelleEvent**: Event container with lazy bank loading
- **JazelleStream**: Low-level binary I/O with VAX floating-point support
- **Family<T>**: Template-based bank manager for type safety
- **Bank Classes**: Concrete implementations (IEVENTH, MCPART, PHCHRG, etc.)

#### 2. **Python Bindings**
- **Cython wrapper**: Zero-copy data transfer where possible
- **Memory management**: Automatic cleanup with context managers
- **NumPy integration**: Direct buffer access for efficiency
- **Display system**: Rich HTML and ASCII rendering

#### 3. **Format Streamers**
- **Modular design**: Each format has dedicated read/write streamer
- **Interoperability**: Seamless conversion between formats
- **Optimization**: Format-specific optimizations (e.g., columnar for Parquet)

### Data Flow

```
Jazelle Binary File
       │
       ├─→ [C++ JazelleStream] ─→ Binary parsing
       │                           VAX float conversion
       │
       ├─→ [C++ JazelleEvent] ─→ Event construction
       │                          Bank instantiation
       │
       ├─→ [Cython Bindings] ─→ Python object wrapping
       │                         Memory views
       │
       └─→ [Python API] ─→ NumPy arrays
                          Awkward arrays
                          Pandas DataFrames
                          HDF5/Parquet/Feather
```

## 📚 Data Structure

### Jazelle Format Overview

Jazelle files store event data in a binary format with logical and physical records:

```
┌─────────────────────────────────────────────────────────┐
│                    Jazelle File                         │
├─────────────────────────────────────────────────────────┤
│  Physical Record 1: [Header] [Logical Records...]       │
│  Physical Record 2: [Header] [Logical Records...]       │
│  ...                                                     │
│  Physical Record N: [Header] [Logical Records...]       │
└─────────────────────────────────────────────────────────┘
```

Each **Event** contains multiple **Banks** organized by **Family**:

```
Event
├── IEVENTH (Event Header)
│   ├── run number
│   ├── event number
│   ├── timestamp
│   └── ...
│
├── PHPSUM (Physics Summary)
│   ├── ncharged (# charged tracks)
│   ├── nneutral (# neutral particles)
│   ├── thrust (event shape)
│   └── ...
│
├── MCPART (Monte Carlo Particles - may be empty)
│   ├── Particle 1: {ptype, e, p[3], ...}
│   ├── Particle 2: {ptype, e, p[3], ...}
│   └── ...
│
├── PHCHRG (Charged Tracks)
│   ├── Track 1: {charge, nhit, p[3], ...}
│   ├── Track 2: {charge, nhit, p[3], ...}
│   └── ...
│
├── PHKLUS (Calorimeter Clusters)
│   ├── Cluster 1: {energy, position[3], ...}
│   └── ...
│
└── ... (other detector banks)
```

### Available Bank Families

| Family | Description | Key Fields |
|--------|-------------|------------|
| **IEVENTH** | Event header | run, event, evttime |
| **PHPSUM** | Particle summary | ncharged, nneutral, thrust, evis |
| **MCHEAD** | Monte Carlo header | weight, sqrts, ... |
| **MCPART** | MC particles | ptype, e, p[3], origin[3] |
| **PHCHRG** | Charged tracks | charge, nhit, hlxpar[6], dhlxpar[15] |
| **PHKLUS** | Calorimeter clusters | eraw, cth, elayer[8] |
| **PHWIC** | Warm iron calorimeter | nhit, t1, t2, t3 |
| **PHKTRK** | NOT USED | -- |
| **PHCRID** | Cherenkov ring imaging | norm, rc, geom |
| **PHKELID** | Calorimeter/Electron ID | prob, phi, theta |

### Rich Display System

```python
# Display in Jupyter notebooks
with JazelleFile('data.jazelle') as f:
    # Show file summary (HTML in Jupyter)
    f.info()
    
    # Show first 5 events in table
    f.display(start=0, count=5)
    
    # Show last 3 events
    total = len(f)
    f.display(start=total-3, count=3)
    
    # Display single event (HTML rendering)
    event = f[0]
    display(event)  # Rich HTML table (Inside Jupyter notebook)
    
    # Display specific family
    display(event.phchrg)
    
    # ASCII display for terminals
    print(event)
    
    # Customize display options globally
    import jazelle
    jazelle.set_display_options(max_rows=20, float_precision=3)
```

## 🧪 Testing

The package includes comprehensive test suites:

### C++ Tests
- Binary I/O operations
- VAX floating-point conversion
- Event parsing
- Bank instantiation

### Python Tests
- Basic file reading
- Parallel processing
- Format conversion (HDF5, Parquet, Feather, JSON)
- Display system
- CLI commands

Run tests with:

```bash
# Python tests
pytest tests/python/test_core.py
pytest tests/python/test_streamers.py
pytest tests/python/test_display.py
pytest tests/python/test_cli.py

# C++ tests (if built from source)
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
ctest --output-on-failure
```

## 📖 Documentation

- **Tutorial Notebook**: `examples/T01_quickstart_jazelle.ipynb` - Comprehensive beginner's guide
- **API Reference**: Full API documentation available in docstrings
- **Benchmark Scripts**: `benchmarks/` - Performance testing code
- **Example Scripts**: `examples/` - Usage examples

## 🤝 Contributing

Contributions are welcome! This project is part of a larger effort to preserve and modernize legacy experimental physics data.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- C++: Follow modern C++20 practices, use clang-format
- Python: Follow PEP 8
- Documentation: Add docstrings for all public APIs

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tony Johnson**: Original Java implementation of the Jazelle reader
- **SLD Collaboration**: For the Jazelle format specification and experimental data
- **SLAC National Accelerator Laboratory**: For the SLD experiment and continued support of data preservation efforts

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AlkaidCheng/jazelle_reader/issues)

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{jazelle_reader,
  author = {Cheng, Alkaid},
  title = {jazelle\_reader: A Modern Python Reader for SLD Jazelle Files},
  year = {2025},
  url = {https://github.com/AlkaidCheng/jazelle_reader},
  note = {High-performance C++20 implementation with Python bindings for
          legacy particle physics data analysis}
}
```

---

**Built with ❤️ for the particle physics community**

*Preserving the past, empowering the future*