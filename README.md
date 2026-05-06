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

## 📖 Tutorials

We provide comprehensive Jupyter notebooks in the `examples/` directory to help you get started:

- T01_Quickstart.ipynb: The basics of opening files, iterating events, inspecting headers, and exporting data.

- T02_Z_Boson_Reconstruction.ipynb: A simple physics analysis demo. Learn how to reconstruct the Z boson resonance peak from lepton pairs using `awkward` arrays with `vector` and visualize it with `quickstats`.

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
    first_100 = f.read_batch(start=0, count=100)
    
    # Last 50 events
    last_50 = f.read_batch(start=len(f)-50, count=50)
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
| 1          | 0.49             | 1.0×              |
| 100        | 31.26            | 63.3×             |
| 500        | 48.06            | 97.4×             |
| **1000**   | **49.13**        | **99.6×**         |
| 5000       | 40.02            | 81.1×             |
| 9994 (all) | 52.59            | 106.6×            |

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
│  ┌─────────────────────────────────────────────────────┐    │
│  │  JazelleFile: File management & parallel reading    │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  JazelleEvent: Event container & bank access        │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Family<T>: Type-safe bank management               │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Banks: IEVENTH, MCPART, PHCHRG, ...                │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  JazelleStream: Binary I/O & VAX conversion         │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  Threading: Lock-free queues, thread pools          │    │
│  └─────────────────────────────────────────────────────┘    │
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
│  ...                                                    │
│  Physical Record N: [Header] [Logical Records...]       │
└─────────────────────────────────────────────────────────┘
```

Each **Event** contains multiple **Banks** organized by **Family**:

```
Event
├── IEVENTH (Event Header)
│   ├── run, event, evttime, trigger, ...
│
├── PHBM (Beam Information)
│   ├── ecm (Center of mass energy)
│   ├── pol (Beam polarization)
│   └── ...
│
├── MCHEAD & MCPART (Monte Carlo Truth)
│   ├── MCHEAD: origin, ipx, ipy, ipz
│   └── MCPART: p[3], e, ptype, charge, parent_id, ...
│
├── PHCHRG (Charged Tracks)
│   ├── hlxpar[6] (Helix parameters)
│   ├── dhlxpar[15] (Error matrix)
│   ├── nhit, chi2, dedx, ...
│
├── PHKLUS (Calorimeter Clusters)
│   ├── eraw (Raw energy)
│   ├── elayer[8] (Energy per layer)
│   └── ...
│
├── PHCRID, PHWIC, PHKELID (Particle ID Subsystems)
│   ├── PHCRID: Cherenkov ring likelihoods
│   ├── PHWIC: Muon iron tracking
│   └── PHKELID: Electron calorimeter matching
│
└── Relational Tables
    ├── PHPOINT: Master pointers between sub-systems
    └── PHKCHRG: Track-to-Cluster matching kinematics
```

### Available Bank Families and Field Descriptions

### Comprehensive Bank Dictionary

Below is the definitive reference for all physics banks successfully extracted and exposed to the Python/NumPy API via the Cython bindings. All multi-dimensional arrays (like `hlxpar[6]`) are exposed as 2D NumPy arrays (`N x D`) in the batched outputs. Every bank automatically includes a unique `id` field (`int32`).

| Bank Family | Field Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **IEVENTH** | `run` / `event` | `int32` | SLD Run number and Event number. |
| *(Event Header)* | `evttime` | `int64` | UTC Timestamp of the event creation. |
| | `evttype` | `int32` | Event generation type (0=PHYSICS, 1=TRUTH, 2=FASTMC, etc.). |
| | `trigger` | `int32` | Hardware trigger mask for the readout. |
| | `weight` | `float32` | Event weight (1.0 for real data). |
| | `header` | `int32` | Internal Jazelle pointer to the header bank. |
| **PHBM** | `ecm` / `decm` | `float32` | Center of mass energy (GeV) and its uncertainty. |
| *(Beam Info)* | `pol` / `dpol` | `float32` | Average Compton beam polarization magnitude and error. |
| | `pos` / `dpos` | `float32[3,6]` | Interaction point (X, Y, Z) and symmetric error matrix. |
| **PHPSUM** | `px`, `py`, `pz` | `float32` | Total momentum vector summary. |
| *(Physics Sum)* | `x`, `y`, `z` | `float32` | Geometric event vertex summary. |
| | `charge` | `float32` | Total measured charge. |
| | `status` | `int32` | Reconstruction status flag. |
| | `ptot` | `float64` | Total scalar momentum magnitude. |
| **MCHEAD** | `ntot` | `int32` | Total number of final state particles generated. |
| *(MC Header)* | `origin` | `int32` | Origin process bitmask (e.g., Z -> uubar, Z -> mumu). |
| | `ipx`, `ipy`, `ipz` | `float32` | Simulated primary vertex momentum. |
| **MCPART** | `ptype` | `int32` | LUND/PDG Particle Identification Code. |
| *(MC Truth)* | `p` / `ptot` / `e` | `float32[3,1,1]`| True (X,Y,Z) momentum, scalar momentum, and total energy. |
| | `charge` | `float32` | True electric charge. |
| | `origin` | `int32` | Simulation history bitmask (decayed, interacted, stopped). |
| | `parent_id` | `int32` | Relational ID pointing to the parent `MCPART` bank. |
| | `xt` | `float32[3]` | True (X,Y,Z) spatial termination or decay coordinate. |
| **PHCHRG** | `hlxpar` | `float32[6]` | Helix parameters: `phi`, `1/pt`, `tan(lambda)`, `x`, `y`, `z`. |
| *(Charged Track)*| `dhlxpar` | `float32[15]` | 5x5 symmetric error matrix for the helix fit. |
| | `bnorm` / `b3norm`| `float32` | 2D and 3D Impact parameters (Distance of closest approach). |
| | `impact` / `impact3`| `float32` | Significance/error of the 2D and 3D impact parameters. |
| | `charge` | `int16` | Reconstructed charge (+1 or -1). |
| | `smwstat` / `status`| `int16`/`int32`| Track swimming status and general reconstruction status. |
| | `tkpar0` | `float32` | Reference point parameter for the track fit. |
| | `tkpar` / `dtkpar` | `float32[5,15]`| Alternative track fit parameters and 5x5 error matrix. |
| | `length` | `float32` | Total reconstructed arc length of the track. |
| | `chi2dt` / `ndfdt` | `float32`/`int16`| Chi-squared and Degrees of Freedom for the tracking fit. |
| | `imc` | `int16` | Pointer/Match to corresponding `MCPART` (for MC). |
| | `nhit` / `nhite` | `int16` | Total tracker hits and Number of expected hits. |
| | `nhitp` / `nmisht` | `int16` | Hits used in the final fit and Number of missing hits. |
| | `nwrght` / `nhitv` | `int16` | Wrong/noise hits and Vertex Detector (VXD) specific hits. |
| | `chi2` / `chi2v` | `float32` | Overall track chi-squared and VXD-specific chi-squared. |
| | `vxdhit` | `int32` | Bitmask indicating which VXD layers were hit. |
| | `mustat` / `estat` | `int16` | Muon (WIC) and Electron (Calorimeter) matching status. |
| | `dedx` | `int32` | Encoded dE/dx (ionization energy loss) for PID. |
| **PHKLUS** | `status` | `int32` | Cluster quality and region status. |
| *(Calo Cluster)* | `eraw` | `float32` | Raw, uncalibrated energy sum of the calorimeter cluster. |
| | `cth` / `wcth` | `float32` | Geometric vs. Energy-weighted cosine(theta) of centroid. |
| | `phi` / `wphi` | `float32` | Geometric vs. Energy-weighted azimuthal angle. |
| | `elayer` | `float32[8]` | Energy deposited at specific depths (EM1, EM2, HAD1, etc.). |
| | `nhit2` / `nhit3` | `int32` | Hit counts isolated to EM and Hadronic sub-clusters. |
| | `cth2` / `wcth2` | `float32` | Geometric vs. Energy-weighted cosine(theta) for EM. |
| | `phi2` / `whphi2` | `float32` | Geometric vs. Energy-weighted phi for EM. |
| | `cth3` / `wcth3` | `float32` | Geometric vs. Energy-weighted cosine(theta) for Hadronic. |
| | `phi3` / `wphi3` | `float32` | Geometric vs. Energy-weighted phi for Hadronic. |
| **PHWIC** | `idstat` | `int16` | Muon identification status and quality flag. |
| *(Muon Iron)* | `nhit` / `nhit45` | `int16` | Total WIC layers hit and specific 45-degree stereo hits. |
| | `npat` / `nhitpat` | `int16` | Distinct patterns found and hits in primary pattern. |
| | `syshit` | `int16` | System bitmask (barrel, endcap, octants). |
| | `qpinit` | `float32` | Initial momentum (q/p) expected as it enters the WIC. |
| | `t1`, `t2`, `t3` | `float32` | Trajectory parameters. |
| | `hitmiss` | `int32` | Bitmask of expected vs. actual layer hits. |
| | `itrlen` | `float32` | Total interaction length (iron amount) penetrated. |
| | `nlayexp` / `nlaybey`| `int16` | Expected layers hit vs. punch-through layers hit. |
| | `missprob` | `float32` | Probability of hadron punch-through misidentified as muon. |
| | `phwicid` | `int32` | WIC internal cluster ID. |
| | `nhitshar` / `nother`| `int16` | WIC hits shared with another track / hits not assigned. |
| | `hitsused` | `int32` | Mask of hits used in the internal WIC track fit. |
| | `pref1` | `float32[3]` | Reference point (X, Y, Z) for the internal WIC track. |
| | `pfit` / `dpfit` | `float32[4,10]`| WIC-only track fit parameters and error matrix. |
| | `chi2` / `ndf` | `float32`/`int16`| Chi-squared and Degrees of Freedom for WIC internal fit. |
| | `punfit` | `int16` | Points deemed unusable and discarded. |
| | `matchChi2`/`matchNdf`| `float32`/`int16`| Chi-squared and NDF of geometric match to CDC track. |
| **PHCRID** | `ctlword` | `int32` | Control word defining successfully read sub-components. |
| *(Cherenkov PID)*| `norm` | `float32` | Normalization factor for the likelihood calculations. |
| | `rc` / `geom` | `int16` | Global reconstruction return code and geometry region. |
| | `trkp` | `int16` | Track momentum bin/flag used during ring resolution. |
| | `nhits` | `int16` | Total Cherenkov photons (Liquid + Gas) for the track. |
| | `liq_*` fields | `int16`/`int32`| `rc`, `nhits`, `besthyp`, `nhexp`, `nhfnd`, `nhbkg`, `mskphot` (Liquid). |
| | `gas_*` fields | `int16`/`int32`| `rc`, `nhits`, `besthyp`, `nhexp`, `nhfnd`, `nhbkg`, `mskphot` (Gas). |
| | `llik_e,mu,pi,k,p` | `float32` | Final combined Log-Likelihoods for PID mass hypotheses. |
| **PHKELID** | `phchrg_id` | `int32` | Relational pointer to the parent `PHCHRG` track. |
| *(Electron PID)* | `idstat` / `prob` | `int16` | Electron ID status and computed probability. |
| | `phi` / `theta` | `float32` | Azimuthal and Polar angles at the calorimeter face. |
| | `qp` | `float32` | Track momentum (q/p). |
| | `dphi`, `dtheta`, `dqp`| `float32` | Error/Spread in angular match and momentum. |
| | `tphi` / `ttheta` | `float32` | Angle of the associated calorimeter cluster centroid. |
| | `isolat` | `float32` | Isolation metric (energy surrounding electron candidate). |
| | `em1` / `em12` | `float32` | Energy in the first EM layer / first two EM layers. |
| | `dem12` | `float32` | Uncertainty on the EM1+2 energy. |
| | `had1` | `float32` | Energy in the first hadronic layer (electron veto). |
| | `emphi` / `emtheta`| `float32` | Width of the EM shower in phi and theta. |
| | `phiwid` / `thewid`| `float32` | Overall transverse width in phi and theta. |
| | `em1x1` | `float32` | Energy in the central 1x1 calorimeter tower. |
| | `em2x2a`, `em2x2b` | `float32` | Energy in the 2x2 tower blocks (configs A & B). |
| | `em3x3a`, `em3x3b` | `float32` | Energy in the 3x3 tower blocks (core vs. extended). |
| **PHKCHRG** | `phchrg_id` | `int32` | Key 1: ID of the linked `PHCHRG` track. |
| *(Cluster Match)*| `phklus_id` | `int32` | Key 2: ID of the linked `PHKLUS` cluster. |
| | `match_distance`| `float32` | Overall spatial matching probability/distance. |
| | `delta_phi` | `float32` | Azimuthal angular residual between track and cluster. |
| | `delta_theta` | `float32` | Polar angular residual between track and cluster. |
| **PHPOINT** | `phpsum_id` | `int32` | Master pointer resolving the associated `PHPSUM` bank. |
| *(Master Keys)* | `phchrg_id` | `int32` | Master pointer resolving the associated `PHCHRG` bank. |
| | `phklus_id` | `int32` | Master pointer resolving the associated `PHKLUS` bank. |
| | `phkelid_id` | `int32` | Master pointer resolving the associated `PHKELID` bank. |
| | `phwic_id` | `int32` | Master pointer resolving the associated `PHWIC` bank. |
| | `phcrid_id` | `int32` | Master pointer resolving the associated `PHCRID` bank. |

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

- **Tutorial Notebook**: Comprehensive beginner's guide
- **API Reference**: Full API documentation available in docstrings
- **Benchmark Scripts**: `benchmarks/` - Performance testing code

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
  author = {Cheng, Chi Lung},
  title = {jazelle\_reader: A Modern Python Reader for SLD Jazelle Files},
  year = {2025},
  url = {https://github.com/HEP-KE/jazelle_reader},
  note = {High-performance C++20 implementation with Python bindings for
          legacy particle physics data analysis}
}
```

---

**Built with ❤️ for the particle physics community**

*Preserving the past, empowering the future*