# Prime Basis Research

Express every prime as **previous_prime + sum of distinct smaller primes**, then explore the resulting structure through 14 interactive visualizations.

```
2  = 1 + 1          11 = 7 + 3 + 1       29 = 23 + 5 + 1
3  = 2 + 1          13 = 11 + 2           31 = 29 + 2
5  = 3 + 2          17 = 13 + 3 + 1       37 = 31 + 5 + 1
7  = 5 + 2          19 = 17 + 2           41 = 37 + 3 + 1
```

## Quick Start

```bash
# Install primesieve (required)
sudo pacman -S primesieve        # Arch
sudo apt install libprimesieve-dev  # Debian/Ubuntu
brew install primesieve          # macOS

# Generate 1M decompositions (~2.5 min, cached to prime_basis.bin)
cargo run --release -- generate 1000000

# Launch the dashboard
cargo run --release --bin viz_launcher
```

The dashboard lets you launch any visualization with one click. Or run them individually:

```bash
cargo run --release --bin viz_phase_space
```

## Researcher's Guide to the Visualizations

Every visualization has a built-in **❓ Help** button explaining what to look for. The tools are organized in a progression from basic structure to deep analysis.

### Is there structure? (Start here)

| Tool | Command | What it reveals |
|------|---------|-----------------|
| Phase Space | `viz_phase_space` | Scatter of gap × component count × next gap. Look for clusters, stratification, or strange attractors. Toggle 3D mode to see delay embedding. |
| Compression Signature | `viz_compression` | Bit cost of decompositions vs log₂(p). If the ratio stays below 1.0, the basis method genuinely compresses primes. Watch the trend line. |

### Modular structure

| Tool | Command | What it reveals |
|------|---------|-----------------|
| Modular Starfield | `viz_starfield` | Polar plot with angle = p mod M. Try M=6, 30, 210 (primorials) to see symmetry snap into place. Empty sectors = factors of M. |
| Resonance Cylinder | `viz_resonance` | Animated sweep of M from 2→300. Watch for the moment chaos locks into geometric order — those are the structurally significant moduli. |

### Temporal patterns

| Tool | Command | What it reveals |
|------|---------|-----------------|
| Comb Spectrogram | `viz_spectrogram` | Heatmap of which base primes are active over time. Vertical bands = workhorse primes. Horizontal gaps = simple regions. |
| Spectral Barcode | `viz_barcode` | Same data as spectrogram, inverted (dark=used). Look for repeating block patterns and periodic activation. |

### Geometric structure

| Tool | Command | What it reveals |
|------|---------|-----------------|
| Successive Vector Distance | `viz_vector_distance` | Euclidean distance between consecutive basis vectors. Smooth trend = continuous flow. Spikes = phase transitions. Adjustable smoothing window. |
| PCA Embedding | `viz_pca` | Projects high-dimensional basis vectors to 2D/3D. Clusters = structurally similar primes. Curves = low-dimensional manifold. Toggle between PC pairs. |

### Hidden periodicity

| Tool | Command | What it reveals |
|------|---------|-----------------|
| Hyper-Crystal Diffraction | `viz_diffraction` | FFT of a composite signal built from decomposition frequencies. Sharp peaks = real periodic structure. Red markers highlight peaks >2σ above mean. |

### Relational structure

| Tool | Command | What it reveals |
|------|---------|-----------------|
| 3D Vector Walk | `viz_vector_walk` | Cumulative trajectory through basis-space. Drag to rotate, scroll to zoom. Remap axes to different base primes via dropdowns. |
| Dependency Network | `viz_network` | Force-directed graph of base prime co-occurrence. Hub nodes = backbone primes. Thick edges = frequently co-occurring pairs. Drag nodes, pause simulation. |

### Phase 1 tools (distributions)

| Tool | Command | What it reveals |
|------|---------|-----------------|
| Gap Waveform | `viz_gap_waveform` | Gap sizes colored by complexity. Blue=1 component, green=2, red=3+. |
| Support Scores | `viz_support_scores` | Base prime usage frequency + power-law test (log-log regression). |
| Distributions | `viz_distributions` | Histograms of component counts and gap sizes + running averages. |

## Decomposition Algorithm

Given primes `1, 2, 3, 5, 7, 11, 13, ...` (1 included as base element):

1. Compute `gap = p − previous_prime`
2. Greedy decomposition: take largest available prime ≤ remainder, repeat
3. If greedy fails, backtracking search guarantees a solution
4. Result: `p = previous_prime + c₁ + c₂ + ...` where all cᵢ are distinct

Greedy handles >99% of gaps. The full dataset of 1M primes has max 3 components per decomposition.

## CLI Reference

```bash
cargo run --release -- generate 1000000   # Generate decompositions (incremental)
cargo run --release -- show --last 50     # View recent decompositions
cargo run --release -- status             # Check database status
cargo run --release -- stats              # Full statistical analysis
cargo run --release -- entropy            # Sliding-window Shannon entropy
cargo run --release -- autocorrelation    # Gap autocorrelation at various lags
cargo run --release -- export --start 0 --count 1000 --output data.json  # Export to JSON
```

All commands accept `--cache path.bin` to use a custom data file.

## Project Structure

```
src/
├── lib.rs              # Core: FFI bindings, PrimeDatabase, decomposition
├── analysis.rs         # Statistics, PCA, co-occurrence, compression bits, basis vectors
├── spectral.rs         # FFT: composite signal, magnitude spectrum, peak detection
├── viz_common.rs       # Shared viz utilities: load_data, color, projection, help panels
├── main.rs             # CLI entry point
└── bin/
    ├── viz_launcher.rs         # 🚀 Dashboard (launches all tools)
    ├── viz_phase_space.rs      # 🌀 Phase Space Plot
    ├── viz_starfield.rs        # ⭐ Modular Starfield
    ├── viz_resonance.rs        # 🎯 Resonance Cylinder
    ├── viz_compression.rs      # 📐 Compression Signature
    ├── viz_spectrogram.rs      # 🎹 Comb Spectrogram
    ├── viz_barcode.rs          # 🔬 Spectral Barcode
    ├── viz_vector_distance.rs  # 📏 Successive Vector Distance
    ├── viz_pca.rs              # 🧬 PCA Embedding
    ├── viz_diffraction.rs      # 💎 Hyper-Crystal Diffraction
    ├── viz_vector_walk.rs      # 🚶 3D Vector Walk
    ├── viz_network.rs          # 🕸 Dependency Network
    ├── viz_gap_waveform.rs     # 🌊 Gap Waveform
    ├── viz_support_scores.rs   # 📈 Support Scores
    └── viz_distributions.rs    # 📊 Distributions
```

## Tests

```bash
cargo test --lib   # 39 tests: unit + 13 property-based tests (proptest)
```

Property-based tests validate correctness properties including:
- Compression bits formula matches specification
- Basis vector construction is correct
- PCA components are orthonormal with bounded reconstruction error
- Co-occurrence matrix is symmetric with consistent diagonal
- FFT detects known frequencies
- Force simulation dissipates energy
- Trajectory cumulative sum is invariant

## Performance

| Primes | Generation | Largest prime | Cache size |
|--------|-----------|---------------|------------|
| 1,000 | 7 ms | 8,837 | ~1 KB |
| 100,000 | 1.5 s | 1,315,309 | ~5 MB |
| 1,000,000 | 2.5 min | 15,503,737 | 52.8 MB |

Stats computation over 1M primes: ~100ms. All visualizations load in <2s.

## Dependencies

- [primesieve](https://github.com/kimwalisch/primesieve) — prime generation via C FFI
- [egui](https://github.com/emilk/egui) / eframe — immediate-mode GUI
- [egui_plot](https://github.com/emilk/egui_plot) — interactive plots
- [rustfft](https://github.com/ejmahler/RustFFT) — FFT computation
- [nalgebra](https://nalgebra.org/) — PCA eigendecomposition
- [proptest](https://github.com/proptest-rs/proptest) — property-based testing

## License

MIT
