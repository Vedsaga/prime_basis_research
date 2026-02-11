# Prime Basis Research

Express every prime as **previous_prime + sum of distinct smaller primes**.

```
2  = 1 + 1
3  = 2 + 1
5  = 3 + 2
7  = 5 + 2
11 = 7 + 3 + 1
13 = 11 + 2
17 = 13 + 3 + 1
19 = 17 + 2
23 = 19 + 3 + 1
29 = 23 + 5 + 1
31 = 29 + 2
37 = 31 + 5 + 1
41 = 37 + 3 + 1
43 = 41 + 2
```

## Decomposition Algorithm

Given the sequence of primes `1, 2, 3, 5, 7, 11, 13, ...` (1 is included as a base element):

```
For each new prime p:
  1.  gap = p − previous_prime
  2.  Decompose gap using known primes (greedy, largest-first):
        remainder = gap
        components = []
        for each known_prime from LARGEST to SMALLEST:
            if known_prime ≤ remainder:
                components.append(known_prime)
                remainder -= known_prime
            if remainder == 0: done ✓
  3.  If greedy fails (remainder ≠ 0), use backtracking search
  4.  Result: p = previous_prime + component₁ + component₂ + ...
```

### Worked example: p = 29

| Step | Action | Remainder |
|------|--------|-----------|
| Start | gap = 29 − 23 = **6** | 6 |
| 1 | Largest prime ≤ 6 → **5** | 1 |
| 2 | Largest prime ≤ 1 → **1** | 0 ✓ |

Result: `29 = 23 + 5 + 1`

## Features

- Incremental generation — generate N more primes at any time; previously computed decompositions are never recomputed
- Persistent binary cache — results stored in compact bincode format (~53 MB for 1M primes)
- primesieve-powered — prime generation via [kimwalisch/primesieve](https://github.com/kimwalisch/primesieve) C library
- Greedy + backtracking — greedy decomposition handles >99% of gaps; backtracking guarantees correctness
- Full statistical analysis — precomputed stats cached to a sidecar JSON file for instant access
- Interactive visualizations — separate egui-based GUI binaries for each visualization type
- Streaming computation — statistics computed over the full dataset; rendering uses adaptive LOD

## Prerequisites

Install the [primesieve](https://github.com/kimwalisch/primesieve) C library:

```bash
# Arch Linux / EndeavourOS
sudo pacman -S primesieve

# Debian / Ubuntu
sudo apt install primesieve libprimesieve-dev

# macOS
brew install primesieve

# Fedora
sudo dnf install primesieve
```

## Build

```bash
# Build everything (CLI + all visualizations)
cargo build --release
```

## CLI Usage

### Generate primes

```bash
# Generate the first 100 decompositions
cargo run --release -- generate 100

# Generate 1000 more (incremental — loads from cache)
cargo run --release -- generate 1000

# Custom cache file path
cargo run --release -- --cache my_data.bin generate 500
```

### View decompositions

```bash
# Show last 20 (default)
cargo run --release -- show

# Show last 50
cargo run --release -- show --last 50

# Show all
cargo run --release -- show --all
```

### Check status

```bash
cargo run --release -- status
```

### Full analysis statistics

Computes and caches comprehensive statistics over the entire dataset:
support scores, gap/component distributions, decomposition uniqueness, first appearances, and more.

```bash
# Compute stats (cached to prime_basis.stats.json)
cargo run --release -- stats

# Force recompute (ignores cache)
cargo run --release -- stats --recompute
```

Example output:
```
╔══════════════════════════════════════════════════╗
║         Prime Basis — Full Analysis Stats        ║
╠══════════════════════════════════════════════════╣
║  Decompositions:                     1,001,101  ║
║  Total primes:                       1,001,102  ║
║  Largest prime:                     15,503,737  ║
╠══════════════════════════════════════════════════╣
║  GAP STATISTICS                                  ║
║  Min gap:                                    1  ║
║  Max gap:                                  154  ║
║  Mean gap:                               15.49  ║
║  Distinct gaps:                             78  ║
╠══════════════════════════════════════════════════╣
║  COMPONENT STATISTICS                            ║
║  Min components:                             1  ║
║  Max components:                             3  ║
║  Mean components:                         1.91  ║
╠══════════════════════════════════════════════════╣
║  BASIS USAGE                                     ║
║  Unique bases:                              37  ║
║  Largest base:                             151  ║
╚══════════════════════════════════════════════════╝

  Component count distribution:
    1 components:   86,130 (  8.6%) ████
    2 components:  914,963 ( 91.4%) █████████████████████████████████████████████
    3 components:        8 (  0.0%)

  Top 20 base primes by support score:
     1.     1     673,383   67.3%  ████████████████████
     2.     3     283,743   28.3%  ████████
     3.     5     190,258   19.0%  █████
     ...
```

### Sliding-window entropy

Computes Shannon entropy of the component distribution in a sliding window,
revealing whether the decomposition process is stationary, drifting, or oscillating.

```bash
# Default window=1000
cargo run --release -- entropy

# Custom window size
cargo run --release -- entropy --window 5000

# Show every Nth value
cargo run --release -- entropy --window 1000 --step 10
```

### Gap autocorrelation

Tests whether gap sizes are correlated at various lags — reveals hidden periodicity.

```bash
# Default max_lag=100
cargo run --release -- autocorrelation

# Custom max lag
cargo run --release -- autocorrelation --max-lag 500
```

### Export data to JSON

Export a window of decompositions as JSON for use in external tools (Python, R, etc.).

```bash
# Export first 1000 decompositions
cargo run --release -- export --start 0 --count 1000 --output data.json

# Export a specific range
cargo run --release -- export --start 500000 --count 5000 --output mid_range.json
```

Output format:
```json
[
  {
    "index": 0,
    "prime": 2,
    "prev_prime": 1,
    "gap": 1,
    "components": [1],
    "num_components": 1
  },
  ...
]
```


## Interactive Visualizations

Each visualization is a separate binary — they are independent, so modifying or adding
one never breaks the others. All visualizations support zoom (scroll), pan (drag), and
interactive legends.

### Gap Waveform

Scatter plot of prime gap sizes over the sequence, colored by decomposition complexity.

```bash
cargo run --release --bin viz_gap_waveform
```

- Blue dots = 1 component (twin primes, gap=2)
- Green dots = 2 components (91.4% of all decompositions)
- Red dots = 3+ components (extremely rare)
- Zoomed out: shows bucketed max/mean lines + running average
- Zoomed in: shows individual colored points

Research questions answered:
- Do large gaps correlate with more components?
- Does the system self-stabilize after chaotic bursts?
- Are there "heart rate variability" patterns?

### Support Scores

Bar chart and log-log plot of base prime usage frequency.

```bash
cargo run --release --bin viz_support_scores
```

- Top panel: usage count per base prime (X = prime value)
- Bottom panel: ln(prime) vs ln(count) with linear regression fit
- Power-law slope shown in header — a straight line on log-log indicates scale-free structure

Research questions answered:
- Does support score follow a power law?
- Which primes are the structural "backbone"?
- Is there a sharp cutoff or gradual decay?

### Distributions

Histograms of component counts and gap sizes, plus running averages.

```bash
cargo run --release --bin viz_distributions
```

- Top left: component count histogram
- Top right: gap size histogram (all 78 distinct gaps)
- Bottom: running averages of gap size and component count (window=5000)

Research questions answered:
- What's the shape of the distributions?
- Do they change as primes get larger?

### Custom cache path

All visualization binaries accept `--cache` to point at a different data file:

```bash
cargo run --release --bin viz_gap_waveform -- --cache my_data.bin
```

## Performance

Benchmarked on release build:

| Primes | Time | Largest prime | Cache size |
|--------|------|---------------|------------|
| 100 | 109 µs | 547 | < 1 KB |
| 1,000 | 7 ms | 8,837 | ~1 KB |
| 100,000 | 1.5 s | 1,315,309 | ~5 MB |
| 1,000,000 | 2.5 min | 15,503,737 | 52.8 MB |

Stats computation over 1M primes: ~100ms (cached to 4.9 KB JSON sidecar).

## Project Structure

```
├── build.rs                    # Links system libprimesieve
├── Cargo.toml                  # Dependencies
├── src/
│   ├── lib.rs                  # Core: FFI bindings, PrimeDatabase, decomposition
│   ├── analysis.rs             # Statistics, streaming, windowing, entropy, autocorrelation
│   ├── viz_common.rs           # Shared visualization utilities
│   ├── main.rs                 # CLI: generate / show / status / stats / entropy / autocorrelation / export
│   └── bin/
│       ├── viz_gap_waveform.rs     # 🌊 Gap sizes colored by complexity
│       ├── viz_support_scores.rs   # 📈 Base prime usage + power-law test
│       └── viz_distributions.rs    # 📊 Histograms + running averages
├── prime_basis.bin             # Binary cache (gitignored)
├── prime_basis.stats.json      # Precomputed statistics cache
└── IMPLEMENTATION_PLAN.md      # Full visualization roadmap (18 planned)
```

## Tests

```bash
cargo test --lib
```

Runs 26 tests:
- 7 original tests (decomposition correctness, FFI, incremental generation)
- 19 analysis tests (statistics, windowing, sampling, entropy, autocorrelation, cache roundtrip)

## Upcoming Visualizations

See `IMPLEMENTATION_PLAN.md` for the full roadmap. Next up:
- 🌌 Modular Starfield (polar plot, p mod 30)
- 🧬 Shannon Entropy over time
- 🌠 Phase Space Plot (strange attractor hunt)
- ✨ Compression Signature (information theory)
- 🎼 Comb Spectrogram (waterfall plot)
- 🕸 Dependency Network (force-directed graph)
- 🔊 Audio Engine (prime drum machine)
- ...and 11 more

## License

MIT
