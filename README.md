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

**Result**: `29 = 23 + 5 + 1`

## Features

- **Incremental generation** — generate N more primes at any time; previously computed decompositions are never recomputed
- **Persistent binary cache** — results stored in compact [bincode](https://github.com/bincode-org/bincode) format (~53 MB for 1M primes)
- **primesieve-powered** — prime generation via [kimwalisch/primesieve](https://github.com/kimwalisch/primesieve) C library (segmented sieve of Eratosthenes, O(n log log n), cache-optimized, multi-threaded)
- **Greedy + backtracking** — greedy decomposition handles >99% of gaps; backtracking guarantees correctness for edge cases

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

Output:
```
╔══════════════════════════════════════════╗
║       Prime Basis Research — Status      ║
╠══════════════════════════════════════════╣
║  Total primes:               1,001,102  ║
║  Decompositions:             1,001,101  ║
║  Largest prime:             15,503,737  ║
║  Cache file size:              52.8 MB  ║
╚══════════════════════════════════════════╝

  Gap statistics:
    Max gap:          154
    Avg gap:          15.5
    Max components:   3 (for a single gap)
```

## Performance

Benchmarked on release build:

| Primes | Time | Largest prime | Cache size |
|--------|------|---------------|------------|
| 100 | 109 µs | 547 | < 1 KB |
| 1,000 | 7 ms | 8,837 | ~1 KB |
| 100,000 | 1.5 s | 1,315,309 | ~5 MB |
| 1,000,000 | 2.5 min | 15,503,737 | 52.8 MB |

## Project Structure

```
├── build.rs        # Links system libprimesieve
├── Cargo.toml      # Dependencies: clap, serde, bincode
├── src/
│   ├── lib.rs      # Core: FFI bindings, PrimeDatabase, decomposition
│   └── main.rs     # CLI: generate / show / status
└── prime_basis.bin # Cache file (gitignored)
```

## Tests

```bash
cargo test
```

Runs 7 tests including verification of the user's example decompositions:
- `test_user_example_decompositions` — checks `3=2+1`, `5=3+2`, `11=7+3+1`, `29=23+5+1`, etc.
- `test_decompose_gap_simple` — greedy decomposition of small gaps
- `test_decompose_gap_needs_backtrack` — backtracking fallback
- `test_database_incremental` — incremental generation preserves existing data
- `test_generate_next_primes` / `test_generate_next_primes_after_10` — primesieve FFI works
- `test_display_format` — output formatting

## License

MIT
