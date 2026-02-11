# Prime Basis Visualization — Complete Implementation Plan

## Philosophy

**Goal**: Explore prime decompositions through every lens possible — visual, auditory, statistical, geometric, spectral, information-theoretic — to discover emergent patterns, rhythms, or hidden structure.

**One at a time**: Each visualization is implemented, tested, and explored before moving to the next. No juggling.

**Efficiency where it matters, full data where it matters**: Rendering optimizations (LOD, downsampling for display) are used to keep the UI responsive. But statistical computations, entropy, support scores, distributions, etc. MUST process the full dataset — patterns that span millions of primes won't show up in a 1000-prime window. The rule is:
- **Compute on ALL data** (streaming through the file if needed)
- **Render smartly** (downsample pixels, not data; aggregate, don't truncate)

**Scalability**: The .bin file is currently ~50MB / 1M primes but will grow. All code must handle arbitrary sizes via streaming reads, never requiring the full dataset in memory simultaneously for any single operation.

---

## Complete Idea Inventory

Every idea extracted from the brainstorming document, organized into implementation phases.

---

## Phase 0: Data Infrastructure

### 0.1 Streaming Binary Reader
- Read `prime_basis.bin` in chunks (e.g., 50k decompositions at a time)
- Iterator-based API: `db.iter_decompositions()` that streams from disk
- Random access by index range: `db.read_range(start, count)`
- Never require full 50MB+ in memory for any single visualization

### 0.2 Precomputed Statistics Cache
- On first load (or after regeneration), compute and cache:
  - Total count, max gap, avg gap, max components
  - Per-base-prime usage counts (support scores)
  - Gap distribution histogram
  - Component count distribution
- Store as a small sidecar file (e.g., `prime_basis.stats.bin`)
- Avoids re-scanning 50MB for every visualization startup

### 0.3 Sampling Utilities
- `every_nth(n)`: For rendering millions of points on a finite screen
- `random_sample(count)`: For statistical validation
- `window(start, count)`: For focused exploration
- These are RENDERING helpers — statistics always use full data

---

## Phase 1: Core Statistical Visualizations

### 1.1 🌊 Prime Gap Waveform (Colored by Complexity)
**Source**: Brainstorm §1 "The Prime Gap Waveform"

**What**: Line/scatter plot of gap sizes over the prime sequence
- X-axis: Prime index n (full range, with zoom/pan)
- Y-axis: Gap size (p_n - p_{n-1})
- Color: Blue = 1 component, Green = 2 components, Red = 3+
- Rendering: At full zoom-out, aggregate (e.g., max gap per 1000-prime bucket). Zoom in for individual points.

**Research questions**:
- Do large gaps correlate with more components?
- Does the system self-stabilize after chaotic bursts?
- Does average component count oscillate?
- Are there "heart rate variability" patterns — flat valleys with sharp spikes?

---

### 1.2 🌈 Basis Usage Heatmap (Emergence Grid)
**Source**: Brainstorm §2 "The Basis Usage Heatmap"

**What**: Binary matrix visualization
- Rows: Primes (time axis, scrollable over full dataset)
- Columns: Base primes used as components (1, 2, 3, 5, 7, 11, 13, ...)
- Cell: Bright if base prime is used in that decomposition, dark otherwise
- Rendering: At zoom-out, show density (% of primes using each base in a block). Zoom in for individual cells.

**Research questions**:
- Is there a dominance hierarchy among base primes?
- Is there a decay curve in usage frequency?
- Are there "shock zones" where larger primes (7, 11, 13) suddenly activate?
- Does the system behave like a harmonic series, turbulent system, or slow-drifting attractor?

---

### 1.3 📊 Component Distribution & Gap Statistics
**Source**: Brainstorm "If I Had to Choose a First Experiment" + general statistics

**What**: Histogram panel
- Histogram: Number of components per decomposition (1, 2, 3, ...)
- Histogram: Gap size distribution
- Running average: Component count over time (sliding window)
- Running average: Gap size over time

**Research questions**:
- What's the shape of the component count distribution?
- Does it change as primes get larger?

---

### 1.4 📈 Support Score Distribution (Log-Log Plot)
**Source**: Brainstorm §4 "Dependency Network" — `support_score(p) = number of times p appears as a component`

**What**: Compute support score for every base prime across the FULL dataset, then:
- Bar chart: Top N base primes by usage count
- Log-log plot: support_score vs prime value
- Test for power-law behavior (fit line on log-log)

**Research questions**:
- Does support score follow a power law? (Would indicate scale-free structure)
- Which primes are the "backbone" of the system?
- Is there a sharp cutoff or gradual decay?

---

## Phase 2: Geometric & Modular Visualizations

### 2.1 🌌 Modular Starfield (Polar Plot)
**Source**: Brainstorm §3 "Modular Starfields" + "Spiral Prime Lattice"

**What**: Polar coordinate scatter plot
- Radius (r): Prime index n
- Angle (θ): p_n mod M (default M=30, but adjustable: 6, 30, 210, custom)
- Dot size: Gap size
- Color: Component count (blue→red temperature scale)
- Rendering: Full dataset, downsample for display density

**Research questions**:
- Do complex decompositions cluster on specific modular spokes?
- Is complexity equidistributed across residue classes, or biased?
- How does the pattern change with different moduli?

---

### 2.2 🔄 Resonance Cylinder (Dynamic Modulus Scanner)
**Source**: Brainstorm "The Resonance Cylinder (Modulus Scanning)"

**What**: Animated polar plot where the modulus sweeps continuously
- Plot primes on a cylinder surface
- Animate circumference (modulus) from 2 to 300
- At primorials (6, 30, 210), chaos should snap into vertical lines
- Interactive: slider to manually control modulus, play/pause animation

**Research questions**:
- At which moduli does order emerge?
- Do decomposition-colored dots align differently than plain prime dots?
- Are there unexpected moduli where structure appears?

---

### 2.3 🧭 Multi-Dimensional Vector Walk (3D/4D Trajectory)
**Source**: Brainstorm "The Multi-Dimensional Vector Walk"

**What**: 3D trajectory through basis-space
- Assign spatial directions to base primes:
  - +1 → X axis
  - +2 → Y axis
  - +3 → Z axis
  - +5 → Color (4th dimension)
- Each gap's decomposition = a displacement vector
- Trace the cumulative path
- Camera controls: rotate, zoom, pan
- Trail rendering with fade

**Research questions**:
- Does the path form a spiral, a coral-reef texture, or random spaghetti?
- Is there a preferred growth direction?
- Does the shape change character at different prime magnitudes?

---

## Phase 3: Dynamical Systems & Information Theory

### 3.1 🌠 Phase Space Plot (Strange Attractor Hunt)
**Source**: Brainstorm §7 "Phase Space Plot (Dynamic Systems View)"

**What**: Scatter plot treating each prime as a dynamical step
- X: gap size (g_n)
- Y: number of components (c_n)
- Plot ALL (g_n, c_n) pairs
- Optional: color by prime index to see temporal evolution
- Optional: 3D with Z = g_{n+1} (next gap) for delay embedding

**Research questions**:
- Does it cluster into distinct regions?
- Does it form a strange attractor?
- Or does it smear randomly? (Would suggest no dynamical structure)

---

### 3.2 🧬 Shannon Entropy Over Time
**Source**: Brainstorm §6 "Entropy Over Time"

**What**: Sliding-window entropy of component distributions
- Window size: configurable (default 1000, but test 100, 5000, 10000)
- For each window: compute Shannon entropy H of the distribution of which base primes appear
- Plot H(n) over the full prime sequence
- Overlay: multiple window sizes on same plot

**Research questions**:
- Does entropy stabilize → statistically stationary process?
- Does entropy drift upward → growing chaos?
- Does entropy oscillate → hidden macro-rhythm?
- How does window size affect the signal?

---

### 3.3 ✨ Prime Compression Signature
**Source**: Brainstorm §8 "The Most Radical Idea"

**What**: Information-theoretic analysis
- For each decomposition, measure: bits_required = log₂(number of possible decompositions) or simply the storage cost of the component list
- Compare against: log₂(p_n) (bits to store the prime directly)
- Plot: bits_per_decomposition(n) vs log₂(p_n)
- Compute: ratio over time

**Research questions**:
- Does average bits per prime stabilize below log₂(p)?
- If yes → the basis method IS a compression scheme for primes
- That's structural information theory, not just visualization

---

## Phase 4: Spectral & Wave Analysis

### 4.1 🎼 Comb Spectrogram (Waterfall Plot)
**Source**: Brainstorm §2 initial "The Comb Spectrogram" (distinct from heatmap)

**What**: Scrolling waterfall display (like audio spectral analysis)
- X-axis: Base primes (frequency bins)
- Y-axis: Time (prime sequence, scrolling)
- Pixel intensity: Whether that base prime is used (binary) or usage density in a window
- Animated scrolling through the sequence

**Research questions**:
- Are there vertical stripes (constant usage of +1)?
- Are there diagonal waves (drifting heavy components)?
- Are there "bands of stability" where {1,2,3} dominates, interspersed with chaotic "breakbeats"?

---

### 4.2 💎 Hyper-Crystal Diffraction (Fourier Slice)
**Source**: Brainstorm "The Hyper-Crystal Diffraction"

**What**: Treat decomposition components as phases/frequencies, compute interference
- Map each base prime to a frequency: 1→f₁, 2→f₂, 3→f₃, ...
- For each decomposition, sum the corresponding sine waves
- Plot the resulting composite waveform
- Compute FFT of the composite signal
- Look for constructive interference peaks ("rogue waves")

**Research questions**:
- Are there sudden amplitude spikes where hidden dimensions align?
- Does the FFT reveal dominant frequencies?
- Do the interference patterns have structure or are they noise?

---

### 4.3 🔬 Spectral Barcode (Energy Levels)
**Source**: Brainstorm "The Energy Level Staircase"

**What**: Render decompositions as spectral absorption lines
- Each decomposition → a barcode (dark bands for used components, light for unused)
- Stack barcodes vertically to create a 2D surface
- Compare visually to atomic spectral lines

**Research questions**:
- Do "spectral lines" emerge (consistent dark bands)?
- Does the pattern resemble known physical spectra?
- Connection to Montgomery-Odlyzko pair correlation?

---

## Phase 5: Higher-Dimensional Analysis

### 5.1 🧠 Basis Vector Embedding + PCA
**Source**: Brainstorm "Define state vector" + "Is There a Manifold?"

**What**: Treat each decomposition as a point in high-dimensional space
- For each prime, build vector: v_n = (uses_1, uses_2, uses_3, uses_5, uses_7, ...)
- Apply PCA to project into 2D/3D
- Color by prime index (time)
- Apply manifold learning (t-SNE, UMAP) if PCA is inconclusive

**Research questions**:
- Do the vectors cluster?
- Do they lie near a low-dimensional manifold?
- Does the PCA projection reveal smooth structure hidden in 1D?

---

### 5.2 📐 Successive Vector Distance (Flatland Test)
**Source**: Brainstorm "A Concrete Experiment" — ||v_n - v_{n-1}||

**What**: Measure smoothness in basis-space
- Build basis usage vectors for all primes
- Compute Euclidean distance between successive vectors: d_n = ||v_n - v_{n-1}||
- Plot d_n over time

**Research questions**:
- Does d_n stabilize → smooth higher-dimensional flow behind jagged 1D surface?
- Does d_n oscillate regularly → periodic structure in basis-space?
- Does d_n grow → diverging trajectories?

---

## Phase 6: Network & Relational Visualizations

### 6.1 🕸 Dependency Network (Force-Directed Graph)
**Source**: Brainstorm §4 "The Dependency Web" / "Prime Memory Graph"

**What**: Force-directed graph of prime dependencies
- Nodes: Base primes (NOT all 1M primes — the base primes that appear as components)
- Edges: Weighted by how often prime A appears as a component alongside prime B
- Node size: Support score
- Layout: Force-directed physics simulation
- Sampling: Can also show a subgraph for a specific prime range

**Research questions**:
- Are 1, 2, 3 super-hubs?
- Do 5, 7 form secondary scaffolding?
- Are there "orphan clusters" relying on rare large bases?
- Does the network have small-world properties?

---

## Phase 7: Auditory Visualization

### 7.1 🔊 Prime Rhythm Audio Engine (Drum Machine)
**Source**: Brainstorm §1 "Prime Drum Machine" + §5 "Prime Rhythm Audio Engine"

**What**: Sonification of decompositions
- Map base primes to sounds: 1→hi-hat, 2→snare, 3→kick, 5→tom, 7→bell, 11+→synth
- Previous prime = bass drone (carrier wave)
- Playback at configurable BPM (default 120)
- Controls: play/pause, speed, range selection, solo/mute individual voices

**Research questions**:
- Is there a repeating "6-beat motif" (5+1)?
- Do twin-prime rhythms (gap=2) create a recognizable pattern?
- Are there chaotic bursts around larger gaps?
- If patterns are perceptible → the structure is compressible

---

## Phase 8: Integration

### 8.1 🖥 Unified Dashboard
- Multi-panel layout with any combination of the above
- Synchronized navigation: selecting a prime range in one panel highlights it in all
- Data window controls shared across panels
- Export: screenshots, data subsets (CSV/JSON), statistics reports

---

## Implementation Order (Recommended)

| Step | Visualization | Why This Order |
|------|--------------|----------------|
| 0 | Data Infrastructure (streaming reader, stats cache) | Everything depends on this |
| 1 | Gap Waveform (1.1) | Simplest plot, immediate feedback, tests infrastructure |
| 2 | Support Score Distribution (1.4) | Quick to compute, answers "is there hierarchy?" |
| 3 | Basis Usage Heatmap (1.2) | Core visual, reveals structure at a glance |
| 4 | Component/Gap Statistics (1.3) | Quantitative grounding |
| 5 | Entropy Over Time (3.2) | Serious math, answers "is there stationarity?" |
| 6 | Phase Space Plot (3.1) | Quick scatter, answers "is there an attractor?" |
| 7 | Modular Starfield (2.1) | Tests modular bias hypothesis |
| 8 | Resonance Cylinder (2.2) | Animated extension of starfield |
| 9 | Compression Signature (3.3) | Information theory, profound if positive |
| 10 | Comb Spectrogram (4.1) | Waterfall view, different angle on heatmap |
| 11 | Spectral Barcode (4.3) | Quick to build after spectrogram |
| 12 | Successive Vector Distance (5.2) | Simple computation, tests "Flatland" hypothesis |
| 13 | PCA / Manifold Embedding (5.1) | Needs linear algebra lib, but high value |
| 14 | Hyper-Crystal Diffraction (4.2) | FFT-based, more complex |
| 15 | 3D Vector Walk (2.3) | Requires 3D rendering |
| 16 | Dependency Network (6.1) | Force-directed layout, complex |
| 17 | Audio Engine (7.1) | Requires audio library, artistic |
| 18 | Unified Dashboard (8.1) | Integration of everything |

---

## Technical Stack Recommendations

| Need | Library | Notes |
|------|---------|-------|
| GUI framework | `egui` + `eframe` | Immediate mode, fast iteration, cross-platform |
| 2D plotting | `egui_plot` | Built into egui, interactive zoom/pan |
| Heatmaps/images | `egui` texture API | Render heatmap as image, display in egui |
| 3D rendering | `three-d` or `bevy` | For vector walk (Phase 2.3) |
| FFT | `rustfft` | For Fourier/spectral analysis |
| Linear algebra | `nalgebra` + `ndarray` | For PCA, vector operations |
| Audio | `rodio` or `cpal` | For drum machine |
| Parallel compute | `rayon` | For statistics over full dataset |
| Serialization | `bincode` (already used) | Extend for stats cache |

---

## Additional Ideas (My Suggestions)

### A. Gap Autocorrelation
- Compute autocorrelation of the gap sequence: does gap_n predict gap_{n+k}?
- Plot autocorrelation function for lags 1..1000
- If there are peaks at specific lags → periodic structure in gaps
- This is standard signal processing and could reveal rhythms invisible to the eye

### B. Transition Matrix (Markov Analysis)
- Build a matrix: P(component_set_n | component_set_{n-1})
- Or simpler: P(gap_n | gap_{n-1})
- Visualize as a heatmap
- If the matrix has strong off-diagonal structure → the process has memory
- If it's roughly uniform → each gap is independent of the previous

### C. Recurrence Plot
- Classic dynamical systems tool
- Plot a dot at (i, j) if ||state_i - state_j|| < threshold
- State can be gap, component vector, or any derived quantity
- Reveals: periodicity (diagonal lines), chaos (texture), regime changes (blocks)
- Particularly powerful for detecting hidden periodicity in noisy sequences

### D. Cumulative Basis Drift
- For each base prime, plot its cumulative usage count over time
- If usage is steady → straight line with constant slope
- If usage drifts → curves, inflection points
- Overlay multiple base primes on same plot
- Reveals: when does the "character" of decompositions shift?

### E. Gap-Component Correlation Matrix
- For each pair (gap_size, base_prime), count co-occurrences
- Visualize as heatmap
- Reveals: which base primes are associated with which gap sizes?
- Are large gaps always decomposed using the same bases?

### F. First Appearance Tracker
- For each base prime, record the first time it appears as a component
- Plot: base_prime_value vs first_appearance_index
- Reveals: how quickly does the system "discover" new tools?
- Is there a pattern to when new base primes enter the vocabulary?

### G. Decomposition Uniqueness Analysis
- For each gap size, how many distinct decompositions exist (across all occurrences)?
- Does the same gap always decompose the same way, or does context matter?
- If context matters → the system has genuine memory, not just arithmetic

---

## Windowing & Performance Strategy (Detailed)

### Principle: Compute everything, render smartly

```
FULL DATA (1M+ primes, 50MB+)
    │
    ├── Statistics: ALWAYS computed on full data (streaming)
    │   ├── support scores
    │   ├── entropy
    │   ├── distributions
    │   └── correlations
    │
    └── Rendering: Adaptive to screen resolution
        ├── Zoom level 1 (full range): aggregate per-pixel bucket
        │   e.g., 1M primes on 1920px = ~520 primes/pixel
        │   Show: max, min, avg per bucket
        │
        ├── Zoom level 2 (100k range): every 50th point
        │
        ├── Zoom level 3 (10k range): every 5th point
        │
        └── Zoom level 4 (1k range): every point, full detail
```

### Memory Budget
- Streaming reader: ~10MB buffer at a time
- Precomputed stats: <1MB
- Render buffer: Proportional to screen pixels, not data size
- Target: <200MB total RAM regardless of dataset size

---

## Research Hypotheses to Test

These are the core questions, ordered by how quickly each visualization can answer them:

1. **Hierarchy**: Do a few small primes dominate all decompositions? → Support Score (step 2)
2. **Stationarity**: Does the statistical character of decompositions change over time? → Entropy (step 5)
3. **Modular Bias**: Does decomposition complexity depend on p mod 30? → Starfield (step 7)
4. **Dynamical Structure**: Is there an attractor in (gap, components) space? → Phase Space (step 6)
5. **Compression**: Is the basis method more efficient than storing primes directly? → Compression Signature (step 9)
6. **Memory**: Does the previous decomposition predict the next? → Transition Matrix (suggestion B)
7. **Manifold**: Do basis vectors lie on a low-dimensional surface? → PCA (step 13)
8. **Power Law**: Is the support score distribution scale-free? → Log-log plot (step 2)
9. **Periodicity**: Are there hidden cycles in the gap sequence? → Autocorrelation (suggestion A)
10. **Context Sensitivity**: Does the same gap decompose differently in different contexts? → Uniqueness Analysis (suggestion G)
