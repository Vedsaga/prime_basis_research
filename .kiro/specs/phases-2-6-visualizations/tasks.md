# Implementation Plan: Phases 2–6 Visualizations

## Overview

Implement 11 new visualization binaries and supporting analysis/spectral modules for the Prime Basis Research tool. Each visualization follows the established pattern: load data, precompute, render with egui. Implementation order follows the recommended sequence from the project plan. New dependencies: `rustfft`, `nalgebra`, `proptest` (dev).

## Tasks

- [x] 1. Add dependencies and shared utility extensions
  - [x] 1.1 Add `rustfft = "6"`, `nalgebra = "0.33"` to `[dependencies]` and `proptest = "1"` to `[dev-dependencies]` in Cargo.toml
    - _Requirements: 9.1, 9.3, 8.1, 8.6_
  - [x] 1.2 Add `polar_to_cartesian()`, `color_by_index()`, and `project_3d()` to `src/viz_common.rs`
    - `polar_to_cartesian(r, theta) -> [f64; 2]` converts polar to Cartesian
    - `color_by_index(index, total) -> egui::Color32` maps index to blue→red gradient
    - `project_3d(point, yaw, pitch, scale) -> [f64; 2]` projects 3D to 2D via rotation
    - _Requirements: 1.3, 1.4, 2.1_
  - [x] 1.3 Write property tests for viz_common helpers
    - **Property 1: Color gradient monotonicity**
    - **Property 2: 3D projection produces finite coordinates**
    - **Property 3: Polar-to-Cartesian round trip**
    - **Validates: Requirements 1.3, 1.4, 2.1**

- [x] 2. Implement core analysis functions
  - [x] 2.1 Add `build_basis_vector()` and `compression_bits()` to `src/analysis.rs`
    - `build_basis_vector(decomp, top_bases) -> Vec<f64>` returns binary vector of length K
    - `compression_bits(decomp) -> f64` computes bit cost per the formula: sum(ceil(log2(c+1))) + ceil(log2(count+1))
    - _Requirements: 12.1, 4.3_
  - [x] 2.2 Add `successive_distances()` to `src/analysis.rs`
    - Computes Euclidean distance between consecutive basis vectors
    - Returns Vec<f64> of length N-1
    - _Requirements: 7.2, 12.5_
  - [x] 2.3 Add `compute_pca()` and `pca_project()` to `src/analysis.rs`
    - Uses `nalgebra` for eigendecomposition of covariance matrix
    - Returns `PcaResult` with components, explained variance, mean
    - _Requirements: 8.1, 8.6, 12.2_
  - [x] 2.4 Add `co_occurrence_matrix()` to `src/analysis.rs`
    - Computes symmetric K×K matrix of base prime pair co-occurrences
    - _Requirements: 11.1, 12.3_
  - [x] 2.5 Write property tests for analysis functions
    - **Property 4: Compression bits formula correctness**
    - **Property 5: Basis vector construction**
    - **Property 6: Successive distances length and non-negativity**
    - **Property 7: PCA mathematical invariants**
    - **Property 12: Co-occurrence matrix symmetry and diagonal consistency**
    - **Validates: Requirements 4.3, 12.1, 7.2, 8.1, 8.6, 11.1**

- [x] 3. Create spectral analysis module
  - [x] 3.1 Create `src/spectral.rs` with `composite_signal()`, `compute_fft()`, and `find_peaks()`
    - Add `pub mod spectral;` to `src/lib.rs`
    - `composite_signal(decompositions, total) -> Vec<f64>` sums sine waves per component frequencies
    - `compute_fft(signal) -> (Vec<f64>, Vec<f64>)` returns (frequencies, magnitudes)
    - `find_peaks(magnitudes, n_sigma) -> Vec<(usize, f64)>` finds peaks above mean + n_sigma * std
    - _Requirements: 9.1, 9.3, 9.5_
  - [x] 3.2 Write property tests for spectral functions
    - **Property 8: Composite signal for single-component decompositions**
    - **Property 9: FFT detects known frequency**
    - **Property 10: Peak finder detects spikes above threshold**
    - **Validates: Requirements 9.1, 9.3, 9.5**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Phase Space Plot
  - [x] 5.1 Create `src/bin/viz_phase_space.rs`
    - Load data via `viz_common::load_data()`
    - Precompute (gap, num_components, next_gap) triples for all decompositions
    - Render 2D scatter plot with egui_plot::Points, color by component count
    - Add toggle for color-by-index mode using `color_by_index()`
    - Add toggle for 3D mode using `project_3d()` with draggable yaw/pitch
    - Display summary stats in header panel
    - Downsample when visible range > 50,000 points
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 6. Implement Modular Starfield
  - [x] 6.1 Create `src/bin/viz_starfield.rs`
    - Load data, precompute (prime, gap, num_components, index) tuples
    - Render polar scatter plot: r=index, θ=2π×(prime mod M)/M, converted to Cartesian
    - Dot size ∝ gap size (clamped [1.0, 6.0]), color by component count (temperature scale)
    - Modulus slider (2–500) with preset buttons for 6, 30, 210
    - Downsample to 10,000 points max
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 7. Implement Resonance Cylinder
  - [x] 7.1 Create `src/bin/viz_resonance.rs`
    - Same data as Starfield, add animation state (playing, speed, current modulus)
    - Play/pause button, speed slider, manual modulus slider
    - Auto-increment modulus each frame when playing
    - Label primorial values (6, 30, 210) when modulus matches
    - Display current modulus prominently
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 8. Implement Compression Signature
  - [x] 8.1 Create `src/bin/viz_compression.rs`
    - Precompute decomp_bits via `compression_bits()` for all decompositions
    - Precompute log₂(prime) for all decompositions
    - Precompute ratios and running averages
    - Top panel: line plot of decomp_bits and log₂(prime) vs index
    - Bottom panel: ratio vs index with running average overlay
    - Header: avg/min/max ratio stats
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 9. Checkpoint — Ensure all binaries compile and run
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Comb Spectrogram
  - [x] 10.1 Create `src/bin/viz_spectrogram.rs`
    - Determine top N base primes from PrecomputedStats.top_support (default N=30)
    - Build texture image: width=N columns (base primes), height=visible rows
    - Each pixel: bright if base prime used in that decomposition, dark otherwise
    - Aggregate rows into density blocks when zoomed out
    - Vertical scroll to navigate through prime sequence
    - Configurable N via slider
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. Implement Spectral Barcode
  - [x] 11.1 Create `src/bin/viz_barcode.rs`
    - Same base prime ordering as Spectrogram (top N by support score)
    - Render each decomposition as horizontal barcode: dark=used, light=unused
    - Stack vertically, prime index increasing downward
    - Aggregate into density blocks when zoomed out
    - Vertical scroll navigation
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 12. Implement Successive Vector Distance
  - [x] 12.1 Create `src/bin/viz_vector_distance.rs`
    - Precompute distances via `successive_distances()` with top K=30 bases
    - Precompute running average via `running_average()`
    - Line plot: index vs distance, with running average overlay
    - Header: mean distance, std dev, trend direction
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 13. Implement PCA Embedding
  - [x] 13.1 Create `src/bin/viz_pca.rs`
    - Precompute PCA via `compute_pca()` with top K=30 bases, 3 components
    - Project all decompositions via `pca_project()`
    - Scatter plot of selected PC pair, colored by prime index
    - Buttons to switch axis pair: PC1vPC2, PC1vPC3, PC2vPC3
    - Display explained variance ratios
    - Downsample to 10,000 points max
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 14. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement Hyper-Crystal Diffraction
  - [x] 15.1 Create `src/bin/viz_diffraction.rs`
    - Precompute composite signal via `composite_signal()`
    - Precompute FFT via `compute_fft()`
    - Precompute peaks via `find_peaks(magnitudes, 2.0)`
    - Top panel: signal amplitude vs prime index (downsampled line plot)
    - Bottom panel: FFT magnitude spectrum with peak markers
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 16. Implement 3D Vector Walk
  - [x] 16.1 Create `src/bin/viz_vector_walk.rs`
    - Compute cumulative trajectory: for each decomposition, displacement = (has_base_x, has_base_y, has_base_z)
    - Default axis mapping: 1→X, 2→Y, 3→Z
    - Project 3D path to 2D via `project_3d()`, render as colored line segments
    - Mouse drag for rotation (yaw/pitch), scroll for zoom
    - Dropdowns to remap base primes to axes
    - Downsample to 20,000 segments max
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  - [x] 16.2 Write property test for trajectory computation
    - **Property 11: Trajectory cumulative sum invariant**
    - **Validates: Requirements 10.1**

- [x] 17. Implement Dependency Network
  - [x] 17.1 Create `src/bin/viz_network.rs`
    - Precompute co-occurrence matrix via `co_occurrence_matrix()` for top N=20 bases
    - Initialize nodes with random positions, support scores as sizes
    - Force-directed simulation: repulsion (Coulomb), attraction (spring along edges), damping 0.95
    - Render edges as lines (thickness ∝ weight), nodes as circles (radius ∝ support score)
    - Hover: highlight connected edges, tooltip with prime value and support score
    - Pause/resume simulation, drag individual nodes
    - Configurable N via slider
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
  - [x] 17.2 Write property test for force simulation
    - **Property 13: Force-directed simulation energy decrease**
    - **Validates: Requirements 11.3**

- [x] 18. Final checkpoint — Ensure no  warnings anywhere and also all tests pass and all binaries compile
  - Ensure all warning are fixed, make sure no warining for all the modules.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- All visualizations follow the established pattern: load via `viz_common::load_data()`, precompute, render with egui
- Implementation order: Phase Space → Starfield → Resonance → Compression → Spectrogram → Barcode → Vector Distance → PCA → Diffraction → Vector Walk → Network
