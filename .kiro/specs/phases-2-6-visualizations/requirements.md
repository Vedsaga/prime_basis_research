# Requirements Document

## Introduction

This feature implements Phases 2 through 6 of the Prime Basis Visualization research tool: a suite of 11 new interactive visualizations exploring prime gap decompositions through geometric, dynamical, spectral, higher-dimensional, and network lenses. Each visualization is a standalone egui binary in `src/bin/viz_*.rs`, following the established patterns. New analysis/computation functions are added to `src/analysis.rs` or new modules. The implementation order follows the recommended sequence from the project plan: Phase Space → Modular Starfield → Resonance Cylinder → Compression Signature → Comb Spectrogram → Spectral Barcode → Successive Vector Distance → PCA → Hyper-Crystal Diffraction → 3D Vector Walk → Dependency Network.

## Glossary

- **Prime_Database**: The in-memory representation of all prime decompositions, loaded from `prime_basis.bin` via `PrimeDatabase::load()`
- **Precomputed_Stats**: Cached aggregate statistics over the full dataset, loaded via `PrecomputedStats::load_or_compute()`
- **Decomposition**: A record `(prime, prev_prime, gap, components)` representing how a prime gap is expressed as a sum of distinct smaller primes
- **Component**: A base prime used in a decomposition's sum
- **Support_Score**: The number of decompositions in which a given base prime appears as a component
- **Basis_Vector**: A binary or count vector over all known base primes, indicating which bases appear in a given decomposition
- **Viz_Binary**: A standalone Rust binary in `src/bin/` that launches an egui window for a single visualization
- **Downsampling**: Reducing the number of rendered points for display while preserving visual fidelity; statistics always use full data
- **Modulus**: An integer M used to compute residue classes (p mod M) for modular visualizations
- **Phase_Space**: A coordinate system where each axis represents a different observable of the decomposition (e.g., gap size, component count)
- **PCA**: Principal Component Analysis — a linear dimensionality reduction technique projecting high-dimensional vectors to 2D/3D
- **FFT**: Fast Fourier Transform — an algorithm for computing the discrete Fourier transform of a signal
- **Force_Directed_Layout**: A graph layout algorithm using simulated physical forces (attraction/repulsion) to position nodes

## Requirements

### Requirement 1: Phase Space Plot

**User Story:** As a researcher, I want to view a scatter plot of (gap_size, num_components) pairs for all decompositions, so that I can search for dynamical structure or strange attractors in the decomposition sequence.

#### Acceptance Criteria

1. WHEN the Phase_Space Viz_Binary launches, THE Phase_Space Viz_Binary SHALL load the Prime_Database and Precomputed_Stats and display a scatter plot with gap size on the X-axis and component count on the Y-axis
2. WHEN rendering the scatter plot, THE Phase_Space Viz_Binary SHALL plot one point per Decomposition using Downsampling when the visible range exceeds 50,000 points
3. WHEN the user enables color-by-index mode, THE Phase_Space Viz_Binary SHALL color each point by its prime index using a gradient from blue (early) to red (late)
4. WHEN the user enables 3D mode, THE Phase_Space Viz_Binary SHALL add a Z-axis representing the next gap (g_{n+1}), displayed as a 2D projection with adjustable viewing angle
5. THE Phase_Space Viz_Binary SHALL display summary statistics including total points, gap range, and component count range in a header panel

### Requirement 2: Modular Starfield

**User Story:** As a researcher, I want to view primes on a polar plot with angle determined by p mod M, so that I can detect modular bias in decomposition complexity.

#### Acceptance Criteria

1. WHEN the Modular_Starfield Viz_Binary launches, THE Modular_Starfield Viz_Binary SHALL display a polar scatter plot with radius equal to the prime index and angle equal to (2π × (prime mod M)) / M, using a default modulus M=30
2. WHEN the user adjusts the modulus slider, THE Modular_Starfield Viz_Binary SHALL recompute all point angles using the new modulus value and update the plot
3. WHEN rendering points, THE Modular_Starfield Viz_Binary SHALL set dot size proportional to gap size and color according to component count using a blue-to-red temperature scale
4. THE Modular_Starfield Viz_Binary SHALL provide preset modulus buttons for common values (6, 30, 210) in addition to the slider
5. THE Modular_Starfield Viz_Binary SHALL apply Downsampling to limit rendered points to at most 10,000 while preserving the angular distribution

### Requirement 3: Resonance Cylinder

**User Story:** As a researcher, I want to see an animated polar plot that sweeps the modulus from 2 to 300, so that I can identify moduli where structural order emerges.

#### Acceptance Criteria

1. WHEN the Resonance_Cylinder Viz_Binary launches, THE Resonance_Cylinder Viz_Binary SHALL display a polar scatter plot identical in structure to the Modular_Starfield, with the modulus initially set to 2
2. WHEN the user presses the play button, THE Resonance_Cylinder Viz_Binary SHALL animate the modulus from its current value up to 300, incrementing by 1 per frame at a configurable speed
3. WHEN the user pauses the animation, THE Resonance_Cylinder Viz_Binary SHALL freeze the current modulus and allow manual adjustment via a slider
4. WHEN the modulus equals a primorial value (6, 30, 210), THE Resonance_Cylinder Viz_Binary SHALL display a label indicating the primorial
5. THE Resonance_Cylinder Viz_Binary SHALL display the current modulus value prominently during animation

### Requirement 4: Compression Signature

**User Story:** As a researcher, I want to compare the information content of decompositions against log₂(p), so that I can determine whether the basis method acts as a compression scheme for primes.

#### Acceptance Criteria

1. WHEN the Compression_Signature Viz_Binary launches, THE Compression_Signature Viz_Binary SHALL compute for each Decomposition the number of bits required to encode its component list and the value log₂(prime)
2. WHEN rendering the plot, THE Compression_Signature Viz_Binary SHALL display a line plot with prime index on the X-axis and two lines: decomposition bits and log₂(prime)
3. THE Compression_Signature Viz_Binary SHALL compute decomposition bits as the sum of ceil(log₂(c + 1)) for each component c, plus ceil(log₂(component_count + 1)) to encode the count
4. THE Compression_Signature Viz_Binary SHALL display a running ratio of decomposition_bits / log₂(prime) as a separate plot panel
5. THE Compression_Signature Viz_Binary SHALL display summary statistics including average ratio, minimum ratio, and maximum ratio across the full dataset
6. WHEN computing the bits-per-decomposition, THE Compression_Signature analysis module SHALL process the full dataset and return per-decomposition bit costs as a vector

### Requirement 5: Comb Spectrogram

**User Story:** As a researcher, I want to see a scrolling waterfall display of base prime usage over the prime sequence, so that I can detect temporal patterns in which bases are active.

#### Acceptance Criteria

1. WHEN the Comb_Spectrogram Viz_Binary launches, THE Comb_Spectrogram Viz_Binary SHALL render a 2D image where the X-axis represents base primes (columns) and the Y-axis represents time (prime index, rows)
2. WHEN rendering each pixel, THE Comb_Spectrogram Viz_Binary SHALL set pixel intensity based on whether the corresponding base prime is used in the Decomposition at that row
3. THE Comb_Spectrogram Viz_Binary SHALL limit the X-axis to the top N most-used base primes (default N=30, configurable) sorted by Support_Score
4. WHEN the user scrolls vertically, THE Comb_Spectrogram Viz_Binary SHALL shift the visible window through the prime sequence
5. THE Comb_Spectrogram Viz_Binary SHALL aggregate rows into blocks when the visible range exceeds the vertical pixel count, showing usage density per block

### Requirement 6: Spectral Barcode

**User Story:** As a researcher, I want to render decompositions as spectral absorption lines stacked vertically, so that I can visually compare decomposition patterns to physical spectral phenomena.

#### Acceptance Criteria

1. WHEN the Spectral_Barcode Viz_Binary launches, THE Spectral_Barcode Viz_Binary SHALL render each Decomposition as a horizontal barcode where dark bands correspond to used components and light bands correspond to unused components
2. THE Spectral_Barcode Viz_Binary SHALL stack barcodes vertically with prime index increasing downward, creating a 2D surface
3. THE Spectral_Barcode Viz_Binary SHALL use the same base prime ordering as the Comb_Spectrogram (top N by Support_Score)
4. WHEN the user scrolls vertically, THE Spectral_Barcode Viz_Binary SHALL shift the visible window through the prime sequence
5. THE Spectral_Barcode Viz_Binary SHALL aggregate rows into density blocks when the visible range exceeds the vertical pixel count

### Requirement 7: Successive Vector Distance

**User Story:** As a researcher, I want to plot the Euclidean distance between successive Basis_Vectors over time, so that I can test whether decompositions form a smooth flow in higher-dimensional space.

#### Acceptance Criteria

1. WHEN the Successive_Distance Viz_Binary launches, THE Successive_Distance Viz_Binary SHALL compute a Basis_Vector for each Decomposition as a binary vector over the top K base primes (default K=30)
2. WHEN computing distances, THE Successive_Distance analysis module SHALL compute the Euclidean distance between each consecutive pair of Basis_Vectors across the full dataset
3. THE Successive_Distance Viz_Binary SHALL display a line plot with prime index on the X-axis and Euclidean distance on the Y-axis
4. THE Successive_Distance Viz_Binary SHALL overlay a running average line (window=5000) on the distance plot
5. THE Successive_Distance Viz_Binary SHALL display summary statistics including mean distance, standard deviation, and trend direction

### Requirement 8: PCA Embedding

**User Story:** As a researcher, I want to project high-dimensional Basis_Vectors into 2D via PCA, so that I can discover whether decompositions cluster or lie on a low-dimensional manifold.

#### Acceptance Criteria

1. WHEN the PCA Viz_Binary launches, THE PCA analysis module SHALL construct Basis_Vectors for all Decompositions using the top K base primes (default K=30) and compute the first 3 principal components
2. WHEN rendering the 2D projection, THE PCA Viz_Binary SHALL display a scatter plot of PC1 vs PC2, with points colored by prime index (blue-to-red gradient)
3. WHEN the user toggles to PC1 vs PC3 or PC2 vs PC3, THE PCA Viz_Binary SHALL update the scatter plot axes accordingly
4. THE PCA Viz_Binary SHALL display the explained variance ratio for each of the first 3 principal components
5. THE PCA Viz_Binary SHALL apply Downsampling to limit rendered points to at most 10,000 while preserving the distribution shape
6. WHEN computing PCA, THE PCA analysis module SHALL center the data by subtracting the mean vector and compute eigenvectors of the covariance matrix

### Requirement 9: Hyper-Crystal Diffraction

**User Story:** As a researcher, I want to compute FFT-based interference patterns from decomposition components treated as frequencies, so that I can detect hidden spectral structure.

#### Acceptance Criteria

1. WHEN the Diffraction Viz_Binary launches, THE Diffraction analysis module SHALL map each base prime to a frequency and for each Decomposition compute a composite signal by summing sine waves at the corresponding frequencies
2. WHEN rendering the composite waveform, THE Diffraction Viz_Binary SHALL display the signal amplitude over the prime sequence as a line plot
3. THE Diffraction analysis module SHALL compute the FFT of the composite signal across the full dataset
4. THE Diffraction Viz_Binary SHALL display the FFT magnitude spectrum as a separate plot panel with frequency on the X-axis and magnitude on the Y-axis
5. THE Diffraction Viz_Binary SHALL highlight peaks in the FFT spectrum that exceed 2 standard deviations above the mean magnitude

### Requirement 10: 3D Vector Walk

**User Story:** As a researcher, I want to trace a 3D trajectory through basis-space where each gap's decomposition defines a displacement vector, so that I can visualize the cumulative path of the decomposition sequence.

#### Acceptance Criteria

1. WHEN the Vector_Walk Viz_Binary launches, THE Vector_Walk Viz_Binary SHALL assign spatial axes to the first 3 base primes (1→X, 2→Y, 3→Z) and compute a cumulative position by summing displacement vectors from each Decomposition
2. WHEN rendering the trajectory, THE Vector_Walk Viz_Binary SHALL display the 3D path as a connected line with color varying by prime index (blue→red gradient)
3. THE Vector_Walk Viz_Binary SHALL provide camera controls for rotation, zoom, and pan of the 3D view
4. THE Vector_Walk Viz_Binary SHALL apply Downsampling to limit rendered path segments to at most 20,000 points
5. WHEN the user adjusts the base-prime-to-axis mapping, THE Vector_Walk Viz_Binary SHALL recompute the trajectory using the selected base primes for X, Y, and Z axes

### Requirement 11: Dependency Network

**User Story:** As a researcher, I want to view a force-directed graph of base prime co-occurrence, so that I can identify structural relationships and hub-spoke patterns among component primes.

#### Acceptance Criteria

1. WHEN the Network Viz_Binary launches, THE Network analysis module SHALL compute a co-occurrence matrix counting how often each pair of base primes appears together in the same Decomposition across the full dataset
2. WHEN rendering the graph, THE Network Viz_Binary SHALL display base primes as nodes with size proportional to Support_Score and edges with thickness proportional to co-occurrence count
3. THE Network Viz_Binary SHALL implement a force-directed layout using repulsion between all nodes and attraction along edges weighted by co-occurrence
4. THE Network Viz_Binary SHALL limit the graph to the top N base primes by Support_Score (default N=20, configurable)
5. WHEN the user hovers over a node, THE Network Viz_Binary SHALL highlight all edges connected to that node and display the base prime value and Support_Score in a tooltip
6. THE Network Viz_Binary SHALL animate the force-directed layout simulation, allowing the user to pause and drag individual nodes

### Requirement 12: Shared Analysis Infrastructure

**User Story:** As a developer, I want reusable analysis functions for basis vector construction, PCA, FFT, and co-occurrence computation, so that visualization binaries remain focused on rendering.

#### Acceptance Criteria

1. THE analysis module SHALL provide a function to construct a Basis_Vector (binary vector over top K base primes) for a given Decomposition
2. THE analysis module SHALL provide a function to compute PCA (mean-centering, covariance matrix, eigendecomposition) on a matrix of Basis_Vectors, returning principal components and explained variance ratios
3. THE analysis module SHALL provide a function to compute the co-occurrence matrix for base prime pairs across all Decompositions
4. THE analysis module SHALL provide a function to compute per-decomposition bit costs for the Compression_Signature visualization
5. THE analysis module SHALL provide a function to compute successive Euclidean distances between consecutive Basis_Vectors
6. WHEN computing any analysis over the full dataset, THE analysis module SHALL process data in a streaming fashion to stay within the 200MB memory budget
