//! Phase 0: Data infrastructure for prime basis analysis.
//!
//! Provides streaming statistics computation over a PrimeDatabase,
//! precomputed statistics caching, and sampling/windowing utilities.
//! All statistics are computed over the FULL dataset — windowing is
//! only for rendering, never for computation.

use crate::{PrimeDatabase, PrimeDecomposition};
use nalgebra::{DMatrix, SymmetricEigen};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ─── Precomputed Statistics ─────────────────────────────────────────────────

/// Complete precomputed statistics over the full dataset.
/// Cached to disk so visualizations can start instantly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputedStats {
    /// Number of decompositions analyzed
    pub total_decompositions: usize,
    /// Number of primes in the database (including 1)
    pub total_primes: usize,
    /// Largest prime in the dataset
    pub largest_prime: u64,

    // ── Gap statistics ──
    pub gap_max: u64,
    pub gap_min: u64,
    pub gap_sum: u64,
    pub gap_mean: f64,
    /// gap_size → count of occurrences
    pub gap_histogram: HashMap<u64, usize>,

    // ── Component statistics ──
    pub component_count_max: usize,
    pub component_count_min: usize,
    pub component_count_mean: f64,
    /// num_components → count of decompositions with that many
    pub component_count_histogram: HashMap<usize, usize>,

    // ── Support scores: how often each base prime appears as a component ──
    /// base_prime → number of decompositions it appears in
    pub support_scores: HashMap<u64, usize>,
    /// Top 30 base primes by support score, sorted descending
    pub top_support: Vec<(u64, usize)>,

    // ── Unique base primes ever used as components ──
    pub unique_bases_used: usize,
    /// The largest base prime ever used as a component
    pub largest_base_used: u64,

    // ── First appearance: when each base prime first appears as a component ──
    /// base_prime → index of first decomposition using it
    pub first_appearance: HashMap<u64, usize>,

    // ── Decomposition uniqueness: does the same gap always decompose the same way? ──
    /// gap_size → number of distinct component sets observed
    pub gap_decomposition_variants: HashMap<u64, usize>,
}

impl PrecomputedStats {
    /// Compute all statistics from a PrimeDatabase in a single streaming pass.
    pub fn compute(db: &PrimeDatabase) -> Self {
        let decomps = &db.decompositions;
        let total_decompositions = decomps.len();
        let total_primes = db.primes.len();
        let largest_prime = db.last_prime();

        let mut gap_max: u64 = 0;
        let mut gap_min: u64 = u64::MAX;
        let mut gap_sum: u64 = 0;

        let mut comp_max: usize = 0;
        let mut comp_min: usize = usize::MAX;
        let mut comp_sum: usize = 0;

        let mut gap_histogram: HashMap<u64, usize> = HashMap::new();
        let mut component_count_histogram: HashMap<usize, usize> = HashMap::new();
        let mut support_scores: HashMap<u64, usize> = HashMap::new();
        let mut first_appearance: HashMap<u64, usize> = HashMap::new();

        // For decomposition uniqueness: gap → set of observed component signatures
        let mut gap_variants: HashMap<u64, Vec<Vec<u64>>> = HashMap::new();

        for (idx, d) in decomps.iter().enumerate() {
            // Gap stats
            gap_max = gap_max.max(d.gap);
            gap_min = gap_min.min(d.gap);
            gap_sum += d.gap;
            *gap_histogram.entry(d.gap).or_insert(0) += 1;

            // Component count stats
            let nc = d.components.len();
            comp_max = comp_max.max(nc);
            comp_min = comp_min.min(nc);
            comp_sum += nc;
            *component_count_histogram.entry(nc).or_insert(0) += 1;

            // Support scores + first appearance
            for &c in &d.components {
                *support_scores.entry(c).or_insert(0) += 1;
                first_appearance.entry(c).or_insert(idx);
            }

            // Decomposition uniqueness (track distinct component sets per gap)
            let variants = gap_variants.entry(d.gap).or_default();
            // Only store if we haven't seen this exact component set before
            // Components are already sorted descending, so direct comparison works
            if !variants.contains(&d.components) {
                variants.push(d.components.clone());
            }
        }

        // Build top support scores
        let mut top_support: Vec<(u64, usize)> = support_scores.iter().map(|(&k, &v)| (k, v)).collect();
        top_support.sort_by(|a, b| b.1.cmp(&a.1));
        top_support.truncate(30);

        // Unique bases
        let unique_bases_used = support_scores.len();
        let largest_base_used = support_scores.keys().copied().max().unwrap_or(0);

        // Gap decomposition variants count
        let gap_decomposition_variants: HashMap<u64, usize> =
            gap_variants.into_iter().map(|(k, v)| (k, v.len())).collect();

        let gap_mean = if total_decompositions > 0 {
            gap_sum as f64 / total_decompositions as f64
        } else {
            0.0
        };
        let component_count_mean = if total_decompositions > 0 {
            comp_sum as f64 / total_decompositions as f64
        } else {
            0.0
        };

        PrecomputedStats {
            total_decompositions,
            total_primes,
            largest_prime,
            gap_max,
            gap_min: if gap_min == u64::MAX { 0 } else { gap_min },
            gap_sum,
            gap_mean,
            gap_histogram,
            component_count_max: comp_max,
            component_count_min: if comp_min == usize::MAX { 0 } else { comp_min },
            component_count_mean,
            component_count_histogram,
            support_scores,
            top_support,
            unique_bases_used,
            largest_base_used,
            first_appearance,
            gap_decomposition_variants,
        }
    }

    /// Load cached stats from a JSON sidecar file.
    pub fn load(path: &Path) -> Option<Self> {
        if !path.exists() {
            return None;
        }
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Save stats to a JSON sidecar file.
    pub fn save(&self, path: &Path) {
        let data = serde_json::to_string_pretty(self).expect("Failed to serialize stats");
        std::fs::write(path, data).expect("Failed to write stats file");
    }

    /// Load from cache if valid, otherwise recompute from database.
    /// Stats are considered stale if the decomposition count doesn't match.
    pub fn load_or_compute(stats_path: &Path, db: &PrimeDatabase) -> Self {
        if let Some(cached) = Self::load(stats_path) {
            if cached.total_decompositions == db.decompositions.len() {
                return cached;
            }
        }
        let stats = Self::compute(db);
        stats.save(stats_path);
        stats
    }
}

// ─── Windowing & Sampling ───────────────────────────────────────────────────

/// A lightweight view into a range of decompositions.
/// Does NOT clone data — holds references.
pub struct DataWindow<'a> {
    pub start_idx: usize,
    pub decompositions: &'a [PrimeDecomposition],
}

impl<'a> DataWindow<'a> {
    /// Total entries in this window.
    pub fn len(&self) -> usize {
        self.decompositions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decompositions.is_empty()
    }

    /// Iterate with global indices.
    pub fn iter_indexed(&self) -> impl Iterator<Item = (usize, &PrimeDecomposition)> {
        self.decompositions
            .iter()
            .enumerate()
            .map(move |(i, d)| (self.start_idx + i, d))
    }
}

impl PrimeDatabase {
    /// Get a zero-copy window into the decompositions.
    pub fn window(&self, start: usize, count: usize) -> DataWindow<'_> {
        let start = start.min(self.decompositions.len());
        let end = (start + count).min(self.decompositions.len());
        DataWindow {
            start_idx: start,
            decompositions: &self.decompositions[start..end],
        }
    }

    /// Get every Nth decomposition (for rendering downsampling).
    /// Returns owned vec of references with their global indices.
    pub fn every_nth(&self, n: usize) -> Vec<(usize, &PrimeDecomposition)> {
        if n == 0 {
            return vec![];
        }
        self.decompositions
            .iter()
            .enumerate()
            .step_by(n)
            .collect()
    }

    /// Get decompositions in a range, sampled to at most `max_points` entries.
    /// If the range has fewer than max_points, returns all of them.
    /// This is the key rendering helper: full fidelity when zoomed in,
    /// automatic downsampling when zoomed out.
    pub fn sampled_range(
        &self,
        start: usize,
        end: usize,
        max_points: usize,
    ) -> Vec<(usize, &PrimeDecomposition)> {
        let start = start.min(self.decompositions.len());
        let end = end.min(self.decompositions.len());
        let range_len = end.saturating_sub(start);

        if range_len == 0 || max_points == 0 {
            return vec![];
        }

        let step = (range_len / max_points).max(1);
        self.decompositions[start..end]
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, d)| (start + i, d))
            .collect()
    }
}

// ─── Aggregated Buckets (for zoom-out rendering) ────────────────────────────

/// Aggregated statistics for a bucket of consecutive decompositions.
/// Used when rendering at zoom levels where individual points can't be shown.
#[derive(Debug, Clone)]
pub struct Bucket {
    /// Global index of the first decomposition in this bucket
    pub start_idx: usize,
    /// Number of decompositions in this bucket
    pub count: usize,
    /// Gap statistics within the bucket
    pub gap_min: u64,
    pub gap_max: u64,
    pub gap_mean: f64,
    /// Component count statistics within the bucket
    pub comp_min: usize,
    pub comp_max: usize,
    pub comp_mean: f64,
    /// Prime value at the start of the bucket
    pub prime_start: u64,
    /// Prime value at the end of the bucket
    pub prime_end: u64,
}

impl PrimeDatabase {
    /// Aggregate decompositions into fixed-size buckets.
    /// Returns one Bucket per `bucket_size` consecutive decompositions.
    /// This is the core of the "compute everything, render smartly" strategy.
    pub fn aggregate_buckets(&self, bucket_size: usize) -> Vec<Bucket> {
        if bucket_size == 0 || self.decompositions.is_empty() {
            return vec![];
        }

        self.decompositions
            .chunks(bucket_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut gap_min = u64::MAX;
                let mut gap_max = 0u64;
                let mut gap_sum = 0u64;
                let mut comp_min = usize::MAX;
                let mut comp_max = 0usize;
                let mut comp_sum = 0usize;

                for d in chunk {
                    gap_min = gap_min.min(d.gap);
                    gap_max = gap_max.max(d.gap);
                    gap_sum += d.gap;
                    let nc = d.components.len();
                    comp_min = comp_min.min(nc);
                    comp_max = comp_max.max(nc);
                    comp_sum += nc;
                }

                let n = chunk.len();
                Bucket {
                    start_idx: chunk_idx * bucket_size,
                    count: n,
                    gap_min,
                    gap_max,
                    gap_mean: gap_sum as f64 / n as f64,
                    comp_min,
                    comp_max,
                    comp_mean: comp_sum as f64 / n as f64,
                    prime_start: chunk.first().unwrap().prime,
                    prime_end: chunk.last().unwrap().prime,
                }
            })
            .collect()
    }
}

// ─── Streaming computation helpers ──────────────────────────────────────────

/// Compute Shannon entropy of a frequency distribution.
pub fn shannon_entropy(counts: &HashMap<u64, usize>) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    counts
        .values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.ln()
        })
        .sum()
}

/// Compute sliding-window entropy over the full decomposition sequence.
/// Returns (window_center_index, entropy) pairs.
pub fn sliding_entropy(
    decompositions: &[PrimeDecomposition],
    window_size: usize,
) -> Vec<(usize, f64)> {
    if decompositions.len() < window_size || window_size == 0 {
        return vec![];
    }

    let mut results = Vec::with_capacity(decompositions.len() - window_size + 1);

    // Initial window
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for d in &decompositions[..window_size] {
        for &c in &d.components {
            *counts.entry(c).or_insert(0) += 1;
        }
    }
    let center = window_size / 2;
    results.push((center, shannon_entropy(&counts)));

    // Slide the window
    for i in 1..=(decompositions.len() - window_size) {
        // Remove the element leaving the window
        let leaving = &decompositions[i - 1];
        for &c in &leaving.components {
            let entry = counts.get_mut(&c).unwrap();
            *entry -= 1;
            if *entry == 0 {
                counts.remove(&c);
            }
        }
        // Add the element entering the window
        let entering = &decompositions[i + window_size - 1];
        for &c in &entering.components {
            *counts.entry(c).or_insert(0) += 1;
        }
        results.push((i + center, shannon_entropy(&counts)));
    }

    results
}

/// Compute gap autocorrelation for lags 1..max_lag.
/// Returns (lag, correlation) pairs.
pub fn gap_autocorrelation(
    decompositions: &[PrimeDecomposition],
    max_lag: usize,
) -> Vec<(usize, f64)> {
    let n = decompositions.len();
    if n < 2 {
        return vec![];
    }

    let gaps: Vec<f64> = decompositions.iter().map(|d| d.gap as f64).collect();
    let mean = gaps.iter().sum::<f64>() / n as f64;
    let variance: f64 = gaps.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-12 {
        return vec![];
    }

    let max_lag = max_lag.min(n - 1);
    (1..=max_lag)
        .map(|lag| {
            let cov: f64 = (0..n - lag)
                .map(|i| (gaps[i] - mean) * (gaps[i + lag] - mean))
                .sum::<f64>()
                / (n - lag) as f64;
            (lag, cov / variance)
        })
        .collect()
}

/// Compute bit sequence autocorrelation for lags 1..max_lag.
/// Returns (lag, correlation) pairs.
pub fn bit_autocorrelation(
    bits: &[u8],
    max_lag: usize,
) -> Vec<(usize, f64)> {
    let n = bits.len();
    if n < 2 {
        return vec![];
    }

    let bits_f64: Vec<f64> = bits.iter().map(|&b| b as f64).collect();
    let mean = bits_f64.iter().sum::<f64>() / n as f64;
    let variance: f64 = bits_f64.iter().map(|&b| (b - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-12 {
        return vec![];
    }

    let max_lag = max_lag.min(n - 1);
    (1..=max_lag)
        .map(|lag| {
            // Pearson autocorrelation for lag k
            // R_k = E[(X_t - mu)(X_{t+k} - mu)] / sigma^2
            let numerator: f64 = (0..n - lag)
                .map(|i| (bits_f64[i] - mean) * (bits_f64[i + lag] - mean))
                .sum();
            let denominator: f64 = bits_f64.iter().map(|&b| (b - mean).powi(2)).sum();
            
            if denominator == 0.0 {
                (lag, 0.0)
            } else {
                (lag, numerator / denominator)
            }
        })
        .collect()
}

// ─── Advanced Metrics (Entopy, LZ, Spectral) ────────────────────────────────

/// Compute LZ76 Complexity of a binary sequence.
/// Based on Lempel-Ziv complexity measure (number of unique patterns).
pub fn lz76_complexity(sequence: &[u8]) -> usize {
    let n = sequence.len();
    if n == 0 {
        return 0;
    }
    
    // LZW-style dictionary approach for complexity estimation
    let mut dict: HashMap<Vec<u8>, usize> = HashMap::new();
    // Initialize dict with "0" and "1"
    dict.insert(vec![0], 0);
    dict.insert(vec![1], 1);
    let mut dict_next_code = 2;
    
    let mut w = vec![sequence[0]];
    let mut complexity = 0; // Number of phrases output
    
    for &k in &sequence[1..] {
        let mut wk = w.clone();
        wk.push(k);
        if dict.contains_key(&wk) {
            w = wk;
        } else {
            complexity += 1;
            dict.insert(wk, dict_next_code);
            dict_next_code += 1;
            w = vec![k];
        }
    }
    complexity += 1; // Last phrase
    complexity
}

/// Compute simple Markov transition matrix for binary sequence.
/// Returns [[P(0->0), P(0->1)], [P(1->0), P(1->1)]].
pub fn transition_matrix(bits: &[u8]) -> [[f64; 2]; 2] {
    let mut counts = [[0usize; 2]; 2];
    
    for window in bits.windows(2) {
        let from = window[0] as usize;
        let to = window[1] as usize;
        if from < 2 && to < 2 {
            counts[from][to] += 1;
        }
    }
    
    let mut probs = [[0.0; 2]; 2];
    for i in 0..2 {
        let total = (counts[i][0] + counts[i][1]) as f64;
        if total > 0.0 {
            probs[i][0] = counts[i][0] as f64 / total;
            probs[i][1] = counts[i][1] as f64 / total;
        }
    }
    probs
}

/// Estimate entropy rate from transition matrix.
/// H = \sum_i \pi_i H(X_n | X_{n-1} = i)
/// where \pi is stationary distribution.
pub fn entropy_rate(matrix: [[f64; 2]; 2]) -> f64 {
    // Stationary distribution [p0, p1]
    // p0 = p0*P00 + p1*P10
    // p1 = p0*P01 + p1*P11
    // p0 + p1 = 1
    
    // Algebraic solution for 2-state:
    // p0 = P10 / (P01 + P10)
    let p01 = matrix[0][1];
    let p10 = matrix[1][0];
    
    if p01 + p10 == 0.0 {
        return 0.0; // Avoid NaN, effectively uniform or static
    }
    
    let pi0 = p10 / (p01 + p10);
    let pi1 = p01 / (p01 + p10);
    
    let h_row = |probs: [f64; 2]| -> f64 {
        probs.iter().map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 }).sum()
    };
    
    pi0 * h_row(matrix[0]) + pi1 * h_row(matrix[1])
}

/// Compute Power Spectrum of a binary sequence using FFT.
/// Returns the magnitude squared of frequencies.
pub fn power_spectrum(bits: &[u8]) -> Vec<f64> {
    let n = bits.len();
    if n == 0 { return vec![]; }
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    
    let mut buffer: Vec<Complex<f64>> = bits.iter()
        .map(|&b| Complex::new(if b == 1 { 1.0 } else { -1.0 }, 0.0)) // Map 0->-1, 1->1 for better symmetry
        .collect();
        
    fft.process(&mut buffer);
    
    // Return magnitude squared (Power), normalized
    // Only return first half (Nyquist)
    buffer.iter().take(n / 2)
        .map(|c| c.norm_sqr() / (n as f64))
        .collect()
}


// ─── Stats display path helper ──────────────────────────────────────────────

/// Given a cache path like "prime_basis.bin", return "prime_basis.stats.json"
pub fn stats_path_for(cache_path: &Path) -> std::path::PathBuf {
    let stem = cache_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    cache_path.with_file_name(format!("{}.stats.json", stem))
}

// ─── Basis Vector & Compression ─────────────────────────────────────────────

/// Build a binary basis vector for a decomposition over the top-K base primes.
/// Returns a Vec<f64> of length K where entry i is 1.0 if top_bases[i]
/// is in the decomposition's components, else 0.0.
pub fn build_basis_vector(decomp: &PrimeDecomposition, top_bases: &[u64]) -> Vec<f64> {
    top_bases
        .iter()
        .map(|base| {
            if decomp.components.contains(base) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute per-decomposition bit cost.
/// bits = sum(ceil(log2(c + 1)) for each component c) + ceil(log2(count + 1))
/// where count is the number of components.
pub fn compression_bits(decomp: &PrimeDecomposition) -> f64 {
    let component_bits: f64 = decomp
        .components
        .iter()
        .map(|&c| ((c as f64) + 1.0).log2().ceil())
        .sum();
    let count_bits = ((decomp.components.len() as f64) + 1.0).log2().ceil();
    component_bits + count_bits
}

// ─── Successive Distances ───────────────────────────────────────────────────

/// Compute Euclidean distances between consecutive basis vectors.
/// Returns a Vec<f64> of length (decompositions.len() - 1).
pub fn successive_distances(
    decompositions: &[PrimeDecomposition],
    top_bases: &[u64],
) -> Vec<f64> {
    if decompositions.len() < 2 {
        return vec![];
    }

    let mut prev = build_basis_vector(&decompositions[0], top_bases);
    decompositions[1..]
        .iter()
        .map(|decomp| {
            let curr = build_basis_vector(decomp, top_bases);
            let dist = prev
                .iter()
                .zip(curr.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            prev = curr;
            dist
        })
        .collect()
}

// ─── PCA ─────────────────────────────────────────────────────────────────────

/// Result of Principal Component Analysis on basis vectors.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Top n_components eigenvectors, each of length K.
    pub components: Vec<Vec<f64>>,
    /// Eigenvalues (descending) for the top n_components.
    pub explained_variance: Vec<f64>,
    /// Eigenvalues normalized so they sum to ≤ 1.0.
    pub explained_variance_ratio: Vec<f64>,
    /// Mean basis vector (length K), subtracted during centering.
    pub mean: Vec<f64>,
}

/// Compute PCA on basis vectors built from decompositions over top_bases.
///
/// Algorithm:
/// 1. Build basis vectors for all decompositions
/// 2. Compute mean vector
/// 3. Center the data (subtract mean)
/// 4. Compute covariance matrix (X^T * X / (n-1))
/// 5. Eigendecompose via nalgebra's symmetric eigen
/// 6. Sort eigenvectors by descending eigenvalue
/// 7. Return top n_components
///
/// Edge cases: returns empty/partial results for empty inputs or
/// fewer decompositions than requested components.
pub fn compute_pca(
    decompositions: &[PrimeDecomposition],
    top_bases: &[u64],
    n_components: usize,
) -> PcaResult {
    let k = top_bases.len();
    let n = decompositions.len();

    // Edge case: no data or no dimensions
    if n == 0 || k == 0 || n_components == 0 {
        return PcaResult {
            components: vec![],
            explained_variance: vec![],
            explained_variance_ratio: vec![],
            mean: vec![0.0; k],
        };
    }

    // 1. Build basis vectors (n × k)
    let vectors: Vec<Vec<f64>> = decompositions
        .iter()
        .map(|d| build_basis_vector(d, top_bases))
        .collect();

    // 2. Compute mean vector
    let mut mean = vec![0.0; k];
    for v in &vectors {
        for (i, &val) in v.iter().enumerate() {
            mean[i] += val;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // 3. Center the data into an n × k matrix
    let mut centered = DMatrix::<f64>::zeros(n, k);
    for (row, v) in vectors.iter().enumerate() {
        for (col, &val) in v.iter().enumerate() {
            centered[(row, col)] = val - mean[col];
        }
    }

    // 4. Covariance matrix (k × k): X^T * X / (n-1)
    let divisor = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let cov = (centered.transpose() * &centered) / divisor;

    // 5. Eigendecompose
    let eigen = SymmetricEigen::new(cov);
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // 6. Sort by descending eigenvalue
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 7. Take top n_components (clamped to available)
    let n_comp = n_components.min(k);
    let total_variance: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();

    let mut components = Vec::with_capacity(n_comp);
    let mut explained_variance = Vec::with_capacity(n_comp);
    let mut explained_variance_ratio = Vec::with_capacity(n_comp);

    for &idx in indices.iter().take(n_comp) {
        let ev = eigenvalues[idx].max(0.0);
        explained_variance.push(ev);
        explained_variance_ratio.push(if total_variance > 0.0 {
            ev / total_variance
        } else {
            0.0
        });

        // Extract eigenvector column
        let component: Vec<f64> = (0..k).map(|row| eigenvectors[(row, idx)]).collect();
        components.push(component);
    }

    PcaResult {
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
    }
}

/// Project a single basis vector onto PCA components.
/// Subtracts the mean, then computes dot product with each component.
/// Returns a Vec<f64> of length n_components.
pub fn pca_project(basis_vector: &[f64], pca: &PcaResult) -> Vec<f64> {
    // Center the vector
    let centered: Vec<f64> = basis_vector
        .iter()
        .zip(pca.mean.iter())
        .map(|(&v, &m)| v - m)
        .collect();

    // Project onto each component
    pca.components
        .iter()
        .map(|comp| {
            centered
                .iter()
                .zip(comp.iter())
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect()
}

// ─── Co-occurrence Matrix ───────────────────────────────────────────────────

/// Compute co-occurrence matrix for base prime pairs.
/// Returns a symmetric K×K matrix where entry (i,j) counts how many
/// decompositions contain both top_bases[i] and top_bases[j].
/// Diagonal entry (i,i) counts how many decompositions contain top_bases[i].
pub fn co_occurrence_matrix(
    decompositions: &[PrimeDecomposition],
    top_bases: &[u64],
) -> Vec<Vec<u64>> {
    let k = top_bases.len();
    if k == 0 {
        return vec![];
    }

    let mut matrix = vec![vec![0u64; k]; k];

    for decomp in decompositions {
        // Find which top_bases are present in this decomposition
        let present: Vec<usize> = top_bases
            .iter()
            .enumerate()
            .filter(|(_, &base)| decomp.components.contains(&base))
            .map(|(i, _)| i)
            .collect();

        // Increment co-occurrence for every pair (including diagonal)
        for &i in &present {
            for &j in &present {
                if j >= i {
                    matrix[i][j] += 1;
                    if i != j {
                        matrix[j][i] += 1;
                    }
                }
            }
        }
    }

    matrix
}

// ─── Modular Arithmetic Analysis ────────────────────────────────────────────

/// Compute the distribution of prime gaps modulo a given modulus (e.g., 6 or 30).
/// Returns a map of residue -> count.
pub fn gap_mod_counts(decompositions: &[PrimeDecomposition], modulus: u64) -> HashMap<u64, usize> {
    let mut counts = HashMap::new();
    for d in decompositions {
        *counts.entry(d.gap % modulus).or_insert(0) += 1;
    }
    counts
}

/// Compute the distribution of prime residues modulo a given modulus.
/// Returns a map of residue -> count.
pub fn residue_distribution(decompositions: &[PrimeDecomposition], modulus: u64) -> HashMap<u64, usize> {
    let mut counts = HashMap::new();
    for d in decompositions {
        *counts.entry(d.prime % modulus).or_insert(0) += 1;
    }
    counts
}

// ─── Trajectory ─────────────────────────────────────────────────────────────

/// Compute cumulative 3D trajectory from decompositions and axis base prime mapping.
/// For each decomposition, displacement = (1.0 if base_x present, 1.0 if base_y present, 1.0 if base_z present).
/// Returns trajectory of length N+1 (starting at origin).
pub fn compute_trajectory(
    decompositions: &[PrimeDecomposition],
    axis_bases: &[u64; 3],
) -> Vec<[f64; 3]> {
    let mut trajectory = Vec::with_capacity(decompositions.len() + 1);
    let mut pos = [0.0f64; 3];
    trajectory.push(pos);

    for decomp in decompositions {
        let dx = if decomp.components.contains(&axis_bases[0]) { 1.0 } else { 0.0 };
        let dy = if decomp.components.contains(&axis_bases[1]) { 1.0 } else { 0.0 };
        let dz = if decomp.components.contains(&axis_bases[2]) { 1.0 } else { 0.0 };
        pos[0] += dx;
        pos[1] += dy;
        pos[2] += dz;
        trajectory.push(pos);
    }

    trajectory
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrimeDatabase;

    fn make_test_db() -> PrimeDatabase {
        let mut db = PrimeDatabase::bootstrap();
        db.generate(20); // primes up to ~73
        db
    }

    #[test]
    fn test_precomputed_stats_basic() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        assert_eq!(stats.total_decompositions, db.decompositions.len());
        assert_eq!(stats.total_primes, db.primes.len());
        assert!(stats.gap_max > 0);
        assert!(stats.gap_min > 0);
        assert!(stats.gap_mean > 0.0);
        assert!(stats.component_count_max >= 1);
        assert!(!stats.support_scores.is_empty());
        assert!(!stats.top_support.is_empty());
        // 1 should be the most-used base prime
        assert_eq!(stats.top_support[0].0, 1);
    }

    #[test]
    fn test_precomputed_stats_support_scores() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        // Every decomposition uses at least one component,
        // so total support should be >= total decompositions
        let total_support: usize = stats.support_scores.values().sum();
        assert!(total_support >= stats.total_decompositions);
    }

    #[test]
    fn test_precomputed_stats_gap_histogram() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        // Sum of histogram counts should equal total decompositions
        let hist_sum: usize = stats.gap_histogram.values().sum();
        assert_eq!(hist_sum, stats.total_decompositions);
    }

    #[test]
    fn test_precomputed_stats_component_histogram() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        let hist_sum: usize = stats.component_count_histogram.values().sum();
        assert_eq!(hist_sum, stats.total_decompositions);
    }

    #[test]
    fn test_precomputed_stats_first_appearance() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        // Base prime 1 should first appear at index 0 (decomposition of 2 = 1+1)
        assert_eq!(stats.first_appearance.get(&1), Some(&0));
    }

    #[test]
    fn test_precomputed_stats_decomposition_uniqueness() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        // Every gap that appears should have at least 1 variant
        for &count in stats.gap_decomposition_variants.values() {
            assert!(count >= 1);
        }
    }

    #[test]
    fn test_window() {
        let db = make_test_db();
        let w = db.window(5, 3);
        assert_eq!(w.len(), 3);
        assert_eq!(w.decompositions[0].prime, db.decompositions[5].prime);
    }
    
    #[test]
    fn test_lz76() {
        let seq = vec![0, 0, 0, 0, 0, 0];
        // 0 -> (0,0) in dict. 
        // LZW: 0 (new 0), 0 (found), 00 (new), 00 (found)...
        // Correct trivial complexity is low.
        let c = lz76_complexity(&seq);
        assert!(c < 4);
        
        let seq2 = vec![0, 1, 0, 1, 0, 1];
        let c2 = lz76_complexity(&seq2);
        assert!(c2 > 1);
    }
}
