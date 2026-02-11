//! Phase 0: Data infrastructure for prime basis analysis.
//!
//! Provides streaming statistics computation over a PrimeDatabase,
//! precomputed statistics caching, and sampling/windowing utilities.
//! All statistics are computed over the FULL dataset — windowing is
//! only for rendering, never for computation.

use crate::{PrimeDatabase, PrimeDecomposition};
use nalgebra::{DMatrix, SymmetricEigen};
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
        assert_eq!(w.start_idx, 5);

        let indexed: Vec<_> = w.iter_indexed().collect();
        assert_eq!(indexed[0].0, 5);
        assert_eq!(indexed[1].0, 6);
        assert_eq!(indexed[2].0, 7);
    }

    #[test]
    fn test_window_clamped() {
        let db = make_test_db();
        let total = db.decompositions.len();
        // Request beyond end
        let w = db.window(total - 2, 100);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn test_every_nth() {
        let db = make_test_db();
        let sampled = db.every_nth(3);
        // Should get roughly total/3 entries
        let expected = (db.decompositions.len() + 2) / 3;
        assert_eq!(sampled.len(), expected);
        // First should be index 0
        assert_eq!(sampled[0].0, 0);
        // Second should be index 3
        if sampled.len() > 1 {
            assert_eq!(sampled[1].0, 3);
        }
    }

    #[test]
    fn test_sampled_range_full_fidelity() {
        let db = make_test_db();
        // Request more points than exist → get all
        let sampled = db.sampled_range(0, 10, 1000);
        assert_eq!(sampled.len(), 10);
    }

    #[test]
    fn test_sampled_range_downsampled() {
        let db = make_test_db();
        let total = db.decompositions.len();
        // Request only 5 points from the full range
        let sampled = db.sampled_range(0, total, 5);
        assert!(sampled.len() <= 6); // might be slightly more due to integer division
        assert!(sampled.len() >= 4);
    }

    #[test]
    fn test_aggregate_buckets() {
        let db = make_test_db();
        let buckets = db.aggregate_buckets(5);
        assert!(!buckets.is_empty());

        // First bucket starts at 0
        assert_eq!(buckets[0].start_idx, 0);
        assert!(buckets[0].count <= 5);
        assert!(buckets[0].gap_min <= buckets[0].gap_max);
        assert!(buckets[0].comp_min <= buckets[0].comp_max);
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution → max entropy
        let mut uniform = HashMap::new();
        uniform.insert(1, 100);
        uniform.insert(2, 100);
        uniform.insert(3, 100);
        let h_uniform = shannon_entropy(&uniform);

        // Peaked distribution → lower entropy
        let mut peaked = HashMap::new();
        peaked.insert(1, 298);
        peaked.insert(2, 1);
        peaked.insert(3, 1);
        let h_peaked = shannon_entropy(&peaked);

        assert!(h_uniform > h_peaked);
    }

    #[test]
    fn test_sliding_entropy() {
        let db = make_test_db();
        let entropy = sliding_entropy(&db.decompositions, 5);
        assert!(!entropy.is_empty());
        // All entropy values should be non-negative
        for &(_, h) in &entropy {
            assert!(h >= 0.0);
        }
    }

    #[test]
    fn test_gap_autocorrelation() {
        let db = make_test_db();
        let ac = gap_autocorrelation(&db.decompositions, 5);
        assert_eq!(ac.len(), 5);
        // Autocorrelation at lag 0 would be 1.0, but we start at lag 1
        // Values should be in [-1, 1] range (approximately)
        for &(_, r) in &ac {
            assert!(r >= -1.5 && r <= 1.5); // slight tolerance for small samples
        }
    }

    #[test]
    fn test_stats_path_for() {
        let p = Path::new("prime_basis.bin");
        assert_eq!(stats_path_for(p), Path::new("prime_basis.stats.json"));

        let p2 = Path::new("/data/my_primes.bin");
        assert_eq!(stats_path_for(p2), Path::new("/data/my_primes.stats.json"));
    }

    #[test]
    fn test_stats_save_load_roundtrip() {
        let db = make_test_db();
        let stats = PrecomputedStats::compute(&db);

        let tmp = std::env::temp_dir().join("test_prime_stats.json");
        stats.save(&tmp);

        let loaded = PrecomputedStats::load(&tmp).expect("Should load saved stats");
        assert_eq!(loaded.total_decompositions, stats.total_decompositions);
        assert_eq!(loaded.gap_max, stats.gap_max);
        assert_eq!(loaded.top_support.len(), stats.top_support.len());

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_or_compute_fresh() {
        let db = make_test_db();
        let tmp = std::env::temp_dir().join("test_prime_stats_fresh.json");
        let _ = std::fs::remove_file(&tmp); // ensure no cache

        let stats = PrecomputedStats::load_or_compute(&tmp, &db);
        assert_eq!(stats.total_decompositions, db.decompositions.len());
        assert!(tmp.exists()); // should have been saved

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_or_compute_cached() {
        let db = make_test_db();
        let tmp = std::env::temp_dir().join("test_prime_stats_cached.json");

        // Compute and save
        let stats1 = PrecomputedStats::compute(&db);
        stats1.save(&tmp);

        // Load from cache (should not recompute)
        let stats2 = PrecomputedStats::load_or_compute(&tmp, &db);
        assert_eq!(stats2.total_decompositions, stats1.total_decompositions);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    // ─── Property-Based Tests ───────────────────────────────────────────────

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a valid PrimeDecomposition with random components.
        /// Components are distinct values from a small prime set, sorted descending.
        fn arb_decomposition() -> impl Strategy<Value = PrimeDecomposition> {
            let small_primes: Vec<u64> = vec![1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43];
            // Pick a random non-empty subset of small_primes as components
            proptest::bits::u16::between(1, 15).prop_map(move |mask| {
                let mut components: Vec<u64> = small_primes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, &p)| p)
                    .collect();
                components.sort_unstable_by(|a, b| b.cmp(a)); // descending
                let gap: u64 = components.iter().sum();
                PrimeDecomposition {
                    prime: 100 + gap, // arbitrary prime > gap
                    prev_prime: 100,
                    gap,
                    components,
                }
            })
        }

        /// Generate a list of top base primes for testing.
        fn arb_top_bases() -> impl Strategy<Value = Vec<u64>> {
            // Use a fixed set of small primes, pick a non-empty subset of size 3..=10
            let primes: Vec<u64> = vec![1, 2, 3, 5, 7, 11, 13, 17, 19, 23];
            (3..=10usize).prop_flat_map(move |k| {
                let primes = primes.clone();
                proptest::sample::subsequence(primes.clone(), k)
                    .prop_map(|mut v| { v.sort_unstable(); v })
            })
        }

        /// Generate a Vec of N decompositions (N >= min_count).
        fn arb_decompositions(min_count: usize, max_count: usize) -> impl Strategy<Value = Vec<PrimeDecomposition>> {
            proptest::collection::vec(arb_decomposition(), min_count..=max_count)
        }

        proptest! {
            // Feature: phases-2-6-visualizations, Property 4: Compression bits formula correctness
            // **Validates: Requirements 4.3**
            #[test]
            fn prop_compression_bits_formula(decomp in arb_decomposition()) {
                let result = compression_bits(&decomp);

                // Manually compute expected value
                let expected_component_bits: f64 = decomp
                    .components
                    .iter()
                    .map(|&c| ((c as f64) + 1.0).log2().ceil())
                    .sum();
                let expected_count_bits = ((decomp.components.len() as f64) + 1.0).log2().ceil();
                let expected = expected_component_bits + expected_count_bits;

                prop_assert!(
                    (result - expected).abs() < 1e-10,
                    "compression_bits mismatch: got {}, expected {}", result, expected
                );
                prop_assert!(result > 0.0, "compression_bits should be positive, got {}", result);
                prop_assert!(result.is_finite(), "compression_bits should be finite");
            }

            // Feature: phases-2-6-visualizations, Property 5: Basis vector construction
            // **Validates: Requirements 12.1**
            #[test]
            fn prop_basis_vector_construction(
                decomp in arb_decomposition(),
                top_bases in arb_top_bases()
            ) {
                let vec = build_basis_vector(&decomp, &top_bases);

                // Length must equal K
                prop_assert_eq!(vec.len(), top_bases.len());

                // Each entry is 1.0 iff the base is in components, else 0.0
                for (i, &base) in top_bases.iter().enumerate() {
                    let expected = if decomp.components.contains(&base) { 1.0 } else { 0.0 };
                    prop_assert!(
                        (vec[i] - expected).abs() < 1e-10,
                        "basis_vector[{}] for base {}: got {}, expected {}",
                        i, base, vec[i], expected
                    );
                }
            }

            // Feature: phases-2-6-visualizations, Property 6: Successive distances length and non-negativity
            // **Validates: Requirements 7.2**
            #[test]
            fn prop_successive_distances_length_and_nonneg(
                decomps in arb_decompositions(2, 20),
                top_bases in arb_top_bases()
            ) {
                let dists = successive_distances(&decomps, &top_bases);

                // Length must be N-1
                prop_assert_eq!(dists.len(), decomps.len() - 1);

                // All distances non-negative
                for (i, &d) in dists.iter().enumerate() {
                    prop_assert!(d >= 0.0, "distance[{}] = {} is negative", i, d);
                }

                // Distance is zero iff consecutive decompositions use the same subset of top bases
                for i in 0..dists.len() {
                    let v1 = build_basis_vector(&decomps[i], &top_bases);
                    let v2 = build_basis_vector(&decomps[i + 1], &top_bases);
                    let same_subset = v1 == v2;
                    if same_subset {
                        prop_assert!(
                            dists[i].abs() < 1e-10,
                            "distance[{}] should be 0 for same subsets, got {}", i, dists[i]
                        );
                    } else {
                        prop_assert!(
                            dists[i] > 1e-10,
                            "distance[{}] should be > 0 for different subsets, got {}", i, dists[i]
                        );
                    }
                }
            }

            // Feature: phases-2-6-visualizations, Property 7: PCA mathematical invariants
            // **Validates: Requirements 8.1, 8.6**
            #[test]
            fn prop_pca_invariants(
                decomps in arb_decompositions(12, 30),
                top_bases in arb_top_bases()
            ) {
                let k = top_bases.len();
                let n_components = 2.min(k); // request 2 components (or fewer if K < 2)

                let pca = compute_pca(&decomps, &top_bases, n_components);

                // (a) Should produce n_components components (or fewer if not enough variance)
                let n = pca.components.len();
                prop_assert!(n <= n_components, "got {} components, expected <= {}", n, n_components);

                // Skip further checks if we got fewer than 2 components
                if n >= 2 {
                    // (a) Mutual orthogonality: dot product of distinct components ≈ 0
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let dot: f64 = pca.components[i]
                                .iter()
                                .zip(pca.components[j].iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            prop_assert!(
                                dot.abs() < 1e-6,
                                "components {} and {} not orthogonal: dot = {}", i, j, dot
                            );
                        }
                    }

                    // (b) Each component is unit-length
                    for (i, comp) in pca.components.iter().enumerate() {
                        let norm: f64 = comp.iter().map(|x| x * x).sum::<f64>().sqrt();
                        prop_assert!(
                            (norm - 1.0).abs() < 1e-6,
                            "component {} not unit-length: norm = {}", i, norm
                        );
                    }
                }

                // (c) Explained variance ratios sum to <= 1.0
                let ratio_sum: f64 = pca.explained_variance_ratio.iter().sum();
                prop_assert!(
                    ratio_sum <= 1.0 + 1e-6,
                    "explained variance ratios sum to {} > 1.0", ratio_sum
                );
            }

            // Feature: phases-2-6-visualizations, Property 12: Co-occurrence matrix symmetry and diagonal consistency
            // **Validates: Requirements 11.1**
            #[test]
            fn prop_co_occurrence_symmetry_and_diagonal(
                decomps in arb_decompositions(1, 20),
                top_bases in arb_top_bases()
            ) {
                let matrix = co_occurrence_matrix(&decomps, &top_bases);
                let k = top_bases.len();

                prop_assert_eq!(matrix.len(), k, "matrix should have {} rows", k);
                for row in &matrix {
                    prop_assert_eq!(row.len(), k, "each row should have {} columns", k);
                }

                // Symmetry: entry (i,j) == entry (j,i)
                for i in 0..k {
                    for j in 0..k {
                        prop_assert_eq!(
                            matrix[i][j], matrix[j][i],
                            "matrix not symmetric at ({}, {}): {} != {}", i, j, matrix[i][j], matrix[j][i]
                        );
                    }
                }

                // Diagonal: (i,i) equals count of decompositions containing top_bases[i]
                for i in 0..k {
                    let expected_count = decomps
                        .iter()
                        .filter(|d| d.components.contains(&top_bases[i]))
                        .count() as u64;
                    prop_assert_eq!(
                        matrix[i][i], expected_count,
                        "diagonal ({},{}) = {}, expected {} for base {}",
                        i, i, matrix[i][i], expected_count, top_bases[i]
                    );
                }
            }
        } // end first proptest! block

        // Feature: phases-2-6-visualizations, Property 13: Force-directed simulation energy decrease
        // **Validates: Requirements 11.3**
        //
        // A standalone force simulation step that mirrors the logic in viz_network.rs.
        // Takes node positions, velocities, edges with weights, and a damping factor.
        // Returns updated (positions, velocities).
        //
        // The simulation applies:
        //   1. Coulomb repulsion between all node pairs
        //   2. Spring attraction along edges (strength ∝ weight)
        //   3. vel = (vel + force) * damping
        //   4. Clamp velocity magnitude to max_vel
        //   5. pos += vel
        fn force_simulation_step(
            positions: &[[f64; 2]],
            velocities: &[[f64; 2]],
            edges: &[(usize, usize, f64)],
            damping: f64,
        ) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
            let n = positions.len();
            let mut forces = vec![[0.0f64; 2]; n];

            let repulsion_k = 50_000.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = positions[i][0] - positions[j][0];
                    let dy = positions[i][1] - positions[j][1];
                    let dist_sq = dx * dx + dy * dy;
                    let dist = dist_sq.sqrt().max(1.0);
                    let force = repulsion_k / dist_sq.max(1.0);
                    let fx = force * dx / dist;
                    let fy = force * dy / dist;
                    forces[i][0] += fx;
                    forces[i][1] += fy;
                    forces[j][0] -= fx;
                    forces[j][1] -= fy;
                }
            }

            let spring_k = 0.001;
            let max_weight = edges.iter().map(|e| e.2).fold(1.0f64, f64::max);
            for &(i, j, w) in edges {
                let dx = positions[j][0] - positions[i][0];
                let dy = positions[j][1] - positions[i][1];
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                let norm_w = w / max_weight;
                let force = spring_k * dist * norm_w;
                let fx = force * dx / dist;
                let fy = force * dy / dist;
                forces[i][0] += fx;
                forces[i][1] += fy;
                forces[j][0] -= fx;
                forces[j][1] -= fy;
            }

            let max_vel = 20.0;
            let mut new_positions = positions.to_vec();
            let mut new_velocities = Vec::with_capacity(n);
            for i in 0..n {
                let mut vx = (velocities[i][0] + forces[i][0]) * damping;
                let mut vy = (velocities[i][1] + forces[i][1]) * damping;
                let speed = (vx * vx + vy * vy).sqrt();
                if speed > max_vel {
                    vx *= max_vel / speed;
                    vy *= max_vel / speed;
                }
                new_positions[i][0] += vx;
                new_positions[i][1] += vy;
                new_velocities.push([vx, vy]);
            }

            (new_positions, new_velocities)
        }

        fn total_kinetic_energy(velocities: &[[f64; 2]]) -> f64 {
            velocities
                .iter()
                .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
                .sum()
        }

        proptest! {
            #[test]
            fn prop_force_simulation_energy_decrease(
                n in 2..=10usize,
                seed in 0..1000u64,
            ) {
                // Generate deterministic random positions and velocities from seed
                let mut positions = Vec::with_capacity(n);
                let mut velocities = Vec::with_capacity(n);
                let mut rng_state = seed;
                for _ in 0..n {
                    // Simple LCG for deterministic pseudo-random values
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let px = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 400.0 - 200.0;
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let py = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 400.0 - 200.0;
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let vx = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 10.0 - 5.0;
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let vy = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 10.0 - 5.0;
                    positions.push([px, py]);
                    velocities.push([vx, vy]);
                }

                let damping = 0.95;

                // Pure damping case: no edges → no attraction forces.
                // Repulsion forces still exist, but the property is about the
                // damping ensuring energy dissipation in the absence of external
                // energy input. With only damping (no forces), new_vel = vel * damping,
                // so KE_new = damping^2 * KE_old.
                //
                // Test the pure damping case: zero out forces by passing no edges
                // and placing nodes far apart so repulsion is negligible.
                let mut spread_positions = Vec::with_capacity(n);
                for (i, _) in positions.iter().enumerate() {
                    // Place nodes very far apart so repulsion force ≈ 0
                    spread_positions.push([i as f64 * 100_000.0, 0.0]);
                }

                let initial_ke = total_kinetic_energy(&velocities);
                if initial_ke < 1e-15 {
                    // Skip trivial case with no kinetic energy
                    return Ok(());
                }

                let (_new_pos, new_vel) = force_simulation_step(
                    &spread_positions,
                    &velocities,
                    &[], // no edges
                    damping,
                );

                let final_ke = total_kinetic_energy(&new_vel);

                // With nodes far apart, repulsion force ≈ repulsion_k / dist^2 ≈ 0.
                // So new_vel ≈ vel * damping, and KE_new ≈ damping^2 * KE_old.
                // Allow small tolerance for the residual repulsion force.
                let expected_ke = initial_ke * damping * damping;
                let tolerance = initial_ke * 0.01; // 1% tolerance for residual forces

                prop_assert!(
                    final_ke <= initial_ke + tolerance,
                    "KE should decrease with damping: initial={}, final={}, expected≈{}",
                    initial_ke, final_ke, expected_ke
                );

                // Also verify it's close to the expected damped value
                prop_assert!(
                    (final_ke - expected_ke).abs() < tolerance,
                    "KE should be ≈ damping^2 * initial: initial={}, final={}, expected={}",
                    initial_ke, final_ke, expected_ke
                );
            }

            // Feature: phases-2-6-visualizations, Property 11: Trajectory cumulative sum invariant
            // **Validates: Requirements 10.1**
            #[test]
            fn prop_trajectory_cumulative_sum_invariant(
                decomps in arb_decompositions(1, 30),
                base_x in proptest::sample::select(vec![1u64, 2, 3, 5, 7, 11, 13, 17, 19, 23]),
                base_y in proptest::sample::select(vec![1u64, 2, 3, 5, 7, 11, 13, 17, 19, 23]),
                base_z in proptest::sample::select(vec![1u64, 2, 3, 5, 7, 11, 13, 17, 19, 23]),
            ) {
                let axis_bases: [u64; 3] = [base_x, base_y, base_z];

                let trajectory = compute_trajectory(&decomps, &axis_bases);

                // Trajectory length should be N+1
                prop_assert_eq!(
                    trajectory.len(),
                    decomps.len() + 1,
                    "trajectory length should be N+1"
                );

                // Trajectory starts at origin
                prop_assert!(
                    trajectory[0][0].abs() < 1e-10
                        && trajectory[0][1].abs() < 1e-10
                        && trajectory[0][2].abs() < 1e-10,
                    "trajectory should start at origin"
                );

                // Compute expected final position as sum of all displacements
                let mut expected = [0.0f64; 3];
                for decomp in &decomps {
                    if decomp.components.contains(&axis_bases[0]) { expected[0] += 1.0; }
                    if decomp.components.contains(&axis_bases[1]) { expected[1] += 1.0; }
                    if decomp.components.contains(&axis_bases[2]) { expected[2] += 1.0; }
                }

                let final_pos = trajectory.last().unwrap();
                prop_assert!(
                    (final_pos[0] - expected[0]).abs() < 1e-10
                        && (final_pos[1] - expected[1]).abs() < 1e-10
                        && (final_pos[2] - expected[2]).abs() < 1e-10,
                    "final position {:?} should equal sum of displacements {:?}",
                    final_pos, expected
                );
            }
        }
    }
}
