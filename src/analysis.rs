//! Phase 0: Data infrastructure for prime basis analysis.
//!
//! Provides streaming statistics computation over a PrimeDatabase,
//! precomputed statistics caching, and sampling/windowing utilities.
//! All statistics are computed over the FULL dataset — windowing is
//! only for rendering, never for computation.

use crate::{PrimeDatabase, PrimeDecomposition};
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
}
