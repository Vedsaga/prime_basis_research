//! Shared utilities for all visualization binaries.
//!
//! Each visualization binary (src/bin/viz_*.rs) imports this module
//! for common data loading, formatting, and precomputation helpers.

use crate::analysis::{self, PrecomputedStats};
use crate::PrimeDatabase;
use std::path::PathBuf;

/// Parse --cache argument from command line args, defaulting to "prime_basis.bin".
pub fn parse_cache_path() -> PathBuf {
    std::env::args()
        .skip_while(|a| a != "--cache")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("prime_basis.bin"))
}

/// Load database and precomputed stats, printing progress to stdout.
pub fn load_data() -> (PrimeDatabase, PrecomputedStats) {
    let cache_path = parse_cache_path();
    println!("Loading database from {:?}...", cache_path);
    let db = PrimeDatabase::load(&cache_path);
    println!("Loaded {} decompositions.", db.decompositions.len());

    let stats_path = analysis::stats_path_for(&cache_path);
    println!("Loading/computing statistics...");
    let stats = PrecomputedStats::load_or_compute(&stats_path, &db);
    println!("Ready.");
    (db, stats)
}

/// Format a number with comma separators: 1000000 → "1,000,000"
pub fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Compute running average over a value sequence.
/// Returns ~2000 evenly-spaced (index, avg) points for smooth line rendering.
pub fn running_average(
    values: impl Iterator<Item = f64>,
    total: usize,
    window: usize,
) -> Vec<[f64; 2]> {
    let vals: Vec<f64> = values.collect();
    if vals.len() < window {
        return vec![];
    }

    let step = (total / 2000).max(1);
    let mut result = Vec::with_capacity(total / step);
    let mut sum: f64 = vals[..window].iter().sum();

    for i in window..vals.len() {
        if (i - window) % step == 0 {
            result.push([(i - window / 2) as f64, sum / window as f64]);
        }
        sum += vals[i] - vals[i - window];
    }
    result
}
