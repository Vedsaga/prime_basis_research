pub mod analysis;

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ─── FFI bindings to libprimesieve ──────────────────────────────────────────

// From primesieve.h: enum type identifiers
const UINT64_PRIMES: i32 = 13;

extern "C" {
    /// Generate the first `n` primes >= `start`.
    /// Returns a heap-allocated array (caller must free with `primesieve_free`).
    fn primesieve_generate_n_primes(n: u64, start: u64, type_: i32) -> *mut u64;

    /// Free an array allocated by primesieve.
    fn primesieve_free(primes: *mut u64);
}

/// Generate the next `count` primes strictly greater than `after`.
/// Uses the primesieve C library for maximum performance.
pub fn generate_next_primes(after: u64, count: usize) -> Vec<u64> {
    if count == 0 {
        return vec![];
    }
    unsafe {
        let ptr = primesieve_generate_n_primes(count as u64, after + 1, UINT64_PRIMES);
        if ptr.is_null() {
            panic!("primesieve_generate_n_primes returned NULL");
        }
        let slice = std::slice::from_raw_parts(ptr, count);
        let result = slice.to_vec();
        primesieve_free(ptr);
        result
    }
}

// ─── Data model ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeDecomposition {
    pub prime: u64,
    pub prev_prime: u64,
    pub gap: u64,
    /// Distinct primes (from the known list) that sum to `gap`.
    /// Sorted descending (largest first).
    pub components: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeDatabase {
    /// Ordered list of known primes: [1, 2, 3, 5, 7, 11, ...]
    /// Note: 1 is included as the base element.
    pub primes: Vec<u64>,
    /// One decomposition per prime starting from prime=2.
    pub decompositions: Vec<PrimeDecomposition>,
}

impl PrimeDatabase {
    /// Create a fresh database with just the bootstrap: primes = [1, 2],
    /// and the decomposition 2 = 1 + 1.
    pub fn bootstrap() -> Self {
        PrimeDatabase {
            primes: vec![1, 2],
            decompositions: vec![PrimeDecomposition {
                prime: 2,
                prev_prime: 1,
                gap: 1,
                components: vec![1],
            }],
        }
    }

    /// Load from a bincode file, or bootstrap if the file doesn't exist.
    pub fn load(path: &Path) -> Self {
        if path.exists() {
            let data = fs::read(path).expect("Failed to read database file");
            bincode::deserialize(&data).expect("Failed to deserialize database")
        } else {
            Self::bootstrap()
        }
    }

    /// Save to a bincode file.
    pub fn save(&self, path: &Path) {
        let data = bincode::serialize(self).expect("Failed to serialize database");
        fs::write(path, data).expect("Failed to write database file");
    }

    /// The largest prime currently known.
    pub fn last_prime(&self) -> u64 {
        *self.primes.last().unwrap()
    }

    /// Total number of primes (including 1).
    pub fn count(&self) -> usize {
        self.primes.len()
    }

    /// Generate the next `n` prime decompositions and append to the database.
    pub fn generate(&mut self, n: usize) {
        if n == 0 {
            return;
        }

        let last = self.last_prime();
        let new_primes = generate_next_primes(last, n);

        for &p in &new_primes {
            let prev = self.last_prime();
            let gap = p - prev;
            let components = decompose_gap(gap, &self.primes);
            self.decompositions.push(PrimeDecomposition {
                prime: p,
                prev_prime: prev,
                gap,
                components,
            });
            self.primes.push(p);
        }
    }
}

// ─── Gap decomposition ─────────────────────────────────────────────────────

/// Decompose `gap` into a sum of distinct primes from `known_primes`.
/// Strategy: greedy from largest to smallest. If greedy fails, falls back
/// to backtracking search.
///
/// `known_primes` must be sorted ascending and must include 1.
pub fn decompose_gap(gap: u64, known_primes: &[u64]) -> Vec<u64> {
    // Filter to primes <= gap, iterate from largest to smallest
    let candidates: Vec<u64> = known_primes.iter().copied().filter(|&p| p <= gap).collect();

    // Try greedy first (fast path)
    if let Some(result) = greedy_decompose(gap, &candidates) {
        return result;
    }

    // Fallback: backtracking (guaranteed to find a solution if one exists)
    if let Some(result) = backtrack_decompose(gap, &candidates) {
        return result;
    }

    panic!(
        "Cannot decompose gap {} using known primes {:?}",
        gap, known_primes
    );
}

/// Greedy: pick the largest prime ≤ remaining, subtract, repeat.
fn greedy_decompose(gap: u64, candidates: &[u64]) -> Option<Vec<u64>> {
    let mut remaining = gap;
    let mut result = Vec::new();

    // Iterate from largest to smallest
    for &p in candidates.iter().rev() {
        if p <= remaining {
            result.push(p);
            remaining -= p;
            if remaining == 0 {
                return Some(result);
            }
        }
    }
    None // Greedy didn't work
}

/// Backtracking: exhaustive search for a subset that sums to gap.
/// Returns the first solution found (smallest cardinality via BFS-like ordering).
fn backtrack_decompose(gap: u64, candidates: &[u64]) -> Option<Vec<u64>> {
    // Try subsets of increasing size
    for size in 1..=candidates.len() {
        let mut result = Vec::with_capacity(size);
        if backtrack_helper(gap, candidates, candidates.len(), size, &mut result) {
            result.sort_unstable_by(|a, b| b.cmp(a)); // descending
            return Some(result);
        }
    }
    None
}

fn backtrack_helper(
    remaining: u64,
    candidates: &[u64],
    max_idx: usize,
    slots_left: usize,
    result: &mut Vec<u64>,
) -> bool {
    if remaining == 0 && slots_left == 0 {
        return true;
    }
    if slots_left == 0 || max_idx == 0 {
        return false;
    }

    // Iterate from largest index downward for better pruning
    for i in (0..max_idx).rev() {
        let p = candidates[i];
        if p > remaining {
            continue;
        }
        result.push(p);
        if backtrack_helper(remaining - p, candidates, i, slots_left - 1, result) {
            return true;
        }
        result.pop();
    }
    false
}

// ─── Display formatting ─────────────────────────────────────────────────────

impl PrimeDecomposition {
    /// Format as "p = prev + a + b + ..."
    pub fn display(&self) -> String {
        let parts: Vec<String> = std::iter::once(self.prev_prime.to_string())
            .chain(self.components.iter().map(|c| c.to_string()))
            .collect();
        format!("{} = {}", self.prime, parts.join(" + "))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_next_primes() {
        let primes = generate_next_primes(1, 5);
        assert_eq!(primes, vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_generate_next_primes_after_10() {
        let primes = generate_next_primes(10, 3);
        assert_eq!(primes, vec![11, 13, 17]);
    }

    #[test]
    fn test_decompose_gap_simple() {
        let known = vec![1, 2, 3, 5];
        // gap=1 → [1]
        assert_eq!(decompose_gap(1, &known), vec![1]);
        // gap=2 → [2]
        assert_eq!(decompose_gap(2, &known), vec![2]);
        // gap=4 → greedy: 3+1
        assert_eq!(decompose_gap(4, &known), vec![3, 1]);
        // gap=6 → greedy: 5+1
        assert_eq!(decompose_gap(6, &known), vec![5, 1]);
    }

    #[test]
    fn test_decompose_gap_needs_backtrack() {
        // gap=8, known = [1, 2, 3, 5, 7]
        // greedy: 7+1=8 ✓
        let known = vec![1, 2, 3, 5, 7];
        assert_eq!(decompose_gap(8, &known), vec![7, 1]);
    }

    #[test]
    fn test_user_example_decompositions() {
        // Verify the user's example decompositions
        let mut db = PrimeDatabase::bootstrap();
        // After bootstrap: primes=[1,2], decompositions=[{2=1+1}]

        // Generate primes 3..43
        db.generate(13); // 3,5,7,11,13,17,19,23,29,31,37,41,43

        // Check specific decompositions from the user's example:
        // 3 = 2 + 1
        assert_eq!(db.decompositions[1].prime, 3);
        assert_eq!(db.decompositions[1].prev_prime, 2);
        assert_eq!(db.decompositions[1].components, vec![1]);

        // 5 = 3 + 2
        assert_eq!(db.decompositions[2].prime, 5);
        assert_eq!(db.decompositions[2].prev_prime, 3);
        assert_eq!(db.decompositions[2].components, vec![2]);

        // 7 = 5 + 2
        assert_eq!(db.decompositions[3].prime, 7);
        assert_eq!(db.decompositions[3].prev_prime, 5);
        assert_eq!(db.decompositions[3].components, vec![2]);

        // 11 = 7 + 3 + 1
        assert_eq!(db.decompositions[4].prime, 11);
        assert_eq!(db.decompositions[4].prev_prime, 7);
        assert_eq!(db.decompositions[4].components, vec![3, 1]);

        // 13 = 11 + 2
        assert_eq!(db.decompositions[5].prime, 13);
        assert_eq!(db.decompositions[5].prev_prime, 11);
        assert_eq!(db.decompositions[5].components, vec![2]);

        // 29 = 23 + 5 + 1
        assert_eq!(db.decompositions[9].prime, 29);
        assert_eq!(db.decompositions[9].prev_prime, 23);
        assert_eq!(db.decompositions[9].components, vec![5, 1]);
    }

    #[test]
    fn test_display_format() {
        let d = PrimeDecomposition {
            prime: 29,
            prev_prime: 23,
            gap: 6,
            components: vec![5, 1],
        };
        assert_eq!(d.display(), "29 = 23 + 5 + 1");
    }

    #[test]
    fn test_database_incremental() {
        let mut db = PrimeDatabase::bootstrap();
        assert_eq!(db.count(), 2); // [1, 2]

        db.generate(5);
        assert_eq!(db.count(), 7); // [1, 2, 3, 5, 7, 11, 13]
        assert_eq!(db.last_prime(), 13);

        db.generate(3);
        assert_eq!(db.count(), 10); // + 17, 19, 23
        assert_eq!(db.last_prime(), 23);
    }
}
