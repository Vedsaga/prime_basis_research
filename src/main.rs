use prime_basis_research::{all_subset_sums, find_min_subset, find_minimal_basis, sieve};
use itertools::Itertools;
use serde::Serialize;
use std::fs::File;
use std::io::Write;

#[derive(Serialize)]
struct DecompositionRecord {
    index: usize,
    prime: usize,
    prev_prime: usize,
    gap: usize,
    basis_subset: Vec<usize>,
}

fn get_prime(primes: &[usize], index: usize, one_indexed: bool) -> usize {
    let i = if one_indexed { index - 1 } else { index };
    if i >= primes.len() {
        panic!("Index {} out of range (have {} primes)", index, primes.len());
    }
    primes[i]
}

fn decompose_prime(
    primes: &[usize],
    basis: &std::collections::HashSet<usize>,
    index: usize,
    one_indexed: bool,
) -> (usize, usize, Vec<usize>) {
    let i = if one_indexed { index - 1 } else { index };
    if i == 0 {
        panic!("No previous prime for index 1");
    }
    let prev = primes[i - 1];
    let curr = primes[i];
    let gap = curr - prev;
    let subset = find_min_subset(gap, basis).expect("Basis should cover gap");
    (prev, curr, subset)
}

fn main() -> std::io::Result<()> {
    // ─── Build once ───────────────────────────────────────────────────────────────
    let prime_limit = 10_000;
    let primes = sieve(prime_limit);
    
    // Calculate gaps
    let mut unique_gaps = std::collections::HashSet::new();
    for i in 1..primes.len() {
        unique_gaps.insert(primes[i] - primes[i - 1]);
    }
    let unique_gaps_vec: Vec<usize> = unique_gaps.into_iter().sorted().collect();
    
    // Find Basis
    let basis_vec = find_minimal_basis(&unique_gaps_vec);
    let basis_set: std::collections::HashSet<usize> = basis_vec.iter().cloned().collect();

    println!("============================================================");
    println!("  PRIME BASIS RESEARCH TOOL (Rust Port)");
    println!("  Primes up to: {:?}  |  Count: {}", prime_limit, primes.len()); // Format with separator manually or crate if needed, simple print here.
    println!("  Basis: {:?}", basis_vec);
    
    // Calculate max gap covered
    let all_sums = all_subset_sums(&basis_vec);
    let max_gap_covered = all_sums.last().unwrap_or(&0);
    println!("  Basis size: {}  |  Max gap covered: {}", basis_vec.len(), max_gap_covered);
    println!("============================================================");

    // ─── Demo: index lookup ───────────────────────────────────────────────────────
    println!("\n── Lookup by index ──");
    for idx in [1, 2, 5, 10, 100, 1000] {
        let p = get_prime(&primes, idx, true);
        println!("  prime({:>5}) = {}", idx, p);
    }

    // ─── Demo: decompose by index ─────────────────────────────────────────────────
    println!("\n── Decomposition by index ──");
    let mut decomposition_records = Vec::new();

    // Start from idx 2 (first prime is 2, second is 3, gap starts there)
    // The Python script decomposes loop range(2, 20) -> indices 2..19 (primes at index 2..19, i.e., 3rd prime onwards?)
    // Wait, Python `range(2, 20)` iterates 2, 3, ..., 19.
    // `get_prime(primes, index, one_indexed=True)`
    // idx=2 -> 2nd prime (3). Prev is 1st prime (2). Gap 1.
    for idx in 2..20 {
        let (prev, curr, subset) = decompose_prime(&primes, &basis_set, idx, true);
        let parts = subset.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" + ");
        println!("  prime({:>3}) = {:>6} = {:>6} + ({})", idx, curr, prev, parts);
        
        decomposition_records.push(DecompositionRecord {
            index: idx,
            prime: curr,
            prev_prime: prev,
            gap: curr - prev,
            basis_subset: subset,
        });
    }

    // Save ALL decompositions to JSON for graphing (up to limit)
    // Not just the demo ones, let's do all up to primes.len()
    println!("\nGenerating full decomposition data...");
    let mut full_records = Vec::new();
    for idx in 2..primes.len() + 1 { // 1-based index up to count
         let (prev, curr, subset) = decompose_prime(&primes, &basis_set, idx, true);
         full_records.push(DecompositionRecord {
            index: idx,
            prime: curr,
            prev_prime: prev,
            gap: curr - prev,
            basis_subset: subset,
        });
    }

    let json_file_path = "prime_decompositions.json";
    let file = File::create(json_file_path)?;
    serde_json::to_writer_pretty(file, &full_records)?;
    println!("  Saved {} records to {}", full_records.len(), json_file_path);


    // ─── Growth table ─────────────────────────────────────────────────────────────
    println!("\n── Basis growth with range ──");
    println!("  {:>8}  {:>8}  {:>7}  {:<40}  Size", "Limit", "#Primes", "MaxGap", "Basis");
    println!("  {:->8}  {:->8}  {:->7}  {:->40}  ----", "", "", "", "");
    
    for limit in [100, 500, 1000, 5000, 10000, 50000, 100000] {
        let p = sieve(limit);
        if p.len() < 2 { continue; }
        
        let mut g_set = std::collections::HashSet::new();
        for i in 1..p.len() {
            g_set.insert(p[i] - p[i-1]);
        }
        let g: Vec<usize> = g_set.into_iter().sorted().collect();
        let b = find_minimal_basis(&g);
        
        let max_g = g.last().unwrap_or(&0);
        // Format basis vector as string
        let b_str = format!("{:?}", b);
        
        println!("  {:>8}  {:>8}  {:>7}  {:<40}  {}", limit, p.len(), max_g, b_str, b.len());
    }

    // ─── Observation: powers of 2 ────────────────────────────────────────────────
    println!("\n── Alternative basis: Powers of 2 ──");
    println!("  Powers of 2 as basis give ALL subset sums without repetition:");
    
    let pow2_bases = vec![
        (4, vec![1, 2, 4, 8]),
        (5, vec![1, 2, 4, 8, 16]),
        (6, vec![1, 2, 4, 8, 16, 32]),
    ];

    for (size, pb) in pow2_bases {
        let sums = all_subset_sums(&pb);
        let max_s = sums.last().unwrap_or(&0);
        println!("  {:?}  →  covers 1..{}  (size {})", pb, max_s, size);
    }
    
    println!("\n  Comparison for max gap (primes up to 10,000):");
    // Recalculate basis for current `basis_vec` max coverage
    let current_max_gap = all_sums.last().unwrap_or(&0);
    println!("  Greedy consecutive basis: {:?} (size {}) covers 1..{}", basis_vec, basis_vec.len(), current_max_gap);
    
    let pow2_for36 = vec![1, 2, 4, 8, 16, 32];
    let p2_sums = all_subset_sums(&pow2_for36);
    println!("  Powers-of-2 basis:        {:?} (size {}) covers 1..{}", pow2_for36, pow2_for36.len(), p2_sums.last().unwrap_or(&0));
    println!("\n  → Powers of 2 use FEWER elements for the same max gap coverage!");

    Ok(())
}
