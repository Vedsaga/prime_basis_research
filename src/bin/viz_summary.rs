//! Summary Statistics Tool
//!
//! Outputs key findings and metrics from the prime basis research.
//!
//! Run: cargo run --release --bin viz_summary [-- --cache path.bin]

use prime_basis_research::viz_common;
use prime_basis_research::analysis::compression_bits;

fn main() {
    let (db, stats) = viz_common::load_data();
    
    println!("\n=== PRIME BASIS RESEARCH SUMMARY ===\n");
    
    println!("Total Decompositions: {}", viz_common::format_num(db.decompositions.len()));
    println!("Unique Bases Used:    {}", viz_common::format_num(stats.unique_bases_used));
    
    println!("\n--- [1] COMPRESSION EFFICIENCY ---");
    // Calculate global compression stats
    let mut total_ratio = 0.0;
    let mut min_ratio = f64::MAX;
    let mut max_ratio = 0.0;
    let mut compressed_count = 0;
    
    for d in &db.decompositions {
        if d.prime < 2 { continue; }
        
        let basis_bits = compression_bits(d);
        let p_log2 = (d.prime as f64).log2();
        let ratio = basis_bits / p_log2;
        
        total_ratio += ratio;
        if ratio < min_ratio { min_ratio = ratio; }
        if ratio > max_ratio { max_ratio = ratio; }
        if ratio < 1.0 { compressed_count += 1; }
    }
    
    let avg_ratio = total_ratio / db.decompositions.len() as f64;
    let compress_pct = (compressed_count as f64 / db.decompositions.len() as f64) * 100.0;
    
    println!("Average Compression Ratio: {:.4} (Lower is better)", avg_ratio);
    println!("Min Ratio: {:.4}", min_ratio);
    println!("Max Ratio: {:.4}", max_ratio);
    println!("Percent Compressed (<1.0): {:.1}%", compress_pct);
    
    println!("\n--- [2] CORE BASIS PRIMES ---");
    println!("Top 5 most frequently used base primes:");
    for (i, (prime, count)) in stats.top_support.iter().take(5).enumerate() {
        let pct = (*count as f64 / db.decompositions.len() as f64) * 100.0;
        println!("  {}. Prime {:<5} | Used in {:.1}% of numbers", i+1, prime, pct);
    }
    
    println!("\n--- [3] GAP STATISTICS ---");
    let mut max_gap = 0;
    let mut gap_sum = 0;
    for d in &db.decompositions {
        if d.gap > max_gap { max_gap = d.gap; }
        gap_sum += d.gap;
    }
    let avg_gap = gap_sum as f64 / db.decompositions.len() as f64;
    println!("Average Gap Size: {:.2}", avg_gap);
    println!("Max Gap Encountered: {}", max_gap);
    
    println!("\n====================================\n");
}
