//! Resonance Analysis CLI
//! 
//! Performs specific falsification tests for physical resonance in prime gaps
//! and investigates arithmetic rhythms (mod 6, mod 30).

use clap::Parser;
use prime_basis_research::{
    analysis::{gap_mod_counts, residue_distribution, PrecomputedStats},
    spectral::compute_fft,
    PrimeDatabase,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of primes to analyze (default: all in cache)
    #[arg(short, long)]
    count: Option<usize>,

    /// Number of shuffle iterations for null hypothesis testing
    #[arg(short, long, default_value_t = 100)]
    shuffles: usize,
}

fn main() {
    let args = Args::parse();

    println!("Loading Prime Database...");
    // Attempt to load from default cache path
    let cache_path = Path::new("prime_basis.bin");
    let db = if cache_path.exists() {
        PrimeDatabase::load(cache_path)
    } else {
        println!("Cache not found, bootstrapping...");
        let mut db = PrimeDatabase::bootstrap();
        db.generate(args.count.unwrap_or(1_000_000));
        db
    };

    let total = db.decompositions.len();
    let count = args.count.unwrap_or(total).min(total);
    let slice = &db.decompositions[..count];

    println!("Analyzing first {} decompositions.", count);

    // ─── 1. Arithmetic Rhythms (The "Likely" Structure) ───
    println!("\n=== Arithmetic Rhythms ===");
    
    // Mod 6 Analysis
    let gap_mod6 = gap_mod_counts(slice, 6);
    println!("\n[Gap Mod 6 Distribution]");
    print_dist(&gap_mod6, count);

    // Mod 30 Analysis
    let gap_mod30 = gap_mod_counts(slice, 30);
    println!("\n[Gap Mod 30 Distribution (Top 5)]");
    print_top_dist(&gap_mod30, count, 5);

    // Residue Mod 6
    let res_mod6 = residue_distribution(slice, 6);
    println!("\n[Prime Residue Mod 6 Distribution]");
    print_dist(&res_mod6, count);

    // ─── 2. Resonance Falsification Test (The "Unlikely" Structure) ───
    println!("\n=== Resonance Falsification Test (FFT) ===");
    println!("Hypothesis: Prime gaps exhibit periodic oscillation (physical resonance).");
    println!("Null Hypothesis: Spectrum is indistinguishable from shuffled noise (colored noise).");

    // Signal: Gap sequence
    let signal: Vec<f64> = slice.iter().map(|d| d.gap as f64).collect();
    
    // Compute Real FFT
    let (_, real_mags) = compute_fft(&signal);
    let real_peak = *real_mags.iter().skip(1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    
    println!("\nSignal: Prime Gap Sequence");
    println!("Real Data Peak Magnitude: {:.6}", real_peak);

    // Compute Shuffled FFTs
    let mut rng = thread_rng();
    let mut shuffled_signal = signal.clone();
    let mut shuffle_peaks = Vec::with_capacity(args.shuffles);

    print!("Running {} shuffles... ", args.shuffles);
    for _ in 0..args.shuffles {
        shuffled_signal.shuffle(&mut rng);
        let (_, shuf_mags) = compute_fft(&shuffled_signal);
        let peak = *shuf_mags.iter().skip(1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        shuffle_peaks.push(peak);
    }
    println!("Done.");

    let shuf_mean: f64 = shuffle_peaks.iter().sum::<f64>() / args.shuffles as f64;
    let shuf_max = *shuffle_peaks.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    
    println!("Shuffle Mean Peak: {:.6}", shuf_mean);
    println!("Shuffle Max Peak:  {:.6}", shuf_max);
    
    let ratio = real_peak / shuf_mean;
    println!("Ratio (Real / Shuffle Mean): {:.2}x", ratio);

    if real_peak > shuf_max * 1.5 {
        println!("RESULT: POTENTIAL ANOMALY. Real peak is significantly higher than shuffled max.");
    } else {
        println!("RESULT: NO RESONANCE. Real spectrum is within expected noise bounds.");
    }

    // ─── 3. Basis "Bit" Resonance ───
    // Check if the presence of the most common base prime oscillates
    let stats = PrecomputedStats::compute(&db);
    if let Some((top_base, _)) = stats.top_support.first() {
        println!("\n=== Basis '{}' Presence Resonance ===", top_base);
        let bit_signal: Vec<f64> = slice.iter().map(|d| {
            if d.components.contains(top_base) { 1.0 } else { 0.0 }
        }).collect();

        let (_, bit_mags) = compute_fft(&bit_signal);
        let bit_peak = *bit_mags.iter().skip(1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        
        println!("Real Data Peak: {:.6}", bit_peak);
        
        // Quick shuffle check
        let mut max_shuf_peak = 0.0f64;
        let mut shuffled_bits = bit_signal.clone();
        for _ in 0..20 { // fewer shuffles for speed
            shuffled_bits.shuffle(&mut rng);
            let (_, s_mags) = compute_fft(&shuffled_bits);
            let p = *s_mags.iter().skip(1).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
            max_shuf_peak = max_shuf_peak.max(p);
        }
        println!("Shuffle Max (20 runs): {:.6}", max_shuf_peak);
        
        if bit_peak > max_shuf_peak * 1.5 {
             println!("RESULT: POTENTIAL ANOMALY in Basis usage.");
        } else {
             println!("RESULT: NO RESONANCE in Basis usage.");
        }
    }
}

fn print_dist(map: &HashMap<u64, usize>, total: usize) {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by_key(|a| a.0);
    for (k, v) in entries {
        let pct = (*v as f64 / total as f64) * 100.0;
        println!("  {}: {} ({:.2}%)", k, v, pct);
    }
}

fn print_top_dist(map: &HashMap<u64, usize>, total: usize, top_n: usize) {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in entries.iter().take(top_n) {
        let pct = (**v as f64 / total as f64) * 100.0;
        println!("  {}: {} ({:.2}%)", k, v, pct);
    }
    if entries.len() > top_n {
        println!("  ... ({} more)", entries.len() - top_n);
    }
}
