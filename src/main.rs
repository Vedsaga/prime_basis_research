use clap::{Parser, Subcommand};
use prime_basis_research::analysis::{
    self, gap_autocorrelation, sliding_entropy, PrecomputedStats,
};
use prime_basis_research::PrimeDatabase;
use std::path::PathBuf;
use std::time::Instant;

/// Prime Basis Research — express each prime as prev_prime + sum of distinct smaller primes
#[derive(Parser)]
#[command(name = "prime-basis", version, about)]
struct Cli {
    /// Path to the cache file
    #[arg(long, default_value = "prime_basis.bin")]
    cache: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate the next N prime decompositions and append to cache
    Generate {
        /// Number of new primes to generate
        n: usize,
    },
    /// Display decompositions in "p = prev + a + b + ..." format
    Show {
        /// Show only the last N entries (default: 20)
        #[arg(long, default_value_t = 20)]
        last: usize,

        /// Show all entries
        #[arg(long, default_value_t = false)]
        all: bool,
    },
    /// Show cache statistics
    Status,
    /// Compute and display full analysis statistics (Phase 0)
    Stats {
        /// Recompute even if cached stats exist
        #[arg(long, default_value_t = false)]
        recompute: bool,
    },
    /// Show sliding-window entropy over the dataset
    Entropy {
        /// Window size for entropy computation
        #[arg(long, default_value_t = 1000)]
        window: usize,
        /// Show every Nth entropy value (for large datasets)
        #[arg(long, default_value_t = 1)]
        step: usize,
    },
    /// Show gap autocorrelation
    Autocorrelation {
        /// Maximum lag to compute
        #[arg(long, default_value_t = 100)]
        max_lag: usize,
    },
    /// Export data window as JSON for external tools
    Export {
        /// Start index
        #[arg(long, default_value_t = 0)]
        start: usize,
        /// Number of entries to export
        #[arg(long, default_value_t = 1000)]
        count: usize,
        /// Output file path
        #[arg(long, default_value = "export.json")]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { n } => cmd_generate(&cli.cache, n),
        Commands::Show { last, all } => cmd_show(&cli.cache, last, all),
        Commands::Status => cmd_status(&cli.cache),
        Commands::Stats { recompute } => cmd_stats(&cli.cache, recompute),
        Commands::Entropy { window, step } => cmd_entropy(&cli.cache, window, step),
        Commands::Autocorrelation { max_lag } => cmd_autocorrelation(&cli.cache, max_lag),
        Commands::Export {
            start,
            count,
            output,
        } => cmd_export(&cli.cache, start, count, &output),
    }
}

fn cmd_generate(cache_path: &PathBuf, n: usize) {
    let mut db = PrimeDatabase::load(cache_path);
    let before_count = db.count();
    let before_last = db.last_prime();

    println!(
        "Cache loaded: {} primes (largest: {})",
        before_count, before_last
    );
    println!("Generating {} new prime decompositions...", n);

    let start = Instant::now();
    db.generate(n);
    let elapsed = start.elapsed();

    println!(
        "Done in {:.2?}. {} → {} primes (largest: {})",
        elapsed,
        before_count,
        db.count(),
        db.last_prime()
    );

    // Print the newly generated decompositions (up to 20 for readability)
    let show_count = n.min(20);
    let start_idx = db.decompositions.len() - n;
    println!("\nFirst {} of {} new decompositions:", show_count, n);
    for d in db.decompositions[start_idx..start_idx + show_count].iter() {
        println!("  {}", d.display());
    }
    if n > show_count {
        println!("  ... ({} more, use `show` command to view)", n - show_count);
    }

    db.save(cache_path);
    println!("\nSaved to {:?}", cache_path);
}

fn cmd_show(cache_path: &PathBuf, last: usize, all: bool) {
    let db = PrimeDatabase::load(cache_path);

    if db.decompositions.is_empty() {
        println!("No decompositions in cache. Run `generate` first.");
        return;
    }

    let entries = if all {
        &db.decompositions[..]
    } else {
        let start = db.decompositions.len().saturating_sub(last);
        &db.decompositions[start..]
    };

    let total = db.decompositions.len();
    if !all {
        println!(
            "Showing last {} of {} decompositions:\n",
            entries.len(),
            total
        );
    } else {
        println!("All {} decompositions:\n", total);
    }

    for d in entries {
        println!("  {}", d.display());
    }
}

fn cmd_status(cache_path: &PathBuf) {
    if !cache_path.exists() {
        println!("No cache file found at {:?}", cache_path);
        println!("Run `generate <N>` to create one.");
        return;
    }

    let db = PrimeDatabase::load(cache_path);
    let file_size = std::fs::metadata(cache_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("╔══════════════════════════════════════════╗");
    println!("║       Prime Basis Research — Status      ║");
    println!("╠══════════════════════════════════════════╣");
    println!(
        "║  Total primes:    {:>20}  ║",
        format_number(db.count() as u64)
    );
    println!(
        "║  Decompositions:  {:>20}  ║",
        format_number(db.decompositions.len() as u64)
    );
    println!(
        "║  Largest prime:   {:>20}  ║",
        format_number(db.last_prime())
    );
    println!(
        "║  Cache file size: {:>20}  ║",
        format_bytes(file_size)
    );
    println!("╚══════════════════════════════════════════╝");

    // Show some gap statistics
    if !db.decompositions.is_empty() {
        let max_gap = db.decompositions.iter().map(|d| d.gap).max().unwrap_or(0);
        let avg_gap = db.decompositions.iter().map(|d| d.gap).sum::<u64>() as f64
            / db.decompositions.len() as f64;
        let max_components = db
            .decompositions
            .iter()
            .map(|d| d.components.len())
            .max()
            .unwrap_or(0);

        println!("\n  Gap statistics:");
        println!("    Max gap:          {}", max_gap);
        println!("    Avg gap:          {:.1}", avg_gap);
        println!("    Max components:   {} (for a single gap)", max_components);
    }
}

fn format_number(n: u64) -> String {
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

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// ─── Phase 0: Analysis commands ─────────────────────────────────────────────

fn cmd_stats(cache_path: &PathBuf, recompute: bool) {
    let db = PrimeDatabase::load(cache_path);
    let stats_path = analysis::stats_path_for(cache_path);

    let start = Instant::now();
    let stats = if recompute {
        println!("Computing statistics over {} decompositions...", db.decompositions.len());
        let s = PrecomputedStats::compute(&db);
        s.save(&stats_path);
        s
    } else {
        println!("Loading/computing statistics...");
        PrecomputedStats::load_or_compute(&stats_path, &db)
    };
    let elapsed = start.elapsed();

    println!("Done in {:.2?}\n", elapsed);

    println!("╔══════════════════════════════════════════════════╗");
    println!("║         Prime Basis — Full Analysis Stats        ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Decompositions:  {:>28}  ║", format_number(stats.total_decompositions as u64));
    println!("║  Total primes:    {:>28}  ║", format_number(stats.total_primes as u64));
    println!("║  Largest prime:   {:>28}  ║", format_number(stats.largest_prime));
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  GAP STATISTICS                                  ║");
    println!("║  Min gap:         {:>28}  ║", stats.gap_min);
    println!("║  Max gap:         {:>28}  ║", stats.gap_max);
    println!("║  Mean gap:        {:>28.2}  ║", stats.gap_mean);
    println!("║  Distinct gaps:   {:>28}  ║", stats.gap_histogram.len());
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  COMPONENT STATISTICS                            ║");
    println!("║  Min components:  {:>28}  ║", stats.component_count_min);
    println!("║  Max components:  {:>28}  ║", stats.component_count_max);
    println!("║  Mean components: {:>28.2}  ║", stats.component_count_mean);
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  BASIS USAGE                                     ║");
    println!("║  Unique bases:    {:>28}  ║", stats.unique_bases_used);
    println!("║  Largest base:    {:>28}  ║", stats.largest_base_used);
    println!("╚══════════════════════════════════════════════════╝");

    // Component count distribution
    println!("\n  Component count distribution:");
    let mut comp_hist: Vec<_> = stats.component_count_histogram.iter().collect();
    comp_hist.sort_by_key(|(&k, _)| k);
    for (&count, &freq) in &comp_hist {
        let pct = freq as f64 / stats.total_decompositions as f64 * 100.0;
        let bar_len = (pct * 0.5) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    {} components: {:>8} ({:>5.1}%) {}", count, format_number(freq as u64), pct, bar);
    }

    // Top support scores
    println!("\n  Top 20 base primes by support score:");
    println!("    {:>8}  {:>10}  {:>6}  {}", "Prime", "Count", "%", "");
    for (i, &(prime, count)) in stats.top_support.iter().take(20).enumerate() {
        let pct = count as f64 / stats.total_decompositions as f64 * 100.0;
        let bar_len = (pct * 0.3) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    {:>2}. {:>5}  {:>10}  {:>5.1}%  {}", i + 1, prime, format_number(count as u64), pct, bar);
    }

    // Gap distribution (top 15)
    println!("\n  Top 15 gap sizes by frequency:");
    let mut gap_hist: Vec<_> = stats.gap_histogram.iter().collect();
    gap_hist.sort_by(|a, b| b.1.cmp(a.1));
    for &(&gap, &freq) in gap_hist.iter().take(15) {
        let pct = freq as f64 / stats.total_decompositions as f64 * 100.0;
        let bar_len = (pct * 0.5) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    gap {:>3}: {:>8} ({:>5.1}%) {}", gap, format_number(freq as u64), pct, bar);
    }

    // Decomposition uniqueness summary
    let total_gaps_with_variants: usize = stats
        .gap_decomposition_variants
        .values()
        .filter(|&&v| v > 1)
        .count();
    let max_variants = stats
        .gap_decomposition_variants
        .values()
        .copied()
        .max()
        .unwrap_or(0);
    println!("\n  Decomposition uniqueness:");
    println!("    Gaps with multiple decomposition variants: {}", total_gaps_with_variants);
    println!("    Max variants for a single gap size:        {}", max_variants);

    // First appearance summary
    println!("\n  First appearance of base primes (first 15):");
    let mut fa: Vec<_> = stats.first_appearance.iter().collect();
    fa.sort_by_key(|(&prime, _)| prime);
    for &(&prime, &idx) in fa.iter().take(15) {
        println!("    base {:>5} first used at decomposition #{}", prime, idx);
    }

    println!("\n  Stats cached to: {:?}", stats_path);
}

fn cmd_entropy(cache_path: &PathBuf, window: usize, step: usize) {
    let db = PrimeDatabase::load(cache_path);

    println!(
        "Computing sliding entropy (window={}, {} decompositions)...",
        window,
        db.decompositions.len()
    );

    let start = Instant::now();
    let entropy = sliding_entropy(&db.decompositions, window);
    let elapsed = start.elapsed();

    println!("Done in {:.2?}. {} entropy values.\n", elapsed, entropy.len());

    if entropy.is_empty() {
        println!("Not enough data for window size {}.", window);
        return;
    }

    // Summary statistics
    let h_values: Vec<f64> = entropy.iter().map(|&(_, h)| h).collect();
    let h_min = h_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let h_max = h_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let h_mean = h_values.iter().sum::<f64>() / h_values.len() as f64;
    let h_std = (h_values.iter().map(|h| (h - h_mean).powi(2)).sum::<f64>() / h_values.len() as f64).sqrt();

    println!("  Entropy summary (window={}):", window);
    println!("    Min:    {:.6}", h_min);
    println!("    Max:    {:.6}", h_max);
    println!("    Mean:   {:.6}", h_mean);
    println!("    StdDev: {:.6}", h_std);
    println!("    Range:  {:.6}", h_max - h_min);

    if h_std < 0.01 {
        println!("\n    → Entropy is very stable. The process appears statistically stationary.");
    } else if h_std > 0.1 {
        println!("\n    → Entropy shows significant variation. Look for oscillation or drift.");
    }

    // Show sampled values
    let step = step.max(1);
    let show_count = 40;
    let display_step = (entropy.len() / show_count).max(1).max(step);

    println!("\n  Sampled entropy values (every {} of {}):", display_step, entropy.len());
    println!("    {:>10}  {:>10}", "Index", "Entropy");
    for &(idx, h) in entropy.iter().step_by(display_step) {
        println!("    {:>10}  {:>10.6}", idx, h);
    }
}

fn cmd_autocorrelation(cache_path: &PathBuf, max_lag: usize) {
    let db = PrimeDatabase::load(cache_path);

    println!(
        "Computing gap autocorrelation (max_lag={}, {} decompositions)...",
        max_lag,
        db.decompositions.len()
    );

    let start = Instant::now();
    let ac = gap_autocorrelation(&db.decompositions, max_lag);
    let elapsed = start.elapsed();

    println!("Done in {:.2?}.\n", elapsed);

    if ac.is_empty() {
        println!("Not enough data.");
        return;
    }

    // Find significant correlations
    let threshold = 2.0 / (db.decompositions.len() as f64).sqrt(); // ~95% significance
    let significant: Vec<_> = ac.iter().filter(|&&(_, r)| r.abs() > threshold).collect();

    println!("  Significance threshold (95%): ±{:.6}", threshold);
    println!("  Significant lags: {}\n", significant.len());

    println!("  {:>6}  {:>12}  {}", "Lag", "Correlation", "");
    for &(lag, r) in &ac {
        let bar_len = (r.abs() * 50.0) as usize;
        let bar: String = if r >= 0.0 {
            "█".repeat(bar_len)
        } else {
            "░".repeat(bar_len)
        };
        let marker = if r.abs() > threshold { " *" } else { "" };
        println!("  {:>6}  {:>12.6}  {}{}", lag, r, bar, marker);
    }

    if !significant.is_empty() {
        println!("\n  * = statistically significant (|r| > {:.4})", threshold);
        println!("\n  Top significant lags:");
        let mut sig_sorted = significant.clone();
        sig_sorted.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        for &(lag, r) in sig_sorted.iter().take(10) {
            println!("    lag {:>4}: r = {:>+.6}", lag, r);
        }
    }
}

fn cmd_export(cache_path: &PathBuf, start: usize, count: usize, output: &PathBuf) {
    let db = PrimeDatabase::load(cache_path);
    let window = db.window(start, count);

    println!(
        "Exporting {} decompositions (index {}..{}) to {:?}...",
        window.len(),
        window.start_idx,
        window.start_idx + window.len(),
        output
    );

    // Build a simple JSON array
    let entries: Vec<serde_json::Value> = window
        .decompositions
        .iter()
        .enumerate()
        .map(|(i, d)| {
            serde_json::json!({
                "index": window.start_idx + i,
                "prime": d.prime,
                "prev_prime": d.prev_prime,
                "gap": d.gap,
                "components": d.components,
                "num_components": d.components.len(),
            })
        })
        .collect();

    let json = serde_json::to_string_pretty(&entries).expect("Failed to serialize");
    std::fs::write(output, json).expect("Failed to write export file");
    println!("Done. Exported {} entries.", entries.len());
}
