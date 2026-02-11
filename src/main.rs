use clap::{Parser, Subcommand};
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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { n } => cmd_generate(&cli.cache, n),
        Commands::Show { last, all } => cmd_show(&cli.cache, last, all),
        Commands::Status => cmd_status(&cli.cache),
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
