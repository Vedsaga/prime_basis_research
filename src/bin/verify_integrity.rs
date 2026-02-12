use prime_basis_research::PrimeDatabase;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let cache_path = PathBuf::from("prime_basis.bin");
    if !cache_path.exists() {
        println!("No cache file found at prime_basis.bin");
        return;
    }

    println!("Loading prime database...");
    let db = PrimeDatabase::load(&cache_path);
    let total = db.decompositions.len();

    println!("Verifying {} decompositions...", total);
    let start_time = Instant::now();
    let mut errors = 0;

    for (i, decomp) in db.decompositions.iter().enumerate() {
        // Verify basic arithmetic
        if decomp.prime != decomp.prev_prime + decomp.gap {
            println!(
                "ERROR at index {}: Arithmetic mismatch. {} != {} + {}",
                i, decomp.prime, decomp.prev_prime, decomp.gap
            );
            errors += 1;
            continue;
        }

        // Verify greedy property with optimized search
        // The available primes are db.primes[0..=i]
        let known_primes = &db.primes[0..=i];
        
        // Re-calculate greedy decomposition efficiently
        let mut remaining = decomp.gap;
        let mut expected = Vec::with_capacity(decomp.components.len());
        
        // Start search at the end of known_primes (since it is sorted)
        // But optimization: use binary search to find the start point
        // Specifically, find the index such that known_primes[idx] <= remaining
        // known_primes is sorted ascending.
        
        let mut search_idx = match known_primes.binary_search(&remaining) {
            Ok(idx) => idx, // Found exact match
            Err(idx) => idx.saturating_sub(1), // Insert point is idx, so largest <= remaining is idx-1
        };

        while remaining > 0 {
            // known_primes[search_idx] is guaranteed <= remaining if search_idx is valid
            // But we must check if we run out of primes (shouldn't happen for greedy)
            
            // Optimization: iterate backwards from search_idx
            // In practice, since we subtract the largest possible prime, remaining decreases rapdily.
            // The next component will be <= new_remaining.
            // So we can just update search_idx for the new remaining.
            
            // Find largest prime <= remaining. Since we are iterating down, we can start linear scan from current search_idx
            // OR do binary search again if the step is large. 
            // Linear scan down is simplest if components are close. binary search is safer if gap drops a lot.
            
            // Actually, simply doing binary search for every component is O(K log N) where K is number of components (small).
            // This is very fast.
            
            let idx = match known_primes[..=search_idx].binary_search(&remaining) {
                Ok(found) => found,
                Err(insert) => insert.saturating_sub(1),
            };
            
            let p = known_primes[idx];
            expected.push(p);
            
            if remaining < p {
                 // Should not happen if logic is correct and 1 is present
                 break;
            }
            remaining -= p;
            search_idx = idx; // Optimization: next prime must be smaller than current
        }
        
        if decomp.components != expected {
             println!(
                "ERROR at index {}: Greedy violation for prime {}.",
                i, decomp.prime
            );
            println!("  Stored:   {:?}", decomp.components);
            println!("  Expected: {:?}", expected);
            errors += 1;
            
            if errors >= 10 {
                println!("Too many errors, stopping verification.");
                break;
            }
        }

        if (i + 1) % 100_000 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }

    let elapsed = start_time.elapsed();
    println!("\nVerification complete in {:.2?}.", elapsed);
    if errors == 0 {
        println!("SUCCESS: All {} decompositions are correct.", total);
    } else {
        println!("FAILURE: Found {} errors.", errors);
        std::process::exit(1);
    }
}
