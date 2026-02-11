use std::collections::HashSet;
use itertools::Itertools;

/// Returns a list of primes up to `n` (inclusive or exclusive depending on impl, standard sieve is inclusive).
pub fn sieve(n: usize) -> Vec<usize> {
    if n < 2 {
        return vec![];
    }
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let limit = (n as f64).sqrt() as usize;
    for i in 2..=limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
    }
    is_prime.iter().enumerate()
        .filter_map(|(i, &p)| if p { Some(i) } else { None })
        .collect()
}

/// Finds the smallest subset of `basis` that sums to `target`.
/// Returns `Some(subset)` or `None`.
/// Since basis is usually small, we iterate size 1..=len.
pub fn find_min_subset(target: usize, basis: &HashSet<usize>) -> Option<Vec<usize>> {
    let sorted_basis: Vec<usize> = basis.iter().cloned().sorted().collect();
    for size in 1..=sorted_basis.len() {
        for combo in sorted_basis.iter().combinations(size) {
            let sum_val: usize = combo.iter().map(|&&x| x).sum();
            if sum_val == target {
                return Some(combo.into_iter().cloned().collect());
            }
        }
    }
    None
}

/// Finds a minimal basis that can form all `required_gaps` via subset sums.
pub fn find_minimal_basis(required_gaps: &[usize]) -> Vec<usize> {
    let mut basis = HashSet::new();
    // Sort gaps to process smaller ones first (though set iteration order doesn't guarantee this,
    // the logic depends on covering gaps progressively). The Python script sorts `required_gaps`.
    // We should iterate over sorted gaps.
    let sorted_gaps: Vec<usize> = required_gaps.iter().cloned().sorted().collect();

    for gap in sorted_gaps {
        if find_min_subset(gap, &basis).is_none() {
            // Need to add an element. Python logic:
            // for v in range(1, gap + 1):
            //   if find_min_subset(gap, basis | {v}):
            //     basis.add(v)
            //     break
            for v in 1..=gap {
                let mut temp_basis = basis.clone();
                temp_basis.insert(v);
                if find_min_subset(gap, &temp_basis).is_some() {
                    basis.insert(v);
                    break;
                }
            }
        }
    }
    basis.into_iter().sorted().collect()
}

/// Returns all possible subset sums of the basis.
pub fn all_subset_sums(basis: &[usize]) -> Vec<usize> {
    let mut sums = HashSet::new();
    for r in 1..=basis.len() {
        for combo in basis.iter().combinations(r) {
            sums.insert(combo.into_iter().sum::<usize>());
        }
    }
    sums.into_iter().sorted().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve() {
        assert_eq!(sieve(10), vec![2, 3, 5, 7]);
        assert_eq!(sieve(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_subset() {
        let basis: HashSet<usize> = [1, 2, 4].iter().cloned().collect();
        assert_eq!(find_min_subset(3, &basis), Some(vec![1, 2])); // sorted output depends on combo impl, but sum is 3
        assert_eq!(find_min_subset(7, &basis), Some(vec![1, 2, 4]));
        assert_eq!(find_min_subset(8, &basis), None);
    }
}
