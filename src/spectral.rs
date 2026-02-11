//! Spectral analysis module for FFT-based interference patterns.
//!
//! Maps decomposition components to frequencies and computes FFT magnitude
//! spectra to detect hidden spectral structure in prime gap decompositions.

use crate::PrimeDecomposition;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::TAU;

/// Build a composite signal from prime decompositions.
///
/// Each decomposition at index `i` contributes `sin(2π × c × i / total)`
/// for every component `c` in its component list. The signal value at
/// index `i` is the sum of all such sine terms.
///
/// Returns an empty vector if `decompositions` is empty or `total` is 0.
pub fn composite_signal(
    decompositions: &[PrimeDecomposition],
    total: usize,
) -> Vec<f64> {
    if decompositions.is_empty() || total == 0 {
        return Vec::new();
    }

    decompositions
        .iter()
        .enumerate()
        .map(|(i, decomp)| {
            let t = i as f64 / total as f64;
            decomp
                .components
                .iter()
                .map(|&c| (TAU * c as f64 * t).sin())
                .sum()
        })
        .collect()
}

/// Compute the FFT magnitude spectrum of a real-valued signal.
///
/// Returns `(frequencies, magnitudes)` where both vectors have length `N/2 + 1`
/// (the positive-frequency half of the spectrum). Frequencies are bin indices
/// in `[0, N/2]`.
///
/// Returns empty vectors if the signal is empty.
pub fn compute_fft(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    fft.process(&mut buffer);

    let half = n / 2 + 1;
    let frequencies: Vec<f64> = (0..half).map(|i| i as f64).collect();
    let magnitudes: Vec<f64> = buffer[..half]
        .iter()
        .map(|c| c.norm() / n as f64)
        .collect();

    (frequencies, magnitudes)
}

/// Find peaks in a magnitude spectrum that exceed `mean + n_sigma * std_dev`.
///
/// Returns a vector of `(index, magnitude)` pairs for each bin that exceeds
/// the threshold. Returns an empty vector if `magnitudes` is empty.
pub fn find_peaks(magnitudes: &[f64], n_sigma: f64) -> Vec<(usize, f64)> {
    if magnitudes.is_empty() {
        return Vec::new();
    }

    let n = magnitudes.len() as f64;
    let mean = magnitudes.iter().sum::<f64>() / n;
    let variance = magnitudes.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let threshold = mean + n_sigma * std_dev;

    magnitudes
        .iter()
        .enumerate()
        .filter(|&(_, &mag)| mag > threshold)
        .map(|(i, &mag)| (i, mag))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: phases-2-6-visualizations, Property 8: Composite signal for single-component decompositions
    // For any decomposition with exactly one component c at index t in a sequence of length N,
    // the composite signal value at index t should equal sin(2π × c × t / N).
    // **Validates: Requirements 9.1**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_composite_signal_single_component(
            c in 1u64..=50,
            n in 2usize..=200,
            t_frac in 0.0f64..1.0
        ) {
            let t = (t_frac * (n as f64 - 1.0)).round() as usize;
            let t = t.min(n - 1);

            // Build a sequence of N decompositions, each with a single component c
            let decompositions: Vec<PrimeDecomposition> = (0..n)
                .map(|_| PrimeDecomposition {
                    prime: 5,
                    prev_prime: 3,
                    gap: 2,
                    components: vec![c],
                })
                .collect();

            let signal = composite_signal(&decompositions, n);
            prop_assert_eq!(signal.len(), n);

            let expected = (TAU * c as f64 * t as f64 / n as f64).sin();
            prop_assert!(
                (signal[t] - expected).abs() < 1e-10,
                "signal[{}] = {}, expected sin(2π × {} × {} / {}) = {}",
                t, signal[t], c, t, n, expected
            );
        }
    }

    // Feature: phases-2-6-visualizations, Property 9: FFT detects known frequency
    // For any frequency f in (0, N/2) and sample count N >= 64, generating a pure sine wave
    // at frequency f and computing compute_fft() should produce a magnitude spectrum with
    // its peak at or adjacent to frequency bin f.
    // **Validates: Requirements 9.3**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_fft_detects_known_frequency(
            n in 64usize..=512,
            f_frac in 0.05f64..0.45
        ) {
            // Pick an integer frequency bin in (0, N/2)
            let half = n / 2;
            let f = ((f_frac * half as f64).round() as usize).max(1).min(half - 1);

            // Generate a pure sine wave at frequency f
            let signal: Vec<f64> = (0..n)
                .map(|i| (TAU * f as f64 * i as f64 / n as f64).sin())
                .collect();

            let (_freqs, magnitudes) = compute_fft(&signal);

            // Find the peak bin (skip DC at index 0)
            let peak_bin = magnitudes
                .iter()
                .enumerate()
                .skip(1)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            // Peak should be at f or adjacent (f-1, f+1) due to spectral leakage
            let diff = if peak_bin >= f { peak_bin - f } else { f - peak_bin };
            prop_assert!(
                diff <= 1,
                "FFT peak at bin {}, expected at or adjacent to bin {} (N={}, diff={})",
                peak_bin, f, n, diff
            );
        }
    }

    // Feature: phases-2-6-visualizations, Property 10: Peak finder detects spikes above threshold
    // For any magnitude array where exactly one element exceeds mean + 2σ and all others are below,
    // find_peaks(magnitudes, 2.0) should return exactly that element's index and value.
    // **Validates: Requirements 9.5**
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_peak_finder_detects_single_spike(
            base_val in 1.0f64..=10.0,
            arr_len in 10usize..=200,
            spike_frac in 0.0f64..1.0
        ) {
            let spike_idx = (spike_frac * (arr_len as f64 - 1.0)).round() as usize;
            let spike_idx = spike_idx.min(arr_len - 1);

            // Build a uniform array with one spike.
            // For a uniform array of value `base_val`, mean = base_val and std_dev ≈ 0.
            // We need the spike to exceed mean + 2σ, and all others to be below.
            // With uniform values, σ depends on the spike value itself.
            //
            // For an array of (arr_len - 1) copies of base_val and one spike s:
            //   mean = ((arr_len - 1) * base_val + s) / arr_len
            //   variance = ((arr_len - 1) * (base_val - mean)^2 + (s - mean)^2) / arr_len
            //
            // We need: s > mean + 2 * std_dev AND base_val < mean + 2 * std_dev
            //
            // A spike that is sufficiently large relative to base_val will satisfy this.
            // We compute the required spike value analytically.
            //
            // Strategy: set spike = base_val + K where K is large enough.
            // After some algebra, a spike of base_val + 10 * base_val works for arr_len >= 10.
            let spike_val = base_val + 10.0 * base_val;

            let mut magnitudes = vec![base_val; arr_len];
            magnitudes[spike_idx] = spike_val;

            // Verify our construction: compute mean and std_dev, check spike is above threshold
            // and all others are below
            let n = arr_len as f64;
            let mean = magnitudes.iter().sum::<f64>() / n;
            let variance = magnitudes.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt();
            let threshold = mean + 2.0 * std_dev;

            // Verify construction invariants
            prop_assert!(
                spike_val > threshold,
                "spike {} should exceed threshold {} (mean={}, std={})",
                spike_val, threshold, mean, std_dev
            );
            prop_assert!(
                base_val < threshold,
                "base {} should be below threshold {} (mean={}, std={})",
                base_val, threshold, mean, std_dev
            );

            let peaks = find_peaks(&magnitudes, 2.0);

            prop_assert_eq!(
                peaks.len(), 1,
                "expected exactly 1 peak, got {} (threshold={}, spike={}, base={})",
                peaks.len(), threshold, spike_val, base_val
            );
            prop_assert_eq!(
                peaks[0].0, spike_idx,
                "peak index {}, expected {}", peaks[0].0, spike_idx
            );
            prop_assert!(
                (peaks[0].1 - spike_val).abs() < 1e-10,
                "peak value {}, expected {}", peaks[0].1, spike_val
            );
        }
    }
}
