//! Shared utilities for all visualization binaries.
//!
//! Each visualization binary (src/bin/viz_*.rs) imports this module
//! for common data loading, formatting, and precomputation helpers.

use crate::analysis::{self, PrecomputedStats};
use crate::PrimeDatabase;
use eframe::egui;
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

/// Convert polar coordinates (r, theta) to Cartesian (x, y).
pub fn polar_to_cartesian(r: f64, theta: f64) -> [f64; 2] {
    [r * theta.cos(), r * theta.sin()]
}

/// Map an index in [0, total) to a blue→red gradient color.
/// index=0 → pure blue, index=total-1 → pure red.
pub fn color_by_index(index: usize, total: usize) -> egui::Color32 {
    if total <= 1 {
        return egui::Color32::from_rgb(0, 0, 255);
    }
    let t = index as f64 / (total - 1) as f64;
    let r = (t * 255.0) as u8;
    let b = ((1.0 - t) * 255.0) as u8;
    egui::Color32::from_rgb(r, 0, b)
}

/// Project a 3D point to 2D using yaw/pitch rotation and uniform scale.
/// Applies Y-axis rotation (yaw) then X-axis rotation (pitch), then
/// takes the resulting (x, y) as the 2D projection.
pub fn project_3d(point: [f64; 3], yaw: f64, pitch: f64, scale: f64) -> [f64; 2] {
    let [x, y, z] = point;

    // Rotate around Y axis (yaw)
    let cos_y = yaw.cos();
    let sin_y = yaw.sin();
    let x1 = x * cos_y + z * sin_y;
    let y1 = y;
    let z1 = -x * sin_y + z * cos_y;

    // Rotate around X axis (pitch)
    let cos_p = pitch.cos();
    let sin_p = pitch.sin();
    let x2 = x1;
    let y2 = y1 * cos_p - z1 * sin_p;

    [x2 * scale, y2 * scale]
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: phases-2-6-visualizations, Property 1: Color gradient monotonicity
    //
    // For any two indices i < j in [0, total), color_by_index(i, total) should
    // produce a color with higher blue and lower red than color_by_index(j, total).
    // **Validates: Requirements 1.3**
    proptest! {
        #[test]
        fn prop_color_gradient_monotonicity(
            total in 2usize..=1000,
            pct_i in 0.0f64..1.0,
            pct_j in 0.0f64..1.0,
        ) {
            // Map percentages to indices within [0, total), ensuring i < j
            let a = (pct_i * (total - 1) as f64) as usize;
            let b = (pct_j * (total - 1) as f64) as usize;
            let (i, j) = if a < b { (a, b) } else { (b, a) };
            // Skip when equal — no ordering to check
            if i == j { return Ok(()); }

            let color_i = color_by_index(i, total);
            let color_j = color_by_index(j, total);

            // Earlier index → more blue, less red
            prop_assert!(color_i.b() >= color_j.b(),
                "Blue should decrease: i={}, j={}, total={}, blue_i={}, blue_j={}",
                i, j, total, color_i.b(), color_j.b());
            prop_assert!(color_i.r() <= color_j.r(),
                "Red should increase: i={}, j={}, total={}, red_i={}, red_j={}",
                i, j, total, color_i.r(), color_j.r());
        }
    }

    // Feature: phases-2-6-visualizations, Property 2: 3D projection produces finite coordinates
    //
    // For any 3D point with finite coordinates and any yaw/pitch in [0, 2π),
    // project_3d should return finite 2D coordinates.
    // **Validates: Requirements 1.4**
    proptest! {
        #[test]
        fn prop_3d_projection_finite(
            x in -1000.0f64..1000.0,
            y in -1000.0f64..1000.0,
            z in -1000.0f64..1000.0,
            yaw in 0.0f64..std::f64::consts::TAU,
            pitch in 0.0f64..std::f64::consts::TAU,
            scale in 0.1f64..10.0,
        ) {
            let result = project_3d([x, y, z], yaw, pitch, scale);
            prop_assert!(result[0].is_finite(),
                "X should be finite for point=[{},{},{}], yaw={}, pitch={}, scale={}",
                x, y, z, yaw, pitch, scale);
            prop_assert!(result[1].is_finite(),
                "Y should be finite for point=[{},{},{}], yaw={}, pitch={}, scale={}",
                x, y, z, yaw, pitch, scale);
        }
    }

    // Feature: phases-2-6-visualizations, Property 3: Polar-to-Cartesian round trip
    //
    // For any r >= 0 and theta in [0, 2π), converting to Cartesian and back
    // should recover the original r and theta within floating-point tolerance.
    // **Validates: Requirements 2.1**
    proptest! {
        #[test]
        fn prop_polar_cartesian_round_trip(
            r in 0.01f64..1000.0,
            theta in 0.01f64..(std::f64::consts::TAU - 0.01),
        ) {
            let [x, y] = polar_to_cartesian(r, theta);

            // Recover r and theta
            let r_recovered = (x * x + y * y).sqrt();
            let theta_recovered = y.atan2(x).rem_euclid(std::f64::consts::TAU);

            let r_err = (r_recovered - r).abs();
            prop_assert!(r_err < 1e-9,
                "r round-trip failed: original={}, recovered={}, err={}",
                r, r_recovered, r_err);

            let theta_err = (theta_recovered - theta).abs();
            prop_assert!(theta_err < 1e-9,
                "theta round-trip failed: original={}, recovered={}, err={}",
                theta, theta_recovered, theta_err);
        }
    }
}
