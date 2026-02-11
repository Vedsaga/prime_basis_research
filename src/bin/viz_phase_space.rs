//! Visualization: Phase Space Plot
//!
//! 2D/3D scatter plot of (gap, num_components, next_gap).
//!
//! Run: cargo run --release --bin viz_phase_space [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Plot, PlotPoints, Points};
use prime_basis_research::viz_common::{self, color_by_index, format_num, project_3d};

struct PhaseSpaceApp {
    /// (gap, num_components, next_gap) for all decompositions
    points: Vec<(f64, f64, f64)>,
    /// Number of components per point (for default coloring)
    comp_counts: Vec<usize>,
    /// Max component count (for color bucketing)
    max_comp: usize,

    // Summary stats
    total_points: usize,
    gap_min: u64,
    gap_max: u64,
    comp_min: usize,
    comp_max: usize,

    // View state
    is_3d: bool,
    yaw: f64,
    pitch: f64,
    scale: f64,
    color_by_idx: bool,
}

impl PhaseSpaceApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, _stats) = viz_common::load_data();

        let n = db.decompositions.len();
        let mut points = Vec::with_capacity(n.saturating_sub(1));
        let mut comp_counts = Vec::with_capacity(n.saturating_sub(1));
        let mut gap_min = u64::MAX;
        let mut gap_max = 0u64;
        let mut comp_min = usize::MAX;
        let mut comp_max = 0usize;

        for i in 0..n.saturating_sub(1) {
            let d = &db.decompositions[i];
            let next_gap = db.decompositions[i + 1].gap as f64;
            let g = d.gap;
            let c = d.components.len();

            if g < gap_min {
                gap_min = g;
            }
            if g > gap_max {
                gap_max = g;
            }
            if c < comp_min {
                comp_min = c;
            }
            if c > comp_max {
                comp_max = c;
            }

            points.push((g as f64, c as f64, next_gap));
            comp_counts.push(c);
        }

        let total_points = points.len();

        Self {
            points,
            comp_counts,
            max_comp: comp_max,
            total_points,
            gap_min,
            gap_max,
            comp_min,
            comp_max,
            is_3d: false,
            yaw: 0.5,
            pitch: 0.3,
            scale: 1.0,
            color_by_idx: false,
        }
    }

    /// Build color buckets by component count. Returns a Vec of (Points, name) per bucket.
    fn points_by_comp_count(
        &self,
        coords: &[[f64; 2]],
        sample_step: usize,
    ) -> Vec<(PlotPoints, egui::Color32, String)> {
        // Group by component count for coloring
        let mut buckets: std::collections::BTreeMap<usize, Vec<[f64; 2]>> =
            std::collections::BTreeMap::new();

        for (idx, pt) in coords.iter().enumerate() {
            let orig_idx = idx * sample_step;
            if orig_idx < self.comp_counts.len() {
                buckets
                    .entry(self.comp_counts[orig_idx])
                    .or_default()
                    .push(*pt);
            }
        }

        let max_c = self.max_comp.max(1);
        buckets
            .into_iter()
            .map(|(c, pts)| {
                // Temperature scale: low comp → blue, high comp → red
                let t = c as f64 / max_c as f64;
                let r = (t * 255.0) as u8;
                let b = ((1.0 - t) * 255.0) as u8;
                let color = egui::Color32::from_rgb(r, 0, b);
                (PlotPoints::new(pts), color, format!("{} comps", c))
            })
            .collect()
    }

    /// Build color buckets by prime index. Returns a Vec of (Points, color, name).
    fn points_by_index(
        &self,
        coords: &[[f64; 2]],
        sample_step: usize,
    ) -> Vec<(PlotPoints, egui::Color32, String)> {
        // Split into ~8 bands for manageable legend
        let n_bands = 8usize;
        let total = self.total_points;
        let band_size = (total / n_bands).max(1);

        let mut bands: Vec<Vec<[f64; 2]>> = vec![vec![]; n_bands];

        for (idx, pt) in coords.iter().enumerate() {
            let orig_idx = idx * sample_step;
            let band = (orig_idx / band_size).min(n_bands - 1);
            bands[band].push(*pt);
        }

        bands
            .into_iter()
            .enumerate()
            .filter(|(_, pts)| !pts.is_empty())
            .map(|(band, pts)| {
                let mid = band * band_size + band_size / 2;
                let color = color_by_index(mid, total);
                let label = format!(
                    "idx {}-{}",
                    format_num(band * band_size),
                    format_num(((band + 1) * band_size).min(total))
                );
                (PlotPoints::new(pts), color, label)
            })
            .collect()
    }
}

impl eframe::App for PhaseSpaceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Header panel with summary stats
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.heading("Phase Space: Gap × Components × Next Gap");
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Points: {} | Gap range: [{}, {}] | Component range: [{}, {}]",
                    format_num(self.total_points),
                    self.gap_min,
                    self.gap_max,
                    self.comp_min,
                    self.comp_max,
                ));
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.is_3d, "3D Mode");
                if self.is_3d {
                    ui.add(
                        egui::Slider::new(&mut self.scale, 0.1..=5.0).text("Scale"),
                    );
                    ui.label(format!(
                        "Yaw: {:.1}° Pitch: {:.1}° (drag plot to rotate)",
                        self.yaw.to_degrees(),
                        self.pitch.to_degrees()
                    ));
                }
                ui.checkbox(&mut self.color_by_idx, "Color by Index");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Determine downsampling based on visible point count
            let sample_step = if self.total_points > 50_000 {
                (self.total_points / 50_000).max(1)
            } else {
                1
            };

            // Build 2D coordinates (possibly projected from 3D)
            let sampled: Vec<[f64; 2]> = if self.is_3d {
                let cx = (self.gap_max as f64 + self.gap_min as f64) / 2.0;
                let cy = (self.comp_max as f64 + self.comp_min as f64) / 2.0;
                let cz = cx; // next_gap has similar range to gap

                self.points
                    .iter()
                    .step_by(sample_step)
                    .map(|&(g, c, ng)| {
                        project_3d(
                            [g - cx, c - cy, ng - cz],
                            self.yaw,
                            self.pitch,
                            self.scale,
                        )
                    })
                    .collect()
            } else {
                self.points
                    .iter()
                    .step_by(sample_step)
                    .map(|&(g, c, _)| [g, c])
                    .collect()
            };

            // Handle drag for 3D rotation
            if self.is_3d {
                let resp = ui.interact(
                    ui.available_rect_before_wrap(),
                    ui.id().with("3d_drag"),
                    egui::Sense::drag(),
                );
                if resp.dragged() {
                    let delta = resp.drag_delta();
                    self.yaw += (delta.x as f64) * 0.01;
                    self.pitch += (delta.y as f64) * 0.01;
                    // Clamp pitch to avoid gimbal issues
                    self.pitch = self.pitch.clamp(
                        -std::f64::consts::FRAC_PI_2 + 0.01,
                        std::f64::consts::FRAC_PI_2 - 0.01,
                    );
                }
            }

            // Build colored point groups
            let groups = if self.color_by_idx {
                self.points_by_index(&sampled, sample_step)
            } else {
                self.points_by_comp_count(&sampled, sample_step)
            };

            let x_label = if self.is_3d {
                "Projected X"
            } else {
                "Gap Size"
            };
            let y_label = if self.is_3d {
                "Projected Y"
            } else {
                "Component Count"
            };

            let plot = Plot::new("phase_space_plot")
                .legend(Legend::default())
                .x_axis_label(x_label)
                .y_axis_label(y_label)
                .show_axes([true, true])
                .allow_drag(!self.is_3d) // disable plot drag in 3D so our rotation works
                .allow_scroll(true);

            plot.show(ui, |plot_ui| {
                for (pp, color, name) in groups {
                    plot_ui.points(
                        Points::new(pp)
                            .radius(2.0)
                            .color(color)
                            .name(name),
                    );
                }
            });

            if sample_step > 1 {
                ui.label(format!(
                    "Showing 1/{} points (downsampled from {})",
                    sample_step,
                    format_num(self.total_points)
                ));
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Phase Space Visualization",
        native_options,
        Box::new(|cc| Ok(Box::new(PhaseSpaceApp::new(cc)))),
    )
}
