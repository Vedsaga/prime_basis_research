//! Visualization: 3D Vector Walk
//!
//! Traces a 3D trajectory through basis-space where each gap's decomposition
//! defines a displacement vector. The cumulative path is projected to 2D
//! and rendered as colored line segments.
//!
//! Run: cargo run --release --bin viz_vector_walk [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use prime_basis_research::analysis::compute_trajectory;
use prime_basis_research::viz_common::{self, color_by_index, format_num, project_3d};

use prime_basis_research::{PrimeDecomposition};

struct VectorWalkApp {
    /// All decompositions (used for recomputation on axis change)
    decompositions: Vec<PrimeDecomposition>,
    /// All available base primes from top_support (for dropdown choices)
    available_bases: Vec<u64>,

    /// Which base primes map to X, Y, Z axes
    axis_bases: [u64; 3],
    /// Index into available_bases for each axis (for dropdown state)
    axis_indices: [usize; 3],

    /// Precomputed trajectory for current axis mapping
    trajectory: Vec<[f64; 3]>,

    // Camera state
    yaw: f64,
    pitch: f64,
    zoom: f64,
}

impl VectorWalkApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, stats) = viz_common::load_data();

        // Collect available base primes from top_support
        let available_bases: Vec<u64> = stats.top_support.iter().map(|(p, _)| *p).collect();

        // Default axis mapping: first 3 base primes from top_support
        let axis_bases = [
            available_bases.get(0).copied().unwrap_or(1),
            available_bases.get(1).copied().unwrap_or(2),
            available_bases.get(2).copied().unwrap_or(3),
        ];

        // Store decompositions for recomputation on axis change
        let decompositions: Vec<PrimeDecomposition> = db.decompositions;

        let trajectory = compute_trajectory(&decompositions, &axis_bases);

        let axis_indices = [
            available_bases.iter().position(|&b| b == axis_bases[0]).unwrap_or(0),
            available_bases.iter().position(|&b| b == axis_bases[1]).unwrap_or(1),
            available_bases.iter().position(|&b| b == axis_bases[2]).unwrap_or(2),
        ];

        Self {
            decompositions,
            available_bases,
            axis_bases,
            axis_indices,
            trajectory,
            yaw: 0.5,
            pitch: 0.3,
            zoom: 1.0,
        }
    }

    /// Recompute trajectory when axis mapping changes
    fn recompute(&mut self) {
        self.axis_bases = [
            self.available_bases[self.axis_indices[0]],
            self.available_bases[self.axis_indices[1]],
            self.available_bases[self.axis_indices[2]],
        ];
        self.trajectory = compute_trajectory(&self.decompositions, &self.axis_bases);
    }
}

impl eframe::App for VectorWalkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let total_segments = self.trajectory.len().saturating_sub(1);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.heading("3D Vector Walk");
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Segments: {} | Yaw: {:.1}° | Pitch: {:.1}° | Zoom: {:.2}",
                    format_num(total_segments),
                    self.yaw.to_degrees(),
                    self.pitch.to_degrees(),
                    self.zoom,
                ));
            });

            // Axis mapping dropdowns
            ui.horizontal(|ui| {
                let mut changed = false;
                let bases = &self.available_bases;

                for (axis_label, idx) in [("X", 0), ("Y", 1), ("Z", 2)] {
                    ui.label(format!("{} axis:", axis_label));
                    let current = self.axis_indices[idx];
                    let label = format!("{}", bases[current]);
                    egui::ComboBox::from_id_salt(format!("axis_{}", axis_label))
                        .selected_text(&label)
                        .show_ui(ui, |ui| {
                            for (bi, &base) in bases.iter().enumerate() {
                                if ui.selectable_value(&mut self.axis_indices[idx], bi, format!("{}", base)).changed() {
                                    changed = true;
                                }
                            }
                        });
                    ui.add_space(8.0);
                }

                if changed {
                    self.recompute();
                }
            });

            // Zoom slider
            ui.horizontal(|ui| {
                ui.label("Zoom:");
                ui.add(egui::Slider::new(&mut self.zoom, 0.01..=10.0).logarithmic(true));
                ui.label("(scroll to zoom, drag to rotate)");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Handle mouse drag for rotation
            let resp = ui.interact(
                ui.available_rect_before_wrap(),
                ui.id().with("walk_drag"),
                egui::Sense::click_and_drag(),
            );
            if resp.dragged() {
                let delta = resp.drag_delta();
                self.yaw += (delta.x as f64) * 0.01;
                self.pitch += (delta.y as f64) * 0.01;
                self.pitch = self.pitch.clamp(
                    -std::f64::consts::FRAC_PI_2 + 0.01,
                    std::f64::consts::FRAC_PI_2 - 0.01,
                );
            }

            // Handle scroll for zoom
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let factor = 1.0 + (scroll as f64) * 0.002;
                self.zoom = (self.zoom * factor).clamp(0.01, 10.0);
            }

            // Downsample to 20,000 segments max
            let step = (total_segments / 20_000).max(1);

            // Render as colored line segments using egui_plot
            // We split into bands for color gradient
            let n_bands = 16usize.min(total_segments);
            if n_bands == 0 {
                ui.label("No data to display.");
                return;
            }
            let band_size = (total_segments / n_bands).max(1);

            let plot = Plot::new("vector_walk_plot")
                .legend(Legend::default())
                .x_axis_label("Projected X")
                .y_axis_label("Projected Y")
                .data_aspect(1.0)
                .allow_drag(false)
                .allow_scroll(false);

            plot.show(ui, |plot_ui| {
                for band in 0..n_bands {
                    let start = band * band_size;
                    let end = ((band + 1) * band_size + 1).min(self.trajectory.len());
                    if start >= end {
                        continue;
                    }

                    let mid_idx = start + band_size / 2;
                    let color = color_by_index(mid_idx, total_segments);

                    let projected: Vec<[f64; 2]> = self.trajectory[start..end]
                        .iter()
                        .step_by(step)
                        .map(|&pt| project_3d(pt, self.yaw, self.pitch, self.zoom))
                        .collect();

                    if projected.len() >= 2 {
                        plot_ui.line(
                            Line::new(PlotPoints::new(projected))
                                .color(color)
                                .name(format!(
                                    "idx {}-{}",
                                    format_num(start),
                                    format_num(end.saturating_sub(1))
                                )),
                        );
                    }
                }
            });

            if step > 1 {
                ui.label(format!(
                    "Showing 1/{} segments (downsampled from {})",
                    step,
                    format_num(total_segments)
                ));
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "3D Vector Walk",
        native_options,
        Box::new(|cc| Ok(Box::new(VectorWalkApp::new(cc)))),
    )
}
