//! Visualization: Successive Vector Distance
//!
//! Line plot of Euclidean distance between consecutive basis vectors.
//!
//! Run: cargo run --release --bin viz_vector_distance [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use prime_basis_research::analysis::successive_distances;
use prime_basis_research::viz_common::{self, running_average};

struct VectorDistanceApp {
    points: Vec<[f64; 2]>, // (index, distance)
    window_size: usize,
    show_help: bool,
    stats_msg: String,
}

impl VectorDistanceApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, stats) = viz_common::load_data();
        
        let n_bases = 30.min(stats.unique_bases_used);
        let top_bases: Vec<u64> = stats.top_support.iter().take(n_bases).map(|(p, _)| *p).collect();
        
        // Compute distances
        let distances = successive_distances(&db.decompositions, &top_bases);
        
        // Prepare plot data
        // distances[i] corresponds to dist(decomp[i], decomp[i+1])
        // It has length N-1
        
        let mut points = Vec::with_capacity(distances.len());
        let mut sum_dist = 0.0;
        let mut min_dist = f64::MAX;
        let mut max_dist = f64::MIN;
        
        for (i, &d) in distances.iter().enumerate() {
            points.push([i as f64, d]);
            sum_dist += d;
            if d < min_dist { min_dist = d; }
            if d > max_dist { max_dist = d; }
        }
        
        let mean_dist = if !distances.is_empty() { sum_dist / distances.len() as f64 } else { 0.0 };
        
        Self {
            points,
            window_size: 100,
            show_help: false,
            stats_msg: format!("Mean Distance: {:.4} | Range: [{:.4}, {:.4}]", mean_dist, min_dist, max_dist),
        }
    }
}

impl eframe::App for VectorDistanceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Successive Vector Distance");
            ui.label(&self.stats_msg);

            ui.horizontal(|ui| {
                ui.label("Smoothing Window:");
                ui.add(egui::Slider::new(&mut self.window_size, 1..=5000).logarithmic(true));
                
                viz_common::show_help_panel(
                    ui,
                    &mut self.show_help,
                    "Vector Distance Help",
                    "Euclidean distance between consecutive basis vectors.",
                    &[
                        ("Low Values", "Structural stability; next prime is similar."),
                        ("Spikes", "Sudden structural shifts (phase transitions)."),
                        ("Trend Up", "Step size in basis space is growing."),
                    ]
                );
            });

            let plot = Plot::new("distance_plot")
                .legend(Legend::default())
                .include_y(0.0);

            plot.show(ui, |plot_ui| {
                // Raw data is noisy, show with low alpha
                plot_ui.line(Line::new(PlotPoints::new(self.points.clone()))
                    .name("Distance")
                    .color(egui::Color32::from_rgba_unmultiplied(100, 100, 100, 50)));
                
                // Trend (running average computed from current window_size)
                let avg_data = running_average(
                    self.points.iter().map(|p| p[1]),
                    self.points.len(),
                    self.window_size,
                );
                plot_ui.line(Line::new(PlotPoints::new(avg_data))
                    .name("Trend")
                    .color(egui::Color32::RED)
                    .width(2.0));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Successive Vector Distance",
        native_options,
        Box::new(|cc| Ok(Box::new(VectorDistanceApp::new(cc)))),
    )
}
