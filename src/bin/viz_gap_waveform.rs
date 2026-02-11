//! Visualization 1: Prime Gap Waveform
//!
//! Scatter plot of gap sizes over the prime sequence, colored by
//! decomposition complexity (component count).
//!
//! Run: cargo run --release --bin viz_gap_waveform [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use prime_basis_research::viz_common::{self, format_num, running_average};

fn main() -> eframe::Result<()> {
    let (db, stats) = viz_common::load_data();

    // Precompute all plot data
    let decomps = &db.decompositions;
    let total = decomps.len();

    let mut gaps_1comp: Vec<[f64; 2]> = Vec::new();
    let mut gaps_2comp: Vec<[f64; 2]> = Vec::new();
    let mut gaps_3plus: Vec<[f64; 2]> = Vec::new();

    for (i, d) in decomps.iter().enumerate() {
        let point = [i as f64, d.gap as f64];
        match d.components.len() {
            1 => gaps_1comp.push(point),
            2 => gaps_2comp.push(point),
            _ => gaps_3plus.push(point),
        }
    }

    // Bucketed data for zoom-out
    let buckets = db.aggregate_buckets(500);
    let bucket_gap_max: Vec<[f64; 2]> = buckets
        .iter()
        .map(|b| [(b.start_idx + b.count / 2) as f64, b.gap_max as f64])
        .collect();
    let bucket_gap_mean: Vec<[f64; 2]> = buckets
        .iter()
        .map(|b| [(b.start_idx + b.count / 2) as f64, b.gap_mean])
        .collect();

    // Running average of gap size
    let running_avg = running_average(
        decomps.iter().map(|d| d.gap as f64),
        total,
        5000,
    );

    let info_line = format!(
        "{} primes | largest: {} | max gap: {} | avg gap: {:.1}",
        format_num(stats.total_decompositions),
        format_num(stats.largest_prime as usize),
        stats.gap_max,
        stats.gap_mean,
    );

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("Prime Basis — Gap Waveform"),
        ..Default::default()
    };

    eframe::run_native(
        "Gap Waveform",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(GapWaveformApp {
                gaps_1comp,
                gaps_2comp,
                gaps_3plus,
                bucket_gap_max,
                bucket_gap_mean,
                running_avg,
                info_line,
            }))
        }),
    )
}

struct GapWaveformApp {
    gaps_1comp: Vec<[f64; 2]>,
    gaps_2comp: Vec<[f64; 2]>,
    gaps_3plus: Vec<[f64; 2]>,
    bucket_gap_max: Vec<[f64; 2]>,
    bucket_gap_mean: Vec<[f64; 2]>,
    running_avg: Vec<[f64; 2]>,
    info_line: String,
}

impl eframe::App for GapWaveformApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🌊 Gap Waveform");
                ui.separator();
                ui.label(&self.info_line);
            });
            ui.label("Blue=1 component (twin primes), Green=2 components, Red=3+. Scroll to zoom, drag to pan.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let plot = Plot::new("gap_waveform")
                .legend(Legend::default())
                .x_axis_label("Prime index")
                .y_axis_label("Gap size")
                .allow_zoom(true)
                .allow_drag(true)
                .allow_scroll(true);

            plot.show(ui, |plot_ui| {
                let bounds = plot_ui.plot_bounds();
                let visible_range = bounds.max()[0] - bounds.min()[0];

                if visible_range > 100_000.0 {
                    // Zoomed out: bucketed view
                    plot_ui.line(
                        Line::new(PlotPoints::new(self.bucket_gap_max.clone()))
                            .name("Max gap (per 500)")
                            .color(egui::Color32::from_rgb(255, 100, 100))
                            .width(1.0),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::new(self.bucket_gap_mean.clone()))
                            .name("Mean gap (per 500)")
                            .color(egui::Color32::from_rgb(100, 200, 100))
                            .width(2.0),
                    );
                    plot_ui.line(
                        Line::new(PlotPoints::new(self.running_avg.clone()))
                            .name("Running avg (window=5000)")
                            .color(egui::Color32::from_rgb(255, 200, 50))
                            .width(2.0),
                    );
                } else {
                    // Zoomed in: individual points
                    plot_ui.points(
                        Points::new(PlotPoints::new(self.gaps_1comp.clone()))
                            .name("1 component")
                            .color(egui::Color32::from_rgb(50, 120, 255))
                            .radius(2.0),
                    );
                    plot_ui.points(
                        Points::new(PlotPoints::new(self.gaps_2comp.clone()))
                            .name("2 components")
                            .color(egui::Color32::from_rgb(50, 220, 80))
                            .radius(2.0),
                    );
                    plot_ui.points(
                        Points::new(PlotPoints::new(self.gaps_3plus.clone()))
                            .name("3+ components")
                            .color(egui::Color32::from_rgb(255, 60, 60))
                            .radius(4.0),
                    );
                }
            });
        });
    }
}
