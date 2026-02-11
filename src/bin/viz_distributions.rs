//! Visualization 3: Component & Gap Distributions
//!
//! Histograms of component counts and gap sizes, plus running
//! averages showing how they evolve over the prime sequence.
//!
//! Run: cargo run --release --bin viz_distributions [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};
use prime_basis_research::viz_common::{self, format_num, running_average};

fn main() -> eframe::Result<()> {
    let (db, stats) = viz_common::load_data();

    // Component count histogram
    let mut comp_hist: Vec<(usize, usize)> = stats
        .component_count_histogram
        .iter()
        .map(|(&k, &v)| (k, v))
        .collect();
    comp_hist.sort_by_key(|&(k, _)| k);
    let comp_bars: Vec<(f64, f64)> = comp_hist
        .iter()
        .map(|&(k, v)| (k as f64, v as f64))
        .collect();

    // Gap histogram (all gaps, sorted by gap size)
    let mut gap_hist: Vec<(u64, usize)> = stats
        .gap_histogram
        .iter()
        .map(|(&k, &v)| (k, v))
        .collect();
    gap_hist.sort_by_key(|&(k, _)| k);
    let gap_bars: Vec<(f64, f64)> = gap_hist
        .iter()
        .map(|&(k, v)| (k as f64, v as f64))
        .collect();

    // Running averages
    let decomps = &db.decompositions;
    let total = decomps.len();
    let running_comp = running_average(
        decomps.iter().map(|d| d.components.len() as f64),
        total,
        5000,
    );
    let running_gap = running_average(
        decomps.iter().map(|d| d.gap as f64),
        total,
        5000,
    );

    let info_line = format!(
        "{} primes | comp range: {}–{} | gap range: {}–{}",
        format_num(stats.total_decompositions),
        stats.component_count_min,
        stats.component_count_max,
        stats.gap_min,
        stats.gap_max,
    );

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Prime Basis — Distributions"),
        ..Default::default()
    };

    eframe::run_native(
        "Distributions",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(DistApp {
                comp_bars,
                gap_bars,
                running_comp,
                running_gap,
                info_line,
            }))
        }),
    )
}

struct DistApp {
    comp_bars: Vec<(f64, f64)>,
    gap_bars: Vec<(f64, f64)>,
    running_comp: Vec<[f64; 2]>,
    running_gap: Vec<[f64; 2]>,
    info_line: String,
}

impl eframe::App for DistApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("📊 Distributions");
                ui.separator();
                ui.label(&self.info_line);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let half_h = ui.available_height() * 0.48;

            // Top row: histograms side by side
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("Component count distribution");
                    Plot::new("comp_hist")
                        .x_axis_label("# Components")
                        .y_axis_label("Count")
                        .allow_zoom(true)
                        .allow_drag(true)
                        .height(half_h)
                        .show(ui, |plot_ui| {
                            let bars: Vec<Bar> = self
                                .comp_bars
                                .iter()
                                .map(|&(k, v)| Bar::new(k, v).width(0.8))
                                .collect();
                            plot_ui.bar_chart(
                                BarChart::new(bars)
                                    .name("Component count")
                                    .color(egui::Color32::from_rgb(100, 200, 150)),
                            );
                        });
                });

                ui.vertical(|ui| {
                    ui.label("Gap size distribution (all gaps)");
                    Plot::new("gap_hist")
                        .x_axis_label("Gap size")
                        .y_axis_label("Count")
                        .allow_zoom(true)
                        .allow_drag(true)
                        .height(half_h)
                        .show(ui, |plot_ui| {
                            let bars: Vec<Bar> = self
                                .gap_bars
                                .iter()
                                .map(|&(k, v)| Bar::new(k, v).width(1.5))
                                .collect();
                            plot_ui.bar_chart(
                                BarChart::new(bars)
                                    .name("Gap frequency")
                                    .color(egui::Color32::from_rgb(200, 130, 80)),
                            );
                        });
                });
            });

            ui.separator();

            // Bottom: running averages
            ui.label("Running averages (window=5000) — how gap size and component count evolve");
            Plot::new("running_avg")
                .legend(Legend::default())
                .x_axis_label("Prime index")
                .y_axis_label("Value")
                .allow_zoom(true)
                .allow_drag(true)
                .allow_scroll(true)
                .height(ui.available_height())
                .show(ui, |plot_ui| {
                    if !self.running_gap.is_empty() {
                        plot_ui.line(
                            Line::new(PlotPoints::new(self.running_gap.clone()))
                                .name("Avg gap")
                                .color(egui::Color32::from_rgb(80, 180, 255))
                                .width(2.0),
                        );
                    }
                    if !self.running_comp.is_empty() {
                        let scaled: Vec<[f64; 2]> = self
                            .running_comp
                            .iter()
                            .map(|&[x, y]| [x, y * 8.0])
                            .collect();
                        plot_ui.line(
                            Line::new(PlotPoints::new(scaled))
                                .name("Avg components ×8")
                                .color(egui::Color32::from_rgb(255, 150, 50))
                                .width(2.0),
                        );
                    }
                });
        });
    }
}
