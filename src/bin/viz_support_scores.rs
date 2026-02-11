//! Visualization 2: Support Score Distribution
//!
//! Bar chart + log-log plot showing how often each base prime
//! appears as a decomposition component. Reveals structural hierarchy
//! and tests for power-law behavior.
//!
//! Run: cargo run --release --bin viz_support_scores [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints, Points};
use prime_basis_research::viz_common::{self, format_num};

fn main() -> eframe::Result<()> {
    let (_db, stats) = viz_common::load_data();

    // Support score bars (all bases, sorted by prime value)
    let mut all_support: Vec<(u64, usize)> = stats
        .support_scores
        .iter()
        .map(|(&k, &v)| (k, v))
        .collect();
    all_support.sort_by_key(|&(k, _)| k);

    let support_bars: Vec<(f64, f64)> = all_support
        .iter()
        .map(|&(p, c)| (p as f64, c as f64))
        .collect();

    // Log-log data
    let log_log: Vec<[f64; 2]> = all_support
        .iter()
        .filter(|&&(p, c)| p > 0 && c > 0)
        .map(|&(p, c)| [(p as f64).ln(), (c as f64).ln()])
        .collect();

    // Simple linear regression on log-log for power-law fit
    let (slope, intercept) = if log_log.len() >= 2 {
        linear_regression(&log_log)
    } else {
        (0.0, 0.0)
    };
    let fit_line: Vec<[f64; 2]> = if log_log.len() >= 2 {
        let x_min = log_log.first().unwrap()[0];
        let x_max = log_log.last().unwrap()[0];
        vec![
            [x_min, slope * x_min + intercept],
            [x_max, slope * x_max + intercept],
        ]
    } else {
        vec![]
    };

    let info_line = format!(
        "{} decompositions | {} unique bases | largest base: {} | power-law slope: {:.2}",
        format_num(stats.total_decompositions),
        stats.unique_bases_used,
        stats.largest_base_used,
        slope,
    );

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Prime Basis — Support Scores"),
        ..Default::default()
    };

    eframe::run_native(
        "Support Scores",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(SupportApp {
                support_bars,
                log_log,
                fit_line,
                slope,
                info_line,
            }))
        }),
    )
}

struct SupportApp {
    support_bars: Vec<(f64, f64)>,
    log_log: Vec<[f64; 2]>,
    fit_line: Vec<[f64; 2]>,
    slope: f64,
    info_line: String,
}

impl eframe::App for SupportApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("📈 Support Scores");
                ui.separator();
                ui.label(&self.info_line);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Top half: bar chart by prime value
            ui.label("Usage count per base prime (X = prime value)");
            let plot = Plot::new("support_by_prime")
                .legend(Legend::default())
                .x_axis_label("Base prime")
                .y_axis_label("Usage count")
                .allow_zoom(true)
                .allow_drag(true)
                .height(ui.available_height() * 0.45);

            plot.show(ui, |plot_ui| {
                let bars: Vec<Bar> = self
                    .support_bars
                    .iter()
                    .map(|&(p, c)| Bar::new(p, c).width(1.5))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new(bars)
                        .name("Support score")
                        .color(egui::Color32::from_rgb(80, 160, 255)),
                );
            });

            ui.separator();

            // Bottom half: log-log plot with fit line
            ui.label(format!(
                "Log-log plot: ln(prime) vs ln(count). Fit slope = {:.3} (power law: count ~ prime^slope)",
                self.slope
            ));
            let log_plot = Plot::new("log_log")
                .legend(Legend::default())
                .x_axis_label("ln(prime)")
                .y_axis_label("ln(count)")
                .allow_zoom(true)
                .allow_drag(true)
                .height(ui.available_height());

            log_plot.show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new(PlotPoints::new(self.log_log.clone()))
                        .name("ln-ln data")
                        .color(egui::Color32::from_rgb(255, 180, 50))
                        .radius(4.0),
                );
                if !self.fit_line.is_empty() {
                    plot_ui.line(
                        Line::new(PlotPoints::new(self.fit_line.clone()))
                            .name(format!("Fit: slope={:.2}", self.slope))
                            .color(egui::Color32::from_rgb(255, 80, 80))
                            .width(2.0),
                    );
                }
            });
        });
    }
}

/// Simple least-squares linear regression on [x, y] points.
fn linear_regression(points: &[[f64; 2]]) -> (f64, f64) {
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|p| p[0]).sum();
    let sum_y: f64 = points.iter().map(|p| p[1]).sum();
    let sum_xy: f64 = points.iter().map(|p| p[0] * p[1]).sum();
    let sum_xx: f64 = points.iter().map(|p| p[0] * p[0]).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return (0.0, sum_y / n);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}
