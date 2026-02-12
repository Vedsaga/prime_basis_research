//! Visualization: Hyper-Crystal Diffraction
//!
//! Top panel: composite signal amplitude vs prime index (downsampled line plot).
//! Bottom panel: FFT magnitude spectrum with peak markers.
//!
//! Run: cargo run --release --bin viz_diffraction [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use prime_basis_research::spectral::{composite_signal, compute_fft, find_peaks};
use prime_basis_research::viz_common;

struct DiffractionApp {
    signal_line: Vec<[f64; 2]>,
    fft_line: Vec<[f64; 2]>,
    peak_points: Vec<[f64; 2]>,
    stats_msg: String,
    show_help: bool,
}

impl DiffractionApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, _stats) = viz_common::load_data();

        let total = db.decompositions.len();

        // Precompute composite signal
        let signal = composite_signal(&db.decompositions, total);

        // Precompute FFT
        let (fft_freqs, fft_mags) = compute_fft(&signal);

        // Precompute peaks
        let peaks = find_peaks(&fft_mags, 2.0);

        // Downsample signal for display (~4000 points max)
        let step = (total / 4000).max(1);
        let signal_line: Vec<[f64; 2]> = signal
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, &v)| [i as f64, v])
            .collect();

        // FFT line (already half-spectrum, typically manageable)
        let fft_step = (fft_freqs.len() / 4000).max(1);
        let fft_line: Vec<[f64; 2]> = fft_freqs
            .iter()
            .zip(fft_mags.iter())
            .step_by(fft_step)
            .map(|(&f, &m)| [f, m])
            .collect();

        // Peak markers
        let peak_points: Vec<[f64; 2]> = peaks
            .iter()
            .map(|&(idx, mag)| [idx as f64, mag])
            .collect();

        let stats_msg = format!(
            "Signal length: {} | FFT bins: {} | Peaks (>2σ): {}",
            viz_common::format_num(total),
            viz_common::format_num(fft_freqs.len()),
            peaks.len()
        );

        Self {
            signal_line,
            fft_line,
            peak_points,
            stats_msg,
            show_help: false,
        }
    }
}

impl eframe::App for DiffractionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Hyper-Crystal Diffraction");
                viz_common::show_help_panel(
                    ui,
                    &mut self.show_help,
                    "Diffraction Help",
                    "Checks if primes use components in a rhythmic, repeating pattern.",
                    &[
                        ("Top Panel", "The 'melody' of component usage over time."),
                        ("Bottom Panel", "The 'frequency spectrum' of that melody."),
                        ("Peaks (Red)", "Strong rhythms. A peak at X means a pattern repeats every 1/X primes."),
                        ("No Peaks?", "Behavior is 'aperiodic' or random-like (common for primes)."),
                    ]
                );
            });
            ui.label(&self.stats_msg);
            ui.separator();

            // Top panel: composite signal amplitude vs prime index
            let signal_plot = Plot::new("signal_plot")
                .legend(Legend::default())
                .height(ui.available_height() / 2.0);

            signal_plot.show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(PlotPoints::new(self.signal_line.clone()))
                        .name("Composite Signal"),
                );
            });

            ui.separator();

            // Bottom panel: FFT magnitude spectrum with peak markers
            let fft_plot = Plot::new("fft_plot")
                .legend(Legend::default())
                .include_y(0.0);

            fft_plot.show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(PlotPoints::new(self.fft_line.clone()))
                        .name("FFT Magnitude")
                        .color(egui::Color32::from_rgb(100, 200, 255)),
                );
                plot_ui.points(
                    Points::new(PlotPoints::new(self.peak_points.clone()))
                        .name("Peaks (>2σ)")
                        .color(egui::Color32::RED)
                        .radius(4.0),
                );
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Hyper-Crystal Diffraction",
        native_options,
        Box::new(|cc| Ok(Box::new(DiffractionApp::new(cc)))),
    )
}
