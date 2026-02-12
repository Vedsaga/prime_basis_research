//! Dashboard Launcher for Prime Basis Visualizations
//!
//! A single entry point to launch all other visualizations.
//!
//! Run: cargo run --release --bin viz_launcher

use eframe::egui;
use std::process::{Command, Stdio};

struct LauncherApp {
    status: String,
}

impl LauncherApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            status: "Ready to launch.".to_owned(),
        }
    }

    fn launch(&mut self, bin_name: &str) {
        self.status = format!("Launching {}...", bin_name);
        // Spawn as a separate process so the launcher doesn't freeze
        // and multiple tools can run at once.
        match Command::new("cargo")
            .args(&["run", "--release", "--bin", bin_name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn() 
        {
            Ok(_) => self.status = format!("Launched {}.", bin_name),
            Err(e) => self.status = format!("Failed to launch {}: {}", bin_name, e),
        }
    }

    fn tool_card(&mut self, ui: &mut egui::Ui, name: &str, bin: &str, desc: &str) {
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            if ui.add(egui::Button::new(egui::RichText::new(name).strong()).min_size([120.0, 30.0].into())).clicked() {
                self.launch(bin);
            }
            ui.label(desc);
        });
        ui.add_space(5.0);
    }
}

impl eframe::App for LauncherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Prime Basis Visualization Dashboard");
            ui.add_space(5.0);

            ui.label(egui::RichText::new(&self.status).color(egui::Color32::LIGHT_BLUE));
            ui.add_space(10.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                // --- Phase 1: Distributions ---
                ui.heading("Phase 1: Distributions");
                ui.add_space(5.0);
                ui.columns(2, |cols| {
                    self.tool_card(&mut cols[0], "📊 Distributions", "viz_distributions", "Component & gap histograms");
                    self.tool_card(&mut cols[1], "🌊 Gap Waveform", "viz_gap_waveform", "Gap sizes colored by complexity");
                });
                ui.columns(2, |cols| {
                    self.tool_card(&mut cols[0], "📈 Support Scores", "viz_support_scores", "Base prime usage frequency");
                    self.tool_card(&mut cols[1], "📋 Summary", "viz_summary", "Key metrics printed to terminal");
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                // --- Phase 2: Structure ---
                ui.heading("Phase 2: Structure");
                ui.add_space(5.0);
                ui.columns(2, |cols| {
                    self.tool_card(&mut cols[0], "🌀 Phase Space Plot", "viz_phase_space", "Gap × components × next gap scatter");
                    self.tool_card(&mut cols[1], "📐 Compression Signature", "viz_compression", "Bit cost vs log₂(p)");
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                // --- Phase 3: Modular ---
                ui.heading("Phase 3: Modular");
                ui.add_space(5.0);
                ui.columns(2, |cols| {
                    self.tool_card(&mut cols[0], "⭐ Modular Starfield", "viz_starfield", "Polar plot of p mod M residues");
                    self.tool_card(&mut cols[1], "🎯 Resonance Cylinder", "viz_resonance", "Animated modulus sweep 2→300");
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                // --- Phase 4: Temporal ---
                ui.heading("Phase 4: Temporal Patterns");
                ui.add_space(5.0);
                ui.columns(2, |cols| {
                    self.tool_card(&mut cols[0], "🎹 Comb Spectrogram", "viz_spectrogram", "Base prime usage heatmap over time");
                    self.tool_card(&mut cols[1], "🔬 Spectral Barcode", "viz_barcode", "Absorption-line style usage stripes");
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                // --- Phase 5: Geometric ---
                ui.heading("Phase 5: Geometric");
                ui.add_space(5.0);
                ui.columns(2, |cols| {
                    self.tool_card(&mut cols[0], "📏 Vector Distance", "viz_vector_distance", "Successive basis vector distances");
                    self.tool_card(&mut cols[1], "🧬 PCA Embedding", "viz_pca", "High-dim projection to 2D/3D");
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(5.0);

                // --- Phase 6: Spectral & Relational ---
                ui.heading("Phase 6: Spectral & Relational");
                ui.add_space(5.0);
                ui.columns(3, |cols| {
                    self.tool_card(&mut cols[0], "💎 Diffraction", "viz_diffraction", "FFT interference patterns");
                    self.tool_card(&mut cols[1], "🚶 3D Vector Walk", "viz_vector_walk", "Cumulative trajectory in basis-space");
                    self.tool_card(&mut cols[2], "🕸 Dependency Network", "viz_network", "Force-directed co-occurrence graph");
                });

                ui.add_space(10.0);
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_resizable(true),
        ..Default::default()
    };
    eframe::run_native(
        "Prime Basis Dashboard",
        native_options,
        Box::new(|cc| Ok(Box::new(LauncherApp::new(cc)))),
    )
}
