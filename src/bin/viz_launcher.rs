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
            ui.add_space(15.0);

            let available_width = ui.available_width();
            let _col_width = (available_width - 20.0) / 2.0;

            ui.columns(2, |cols| {
                // Phase 2 Column
                cols[0].vertical(|ui| {
                    ui.heading("Phase 2: Structure");
                    ui.add_space(10.0);
                    
                    self.tool_card(ui, "Phase Space Plot", "viz_phase_space", "Gap vs Component stats");
                    self.tool_card(ui, "Modular Starfield", "viz_starfield", "Polar plot of residues");
                    self.tool_card(ui, "Resonance Cylinder", "viz_resonance", "Animated modulus sweep");
                    self.tool_card(ui, "Compression Sig", "viz_compression", "Bit efficiency analysis");
                });

                // Phase 3 Column
                cols[1].vertical(|ui| {
                    ui.heading("Phase 3: Patterns");
                    ui.add_space(10.0);
                    
                    self.tool_card(ui, "Comb Spectrogram", "viz_spectrogram", "Basis usage over time");
                    self.tool_card(ui, "Spectral Barcode", "viz_barcode", "Discrete usage stripes");
                    self.tool_card(ui, "Vector Distance", "viz_vector_distance", "Structural change rate");
                    self.tool_card(ui, "PCA Embedding", "viz_pca", "High-dim projection");
                });
            });
            
            ui.add_space(20.0);
            ui.separator();
            ui.label("Existing Phase 1 Tools:");
            ui.horizontal(|ui| {
                if ui.button("Distributions").clicked() { self.launch("viz_distributions"); }
                if ui.button("Gap Waveform").clicked() { self.launch("viz_gap_waveform"); }
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
