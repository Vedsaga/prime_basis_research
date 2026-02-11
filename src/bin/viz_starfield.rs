//! Visualization: Modular Starfield
//!
//! Polar scatter plot: r = index, theta = 2*pi * (prime % M) / M.
//! Shows emergent symmetries in prime distribution modulo M.
//!
//! Run: cargo run --release --bin viz_starfield [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Plot, Points};
use prime_basis_research::viz_common::{self, polar_to_cartesian};

struct StarfieldApp {
    // Data references (held via index to avoid huge clones if possible, but for simplicity we copy points)
    // Actually, let's store the raw data we need: (prime, gap, index)
    data: Vec<(u64, u64, usize)>,
    
    // View state
    modulus: u64,
    point_size_scale: f64,
    max_points: usize,
    show_help: bool,
}

impl StarfieldApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, _stats) = viz_common::load_data();
        
        let data = db.decompositions.iter().enumerate().map(|(i, d)| {
            (d.prime, d.gap, i)
        }).collect();

        Self {
            data,
            modulus: 6, // Default to primorial 6
            point_size_scale: 1.0,
            max_points: 10_000,
            show_help: false,
        }
    }
}

impl eframe::App for StarfieldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Modular Starfield: Prime Residues");

            ui.horizontal(|ui| {
                ui.label("Modulus:");
                ui.add(egui::Slider::new(&mut self.modulus, 2..=500).text("M"));
                
                if ui.button("M=6").clicked() { self.modulus = 6; }
                if ui.button("M=30").clicked() { self.modulus = 30; }
                if ui.button("M=210").clicked() { self.modulus = 210; }

                ui.separator();
                ui.label("Size:");
                ui.add(egui::Slider::new(&mut self.point_size_scale, 0.5..=5.0));

                viz_common::show_help_panel(
                    ui,
                    &mut self.show_help,
                    "Starfield Help",
                    "Polar plot of primes modulo M. Radius = Index, Angle = Residue.",
                    &[
                        ("Spokes/Rays", "Primes prefer certain residues (e.g. coprime to M)."),
                        ("Empty Sectors", "Residues that primes never occupy (factors of M)."),
                        ("Spirals", "Drift in residue classes over time."),
                    ]
                );
            });

            let plot = Plot::new("starfield_plot")
                .data_aspect(1.0)
                .show_axes([false, false])
                .show_grid([false, false]);

            plot.show(ui, |plot_ui| {
                // Downsample for performance if needed
                let step = (self.data.len() / self.max_points).max(1);
                
                let points: Vec<[f64; 2]> = self.data.iter().step_by(step).map(|(prime, _gap, i)| {
                    let r = *i as f64;
                    // theta = 2pi * (prime % M) / M
                    let residue = prime % self.modulus;
                    let theta = std::f64::consts::TAU * (residue as f64) / (self.modulus as f64);
                    
                    polar_to_cartesian(r, theta)
                }).collect();

                // Create points element
                // Color by index (time) to show evolution
                // For simplicity in this high-point-count plot, we might use a single color or simple gradient logic 
                // if we had per-point color support in a performant way. 
                // egui_plot::Points doesn't support per-point color in a data-driven way easily without multiple draw calls.
                // We'll use scaling alpha or just a fixed color for the "starfield" look.
                
                plot_ui.points(Points::new(points).radius((1.5 * self.point_size_scale) as f32).color(egui::Color32::LIGHT_BLUE).name("Primes"));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Modular Starfield",
        native_options,
        Box::new(|cc| Ok(Box::new(StarfieldApp::new(cc)))),
    )
}
