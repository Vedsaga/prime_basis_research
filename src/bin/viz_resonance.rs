//! Visualization: Resonance Cylinder
//!
//! Animated polar scatter plot that sweeps the modulus from 2 to 300,
//! revealing moduli where structural order emerges in prime residues.
//!
//! Run: cargo run --release --bin viz_resonance [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Plot, Points};
use prime_basis_research::viz_common::{self, polar_to_cartesian};

/// Primorial values and their labels.
const PRIMORIALS: &[(u64, &str)] = &[
    (6, "2×3 = 6"),
    (30, "2×3×5 = 30"),
    (210, "2×3×5×7 = 210"),
];

struct ResonanceApp {
    /// Raw data: (prime, gap, index) for every decomposition.
    data: Vec<(u64, u64, usize)>,

    /// Current modulus (fractional for smooth animation).
    modulus: f32,
    /// Whether the animation is playing.
    playing: bool,
    /// Animation speed: modulus increment per frame.
    speed: f32,
    /// Point size multiplier.
    point_size_scale: f64,
    /// Max rendered points (downsampling).
    max_points: usize,
}

impl ResonanceApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, _stats) = viz_common::load_data();

        let data = db
            .decompositions
            .iter()
            .enumerate()
            .map(|(i, d)| (d.prime, d.gap, i))
            .collect();

        Self {
            data,
            modulus: 2.0,
            playing: false,
            speed: 0.1,
            point_size_scale: 1.0,
            max_points: 10_000,
        }
    }

    /// Return a primorial label if the current integer modulus matches one.
    fn primorial_label(&self) -> Option<&'static str> {
        let m = self.modulus.round() as u64;
        PRIMORIALS.iter().find(|(v, _)| *v == m).map(|(_, l)| *l)
    }
}

impl eframe::App for ResonanceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Advance modulus when playing.
        if self.playing {
            self.modulus += self.speed;
            if self.modulus > 300.0 {
                self.modulus = 300.0;
                self.playing = false;
            }
            // Request continuous repaints while animating.
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Prominent modulus display ---
            let m_int = (self.modulus.round() as u64).max(2);
            ui.heading("Resonance Cylinder: Animated Prime Residues");
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Modulus: {}", m_int))
                        .size(28.0)
                        .strong(),
                );
                if let Some(label) = self.primorial_label() {
                    ui.label(
                        egui::RichText::new(format!("  ★ Primorial: {}", label))
                            .size(20.0)
                            .color(egui::Color32::GOLD),
                    );
                }
            });

            ui.separator();

            // --- Controls ---
            ui.horizontal(|ui| {
                // Play / Pause
                if self.playing {
                    if ui.button("⏸ Pause").clicked() {
                        self.playing = false;
                    }
                } else if ui.button("▶ Play").clicked() {
                    self.playing = true;
                }

                ui.separator();
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.speed, 0.01..=5.0).logarithmic(true));

                ui.separator();
                ui.label("Modulus:");
                let slider_resp =
                    ui.add(egui::Slider::new(&mut self.modulus, 2.0..=300.0).text("M"));
                // Manual slider interaction pauses animation.
                if slider_resp.changed() {
                    self.playing = false;
                }

                ui.separator();
                if ui.button("M=6").clicked() {
                    self.modulus = 6.0;
                    self.playing = false;
                }
                if ui.button("M=30").clicked() {
                    self.modulus = 30.0;
                    self.playing = false;
                }
                if ui.button("M=210").clicked() {
                    self.modulus = 210.0;
                    self.playing = false;
                }

                ui.separator();
                ui.label("Size:");
                ui.add(egui::Slider::new(&mut self.point_size_scale, 0.5..=5.0));
            });

            // --- Plot ---
            let plot = Plot::new("resonance_plot")
                .data_aspect(1.0)
                .show_axes([false, false])
                .show_grid([false, false]);

            plot.show(ui, |plot_ui| {
                let step = (self.data.len() / self.max_points).max(1);

                let points: Vec<[f64; 2]> = self
                    .data
                    .iter()
                    .step_by(step)
                    .map(|(prime, _gap, i)| {
                        let r = *i as f64;
                        let residue = prime % m_int;
                        let theta =
                            std::f64::consts::TAU * (residue as f64) / (m_int as f64);
                        polar_to_cartesian(r, theta)
                    })
                    .collect();

                plot_ui.points(
                    Points::new(points)
                        .radius((1.5 * self.point_size_scale) as f32)
                        .color(egui::Color32::LIGHT_BLUE)
                        .name("Primes"),
                );
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Resonance Cylinder",
        native_options,
        Box::new(|cc| Ok(Box::new(ResonanceApp::new(cc)))),
    )
}
