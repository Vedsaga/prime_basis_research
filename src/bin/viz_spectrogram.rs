//! Visualization: Comb Spectrogram
//!
//! Texture-based view of basis prime usage.
//!
//! Run: cargo run --release --bin viz_spectrogram [-- --cache path.bin]

use eframe::egui;
use egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use prime_basis_research::viz_common::{self};

struct SpectrogramApp {
    texture: Option<egui::TextureHandle>,
    height: usize,
    width: usize,
    image_data: ColorImage,
    show_help: bool,
}

impl SpectrogramApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, stats) = viz_common::load_data();
        
        // Use top N bases, default N=30 or more to fit screen width
        let n_bases = 50.min(stats.unique_bases_used);
        let top_bases: Vec<u64> = stats.top_support.iter().take(n_bases).map(|(p, _)| *p).collect();
        
        let height = db.decompositions.len();
        
        // Debugging dimensions
        println!("Spectrogram Debug: n_bases={}, top_support_len={}, top_bases_len={}", 
                 n_bases, stats.top_support.len(), top_bases.len());

        // Ensure width matches actual data available
        let width = top_bases.len();
        
        // Downsample to max 4096 rows to stay within texture limits.
        let max_rows = 4096;
        let step = (height / max_rows).max(1);
        
        let mut pixels = Vec::new();
        let mut actual_rows = 0usize;
        
        let mut i = 0;
        while i * step < db.decompositions.len() {
            let actual_idx = i * step;
            let d = &db.decompositions[actual_idx];
            
            for &base in &top_bases {
                if d.components.contains(&base) {
                    pixels.extend_from_slice(&[255, 255, 255, 255]);
                } else {
                    pixels.extend_from_slice(&[0, 0, 0, 255]);
                }
            }
            actual_rows += 1;
            i += 1;
        }
        
        println!("Spectrogram Debug: pixels_len={}, expected={}", pixels.len(), width * actual_rows * 4);
        
        let image = ColorImage::from_rgba_unmultiplied([width, actual_rows], &pixels);

        Self {
            texture: None,
            height: actual_rows,
            width,
            image_data: image,
            show_help: false,
        }
    }
}

impl eframe::App for SpectrogramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.texture.is_none() {
            self.texture = Some(ctx.load_texture(
                "spectrogram_tex",
                self.image_data.clone(),
                egui::TextureOptions::NEAREST, // Pixelated look is desired for "barcode" effect
            ));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Comb Spectrogram");
                viz_common::show_help_panel(
                    ui,
                    &mut self.show_help,
                    "Spectrogram Help",
                    "Base Prime usage (X) over Time (Y).",
                    &[
                        ("Vertical Bands", "A base prime is used consistently (workhorse)."),
                        ("Horizontal Gaps", "Ranges where few base primes are needed."),
                        ("Drift", "Active basis changes over time."),
                    ]
                );
            });
            ui.label(format!("Top {} bases (X) vs Time (Y, {} rows)", self.width, self.height));

            let plot = Plot::new("spectrogram_plot")
                .show_axes([false, true])
                .show_grid([false, false])
                .data_aspect(1.0);

            plot.show(ui, |plot_ui| {
                if let Some(tex) = &self.texture {
                    plot_ui.image(PlotImage::new(
                        tex.id(),
                        PlotPoint::new(self.width as f64 / 2.0, self.height as f64 / 2.0),
                        [self.width as f32, self.height as f32],
                    ));
                }
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Comb Spectrogram",
        native_options,
        Box::new(|cc| Ok(Box::new(SpectrogramApp::new(cc)))),
    )
}
