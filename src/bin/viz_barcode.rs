//! Visualization: Spectral Barcode
//!
//! Texture-based view of basis prime usage with barcode styling.
//! Dark bands = used components, light bands = unused (inverted from spectrogram).
//!
//! Run: cargo run --release --bin viz_barcode [-- --cache path.bin]

use eframe::egui;
use egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use prime_basis_research::viz_common;

struct BarcodeApp {
    texture: Option<egui::TextureHandle>,
    height: usize,
    width: usize,
    image_data: ColorImage,
}

impl BarcodeApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, stats) = viz_common::load_data();

        // Top N bases by support score, default N=50 (capped by available)
        let n_bases = 50.min(stats.unique_bases_used);
        let top_bases: Vec<u64> = stats
            .top_support
            .iter()
            .take(n_bases)
            .map(|(p, _)| *p)
            .collect();

        let width = top_bases.len();

        // Downsample to max 4096 rows to stay within texture limits.
        let max_rows = 4096;
        let step = (db.decompositions.len() / max_rows).max(1);

        let mut pixels = Vec::new();
        let mut actual_rows = 0usize;

        // Use while-loop to track actual rows written and avoid dimension mismatch.
        let mut i = 0;
        while i * step < db.decompositions.len() {
            let idx = i * step;

            if step == 1 {
                // No aggregation needed — one row per decomposition.
                let d = &db.decompositions[idx];
                for &base in &top_bases {
                    if d.components.contains(&base) {
                        // Dark = used (barcode absorption line)
                        pixels.extend_from_slice(&[0, 0, 0, 255]);
                    } else {
                        // Light = unused
                        pixels.extend_from_slice(&[255, 255, 255, 255]);
                    }
                }
            } else {
                // Aggregate block: compute usage density over the step range.
                let block_end = (idx + step).min(db.decompositions.len());
                let block_len = block_end - idx;
                for (col, &base) in top_bases.iter().enumerate() {
                    let _ = col;
                    let count = db.decompositions[idx..block_end]
                        .iter()
                        .filter(|d| d.components.contains(&base))
                        .count();
                    // Density: 0.0 (never used) → 1.0 (always used).
                    // Map to inverted brightness: high density → darker.
                    let density = count as f64 / block_len as f64;
                    let brightness = ((1.0 - density) * 255.0) as u8;
                    pixels.extend_from_slice(&[brightness, brightness, brightness, 255]);
                }
            }

            actual_rows += 1;
            i += 1;
        }

        let image = ColorImage::from_rgba_unmultiplied([width, actual_rows], &pixels);

        Self {
            texture: None,
            height: actual_rows,
            width,
            image_data: image,
        }
    }
}

impl eframe::App for BarcodeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.texture.is_none() {
            self.texture = Some(ctx.load_texture(
                "barcode_tex",
                self.image_data.clone(),
                egui::TextureOptions::NEAREST,
            ));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Spectral Barcode");
            ui.label(format!(
                "Top {} bases (X) vs Time (Y, {} rows)",
                self.width, self.height
            ));

            let plot = Plot::new("barcode_plot")
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
        "Spectral Barcode",
        native_options,
        Box::new(|cc| Ok(Box::new(BarcodeApp::new(cc)))),
    )
}
