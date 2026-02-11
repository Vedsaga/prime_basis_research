//! Visualization: Compression Signature
//!
//! Plots bit cost of prime decompositions vs index, and ratio to log2(prime).
//!
//! Run: cargo run --release --bin viz_compression [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use prime_basis_research::analysis::compression_bits;
use prime_basis_research::viz_common::{self, running_average};

struct CompressionApp {
    lines: Vec<[f64; 2]>, // (index, bits)
    log_lines: Vec<[f64; 2]>, // (index, log2(prime))
    ratio_lines: Vec<[f64; 2]>, // (index, bits/log2(prime))
    
    avg_ratio_line: Vec<[f64; 2]>,
    
    stats_msg: String,
}

impl CompressionApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, _stats) = viz_common::load_data();
        
        let total = db.decompositions.len();
        // Downsample for display if needed or keep full fidelity for zoom
        // Let's store full fidelity but running average is pre-downsampled usually
        
        let mut bits_vec = Vec::with_capacity(total);
        let mut logs_vec = Vec::with_capacity(total);
        let mut ratio_vec = Vec::with_capacity(total);
        
        let mut sum_ratio = 0.0;
        let mut min_ratio = f64::MAX;
        let mut max_ratio = f64::MIN;

        for (i, d) in db.decompositions.iter().enumerate() {
            let bits = compression_bits(d);
            let log_p = (d.prime as f64).log2();
            let ratio = if log_p > 0.0 { bits / log_p } else { 0.0 };
            
            bits_vec.push([i as f64, bits]);
            logs_vec.push([i as f64, log_p]);
            ratio_vec.push([i as f64, ratio]);
            
            sum_ratio += ratio;
            if ratio < min_ratio { min_ratio = ratio; }
            if ratio > max_ratio { max_ratio = ratio; }
        }
        
        let avg_ratio = if total > 0 { sum_ratio / total as f64 } else { 0.0 };
        
        // Compute running average for ratio
        let avg_ratio_line = running_average(ratio_vec.iter().map(|p| p[1]), total, total / 100);

        Self {
            lines: bits_vec,
            log_lines: logs_vec,
            ratio_lines: ratio_vec,
            avg_ratio_line,
            stats_msg: format!("Avg Ratio: {:.3} | Min: {:.3} | Max: {:.3}", avg_ratio, min_ratio, max_ratio),
        }
    }
}

impl eframe::App for CompressionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Compression Efficiency");
            ui.label(&self.stats_msg);

            ui.separator();
            
            // Top plot: Bits vs Log2(Prime)
            let plot_bits = Plot::new("bits_plot")
                .legend(Legend::default())
                .height(ui.available_height() / 2.0);
                
            plot_bits.show(ui, |plot_ui| {
               plot_ui.line(Line::new(PlotPoints::new(self.lines.clone())).name("Decomp Bits"));
               plot_ui.line(Line::new(PlotPoints::new(self.log_lines.clone())).name("Log2(Prime)").color(egui::Color32::from_rgb(100, 255, 100)));
            });
            
            ui.separator();

            // Bottom plot: Ratio
            let plot_ratio = Plot::new("ratio_plot")
                .legend(Legend::default())
                .include_y(0.0)
                .include_y(2.0);

            plot_ratio.show(ui, |plot_ui| {
                 // The raw ratio is very noisy, maybe plot dots?
                 // Or line with transparency.
                 plot_ui.line(Line::new(PlotPoints::new(self.ratio_lines.clone())).name("Ratio").color(egui::Color32::from_rgba_unmultiplied(200, 200, 255, 50)));
                 
                 // Running average
                 plot_ui.line(Line::new(PlotPoints::new(self.avg_ratio_line.clone())).name("Avg Ratio").color(egui::Color32::RED).width(2.0));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Compression Signature",
        native_options,
        Box::new(|cc| Ok(Box::new(CompressionApp::new(cc)))),
    )
}
