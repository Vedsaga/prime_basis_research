//! Visualization: PCA Embedding
//!
//! Scatter plot of decompositions projected onto principal components.
//!
//! Run: cargo run --release --bin viz_pca [-- --cache path.bin]

use eframe::egui;
use egui_plot::{Legend, Plot, PlotPoints, Points};
use prime_basis_research::analysis::{compute_pca, pca_project, build_basis_vector};
use prime_basis_research::viz_common;

struct PcaApp {
    points_pc12: Vec<[f64; 2]>,
    points_pc13: Vec<[f64; 2]>,
    points_pc23: Vec<[f64; 2]>,
    
    explained_variance: Vec<f64>,
    
    view_mode: usize, // 0=1v2, 1=1v3, 2=2v3
}

impl PcaApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, stats) = viz_common::load_data();
        
        // Compute PCA
        let n_bases = 30.min(stats.unique_bases_used);
        let top_bases: Vec<u64> = stats.top_support.iter().take(n_bases).map(|(p, _)| *p).collect();
        
        let pca = compute_pca(&db.decompositions, &top_bases, 3);
        
        let total = db.decompositions.len();
        let mut p12 = Vec::with_capacity(total);
        let mut p13 = Vec::with_capacity(total);
        let mut p23 = Vec::with_capacity(total);
        
        for d in &db.decompositions {
            let basis_vec = build_basis_vector(d, &top_bases);
            let proj = pca_project(&basis_vec, &pca);
            
            // proj might have fewer than 3 components if input data dimension was small
            let x = proj.get(0).copied().unwrap_or(0.0);
            let y = proj.get(1).copied().unwrap_or(0.0);
            let z = proj.get(2).copied().unwrap_or(0.0);
            
            p12.push([x, y]);
            p13.push([x, z]);
            p23.push([y, z]);
        }

        Self {
            points_pc12: p12,
            points_pc13: p13,
            points_pc23: p23,
            explained_variance: pca.explained_variance_ratio,
            view_mode: 0,
        }
    }
}

impl eframe::App for PcaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("PCA Embedding");
            
            ui.horizontal(|ui| {
                ui.label("View:");
                ui.radio_value(&mut self.view_mode, 0, "PC1 vs PC2");
                ui.radio_value(&mut self.view_mode, 1, "PC1 vs PC3");
                ui.radio_value(&mut self.view_mode, 2, "PC2 vs PC3");
                
                if !self.explained_variance.is_empty() {
                    ui.separator();
                    ui.label("Variance Ratio:");
                    for (i, v) in self.explained_variance.iter().enumerate() {
                        ui.label(format!("PC{}: {:.1}%", i+1, v * 100.0));
                    }
                }
            });

            let plot = Plot::new("pca_plot")
                .legend(Legend::default())
                .show_axes([true, true])
                //.data_aspect(1.0) // PCA units are arbitrary but usually isotropic scaling is good
                ;

            plot.show(ui, |plot_ui| {
                let data = match self.view_mode {
                    0 => &self.points_pc12,
                    1 => &self.points_pc13,
                    2 => &self.points_pc23,
                    _ => &self.points_pc12,
                };
                
                // Downsample
                let step = (data.len() / 20_000).max(1);
                
                let points: PlotPoints = data.iter().step_by(step).copied().collect();
                
                // Color? Ideally by index (time)
                // egui_plot::Points is single color. 
                // We'll just use a solid color for now.
                plot_ui.points(Points::new(points).radius(1.5).color(egui::Color32::LIGHT_GREEN));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "PCA Embedding",
        native_options,
        Box::new(|cc| Ok(Box::new(PcaApp::new(cc)))),
    )
}
