use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use prime_basis_research::PrimeDatabase;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

const WINDOW_SIZE: usize = 1000; // For sliding window stats if needed

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Attribute {
    Bit,             // 0 or 1
    Gap,             // Gap size
    NumComponents,   // Number of primes used
    LargestPrime,    // Largest prime used
    TailSum,         // Gap - LargestPrime
    PrimeIndex,      // Sequential index
}

impl Attribute {
    fn name(&self) -> &'static str {
        match self {
            Attribute::Bit => "Bit (0/1)",
            Attribute::Gap => "Gap Size",
            Attribute::NumComponents => "Component Count",
            Attribute::LargestPrime => "Largest Component",
            Attribute::TailSum => "Tail Sum (Gap - Largest)",
            Attribute::PrimeIndex => "Prime Index",
        }
    }

    fn all() -> [Attribute; 6] {
        [
            Attribute::PrimeIndex,
            Attribute::Bit,
            Attribute::Gap,
            Attribute::NumComponents,
            Attribute::LargestPrime,
            Attribute::TailSum,
        ]
    }
}

struct Entry {
    prime_idx: usize,
    gap: u64,
    num_components: usize,
    largest_component: u64,
    tail_sum: u64,
    bit: u8,
}

struct Stats {
    total_entries: usize,
    ones_count: usize,
    zeros_count: usize,
    ratio_ones: f64,
    max_run_0: usize,
    max_run_1: usize,
    // Pearson correlations
    corr_gap_comp: f64,
    corr_gap_bit: f64,
    corr_comp_bit: f64,
}

enum AppState {
    Loading,
    Loaded(LoadedState),
    Error(String),
}

struct LoadedState {
    entries: Vec<Entry>,
    runs: Vec<(u8, usize)>, // RLE of bits
    stats: Stats,
    
    // UI State
    selected_tab: Tab,
    
    // Timeline View
    timeline_attr: Attribute,
    timeline_cache: Option<(Attribute, Vec<[f64; 2]>)>, // Cache key + data
    
    // Scatter View
    scatter_x: Attribute,
    scatter_y: Attribute,
    scatter_color: Attribute,
    max_points: usize,
    scatter_cache: Option<(Attribute, Attribute, Attribute, usize, Vec<[f64; 2]>, Vec<[f64; 2]>)>, // Key + Series 0 + Series 1
}

#[derive(PartialEq, Eq)]
enum Tab {
    Timeline, // Pulse wave / Strip
    Scatter,  // Correlations
    Stats,    // Detailed text stats
}

struct VizAttributesApp {
    state: AppState,
    loader: Option<Receiver<Result<LoadedState, String>>>,
}

impl VizAttributesApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        
        thread::spawn(move || {
            load_data(tx);
        });

        Self {
            state: AppState::Loading,
            loader: Some(rx),
        }
    }
}

fn load_data(tx: Sender<Result<LoadedState, String>>) {
    let path = PathBuf::from("prime_basis.bin");
    if !path.exists() {
        tx.send(Err("prime_basis.bin not found. Run `cargo run --release generate 1000` first.".to_string())).ok();
        return;
    }

    let db = match std::panic::catch_unwind(|| PrimeDatabase::load(&path)) {
        Ok(db) => db,
        Err(_) => {
            tx.send(Err("Failed to load database (panic).".to_string())).ok();
            return;
        }
    };

    let mut entries = Vec::with_capacity(db.decompositions.len());
    let mut runs = Vec::new();
    let mut current_bit = 2; // Invalid start
    let mut current_run = 0;

    let mut ones = 0;
    let mut zeros = 0;
    let mut max_run_0 = 0;
    let mut max_run_1 = 0;
    
    // Correlation accumulators
    let mut sum_gap = 0.0;
    let mut sum_comp = 0.0;
    let mut sum_bit = 0.0;
    let mut sum_gap_sq = 0.0;
    let mut sum_comp_sq = 0.0;
    let mut sum_bit_sq = 0.0;
    let mut sum_gap_comp = 0.0;
    let mut sum_gap_bit = 0.0;
    let mut sum_comp_bit = 0.0;

    let n = db.decompositions.len() as f64;

    for (i, d) in db.decompositions.iter().enumerate() {
        let bit = if d.components.contains(&1) { 1 } else { 0 };
        let largest = d.components.first().copied().unwrap_or(0);
        let tail_sum = d.gap.saturating_sub(largest);

        entries.push(Entry {
            prime_idx: i,
            gap: d.gap,
            num_components: d.components.len(),
            largest_component: largest,
            tail_sum,
            bit,
        });

        // RLE
        if bit == current_bit {
            current_run += 1;
        } else {
            if current_bit != 2 {
                runs.push((current_bit, current_run));
                if current_bit == 0 {
                    max_run_0 = max_run_0.max(current_run);
                } else {
                    max_run_1 = max_run_1.max(current_run);
                }
            }
            current_bit = bit;
            current_run = 1;
        }

        if bit == 1 { ones += 1; } else { zeros += 1; }

        // Stats accumulation
        let g = d.gap as f64;
        let c = d.components.len() as f64;
        let b = bit as f64;

        sum_gap += g;
        sum_comp += c;
        sum_bit += b;
        sum_gap_sq += g * g;
        sum_comp_sq += c * c;
        sum_bit_sq += b * b;
        sum_gap_comp += g * c;
        sum_gap_bit += g * b;
        sum_comp_bit += c * b;
    }

    // Push last run
    runs.push((current_bit, current_run));
    if current_bit == 0 {
        max_run_0 = max_run_0.max(current_run);
    } else {
        max_run_1 = max_run_1.max(current_run);
    }

    // Compute Correlations
    let pearson = |sum_x: f64, sum_y: f64, sum_xy: f64, sum_x2: f64, sum_y2: f64| -> f64 {
        let num = n * sum_xy - sum_x * sum_y;
        let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    };

    let stats = Stats {
        total_entries: entries.len(),
        ones_count: ones,
        zeros_count: zeros,
        ratio_ones: ones as f64 / entries.len() as f64,
        max_run_0,
        max_run_1,
        corr_gap_comp: pearson(sum_gap, sum_comp, sum_gap_comp, sum_gap_sq, sum_comp_sq),
        corr_gap_bit: pearson(sum_gap, sum_bit, sum_gap_bit, sum_gap_sq, sum_bit_sq),
        corr_comp_bit: pearson(sum_comp, sum_bit, sum_comp_bit, sum_comp_sq, sum_bit_sq),
    };

    tx.send(Ok(LoadedState {
        entries,
        runs,
        stats,
        selected_tab: Tab::Timeline,
        timeline_attr: Attribute::Bit,
        timeline_cache: None,
        scatter_x: Attribute::Gap,
        scatter_y: Attribute::NumComponents,
        scatter_color: Attribute::Bit,
        max_points: 5000,
        scatter_cache: None,
    })).ok();
}

impl eframe::App for VizAttributesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check loader
        if let Some(rx) = &self.loader {
            match rx.try_recv() {
                Ok(result) => {
                    self.state = match result {
                        Ok(data) => AppState::Loaded(data),
                        Err(e) => AppState::Error(e),
                    };
                    self.loader = None;
                }
                Err(_) => {} // Loading...
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            match &mut self.state {
                AppState::Loading => {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Loading database & computing statistics...");
                        ui.spinner();
                    });
                }
                AppState::Error(e) => {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(egui::Color32::RED, format!("Error: {}", e));
                    });
                }
                AppState::Loaded(data) => {
                    render_app(ui, data);
                }
            }
        });
    }
}

fn render_app(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        ui.heading("Prime Attributes Explorer");
        ui.separator();
        ui.selectable_value(&mut data.selected_tab, Tab::Timeline, "Timeline (Pulse)");
        ui.selectable_value(&mut data.selected_tab, Tab::Scatter, "Correlations (Scatter)");
        ui.selectable_value(&mut data.selected_tab, Tab::Stats, "Statistics");
    });
    ui.separator();

    match data.selected_tab {
        Tab::Timeline => render_timeline(ui, data),
        Tab::Scatter => render_scatter(ui, data),
        Tab::Stats => render_stats(ui, data),
    }
}

fn render_timeline(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        ui.label("Attribute:");
        egui::ComboBox::from_id_source("timeline_attr")
            .selected_text(data.timeline_attr.name())
            .show_ui(ui, |ui| {
                for attr in Attribute::all() {
                    ui.selectable_value(&mut data.timeline_attr, attr, attr.name());
                }
            });
    });

    // Check cache
    if data.timeline_cache.as_ref().map(|(a, _)| *a) != Some(data.timeline_attr) {
        let points: Vec<[f64; 2]> = match data.timeline_attr {
            Attribute::Bit => {
                // Pulse wave: x = running index, y = run length * (1 or -1)
                // Optimization: downsample if too many runs
                let step = (data.runs.len() / 20000).max(1);
                
                let mut pts = Vec::with_capacity(data.runs.len() * 4 / step);
                let mut x = 0.0;
                
                // If stepping, we can't just skip runs because x depends on sum of lengths.
                // But we can aggregate runs. Or just plot every run but skip rendering if excessive.
                // Better: iterate all runs to track X, but only push points if significant or periodic.
                
                // Let's iterate all runs to be correct on X axis
                for (i, &(bit, len)) in data.runs.iter().enumerate() {
                    let len_f = len as f64;
                    // Only add points if within budget or detailed enough?
                    // Actually, simple step_by is risky for pulse wave as we lose "x" sync.
                    // Correct approach: accum x.
                    
                    if i % step == 0 {
                        let y = if bit == 1 { len_f } else { -(len_f) };
                        pts.push([x, 0.0]);      // Start at 0
                        pts.push([x, y]);        // Jump to height
                        pts.push([x + len_f, y]); // Stay at height
                        pts.push([x + len_f, 0.0]); // Drop to 0
                    }
                    x += len_f;
                }
                pts
            }
            attr => {
                let step = (data.entries.len() / 2000).max(1);
                data.entries.iter().step_by(step).map(|e| {
                    let val = get_value(e, attr);
                    [e.prime_idx as f64, val]
                }).collect()
            }
        };
        data.timeline_cache = Some((data.timeline_attr, points));
    }

    let points = &data.timeline_cache.as_ref().unwrap().1;
    let points_cloned = points.clone(); // Clone is cheap for 2000 points, acceptable for 20k

    Plot::new("timeline_plot")
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            if data.timeline_attr == Attribute::Bit {
                plot_ui.line(Line::new(points_cloned).name("Bit Run Length").color(egui::Color32::LIGHT_BLUE));
            } else {
                plot_ui.line(Line::new(points_cloned).name(data.timeline_attr.name()));
            }
        });
}

fn render_scatter(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        ui.label("X Axis:");
        combo_attr(ui, &mut data.scatter_x, "sc_x");
        ui.label("Y Axis:");
        combo_attr(ui, &mut data.scatter_y, "sc_y");
        ui.label("Color:");
        combo_attr(ui, &mut data.scatter_color, "sc_c");
        
        ui.separator();
        ui.label("Points:");
        ui.add(egui::DragValue::new(&mut data.max_points).range(100..=50000));
    });

    let step = (data.entries.len() / data.max_points).max(1);
    let is_bit_color = data.scatter_color == Attribute::Bit;
    
    // Check cache
    let cache_valid = if let Some((gx, gy, gc, pts, _, _)) = &data.scatter_cache {
        *gx == data.scatter_x && *gy == data.scatter_y && *gc == data.scatter_color && *pts == data.max_points
    } else {
        false
    };

    if !cache_valid {
       let pts0;
       let pts1;
       
       if is_bit_color {
            pts0 = data.entries.iter().step_by(step)
                .filter(|e| e.bit == 0)
                .map(|e| [get_value(e, data.scatter_x), get_value(e, data.scatter_y)])
                .collect();
            pts1 = data.entries.iter().step_by(step)
                .filter(|e| e.bit == 1)
                .map(|e| [get_value(e, data.scatter_x), get_value(e, data.scatter_y)])
                .collect();
       } else {
            pts0 = data.entries.iter().step_by(step).map(|e| {
                [get_value(e, data.scatter_x), get_value(e, data.scatter_y)]
            }).collect();
            pts1 = Vec::new();
       }
       data.scatter_cache = Some((data.scatter_x, data.scatter_y, data.scatter_color, data.max_points, pts0, pts1));
    }
    
    let (_, _, _, _, pts0, pts1) = data.scatter_cache.as_ref().unwrap();
    let p0_cloned = pts0.clone();
    let p1_cloned = pts1.clone();

    Plot::new("scatter_plot")
        .show(ui, |plot_ui| {
            if is_bit_color {
                 plot_ui.points(Points::new(p0_cloned).color(egui::Color32::RED).name("Bit 0"));
                 plot_ui.points(Points::new(p1_cloned).color(egui::Color32::GREEN).name("Bit 1"));
            } else {
                 plot_ui.points(Points::new(p0_cloned).name("Entries"));
            }
        });
}

fn render_stats(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.heading("Statistical Summary");
    
    ui.collapsing("Binary Sequence (0 vs 1)", |ui| {
        ui.label(format!("Total entries: {}", data.stats.total_entries));
        ui.label(format!("Ones (Remainder 1): {} ({:.2}%)", data.stats.ones_count, data.stats.ratio_ones * 100.0));
        ui.label(format!("Zeros (Exact Sum): {} ({:.2}%)", data.stats.zeros_count, (1.0 - data.stats.ratio_ones) * 100.0));
        ui.label(format!("Max Run of 0s: {}", data.stats.max_run_0));
        ui.label(format!("Max Run of 1s: {}", data.stats.max_run_1));
    });

    ui.collapsing("Correlations (Pearson)", |ui| {
        ui.label(format!("Gap vs. Component Count: {:.4}", data.stats.corr_gap_comp));
        ui.label(format!("Gap vs. Bit: {:.4}", data.stats.corr_gap_bit));
        ui.label(format!("Component Count vs. Bit: {:.4}", data.stats.corr_comp_bit));
        
        ui.label("Note: Interpretation of Bit correlation:");
        ui.label("  Positive -> Higher values associated with Bit 1");
        ui.label("  Negative -> Higher values associated with Bit 0");
    });
}

fn combo_attr(ui: &mut egui::Ui, current: &mut Attribute, id: &str) {
    egui::ComboBox::from_id_source(id)
        .selected_text(current.name())
        .show_ui(ui, |ui| {
            for attr in Attribute::all() {
                ui.selectable_value(current, attr, attr.name());
            }
        });
}

fn get_value(e: &Entry, attr: Attribute) -> f64 {
    match attr {
        Attribute::Bit => e.bit as f64,
        Attribute::Gap => e.gap as f64,
        Attribute::NumComponents => e.num_components as f64,
        Attribute::LargestPrime => e.largest_component as f64,
        Attribute::TailSum => e.tail_sum as f64,
        Attribute::PrimeIndex => e.prime_idx as f64,
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Attribute Explorer",
        options,
        Box::new(|cc| Ok(Box::new(VizAttributesApp::new(cc)))),
    )
}
