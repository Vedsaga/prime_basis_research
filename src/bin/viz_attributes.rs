use eframe::egui;
use egui_plot::{Legend, Line, Bar, BarChart, Plot, Points};
use prime_basis_research::PrimeDatabase;
use prime_basis_research::analysis::{
    lz76_complexity, transition_matrix, entropy_rate, power_spectrum, shannon_entropy, bit_autocorrelation
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

// const WINDOW_SIZE: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    #[allow(dead_code)]
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
    
    // Conditional Means
    mean_gap_bit0: f64,
    mean_gap_bit1: f64,
    mean_comp_bit0: f64,
    mean_comp_bit1: f64,
    
    // Advanced
    transition_matrix: [[f64; 2]; 2], // P(i->j)
    bit_entropy: f64,
    entropy_rate: f64,
    lz_complexity: usize,
    #[allow(dead_code)]
    power_spectrum: Vec<f64>, // First 1000 coeffs or so? Or buckets?

    // New Advanced Metrics
    tail_sum_mean: f64,
    tail_sum_std: f64,
    tail_sum_skew: f64,
    tail_sum_kurt: f64,
    
    bit_autocorrelation: Vec<(usize, f64)>,
    prob_bit1_given_gap: Vec<(u64, f64, usize)>, // (gap, prob, count)
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
    current_report: String,
    
    // Timeline View
    timeline_attr: Attribute,
    timeline_cache: Option<(Attribute, Vec<[f64; 2]>)>,
    
    // Scatter View
    scatter_x: Attribute,
    scatter_y: Attribute,
    scatter_color: Attribute,
    max_points: usize,
    scatter_cache: Option<(Attribute, Attribute, Attribute, usize, Vec<[f64; 2]>, Vec<[f64; 2]>)>,
    
    // Histogram View
    hist_attr: Attribute,
    hist_bins: usize,
    hist_log: bool,
    hist_cache: Option<(Attribute, usize, bool, Vec<Bar>, String)>, // Bars + Stats Text
    
    // Windowed View
    win_attr: Attribute,
    win_size: usize,
    win_cache: Option<(Attribute, usize, Vec<[f64; 2]>, String)>, // Points + Stats Text
}

#[derive(PartialEq, Eq)]
enum Tab {
    Timeline,
    Scatter,
    Histogram,
    Windowed,
    Stats,
    Report,
}

struct VizAttributesApp {
    state: AppState,
    loader: Option<Receiver<Result<LoadedState, String>>>,
    // Persistent options across reloads could go here
}

impl VizAttributesApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
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
    
    let mut bits_raw = Vec::with_capacity(db.decompositions.len());

    let mut ones = 0;
    let mut zeros = 0;
    let mut max_run_0 = 0;
    let mut max_run_1 = 0;
    
    // Accumulators
    let mut sum_gap = 0.0;
    let mut sum_comp = 0.0;
    let mut sum_bit = 0.0;
    let mut sum_gap_sq = 0.0;
    let mut sum_comp_sq = 0.0;
    let mut sum_bit_sq = 0.0;
    let mut sum_gap_comp = 0.0;
    let mut sum_gap_bit = 0.0;
    let mut sum_comp_bit = 0.0;
    
    // Conditional Accumulators
    let mut sum_gap_b0 = 0.0;
    let mut sum_gap_b1 = 0.0;
    let mut sum_comp_b0 = 0.0;
    let mut sum_comp_b1 = 0.0;
    let mut count_b0 = 0;
    let mut count_b1 = 0;

    let n = db.decompositions.len() as f64;

    for (i, d) in db.decompositions.iter().enumerate() {
        let bit = if d.components.contains(&1) { 1 } else { 0 };
        bits_raw.push(bit);
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
        
        if bit == 1 {
            sum_gap_b1 += g;
            sum_comp_b1 += c;
            count_b1 += 1;
        } else {
            sum_gap_b0 += g;
            sum_comp_b0 += c;
            count_b0 += 1;
        }
    }

    // Push last run
    runs.push((current_bit, current_run));
    if current_bit == 0 {
        max_run_0 = max_run_0.max(current_run);
    } else {
        max_run_1 = max_run_1.max(current_run);
    }

    let pearson = |sum_x: f64, sum_y: f64, sum_xy: f64, sum_x2: f64, sum_y2: f64| -> f64 {
        let num = n * sum_xy - sum_x * sum_y;
        let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    };
    
    // Advanced Metrics
    let trans_mat = transition_matrix(&bits_raw);
    let lz = lz76_complexity(&bits_raw);
    // Shannon entropy of bit distribution
    let mut bit_counts = HashMap::new();
    bit_counts.insert(0, zeros);
    bit_counts.insert(1, ones);
    let h_bit = shannon_entropy(&bit_counts);
    let h_rate = entropy_rate(trans_mat);
    
    // Power spectrum (expensive? 1M FFT is ~50-100ms usually)
    // We only keep low frequency part for report/viz usually, but let's store basics?
    // Actually full spectrum is huge. Just store first 1000 for now or summary?
    // User asked for "Power spectrum of Bit sequence".
    // Computing it is fine, storing 1M floats is fine (8MB).
    // Let's compute it but maybe only store a summary or the whole thing if needed for a plot later.
    // For now, let's just store the first 1000 coefficients.
    let spectrum_full = power_spectrum(&bits_raw);
    let spectrum_summary = spectrum_full.into_iter().take(1000).collect();

    // Tail Sum Stats
    let tail_sums: Vec<f64> = entries.iter().map(|e| e.tail_sum as f64).collect();
    let ts_mean = tail_sums.iter().sum::<f64>() / n;
    let ts_var = tail_sums.iter().map(|v| (v - ts_mean).powi(2)).sum::<f64>() / n;
    let ts_std = ts_var.sqrt();
    let ts_skew = if ts_std > 0.0 { tail_sums.iter().map(|v| ((v - ts_mean)/ts_std).powi(3)).sum::<f64>() / n } else { 0.0 };
    let ts_kurt = if ts_std > 0.0 { tail_sums.iter().map(|v| ((v - ts_mean)/ts_std).powi(4)).sum::<f64>() / n - 3.0 } else { 0.0 };

    // Bit Autocorrelation
    let bit_ac = bit_autocorrelation(&bits_raw, 20);

    // P(Bit=1 | Gap=g)
    let mut gap_bit_counts: HashMap<u64, (usize, usize)> = HashMap::new(); // gap -> (count_total, count_ones)
    for e in &entries {
        if e.gap <= 50 {
            let entry = gap_bit_counts.entry(e.gap).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += e.bit as usize;
        }
    }
    let mut prob_bit1_given_gap: Vec<(u64, f64, usize)> = gap_bit_counts.into_iter()
        .map(|(gap, (total, ones))| (gap, ones as f64 / total as f64, total))
        .collect();
    prob_bit1_given_gap.sort_by_key(|k| k.0);

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
        
        mean_gap_bit0: if count_b0 > 0 { sum_gap_b0 / count_b0 as f64 } else { 0.0 },
        mean_gap_bit1: if count_b1 > 0 { sum_gap_b1 / count_b1 as f64 } else { 0.0 },
        mean_comp_bit0: if count_b0 > 0 { sum_comp_b0 / count_b0 as f64 } else { 0.0 },
        mean_comp_bit1: if count_b1 > 0 { sum_comp_b1 / count_b1 as f64 } else { 0.0 },
        
        transition_matrix: trans_mat,
        bit_entropy: h_bit,
        entropy_rate: h_rate,
        lz_complexity: lz,
        power_spectrum: spectrum_summary,
        tail_sum_mean: ts_mean,
        tail_sum_std: ts_std,
        tail_sum_skew: ts_skew,
        tail_sum_kurt: ts_kurt,
        bit_autocorrelation: bit_ac,
        prob_bit1_given_gap: prob_bit1_given_gap,
    };
    
    // Generate initial report
    let report = generate_report_text(&entries, &stats, Tab::Timeline, Attribute::Bit, Attribute::Gap, Attribute::NumComponents, 5000, 1000, 50);

    tx.send(Ok(LoadedState {
        entries,
        runs,
        stats,
        selected_tab: Tab::Timeline,
        current_report: report,
        timeline_attr: Attribute::Bit,
        timeline_cache: None,
        scatter_x: Attribute::Gap,
        scatter_y: Attribute::NumComponents,
        scatter_color: Attribute::Bit,
        max_points: 5000,
        scatter_cache: None,
        hist_attr: Attribute::Gap,
        hist_bins: 50,
        hist_log: true,
        hist_cache: None,
        win_attr: Attribute::Bit,
        win_size: 1000,
        win_cache: None,
    })).ok();
}

impl eframe::App for VizAttributesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(rx) = &self.loader {
            match rx.try_recv() {
                Ok(result) => {
                    self.state = match result {
                        Ok(data) => AppState::Loaded(data),
                        Err(e) => AppState::Error(e),
                    };
                    self.loader = None;
                }
                Err(_) => {} 
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            match &mut self.state {
                AppState::Loading => {
                    ui.centered_and_justified(|ui| {
                        ui.heading("Loading database & computing research metrics (FFT, LZ76)...");
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
        // Tabs
        ui.selectable_value(&mut data.selected_tab, Tab::Timeline, "Timeline");
        ui.selectable_value(&mut data.selected_tab, Tab::Scatter, "Scatter");
        ui.selectable_value(&mut data.selected_tab, Tab::Histogram, "Histogram");
        ui.selectable_value(&mut data.selected_tab, Tab::Windowed, "Windowed");
        ui.selectable_value(&mut data.selected_tab, Tab::Stats, "Stats");
        ui.selectable_value(&mut data.selected_tab, Tab::Report, "📄 Report");
    });
    ui.separator();

    match data.selected_tab {
        Tab::Timeline => render_timeline(ui, data),
        Tab::Scatter => render_scatter(ui, data),
        Tab::Histogram => render_histogram(ui, data),
        Tab::Windowed => render_windowed(ui, data),
        Tab::Stats => render_stats(ui, data),
        Tab::Report => render_report(ui, data),
    }
}

fn render_timeline(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        ui.label("Attribute:");
        let prev = data.timeline_attr;
        combo_attr(ui, &mut data.timeline_attr, "timeline_attr");
        if prev != data.timeline_attr {
            data.timeline_cache = None; // Invalidate
        }
    });

    // Check cache
    if data.timeline_cache.as_ref().map(|(a, _)| *a) != Some(data.timeline_attr) {
        let points: Vec<[f64; 2]> = match data.timeline_attr {
            Attribute::Bit => {
                let step = (data.runs.len() / 20000).max(1);
                let mut pts = Vec::with_capacity(data.runs.len() * 4 / step);
                let mut x = 0.0;
                for (i, &(bit, len)) in data.runs.iter().enumerate() {
                    let len_f = len as f64;
                    if i % step == 0 {
                        let y = if bit == 1 { len_f } else { -(len_f) };
                        pts.push([x, 0.0]);      
                        pts.push([x, y]);        
                        pts.push([x + len_f, y]); 
                        pts.push([x + len_f, 0.0]); 
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
    let points_cloned = points.clone(); 

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
        let p_x = data.scatter_x;
        let p_y = data.scatter_y;
        let p_c = data.scatter_color;
        let p_m = data.max_points;
        
        ui.label("X:"); combo_attr(ui, &mut data.scatter_x, "sc_x");
        ui.label("Y:"); combo_attr(ui, &mut data.scatter_y, "sc_y");
        ui.label("Color:"); combo_attr(ui, &mut data.scatter_color, "sc_c");
        ui.separator();
        ui.label("Points:"); ui.add(egui::DragValue::new(&mut data.max_points).range(100..=50000));
        
        if p_x != data.scatter_x || p_y != data.scatter_y || p_c != data.scatter_color || p_m != data.max_points {
            data.scatter_cache = None;
        }
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

fn render_histogram(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        let p_attr = data.hist_attr;
        let p_bins = data.hist_bins;
        let p_log = data.hist_log;
        
        ui.label("Attribute:"); combo_attr(ui, &mut data.hist_attr, "hist_attr");
        ui.label("Bins:"); ui.add(egui::DragValue::new(&mut data.hist_bins).range(5..=500));
        ui.checkbox(&mut data.hist_log, "Log Scale X");
        
        if p_attr != data.hist_attr || p_bins != data.hist_bins || p_log != data.hist_log {
            data.hist_cache = None;
        }
    });

    if data.hist_cache.is_none() {
        let values: Vec<f64> = data.entries.iter().map(|e| get_value(e, data.hist_attr)).collect();
        // Compute basic stats
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        let std_dev = var.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let skewness = values.iter().map(|v| ((v - mean)/std_dev).powi(3)).sum::<f64>() / count;
        let kurtosis = values.iter().map(|v| ((v - mean)/std_dev).powi(4)).sum::<f64>() / count - 3.0;

        let stats_text = format!(
            "Mean: {:.4}\nStd Dev: {:.4}\nMin: {:.2}\nMax: {:.2}\nSkewness: {:.4}\nKurtosis: {:.4}",
            mean, std_dev, min, max, skewness, kurtosis
        );
        
        // Binning
        let min_val = if data.hist_log && min <= 0.0 { 1.0 } else { min };
        let max_val = max;
        // if log, transform values
        let (eff_min, eff_max) = if data.hist_log {
            (min_val.ln(), max_val.ln())
        } else {
            (min_val, max_val)
        };
        
        let width = (eff_max - eff_min) / data.hist_bins as f64;
        let mut bins = vec![0u64; data.hist_bins];
        
        for &v in &values {
            let val = if data.hist_log { if v <= 0.0 { continue } else { v.ln() } } else { v };
            if val < eff_min || val > eff_max { continue; }
            let idx = ((val - eff_min) / width).floor() as usize;
            if idx < data.hist_bins {
                bins[idx] += 1;
            } else if idx == data.hist_bins {
                bins[idx - 1] += 1;
            }
        }
        
        let bars: Vec<Bar> = bins.iter().enumerate().map(|(i, &c)| {
            let center = eff_min + (i as f64 + 0.5) * width;
            let display_x = if data.hist_log { center.exp() } else { center };
            // For log scale bars widths are non-uniform in linear space
            // BarChart expects uniform bars usually, so maybe just plot as linear on log axis X?
            // egui_plot doesn't auto-log X for bars easily. 
            // Let's just plot 'center' as X argument.
            Bar::new(display_x, c as f64).width(width * (if data.hist_log { display_x } else { 1.0 })) 
        }).collect();
        
        data.hist_cache = Some((data.hist_attr, data.hist_bins, data.hist_log, bars, stats_text));
    }
    
    let (_, _, _, bars, text) = data.hist_cache.as_ref().unwrap();
    let bars_cloned = bars.clone();
    
    ui.columns(2, |cols| {
        cols[0].add(egui::Label::new(text));
        Plot::new("hist_plot")
            .show(&mut cols[1], |plot_ui| {
                plot_ui.bar_chart(BarChart::new(bars_cloned).color(egui::Color32::LIGHT_BLUE));
            });
    });
}

fn render_windowed(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        let p_attr = data.win_attr;
        let p_size = data.win_size;
        
        ui.label("Attribute:"); combo_attr(ui, &mut data.win_attr, "win_attr");
        ui.label("Window Size:"); ui.add(egui::DragValue::new(&mut data.win_size).range(100..=50000));
        
        if p_attr != data.win_attr || p_size != data.win_size {
            data.win_cache = None;
        }
    });

    if data.win_cache.is_none() {
        let values: Vec<f64> = data.entries.iter().map(|e| get_value(e, data.win_attr)).collect();
        let window = data.win_size;
        let mut rolling_means = Vec::with_capacity(values.len().saturating_sub(window));
        
        if values.len() >= window {
            let mut sum: f64 = values.iter().take(window).sum();
            rolling_means.push([0.0, sum / window as f64]);
            
            for i in 1..=(values.len() - window) {
                sum -= values[i - 1];
                sum += values[i + window - 1];
                rolling_means.push([i as f64, sum / window as f64]);
            }
        }
        
        // Simple stats on the rolling means
        let rm_vals: Vec<f64> = rolling_means.iter().map(|p| p[1]).collect();
        let min = rm_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = rm_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean = rm_vals.iter().sum::<f64>() / rm_vals.len().max(1) as f64;
        
        let text = format!("Window Check:\nMin Rolling Mean: {:.4}\nMax Rolling Mean: {:.4}\nAverage Rolling Mean: {:.4}", min, max, mean);
        
        data.win_cache = Some((data.win_attr, data.win_size, rolling_means, text));
    }
    
    let (_, _, points, text) = data.win_cache.as_ref().unwrap();
    let pts_cloned = points.clone();
    
    ui.label(text);
    Plot::new("window_plot")
        .show(ui, |plot_ui| {
             plot_ui.line(Line::new(pts_cloned).name("Rolling Mean"));
        });
}

fn render_stats(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.heading("Statistical Summary");
    
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.collapsing("Binary Sequence", |ui| {
            ui.label(format!("Count 1s: {} ({:.2}%)", data.stats.ones_count, data.stats.ratio_ones * 100.0));
            ui.label(format!("Count 0s: {} ({:.2}%)", data.stats.zeros_count, (1.0 - data.stats.ratio_ones) * 100.0));
            ui.label(format!("Max Run (0): {}", data.stats.max_run_0));
            ui.label(format!("Max Run (1): {}", data.stats.max_run_1));
        });

        ui.collapsing("Advanced Information", |ui| {
            ui.label(format!("Binary Entropy: {:.4} bits", data.stats.bit_entropy));
            ui.label(format!("Entropy Rate (Markov): {:.4} bits", data.stats.entropy_rate));
            ui.label(format!("LZ76 Complexity: {}", data.stats.lz_complexity));
            ui.label(format!("Transition [0->0]: {:.4}", data.stats.transition_matrix[0][0]));
            ui.label(format!("Transition [0->1]: {:.4}", data.stats.transition_matrix[0][1]));
            ui.label(format!("Transition [1->0]: {:.4}", data.stats.transition_matrix[1][0]));
            ui.label(format!("Transition [1->1]: {:.4}", data.stats.transition_matrix[1][1]));
        });

        ui.collapsing("Conditional Statistics", |ui| {
             ui.label("Mean Gap:");
             ui.label(format!("  | Bit=0: {:.4}", data.stats.mean_gap_bit0));
             ui.label(format!("  | Bit=1: {:.4}", data.stats.mean_gap_bit1));
             ui.label("Mean Components:");
             ui.label(format!("  | Bit=0: {:.4}", data.stats.mean_comp_bit0));
             ui.label(format!("  | Bit=1: {:.4}", data.stats.mean_comp_bit1));
        });
        
        ui.collapsing("Correlations", |ui| {
             ui.label(format!("Gap vs Comp: {:.4}", data.stats.corr_gap_comp));
             ui.label(format!("Gap vs Bit: {:.4}", data.stats.corr_gap_bit));
             ui.label(format!("Comp vs Bit: {:.4}", data.stats.corr_comp_bit));
        });

        ui.collapsing("Tail Sum Analysis", |ui| {
            ui.label(format!("Mean: {:.4}", data.stats.tail_sum_mean));
            ui.label(format!("Std Dev: {:.4}", data.stats.tail_sum_std));
            ui.label(format!("Skewness: {:.4}", data.stats.tail_sum_skew));
            ui.label(format!("Kurtosis: {:.4}", data.stats.tail_sum_kurt));
            ui.label("Interpretation:");
            if data.stats.tail_sum_skew > 1.0 { ui.label("  - Highly skewed (Exponential/Geometric like)"); }
            if data.stats.tail_sum_kurt > 0.0 { ui.label("  - Leptokurtic (Heavy tails/Peaked)"); }
        });

        ui.collapsing("Bit Autocorrelation (Lags 1-20)", |ui| {
            egui::Grid::new("ac_grid").striped(true).show(ui, |ui| {
                ui.label("Lag"); ui.label("Correlation"); ui.end_row();
                for (lag, val) in &data.stats.bit_autocorrelation {
                    ui.label(format!("{}", lag));
                    ui.label(format!("{:.6}", val));
                    ui.end_row();
                }
            });
        });

        ui.collapsing("P(Bit=1 | Gap=g) (g <= 50)", |ui| {
            egui::Grid::new("cond_prob_grid").striped(true).show(ui, |ui| {
                ui.label("Gap"); ui.label("P(Bit=1)"); ui.label("Count"); ui.end_row();
                for (gap, prob, count) in &data.stats.prob_bit1_given_gap {
                    ui.label(format!("{}", gap));
                    ui.label(format!("{:.4}", prob));
                    ui.label(format!("{}", count));
                    ui.end_row();
                }
            });
        });
    });
}

fn render_report(ui: &mut egui::Ui, data: &mut LoadedState) {
    ui.horizontal(|ui| {
        if ui.button("Generate Shareable Report (Refresh)").clicked() {
             data.current_report = generate_report_text(
                 &data.entries, &data.stats, 
                 // Pass simple current config, not view-state specific just yet for simplicity
                 Tab::Timeline, Attribute::Bit, Attribute::Gap, Attribute::NumComponents, 1000, 1000, 50 
             );
        }
        if ui.button("Export to File").clicked() {
            std::fs::write("analysis_report.txt", &data.current_report).ok();
        }
    });

    ui.add(
        egui::TextEdit::multiline(&mut data.current_report)
            .font(egui::TextStyle::Monospace)
            .desired_rows(30)
            .lock_focus(true)
    );
}

fn generate_report_text(
    entries: &[Entry], stats: &Stats, 
    _tab: Tab, _x: Attribute, _y: Attribute, _c: Attribute, 
    _max_pts: usize, _win: usize, _bins: usize
) -> String {
    let mut s = String::new();
    s.push_str("=== ANALYSIS REPORT ===\n");
    s.push_str(&format!("Dataset Size: {}\n", entries.len()));
    s.push_str(&format!("Prime Index Range: 0 - {}\n\n", entries.len().saturating_sub(1)));
    
    s.push_str("GLOBAL STATS:\n");
    s.push_str(&format!("  Ones Count: {}\n", stats.ones_count));
    s.push_str(&format!("  Zeros Count: {}\n", stats.zeros_count));
    s.push_str(&format!("  Ratio Ones: {:.4}\n", stats.ratio_ones));
    s.push_str(&format!("  Max Run 0: {}\n", stats.max_run_0));
    s.push_str(&format!("  Max Run 1: {}\n", stats.max_run_1));
    s.push_str(&format!("  Corr Gap vs Comp: {:.4}\n", stats.corr_gap_comp));
    s.push_str(&format!("  Corr Gap vs Bit: {:.4}\n", stats.corr_gap_bit));
    s.push_str(&format!("  Corr Comp vs Bit: {:.4}\n", stats.corr_comp_bit));
    
    s.push_str("\nCONDITIONAL MEANS:\n");
    s.push_str(&format!("  Mean Gap | Bit=0: {:.4}\n", stats.mean_gap_bit0));
    s.push_str(&format!("  Mean Gap | Bit=1: {:.4}\n", stats.mean_gap_bit1));
    s.push_str(&format!("  Mean Components | Bit=0: {:.4}\n", stats.mean_comp_bit0));
    s.push_str(&format!("  Mean Components | Bit=1: {:.4}\n", stats.mean_comp_bit1));
    
    s.push_str("\nINFORMATION METRICS:\n");
    s.push_str(&format!("  Bit Entropy: {:.4}\n", stats.bit_entropy));
    s.push_str(&format!("  Estimated Entropy Rate: {:.4}\n", stats.entropy_rate));
    s.push_str(&format!("  LZ76 Complexity: {}\n", stats.lz_complexity));
    
    s.push_str("\nTRANSITION MATRIX:\n");
    s.push_str(&format!("  0->0: {:.4}\n", stats.transition_matrix[0][0]));
    s.push_str(&format!("  0->1: {:.4}\n", stats.transition_matrix[0][1]));
    s.push_str(&format!("  1->0: {:.4}\n", stats.transition_matrix[1][0]));
    s.push_str(&format!("  1->1: {:.4}\n", stats.transition_matrix[1][1]));
    
    s.push_str("\nNOTES:\n");
    if stats.corr_gap_bit.abs() > 0.2 {
        s.push_str("  [!] Moderate correlation between Gap Size and Bit detected.\n");
    }
    if stats.entropy_rate < 0.95 {
        s.push_str("  [!] Entropy rate suggests structure (non-randomness) in binary sequence.\n");
    }
    
    s.push_str("=== END REPORT ===\n");

    s.push_str("\nTAIL SUM STATISTICS:\n");
    s.push_str(&format!("  Mean: {:.4}\n", stats.tail_sum_mean));
    s.push_str(&format!("  Std Dev: {:.4}\n", stats.tail_sum_std));
    s.push_str(&format!("  Skewness: {:.4}\n", stats.tail_sum_skew));
    s.push_str(&format!("  Kurtosis: {:.4}\n", stats.tail_sum_kurt));

    s.push_str("\nBIT AUTOCORRELATION (First 20 Lags):\n");
    for (lag, val) in &stats.bit_autocorrelation {
        s.push_str(&format!("  Lag {:2}: {:.6}\n", lag, val));
    }

    s.push_str("\nCONDITIONAL PROBABILITY P(Bit=1 | Gap=g):\n");
    for (gap, prob, count) in &stats.prob_bit1_given_gap {
        if *count > 10 { // Only show significant ones in text report to save space
             s.push_str(&format!("  Gap {:3}: {:.4} (n={})\n", gap, prob, count));
        }
    }
    s
}

fn combo_attr(ui: &mut egui::Ui, current: &mut Attribute, id: &str) {
    egui::ComboBox::from_id_salt(id)
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
