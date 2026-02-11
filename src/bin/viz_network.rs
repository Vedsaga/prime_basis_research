//! Visualization: Dependency Network
//!
//! Force-directed graph of base prime co-occurrence. Nodes are base primes
//! sized by support score; edges have thickness proportional to co-occurrence.
//! Uses egui::Painter for custom rendering with interactive hover/drag.
//!
//! Run: cargo run --release --bin viz_network [-- --cache path.bin]

use eframe::egui;
use prime_basis_research::analysis::co_occurrence_matrix;
use prime_basis_research::viz_common;
use prime_basis_research::PrimeDecomposition;

// ─── Data types ─────────────────────────────────────────────────────────────

struct NetworkNode {
    base_prime: u64,
    support_score: usize,
    pos: [f64; 2],
    vel: [f64; 2],
}

struct NetworkApp {
    nodes: Vec<NetworkNode>,
    /// (node_i, node_j, co-occurrence count)
    edges: Vec<(usize, usize, u64)>,
    /// All top bases from stats (up to 30), sorted by support descending
    all_top_bases: Vec<(u64, usize)>,
    /// Stored decompositions for recomputing co-occurrence on N change
    decompositions: Vec<PrimeDecomposition>,

    n_nodes: usize,
    paused: bool,
    hovered_node: Option<usize>,
    dragged_node: Option<usize>,
}

impl NetworkApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (db, stats) = viz_common::load_data();

        let all_top_bases: Vec<(u64, usize)> = stats.top_support.clone();
        let n_nodes = 20usize.min(all_top_bases.len());

        let mut app = Self {
            nodes: vec![],
            edges: vec![],
            all_top_bases,
            decompositions: db.decompositions,
            n_nodes,
            paused: false,
            hovered_node: None,
            dragged_node: None,
        };
        app.rebuild(n_nodes);
        app
    }

    /// Rebuild nodes and edges for the top `n` bases.
    fn rebuild(&mut self, n: usize) {
        let n = n.min(self.all_top_bases.len()).max(2);
        self.n_nodes = n;

        let top_bases: Vec<u64> = self.all_top_bases.iter().take(n).map(|(p, _)| *p).collect();
        let co_occ = co_occurrence_matrix(&self.decompositions, &top_bases);

        // Initialize nodes in a circle with random-ish offsets
        self.nodes.clear();
        for (i, &(prime, score)) in self.all_top_bases.iter().take(n).enumerate() {
            let angle = std::f64::consts::TAU * i as f64 / n as f64;
            let r = 200.0;
            self.nodes.push(NetworkNode {
                base_prime: prime,
                support_score: score,
                pos: [r * angle.cos(), r * angle.sin()],
                vel: [0.0, 0.0],
            });
        }

        // Build edges from upper triangle of co-occurrence matrix
        self.edges.clear();
        for i in 0..n {
            for j in (i + 1)..n {
                let w = co_occ[i][j];
                if w > 0 {
                    self.edges.push((i, j, w));
                }
            }
        }
    }

    /// Run one step of force-directed simulation.
    fn step_simulation(&mut self) {
        let n = self.nodes.len();
        if n < 2 {
            return;
        }

        // Accumulate forces
        let mut forces = vec![[0.0f64; 2]; n];

        // Repulsion: Coulomb-like between all pairs
        let repulsion_k = 50_000.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.nodes[i].pos[0] - self.nodes[j].pos[0];
                let dy = self.nodes[i].pos[1] - self.nodes[j].pos[1];
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt().max(1.0);
                let force = repulsion_k / dist_sq.max(1.0);
                let fx = force * dx / dist;
                let fy = force * dy / dist;
                forces[i][0] += fx;
                forces[i][1] += fy;
                forces[j][0] -= fx;
                forces[j][1] -= fy;
            }
        }

        // Attraction: spring along edges, strength ∝ co-occurrence weight
        let spring_k = 0.001;
        let max_weight = self.edges.iter().map(|e| e.2).max().unwrap_or(1) as f64;
        for &(i, j, w) in &self.edges {
            let dx = self.nodes[j].pos[0] - self.nodes[i].pos[0];
            let dy = self.nodes[j].pos[1] - self.nodes[i].pos[1];
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            let norm_w = w as f64 / max_weight;
            let force = spring_k * dist * norm_w;
            let fx = force * dx / dist;
            let fy = force * dy / dist;
            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[j][0] -= fx;
            forces[j][1] -= fy;
        }

        // Apply forces, update velocity with damping 0.95
        let damping = 0.95;
        let max_vel = 20.0;
        for (i, node) in self.nodes.iter_mut().enumerate() {
            // Skip dragged node
            if self.dragged_node == Some(i) {
                node.vel = [0.0, 0.0];
                continue;
            }
            node.vel[0] = (node.vel[0] + forces[i][0]) * damping;
            node.vel[1] = (node.vel[1] + forces[i][1]) * damping;
            // Clamp velocity
            let speed = (node.vel[0] * node.vel[0] + node.vel[1] * node.vel[1]).sqrt();
            if speed > max_vel {
                node.vel[0] *= max_vel / speed;
                node.vel[1] *= max_vel / speed;
            }
            node.pos[0] += node.vel[0];
            node.pos[1] += node.vel[1];
        }
    }

    /// Compute node radius from support score.
    fn node_radius(&self, score: usize) -> f32 {
        let max_score = self.nodes.iter().map(|n| n.support_score).max().unwrap_or(1) as f64;
        let min_r = 8.0f64;
        let max_r = 30.0;
        let t = score as f64 / max_score.max(1.0);
        (min_r + t * (max_r - min_r)) as f32
    }

    /// Compute edge thickness from weight.
    fn edge_thickness(&self, weight: u64) -> f32 {
        let max_w = self.edges.iter().map(|e| e.2).max().unwrap_or(1) as f64;
        let min_t = 0.5f64;
        let max_t = 6.0;
        let t = weight as f64 / max_w.max(1.0);
        (min_t + t * (max_t - min_t)) as f32
    }
}

impl eframe::App for NetworkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Run simulation step if not paused
        if !self.paused {
            self.step_simulation();
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.heading("Dependency Network");
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Nodes: {} | Edges: {} | Decompositions: {}",
                    self.nodes.len(),
                    self.edges.len(),
                    viz_common::format_num(self.decompositions.len()),
                ));
            });

            ui.horizontal(|ui| {
                // Pause/resume
                if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    self.paused = !self.paused;
                    if !self.paused {
                        ctx.request_repaint();
                    }
                }

                ui.separator();

                // N slider
                ui.label("Nodes (N):");
                let max_n = self.all_top_bases.len().min(30);
                let mut n = self.n_nodes;
                if ui.add(egui::Slider::new(&mut n, 2..=max_n)).changed() {
                    self.rebuild(n);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_rect_before_wrap();
            let center = available.center();

            // Interaction: detect hover and drag on the canvas
            let resp = ui.allocate_rect(available, egui::Sense::click_and_drag());
            let pointer_pos = resp.hover_pos();

            // Handle drag start/end
            if resp.drag_started() {
                if let Some(pos) = pointer_pos {
                    // Find closest node to pointer
                    let mut best = None;
                    let mut best_dist = f64::MAX;
                    for (i, node) in self.nodes.iter().enumerate() {
                        let nx = center.x as f64 + node.pos[0];
                        let ny = center.y as f64 + node.pos[1];
                        let dx = pos.x as f64 - nx;
                        let dy = pos.y as f64 - ny;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let r = self.node_radius(node.support_score) as f64;
                        if dist < r + 5.0 && dist < best_dist {
                            best = Some(i);
                            best_dist = dist;
                        }
                    }
                    self.dragged_node = best;
                }
            }
            if resp.drag_stopped() {
                self.dragged_node = None;
            }

            // Move dragged node to pointer
            if let (Some(di), Some(pos)) = (self.dragged_node, pointer_pos) {
                self.nodes[di].pos[0] = pos.x as f64 - center.x as f64;
                self.nodes[di].pos[1] = pos.y as f64 - center.y as f64;
                self.nodes[di].vel = [0.0, 0.0];
                if self.paused {
                    ctx.request_repaint();
                }
            }

            // Detect hover
            self.hovered_node = None;
            if let Some(pos) = pointer_pos {
                for (i, node) in self.nodes.iter().enumerate() {
                    let nx = center.x as f64 + node.pos[0];
                    let ny = center.y as f64 + node.pos[1];
                    let dx = pos.x as f64 - nx;
                    let dy = pos.y as f64 - ny;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let r = self.node_radius(node.support_score) as f64;
                    if dist < r + 3.0 {
                        self.hovered_node = Some(i);
                        break;
                    }
                }
            }

            // ─── Render with Painter ────────────────────────────────────
            let painter = ui.painter_at(available);

            // Draw edges
            for &(i, j, w) in &self.edges {
                let p1 = egui::pos2(
                    center.x + self.nodes[i].pos[0] as f32,
                    center.y + self.nodes[i].pos[1] as f32,
                );
                let p2 = egui::pos2(
                    center.x + self.nodes[j].pos[0] as f32,
                    center.y + self.nodes[j].pos[1] as f32,
                );
                let thickness = self.edge_thickness(w);

                // Highlight edges connected to hovered node
                let color = if self.hovered_node == Some(i) || self.hovered_node == Some(j) {
                    egui::Color32::from_rgb(255, 200, 50)
                } else {
                    egui::Color32::from_rgba_premultiplied(120, 120, 120, 100)
                };

                painter.line_segment([p1, p2], egui::Stroke::new(thickness, color));
            }

            // Draw nodes
            for (i, node) in self.nodes.iter().enumerate() {
                let pos = egui::pos2(
                    center.x + node.pos[0] as f32,
                    center.y + node.pos[1] as f32,
                );
                let r = self.node_radius(node.support_score);

                // Color: highlight hovered node
                let fill = if self.hovered_node == Some(i) {
                    egui::Color32::from_rgb(255, 100, 50)
                } else {
                    // Blue-ish gradient by index
                    let t = i as f32 / self.nodes.len().max(1) as f32;
                    egui::Color32::from_rgb(
                        (50.0 + t * 150.0) as u8,
                        (100.0 + t * 50.0) as u8,
                        (200.0 - t * 100.0) as u8,
                    )
                };

                painter.circle(pos, r, fill, egui::Stroke::new(1.5, egui::Color32::WHITE));

                // Label: base prime value
                painter.text(
                    pos,
                    egui::Align2::CENTER_CENTER,
                    format!("{}", node.base_prime),
                    egui::FontId::proportional(11.0),
                    egui::Color32::WHITE,
                );
            }

            // Tooltip on hover
            if let Some(hi) = self.hovered_node {
                let node = &self.nodes[hi];
                let tip_pos = egui::pos2(
                    center.x + node.pos[0] as f32 + self.node_radius(node.support_score) + 8.0,
                    center.y + node.pos[1] as f32 - 10.0,
                );
                let text = format!(
                    "Prime: {}\nSupport: {}",
                    node.base_prime,
                    viz_common::format_num(node.support_score),
                );
                let galley = painter.layout_no_wrap(
                    text,
                    egui::FontId::proportional(13.0),
                    egui::Color32::WHITE,
                );
                let rect = egui::Rect::from_min_size(
                    tip_pos,
                    galley.size() + egui::vec2(8.0, 4.0),
                );
                painter.rect_filled(
                    rect.expand(4.0),
                    4.0,
                    egui::Color32::from_rgba_premultiplied(30, 30, 30, 220),
                );
                painter.galley(tip_pos + egui::vec2(4.0, 2.0), galley, egui::Color32::WHITE);
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Dependency Network",
        native_options,
        Box::new(|cc| Ok(Box::new(NetworkApp::new(cc)))),
    )
}
