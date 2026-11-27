use crate::{AnyError, Point};
use plotters::prelude::*;
use std::{fs, path::Path};

#[derive(Clone, Copy)]
enum Objective {
    Throughput,
    Goodput,
}

impl Objective {
    fn metrics(self, point: &Point) -> Option<(f64, f64, f64)> {
        let ttft = point.p95_ttft_ms?;
        match self {
            Self::Throughput if point.failures == 0 =>
                Some((ttft, point.output_tokens_per_second?, point.usd_per_million_output_tokens?)),
            Self::Throughput => None,
            Self::Goodput => Some((ttft, point.slo_output_tokens_per_second?, point.usd_per_million_slo_tokens?)),
        }
    }

    fn is_frontier(self, point: &Point) -> bool {
        match self {
            Self::Throughput => point.pareto,
            Self::Goodput => point.slo_pareto.unwrap_or(false),
        }
    }
}

fn frontier(points: &[Point], objective: Objective) -> Vec<bool> {
    let rows: Vec<_> = points.iter().map(|point| objective.metrics(point)).collect();
    rows.iter().map(|own| own.is_some_and(|(ttft, rate, cost)| {
        !rows.iter().flatten().any(|&(b_ttft, b_rate, b_cost)| {
            b_ttft <= ttft && b_rate >= rate && b_cost <= cost
                && (b_ttft < ttft || b_rate > rate || b_cost < cost)
        })
    })).collect()
}

fn label(value: Option<f64>) -> String {
    value.map(|x| format!("{x:.2}")).unwrap_or_else(|| "—".into())
}

fn plot(path: &Path, points: &[Point], demo: bool, objective: Objective) -> Result<(), AnyError> {
    let points: Vec<_> = points.iter().filter_map(|point| objective.metrics(point).map(|row| (point, row))).collect();
    let root = SVGBackend::new(path, (1350, 565)).into_drawing_area();
    root.fill(&WHITE)?;
    let (heading, throughput_axis, cost_axis) = match objective {
        Objective::Goodput => ("SLO goodput", "Qualified output tokens/sec", "USD per million qualified tokens"),
        Objective::Throughput => (
            if demo { "Inference benchmark — synthetic fixture" } else { "Inference benchmark" },
            "Output throughput (tokens/sec)", "USD per million output tokens",
        ),
    };
    let root = root.titled(heading, ("sans-serif", 28))?;
    let mut engines: Vec<_> = points.iter().map(|(p, _)| &p.engine).collect();
    engines.sort();
    engines.dedup();
    let palettes = [BLUE, RED, GREEN, MAGENTA, CYAN, BLACK];
    let xmax = points.iter().map(|(_, (_, rate, _))| *rate).fold(1.0, f64::max) * 1.16;
    for (index, pane) in root.split_evenly((1, 2)).into_iter().enumerate() {
        let axis = if index == 0 { "P95 TTFT (ms)" } else { cost_axis };
        let y = |ttft, cost| if index == 0 { ttft } else { cost };
        let ymax = points.iter().map(|(_, (ttft, _, cost))| y(*ttft, *cost)).fold(1.0, f64::max) * 1.18;
        let mut chart = ChartBuilder::on(&pane).margin(22).x_label_area_size(45).y_label_area_size(65)
            .build_cartesian_2d(0.0..xmax, 0.0..ymax)?;
        chart.configure_mesh().x_desc(throughput_axis).y_desc(axis).draw()?;
        for (number, engine) in engines.iter().enumerate() {
            let color = palettes[number % palettes.len()];
            let series = points.iter().filter(|(point, _)| &point.engine == *engine);
            chart.draw_series(series.map(|(point, (ttft, rate, cost))| {
                let at = (*rate, y(*ttft, *cost));
                let mark = if objective.is_frontier(point) { color.filled() } else { color.stroke_width(2) };
                EmptyElement::at(at) + Circle::new((0, 0), 5, mark) +
                    Text::new(point.concurrency.to_string(), (7, -7), ("sans-serif", 12))
            }))?.label(engine.as_str())
                .legend(move |(x, y)| Circle::new((x + 7, y), 4, color.filled()));
        }
        if !engines.is_empty() {
            chart.configure_series_labels().position(SeriesLabelPosition::UpperLeft)
                .background_style(WHITE.mix(0.85)).border_style(BLACK.mix(0.25)).draw()?;
        }
    }
    root.present()?;
    Ok(())
}

pub fn write(output: &Path, mut points: Vec<Point>, demo: bool, notes: &str) -> Result<(), AnyError> {
    let throughput_frontier = frontier(&points, Objective::Throughput);
    let has_slo = points.iter().any(|point| point.slo_met_requests.is_some());
    let goodput_frontier = has_slo.then(|| frontier(&points, Objective::Goodput));
    for (index, point) in points.iter_mut().enumerate() {
        point.pareto = throughput_frontier[index];
        point.slo_pareto = goodput_frontier.as_ref().map(|frontier| frontier[index]);
    }

    fs::create_dir_all(output)?;
    fs::write(output.join("summary.json"), serde_json::to_vec_pretty(&points)?)?;
    let mut csv = csv::Writer::from_path(output.join("summary.csv"))?;
    let mut markdown = String::from("# Inference benchmark\n\n");
    if demo {
        markdown.push_str("**Synthetic fixture. These values are illustrative.**\n\n");
    }
    markdown.push_str(notes);
    markdown.push_str("\n\n![Latency and cost vs throughput](pareto.svg)\n\n");
    markdown.push_str("Filled markers show the three-objective Pareto frontier. Labels show concurrency. Points with failed requests are excluded from the frontier and plot.\n\n");
    markdown.push_str("| Engine | Version | Hardware | Model | Concurrent | Success | Failed | P95 TTFT ms | P95 estimated TPOT ms | P95 max frame gap ms | P95 total ms | Output tokens/s | USD/M output tokens | Pareto |\n");
    markdown.push_str("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    let escape = |text: &str| text.replace('|', "\\|").replace(['\n', '\r'], " ");
    for point in &points {
        csv.serialize(point)?;
        markdown.push_str(&format!("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            escape(&point.engine), escape(&point.version), escape(&point.hardware), escape(&point.model),
            point.concurrency, point.successes, point.failures, label(point.p95_ttft_ms),
            label(point.p95_estimated_tpot_ms), label(point.p95_max_frame_gap_ms),
            label(point.p95_total_ms), label(point.output_tokens_per_second),
            label(point.usd_per_million_output_tokens), if point.pareto { "Yes" } else { "—" }));
    }
    csv.flush()?;
    plot(&output.join("pareto.svg"), &points, demo, Objective::Throughput)?;
    if has_slo {
        plot(&output.join("slo-pareto.svg"), &points, demo, Objective::Goodput)?;
        markdown.push_str("\n## SLO goodput\n\n![SLO goodput and cost](slo-pareto.svg)\n\n");
        markdown.push_str("Goodput counts tokens from requests that meet every budget. Failed and late requests still contribute to wall time and cost. The axes use p95 TTFT across successful requests and cost per million budget-qualified tokens. Points at zero goodput have no unit cost or frontier position.\n\n");
        markdown.push_str("| Engine | Concurrent | Requests meeting SLO | Qualified tokens | Qualified tokens/s | USD/M qualified tokens | Pareto |\n");
        markdown.push_str("|---|---:|---:|---:|---:|---:|---|\n");
        for point in &points {
            markdown.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                escape(&point.engine), point.concurrency, point.slo_met_requests.unwrap_or(0),
                point.slo_output_tokens.unwrap_or(0), label(point.slo_output_tokens_per_second),
                label(point.usd_per_million_slo_tokens),
                if point.slo_pareto == Some(true) { "Yes" } else { "—" },
            ));
        }
    }
    fs::write(output.join("REPORT.md"), markdown)?;
    println!("Report written to {}", output.display());
    Ok(())
}
