use crate::{AnyError, Point};
use plotters::prelude::*;
use std::{fs, path::Path};

fn metrics(point: &Point) -> Option<(f64, f64, f64)> {
    if point.failures > 0 {
        return None;
    }
    Some((point.p95_ttft_ms?, point.output_tokens_per_second?, point.usd_per_million_output_tokens?))
}

fn label(value: Option<f64>) -> String {
    value.map(|x| format!("{x:.2}")).unwrap_or_else(|| "—".into())
}

fn plot(path: &Path, points: &[Point], demo: bool) -> Result<(), AnyError> {
    let points: Vec<_> = points.iter().filter_map(|point| metrics(point).map(|row| (point, row))).collect();
    let root = SVGBackend::new(path, (1350, 565)).into_drawing_area();
    root.fill(&WHITE)?;
    let heading = if demo { "Inference benchmark — synthetic fixture" } else { "Inference benchmark" };
    let root = root.titled(heading, ("sans-serif", 28))?;
    let mut engines: Vec<_> = points.iter().map(|(p, _)| &p.engine).collect();
    engines.sort();
    engines.dedup();
    let palettes = [BLUE, RED, GREEN, MAGENTA, CYAN, BLACK];
    let xmax = points.iter().map(|(_, (_, rate, _))| *rate).fold(1.0, f64::max) * 1.16;
    for (index, pane) in root.split_evenly((1, 2)).into_iter().enumerate() {
        let axis = if index == 0 { "P95 TTFT (ms)" } else { "USD per million output tokens" };
        let y = |ttft, cost| if index == 0 { ttft } else { cost };
        let ymax = points.iter().map(|(_, (ttft, _, cost))| y(*ttft, *cost)).fold(1.0, f64::max) * 1.18;
        let mut chart = ChartBuilder::on(&pane).margin(22).x_label_area_size(45).y_label_area_size(65)
            .build_cartesian_2d(0.0..xmax, 0.0..ymax)?;
        chart.configure_mesh().x_desc("Output throughput (tokens/sec)").y_desc(axis).draw()?;
        for (number, engine) in engines.iter().enumerate() {
            let color = palettes[number % palettes.len()];
            let series = points.iter().filter(|(point, _)| &point.engine == *engine);
            chart.draw_series(series.map(|(point, (ttft, rate, cost))| {
                let at = (*rate, y(*ttft, *cost));
                let mark = if point.pareto { color.filled() } else { color.stroke_width(2) };
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
    let rows: Vec<_> = points.iter().map(metrics).collect();
    for (point, own) in points.iter_mut().zip(&rows) {
        point.pareto = own.is_some_and(|(ttft, rate, cost)| {
            !rows.iter().flatten().any(|&(b_ttft, b_rate, b_cost)| {
                b_ttft <= ttft && b_rate >= rate && b_cost <= cost
                    && (b_ttft < ttft || b_rate > rate || b_cost < cost)
            })
        });
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
    markdown.push_str("| Engine | Version | Hardware | Model | Concurrent | Success | Failed | P95 TTFT ms | P95 total ms | Output tokens/s | USD/M output tokens | Pareto |\n");
    markdown.push_str("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    let escape = |text: &str| text.replace('|', "\\|").replace(['\n', '\r'], " ");
    for point in &points {
        csv.serialize(point)?;
        markdown.push_str(&format!("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            escape(&point.engine), escape(&point.version), escape(&point.hardware), escape(&point.model),
            point.concurrency, point.successes, point.failures, label(point.p95_ttft_ms),
            label(point.p95_total_ms), label(point.output_tokens_per_second),
            label(point.usd_per_million_output_tokens), if point.pareto { "Yes" } else { "—" }));
    }
    csv.flush()?;
    plot(&output.join("pareto.svg"), &points, demo)?;
    fs::write(output.join("REPORT.md"), markdown)?;
    println!("Report written to {}", output.display());
    Ok(())
}
