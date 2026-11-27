mod manifest;
mod report;
mod sse;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    env, fs,
    io::{self, BufReader},
    path::Path,
    sync::{Barrier, atomic::{AtomicUsize, Ordering}},
    thread,
    time::{Duration, Instant},
};

type AnyError = Box<dyn std::error::Error>;

const SAMPLING_TEMPERATURE: u8 = 0;

#[derive(Deserialize)]
struct Config {
    prompts: Vec<String>,
    concurrency: Vec<usize>,
    requests: usize,
    max_tokens: usize,
    #[serde(default = "default_timeout")]
    timeout_seconds: u64,
    #[serde(default)]
    warmup_requests: usize,
    endpoints: Vec<Endpoint>,
}

fn default_timeout() -> u64 { 120 }

#[derive(Deserialize)]
struct Endpoint {
    engine: String,
    version: String,
    hardware: String,
    model: String,
    model_revision: Option<String>,
    precision: Option<String>,
    base_url: String,
    hourly_usd: f64,
    api_key_env: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct Point {
    engine: String,
    version: String,
    hardware: String,
    model: String,
    concurrency: usize,
    successes: usize,
    failures: usize,
    p95_ttft_ms: Option<f64>,
    p95_estimated_tpot_ms: Option<f64>,
    p95_max_frame_gap_ms: Option<f64>,
    p95_total_ms: Option<f64>,
    output_tokens_per_second: Option<f64>,
    usd_per_million_output_tokens: Option<f64>,
    pareto: bool,
}

#[derive(Serialize)]
struct Sample {
    engine: String,
    concurrency: usize,
    request: usize,
    prompt_index: usize,
    ttft_ms: Option<f64>,
    estimated_tpot_ms: Option<f64>,
    max_frame_gap_ms: Option<f64>,
    text_frames: usize,
    total_ms: f64,
    completion_tokens: Option<u64>,
    error: Option<String>,
}

struct Target<'a> {
    endpoint: &'a Endpoint,
    url: reqwest::Url,
    key: Option<String>,
}

impl<'a> Target<'a> {
    fn new(endpoint: &'a Endpoint) -> Result<Self, AnyError> {
        let url = reqwest::Url::parse(&format!("{}/chat/completions", endpoint.base_url.trim_end_matches('/')))?;
        if !matches!(url.scheme(), "http" | "https") || url.host().is_none() {
            return Err("Endpoint needs an HTTP(S) URL".into());
        }
        let key = endpoint.api_key_env.as_deref().map(env::var).transpose()
            .map_err(|_| format!("Missing API key environment variable for {}", endpoint.engine))?;
        Ok(Self { endpoint, url, key })
    }
}

struct StreamStats {
    ttft_ms: f64,
    estimated_tpot_ms: Option<f64>,
    max_frame_gap_ms: Option<f64>,
    text_frames: usize,
    completion_tokens: u64,
}

fn stream_chat(client: &Client, target: &Target<'_>, prompt: &str, max_tokens: usize,
               start: Instant) -> Result<StreamStats, String> {
    let mut request = client.post(target.url.clone()).json(&json!({
        "model": target.endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": SAMPLING_TEMPERATURE,
        "stream": true,
        "stream_options": {"include_usage": true}
    }));
    if let Some(key) = &target.key {
        request = request.bearer_auth(key);
    }
    let response = request.send().map_err(|error| {
        if error.is_timeout() { "Request timed out" } else { "Request failed" }
    })?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status().as_u16()));
    }
    let mut first_text_ms = None;
    let mut last_text_ms = None;
    let mut max_frame_gap_ms: Option<f64> = None;
    let mut text_frames = 0;
    let mut completion_tokens = None;
    let mut done = false;
    let mut events = sse::Events::new(BufReader::new(response));
    while let Some(frame) = events.next().map_err(|error| match error.kind() {
        io::ErrorKind::InvalidData => format!("Invalid stream: {error}"),
        _ => "Stream read failed".to_string(),
    })? {
        if frame.kind == "error" {
            return Err("Upstream returned an error event".into());
        }
        let data = frame.data.trim();
        if data.is_empty() || frame.kind == "ping" {
            continue;
        }
        if data == "[DONE]" {
            done = true;
            break;
        }
        let event: Value = serde_json::from_str(data).map_err(|_| "Invalid JSON stream frame")?;
        if event.get("error").is_some_and(|v| !v.is_null()) {
            return Err("Upstream returned an error frame".into());
        }
        let delta = &event["choices"][0]["delta"];
        if ["content", "reasoning_content", "reasoning"].iter().any(|key| {
            delta[*key].as_str().is_some_and(|text| !text.is_empty())
        }) {
            let now_ms = start.elapsed().as_secs_f64() * 1_000.0;
            first_text_ms.get_or_insert(now_ms);
            if let Some(previous) = last_text_ms {
                let gap = now_ms - previous;
                max_frame_gap_ms = Some(max_frame_gap_ms.unwrap_or(0.0).max(gap));
            }
            last_text_ms = Some(now_ms);
            text_frames += 1;
        }
        if let Some(tokens) = event["usage"]["completion_tokens"].as_u64() {
            completion_tokens = Some(tokens);
        }
    }
    if !done {
        return Err("Stream ended before [DONE]".into());
    }
    let ttft_ms = first_text_ms.ok_or("No generated text in stream")?;
    let completion_tokens = completion_tokens.filter(|&tokens| tokens > 0)
        .ok_or("Missing positive usage.completion_tokens")?;
    let estimated_tpot_ms = if text_frames >= 2 && completion_tokens > 1 {
        last_text_ms.map(|last| (last - ttft_ms) / (completion_tokens - 1) as f64)
    } else {
        None
    };
    Ok(StreamStats { ttft_ms, estimated_tpot_ms, max_frame_gap_ms, text_frames, completion_tokens })
}

fn percentile95(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[(values.len() * 95).div_ceil(100) - 1])
}

fn measure(client: &Client, target: &Target<'_>, config: &Config, concurrency: usize)
    -> Result<(Point, Vec<Sample>), AnyError> {
    for _ in 0..config.warmup_requests {
        stream_chat(client, target, "Reply to this benchmark warmup in one short sentence.",
                    config.max_tokens, Instant::now())
            .map_err(|error| format!("Warmup for {}: {error}", target.endpoint.engine))?;
    }
    let counter = AtomicUsize::new(0);
    let barrier = Barrier::new(concurrency + 1);
    let (elapsed, mut samples) = thread::scope(|scope| {
        let workers: Vec<_> = (0..concurrency).map(|_| {
            let counter = &counter;
            let barrier = &barrier;
            scope.spawn(move || {
                let mut samples = Vec::new();
                barrier.wait();
                loop {
                    let request = counter.fetch_add(1, Ordering::Relaxed);
                    if request >= config.requests {
                        break;
                    }
                    let prompt_index = request % config.prompts.len();
                    let start = Instant::now();
                    let result = stream_chat(client, target, &config.prompts[prompt_index], config.max_tokens, start);
                    let (stats, error) = match result {
                        Ok(stats) => (Some(stats), None),
                        Err(error) => (None, Some(error)),
                    };
                    samples.push(Sample {
                        engine: target.endpoint.engine.clone(),
                        concurrency,
                        request,
                        prompt_index,
                        ttft_ms: stats.as_ref().map(|s| s.ttft_ms),
                        estimated_tpot_ms: stats.as_ref().and_then(|s| s.estimated_tpot_ms),
                        max_frame_gap_ms: stats.as_ref().and_then(|s| s.max_frame_gap_ms),
                        text_frames: stats.as_ref().map_or(0, |s| s.text_frames),
                        total_ms: start.elapsed().as_secs_f64() * 1_000.0,
                        completion_tokens: stats.map(|s| s.completion_tokens),
                        error,
                    });
                }
                samples
            })
        }).collect();
        let start = Instant::now();
        barrier.wait();
        let mut samples = Vec::with_capacity(config.requests);
        for worker in workers {
            samples.extend(worker.join().expect("benchmark worker panicked"));
        }
        (start.elapsed().as_secs_f64(), samples)
    });
    samples.sort_by_key(|sample| sample.request);

    let tokens = samples.iter().filter_map(|sample| sample.completion_tokens).sum::<u64>();
    let successes = samples.iter().filter(|sample| sample.error.is_none()).count();
    let rate = if tokens > 0 { Some(tokens as f64 / elapsed) } else { None };
    let point = Point {
        engine: target.endpoint.engine.clone(),
        version: target.endpoint.version.clone(),
        hardware: target.endpoint.hardware.clone(),
        model: target.endpoint.model.clone(),
        concurrency,
        successes,
        failures: config.requests - successes,
        p95_ttft_ms: percentile95(samples.iter().filter_map(|s| s.ttft_ms).collect()),
        p95_estimated_tpot_ms: percentile95(samples.iter().filter_map(|s| s.estimated_tpot_ms).collect()),
        p95_max_frame_gap_ms: percentile95(samples.iter().filter_map(|s| s.max_frame_gap_ms).collect()),
        p95_total_ms: percentile95(samples.iter().filter(|s| s.error.is_none()).map(|s| s.total_ms).collect()),
        output_tokens_per_second: rate,
        usd_per_million_output_tokens: rate.map(|rate| target.endpoint.hourly_usd * 1_000_000.0 / 3_600.0 / rate),
        pareto: false,
    };
    Ok((point, samples))
}

fn valid_config(config: &Config) -> bool {
    !config.endpoints.is_empty() && config.endpoints.iter().all(|e| {
        !e.engine.trim().is_empty() && !e.version.trim().is_empty() && !e.hardware.trim().is_empty()
            && !e.model.trim().is_empty() && e.hourly_usd.is_finite() && e.hourly_usd >= 0.0
    }) && !config.prompts.is_empty() && config.prompts.iter().all(|p| !p.trim().is_empty())
        && config.requests > 0 && config.max_tokens > 0 && config.timeout_seconds > 0
        && !config.concurrency.is_empty()
        && config.concurrency.iter().all(|&c| c > 0 && c <= config.requests && c <= 512)
}

fn run(config_path: &Path, output: &Path) -> Result<(), AnyError> {
    let config: Config = serde_json::from_reader(fs::File::open(config_path)?)?;
    if !valid_config(&config) {
        return Err("Provide endpoints, nonempty prompts, positive workloads, and concurrency from 1 to min(requests, 512)".into());
    }
    let workload = manifest::Workload::from_config(&config)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(config.timeout_seconds))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let targets = config.endpoints.iter().map(Target::new).collect::<Result<Vec<_>, _>>()?;
    manifest::prepare_output(output)?;
    let started = manifest::timestamp()?;
    let mut report = Vec::new();
    let samples = fs::OpenOptions::new().write(true).create_new(true).open(output.join("samples.csv"))?;
    let mut samples_file = csv::Writer::from_writer(samples);
    for target in &targets {
        for &concurrency in &config.concurrency {
            let (point, samples) = measure(&client, target, &config, concurrency)?;
            eprintln!("{} @ {} concurrent: {} success, {} failed",
                      point.engine, point.concurrency, point.successes, point.failures);
            for sample in samples {
                samples_file.serialize(sample)?;
            }
            samples_file.flush()?;
            report.push(point);
        }
    }
    report::write(output, report, false, &workload.notes())?;
    manifest::write_run(output, workload, &config.endpoints, started)
}

fn demo(output: &Path) -> Result<(), AnyError> {
    manifest::prepare_output(output)?;
    let fixtures = [
        ("vLLM", 1, 39.0, 70.0, 28.0), ("vLLM", 8, 67.0, 290.0, 6.8), ("vLLM", 32, 195.0, 475.0, 4.1),
        ("SGLang", 1, 48.0, 75.0, 26.0), ("SGLang", 8, 70.0, 325.0, 6.0), ("SGLang", 32, 214.0, 490.0, 4.0),
        ("TensorRT-LLM", 1, 42.0, 88.0, 22.0), ("TensorRT-LLM", 8, 89.0, 310.0, 6.4),
        ("TensorRT-LLM", 32, 205.0, 520.0, 3.8),
    ];
    let points = fixtures.into_iter().map(|(engine, concurrency, ttft, rate, cost)| Point {
        engine: engine.into(), version: "fixture".into(), hardware: "fixture".into(), model: "fixture".into(),
        concurrency, successes: 64, failures: 0, p95_ttft_ms: Some(ttft),
        p95_estimated_tpot_ms: None, p95_max_frame_gap_ms: None, p95_total_ms: None,
        output_tokens_per_second: Some(rate), usd_per_million_output_tokens: Some(cost), pareto: false,
    }).collect();
    report::write(output, points, true, "Illustrative numbers for checking the report format.")
}

fn main() -> Result<(), AnyError> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["run", config, output] => run(Path::new(config), Path::new(output)),
        ["demo", output] => demo(Path::new(output)),
        ["compare", output, inputs @ ..] if !inputs.is_empty() => {
            manifest::Comparison::load(inputs)?.write(Path::new(output))
        }
        _ => Err("usage: inference-benchmark demo OUTPUT_DIR | run CONFIG.json OUTPUT_DIR | compare OUTPUT_DIR SUMMARY.json ...".into()),
    }
}
