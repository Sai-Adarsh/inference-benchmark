use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
const REQUEST_LIMIT: Duration = Duration::from_secs(5);
const TEXT: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n";
const USAGE: &str = "data: {\"choices\":[],\"usage\":{\"completion_tokens\":3}}\n\n";
const DONE: &str = "data: [DONE]\n\n";

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "inference-bench-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cli(dir: &Temp, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_inference-benchmark"))
        .current_dir(&dir.0)
        .env("BENCHMARK_TEST_API_KEY", "integration-test")
        .env_remove("BENCHMARK_TEST_MISSING_KEY")
        .env("NO_PROXY", "127.0.0.1")
        .env("no_proxy", "127.0.0.1")
        .args(args)
        .output()
        .unwrap()
}

fn assert_ok(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_error(output: Output, message: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && stderr.contains(message),
        "{stderr}"
    );
}

fn load(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn save(path: &Path, contents: &Value) {
    fs::write(path, serde_json::to_vec(contents).unwrap()).unwrap();
}

fn config(url: &str, requests: usize) -> Value {
    json!({
        "prompts": (0..requests).map(|i| format!("case-{i}")).collect::<Vec<_>>(),
        "requests": requests,
        "max_tokens": 16,
        "concurrency": [requests.min(3)],
        "timeout_seconds": 2,
        "endpoints": [{
            "engine": "vLLM | loopback", "version": "test-1", "hardware": "CPU",
            "model": "local-test", "model_revision": "commit-1", "precision": "bf16",
            "hourly_usd": 2.0, "base_url": url, "api_key_env": "BENCHMARK_TEST_API_KEY"
        }]
    })
}

fn receive(stream: &mut TcpStream) -> String {
    stream.set_nonblocking(false).unwrap();
    stream.set_read_timeout(Some(REQUEST_LIMIT)).unwrap();
    let mut reader = BufReader::new(stream);
    let mut request = String::new();
    reader.read_line(&mut request).unwrap();
    assert_eq!(request, "POST /v1/chat/completions HTTP/1.1\r\n");
    let mut headers = BTreeMap::new();
    loop {
        let mut line = String::new();
        assert_ne!(reader.read_line(&mut line).unwrap(), 0);
        if line == "\r\n" {
            break;
        }
        let (name, value) = line.split_once(':').unwrap();
        headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
    }
    assert_eq!(headers["authorization"], "Bearer integration-test");
    let mut body = vec![0; headers["content-length"].parse().unwrap()];
    reader.read_exact(&mut body).unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["model"], "local-test");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["temperature"], 0);
    body["messages"][0]["content"].as_str().unwrap().to_string()
}

fn mock<F>(count: usize, handle: F) -> (String, thread::JoinHandle<()>)
where
    F: Fn(String, TcpStream) + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let worker = thread::spawn(move || {
        let until = Instant::now() + REQUEST_LIMIT;
        thread::scope(|scope| {
            for _ in 0..count {
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < until =>
                        {
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("server expected {count} requests: {error}"),
                    }
                };
                let handle = &handle;
                scope.spawn(move || handle(receive(&mut stream), stream));
            }
        });
    });
    (url, worker)
}

fn respond(stream: &mut TcpStream, status: &str, body: &str, suffix: &str) {
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
           body.len() + suffix.len()).unwrap();
    if !suffix.is_empty() {
        thread::sleep(Duration::from_millis(15));
        stream.write_all(suffix.as_bytes()).unwrap();
    }
}

fn successful_run(dir: &Temp, name: &str, change: impl FnOnce(&mut Value)) {
    let (url, worker) = mock(2, |_prompt, mut stream| {
        respond(
            &mut stream,
            "200 OK",
            "\u{feff}: keepalive\r\n\r\nevent: ping\r\ndata: ignored\r\n\r\ndata: {\"choices\":[\r\ndata: {\"delta\":{\"reasoning_content\":\"think\"}}]}\r\n\r\n",
            &format!("{TEXT}{USAGE}{DONE}"),
        );
    });
    let mut workload = config(&url, 1);
    workload["warmup_requests"] = 1.into();
    workload.as_object_mut().unwrap().remove("timeout_seconds");
    change(&mut workload);
    save(&dir.path("input.json"), &workload);
    assert_ok(cli(dir, &["run", "input.json", name]));
    worker.join().unwrap();
}

#[test]
fn records_and_compares_runs_with_workload_and_checksum_validation() {
    let dir = Temp::new();
    successful_run(&dir, "first", |_| {});
    successful_run(&dir, "second", |cfg| {
        cfg["endpoints"][0]["engine"] = "SGLang".into()
    });
    let first = load(dir.path("first/run.json"));
    assert_eq!(first["kind"], "live");
    assert_eq!(first["workload"]["warmup_requests_per_point"], 1);
    assert_eq!(first["workload"]["concurrency"], json!([1]));
    assert_eq!(first["workload"]["timeout_seconds"], 120);
    assert_eq!(
        first["workload"]["prompts_sha256"].as_str().unwrap().len(),
        64
    );
    let summary = load(dir.path("first/summary.json"));
    assert_eq!(summary[0]["successes"], 1);
    assert_eq!(summary[0]["pareto"], true);
    let gap = summary[0]["p95_max_frame_gap_ms"].as_f64().unwrap();
    let tpot = summary[0]["p95_estimated_tpot_ms"].as_f64().unwrap();
    assert!((gap / 2.0 - tpot).abs() < 1e-9);
    assert!(
        fs::read_to_string(dir.path("first/REPORT.md"))
            .unwrap()
            .contains(r"vLLM \| loopback")
    );
    let samples = fs::read_to_string(dir.path("first/samples.csv")).unwrap();
    assert_eq!(
        samples.lines().count(),
        2,
        "warmup must not become a data row"
    );

    let original = "first/summary.json";
    let second = "second/summary.json";
    assert_ok(cli(&dir, &["compare", "output/nested", original, second]));
    let combined = load(dir.path("output/nested/run.json"));
    assert_eq!(combined["kind"], "comparison");
    assert_eq!(combined["input_runs"].as_array().unwrap().len(), 2);
    assert_eq!(
        load(dir.path("output/nested/summary.json"))
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_error(
        cli(&dir, &["compare", "again", "output/nested/summary.json"]),
        "live run",
    );
    assert_error(
        cli(&dir, &["compare", "again", original, original]),
        "more than once",
    );
    assert_error(
        cli(&dir, &["compare", "output/nested", original]),
        "already exists",
    );

    let manifest = dir.path("second/run.json");
    let mut modified = load(manifest.clone());
    modified["workload"]["slo"] = json!({"ttft_ms": 100, "total_ms": 200});
    save(&manifest, &modified);
    assert_error(
        cli(&dir, &["compare", "again", original, second]),
        "settings differ",
    );
    modified["workload"].as_object_mut().unwrap().remove("slo");
    modified["workload"]["model_revision"] = "different".into();
    save(&manifest, &modified);
    assert_error(
        cli(&dir, &["compare", "again", original, second]),
        "settings differ",
    );
    modified["workload"]["precision"] = Value::Null;
    save(&manifest, &modified);
    assert_error(
        cli(&dir, &["compare", "again", original, second]),
        "model_revision and precision",
    );
    modified["workload"]["precision"] = "bf16".into();
    save(&manifest, &modified);
    fs::write(dir.path(second), "[]").unwrap();
    assert_error(cli(&dir, &["compare", "again", second]), "summary checksum");
    modified["summary_sha256"] =
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945".into();
    save(&manifest, &modified);
    assert_error(
        cli(&dir, &["compare", "again", second]),
        "expected nonempty",
    );
}

#[test]
fn counts_stream_errors_and_excludes_all_failed_points_from_pareto_frontier() {
    let (url, worker) = mock(8, |prompt, mut stream| match prompt.as_str() {
        "case-0" => respond(&mut stream, "503 Service Unavailable", "", ""),
        "case-1" => respond(&mut stream, "200 OK", "event: error\ndata: failure\n\n", ""),
        "case-2" => respond(
            &mut stream,
            "200 OK",
            "data: {\"error\":{\"message\":\"fail\"}}\n\n",
            "",
        ),
        "case-3" => respond(&mut stream, "200 OK", "data: incomplete-json\n\n", ""),
        "case-4" => respond(&mut stream, "200 OK", TEXT, "data: [DONE]\n"),
        "case-5" => respond(&mut stream, "200 OK", DONE, ""),
        "case-6" => respond(&mut stream, "200 OK", TEXT, DONE),
        "case-7" => stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\ndata: \xff\n\n",
            )
            .unwrap(),
        _ => panic!("unexpected {prompt}"),
    });
    let dir = Temp::new();
    save(&dir.path("input.json"), &config(&url, 8));
    assert_ok(cli(&dir, &["run", "input.json", "errors"]));
    worker.join().unwrap();
    let expected = [
        "HTTP 503",
        "error event",
        "error frame",
        "Invalid JSON",
        "before [DONE]",
        "No generated text",
        "Missing positive usage",
        "Invalid stream",
    ];
    let mut rows = csv::Reader::from_path(dir.path("errors/samples.csv")).unwrap();
    let headers = rows.headers().unwrap().clone();
    let error = headers.iter().position(|name| name == "error").unwrap();
    let request = headers.iter().position(|name| name == "request").unwrap();
    let records: Vec<_> = rows.records().map(Result::unwrap).collect();
    assert_eq!(records.len(), expected.len());
    for (i, (row, expected)) in records.iter().zip(expected).enumerate() {
        assert_eq!(&row[request], i.to_string());
        assert!(row[error].contains(expected), "{:?}", row);
    }
    let summary = load(dir.path("errors/summary.json"));
    assert_eq!(summary[0]["failures"], 8);
    assert_eq!(summary[0]["successes"], 0);
    assert_eq!(summary[0]["pareto"], false);
    assert_eq!(summary[0]["output_tokens_per_second"], Value::Null);
}

#[test]
fn counts_conflicting_stream_usage_as_a_failed_sample() {
    let (url, worker) = mock(2, |prompt, mut stream| {
        let second_usage = match prompt.as_str() {
            "case-0" => USAGE.replace(":3", ":4"),
            "case-1" => USAGE.to_owned(),
            _ => panic!("unexpected {prompt}"),
        };
        respond(&mut stream, "200 OK", &format!("{TEXT}{USAGE}{second_usage}{DONE}"), "");
    });
    let dir = Temp::new();
    save(&dir.path("input.json"), &config(&url, 2));
    assert_ok(cli(&dir, &["run", "input.json", "usage"]));
    worker.join().unwrap();

    let summary = load(dir.path("usage/summary.json"));
    assert_eq!(summary[0]["failures"], 1);
    assert_eq!(summary[0]["successes"], 1);
    let mut reader = csv::Reader::from_path(dir.path("usage/samples.csv")).unwrap();
    let error_column = reader.headers().unwrap().iter().position(|name| name == "error").unwrap();
    let samples: Vec<_> = reader.records().map(Result::unwrap).collect();
    assert_eq!(&samples[0][error_column], "Conflicting usage.completion_tokens");
    assert_eq!(&samples[1][error_column], "");
}

#[test]
fn validates_cli_and_refuses_synthetic_or_incompatible_inputs() {
    let dir = Temp::new();
    assert_error(cli(&dir, &[]), "usage");
    assert_error(cli(&dir, &["compare", "output"]), "usage");
    assert_error(cli(&dir, &["demo", ""]), "NotFound");
    assert_ok(cli(&dir, &["demo", "demo"]));
    let summary = load(dir.path("demo/summary.json"));
    assert_eq!(summary.as_array().unwrap().len(), 9);
    assert!(
        summary
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["pareto"] == true)
    );
    assert!(
        summary
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["pareto"] == false)
    );
    assert!(
        fs::read_to_string(dir.path("demo/pareto.svg"))
            .unwrap()
            .contains("synthetic fixture")
    );
    assert_error(
        cli(&dir, &["compare", "output", "demo/summary.json"]),
        "completed live runs",
    );

    let run = |cfg: &Value, error: &str| {
        save(&dir.path("input.json"), cfg);
        assert_error(cli(&dir, &["run", "input.json", "output"]), error);
        assert!(!dir.path("output").exists());
    };
    let original = config("http://127.0.0.1:1/v1", 3);
    for (key, invalid) in [
        ("endpoints", json!([])),
        ("prompts", json!([])),
        ("prompts", json!([" "])),
        ("requests", 0.into()),
        ("max_tokens", 0.into()),
        ("timeout_seconds", 0.into()),
        ("concurrency", json!([])),
        ("concurrency", json!([0])),
        ("concurrency", json!([4])),
    ] {
        let mut cfg = original.clone();
        cfg[key] = invalid;
        run(&cfg, "positive workloads");
    }
    for (key, invalid) in [
        ("engine", " ".into()),
        ("hardware", "".into()),
        ("model", "".into()),
        ("version", " ".into()),
        ("hourly_usd", (-1).into()),
    ] {
        let mut cfg = original.clone();
        cfg["endpoints"][0][key] = invalid;
        run(&cfg, "positive workloads");
    }
    let mut cfg = original.clone();
    cfg["endpoints"][0]["base_url"] = "file:///tmp".into();
    run(&cfg, "HTTP(S) URL");
    cfg = original.clone();
    cfg["endpoints"][0]["api_key_env"] = "BENCHMARK_TEST_MISSING_KEY".into();
    run(&cfg, "Missing API key");
    cfg = original.clone();
    let endpoint = cfg["endpoints"][0].clone();
    cfg["endpoints"].as_array_mut().unwrap().push(endpoint);
    cfg["endpoints"][1]["model_revision"] = "different".into();
    run(&cfg, "same model ID");
    cfg["endpoints"][0]["model_revision"] = Value::Null;
    cfg["endpoints"][1]["model_revision"] = Value::Null;
    run(&cfg, "benchmarking multiple endpoints");
}

#[test]
fn distinguishes_request_and_body_failures_from_a_single_frame_success() {
    let (url, worker) = mock(4, |prompt, mut stream| {
        match prompt.as_str() {
            "case-0" => {} // Disconnect before sending response headers.
            "case-1" => thread::sleep(Duration::from_millis(1_250)),
            "case-2" => write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: 65536\r\nConnection: close\r\n\r\n{TEXT}"
            )
            .unwrap(),
            "case-3" => respond(&mut stream, "200 OK", &format!("{TEXT}{USAGE}{DONE}"), ""),
            _ => panic!("unexpected {prompt}"),
        }
    });
    let dir = Temp::new();
    let mut cfg = config(&url, 4);
    cfg["timeout_seconds"] = 1.into();
    save(&dir.path("input.json"), &cfg);
    assert_ok(cli(&dir, &["run", "input.json", "mixed"]));
    worker.join().unwrap();
    let samples = fs::read_to_string(dir.path("mixed/samples.csv")).unwrap();
    for failure in ["Request failed", "Request timed out", "Stream read failed"] {
        assert!(samples.contains(failure), "{samples}");
    }
    let points = load(dir.path("mixed/summary.json"));
    assert_eq!(points[0]["successes"], 1);
    assert_eq!(points[0]["failures"], 3);
    assert_eq!(points[0]["pareto"], false);
    assert!(points[0]["p95_ttft_ms"].is_number());
    assert_eq!(points[0]["p95_estimated_tpot_ms"], Value::Null);
    assert_eq!(points[0]["p95_max_frame_gap_ms"], Value::Null);
}

#[test]
fn a_failed_warmup_cannot_produce_a_complete_run() {
    let (url, worker) = mock(1, |prompt, mut stream| {
        assert!(prompt.contains("benchmark warmup"));
        respond(&mut stream, "503 Service Unavailable", "", "");
    });
    let dir = Temp::new();
    let mut cfg = config(&url, 1);
    cfg["warmup_requests"] = 1.into();
    save(&dir.path("input.json"), &cfg);
    assert_error(
        cli(&dir, &["run", "input.json", "failed-run"]),
        "Warmup for vLLM | loopback: HTTP 503",
    );
    worker.join().unwrap();
    assert!(!dir.path("failed-run/run.json").exists());
    assert!(!dir.path("failed-run/summary.json").exists());
}

#[test]
fn slo_goodput_counts_only_proven_budget_compliance() {
    let (url, worker) = mock(5, |prompt, mut stream| {
        let single_usage = USAGE.replace(":3", ":1");
        let response = match prompt.as_str() {
            "case-0" => format!("{TEXT}{single_usage}{DONE}"),
            "case-1" => format!("{TEXT}{USAGE}{DONE}"),
            "case-2" => format!("{TEXT}{TEXT}{USAGE}{DONE}"),
            "case-4" => {
                let invalid = USAGE.replace(":3", ":17");
                format!("{TEXT}{TEXT}{invalid}{DONE}")
            }
            "case-3" => {
                respond(&mut stream, "503 Service Unavailable", "", "");
                return;
            }
            _ => panic!("unexpected request {prompt}"),
        };
        respond(&mut stream, "200 OK", &response, "");
    });
    let dir = Temp::new();
    let mut workload = config(&url, 5);
    workload["slo"] = json!({"ttft_ms": 5000.0, "total_ms": 5000.0, "estimated_tpot_ms": 5000.0});
    save(&dir.path("input.json"), &workload);
    assert_ok(cli(&dir, &["run", "input.json", "budgeted"]));
    worker.join().unwrap();

    let point = &load(dir.path("budgeted/summary.json"))[0];
    assert_eq!(point["successes"], 3);
    assert_eq!(point["slo_met_requests"], 2);
    assert_eq!(point["slo_output_tokens"], 4);
    let rate = point["output_tokens_per_second"].as_f64().unwrap();
    let goodput = point["slo_output_tokens_per_second"].as_f64().unwrap();
    assert!((goodput / rate - 4.0 / 7.0).abs() < 1e-12);
    let cost = point["usd_per_million_slo_tokens"].as_f64().unwrap();
    assert!((cost * goodput - 2.0 * 1e6 / 3600.0).abs() < 1e-9);
    assert_eq!(
        load(dir.path("budgeted/run.json"))["workload"]["slo"],
        workload["slo"]
    );

    let mut samples = csv::Reader::from_path(dir.path("budgeted/samples.csv")).unwrap();
    let column = samples
        .headers()
        .unwrap()
        .iter()
        .position(|h| h == "slo_met")
        .unwrap();
    let values: Vec<_> = samples
        .records()
        .map(|row| row.unwrap()[column].to_owned())
        .collect();
    assert_eq!(values, ["true", "false", "true", "false", "false"]);
    assert_eq!(point["slo_pareto"], true);
    assert_eq!(point["pareto"], false);
    assert!(
        fs::read_to_string(dir.path("budgeted/samples.csv"))
            .unwrap()
            .contains("exceed the request budget")
    );
    assert!(dir.path("budgeted/slo-pareto.svg").exists());
    let report = fs::read_to_string(dir.path("budgeted/REPORT.md")).unwrap();
    let slo_table = report.split_once("## SLO goodput").unwrap().1;
    assert!(slo_table.contains(r"vLLM \| loopback | test-1 | CPU"));
    let plot = fs::read_to_string(dir.path("budgeted/slo-pareto.svg")).unwrap();
    assert!(plot.contains("test-1 (CPU)"));
}

#[test]
fn rejects_a_nonpositive_latency_budget_before_creating_output() {
    let dir = Temp::new();
    let mut workload = config("http://127.0.0.1:1/v1/", 1);
    workload["slo"] = json!({"ttft_ms": -1, "total_ms": 100});
    save(&dir.path("input.json"), &workload);
    assert_error(
        cli(&dir, &["run", "input.json", "bad-slo"]),
        "SLO limits must be finite and positive",
    );
    assert!(!dir.path("bad-slo").exists());
}

#[test]
fn zero_qualified_tokens_has_no_unit_cost_or_frontier_position() {
    let (url, worker) = mock(1, |_prompt, mut stream| {
        respond(&mut stream, "200 OK", &format!("{TEXT}{USAGE}{DONE}"), "");
    });
    let dir = Temp::new();
    let mut workload = config(&url, 1);
    workload["slo"] = json!({"ttft_ms": 5000, "total_ms": 5000, "estimated_tpot_ms": 100});
    save(&dir.path("input.json"), &workload);
    assert_ok(cli(&dir, &["run", "input.json", "zero"]));
    worker.join().unwrap();
    let report = load(dir.path("zero/summary.json"));
    assert_eq!(report[0]["slo_met_requests"], 0);
    assert_eq!(report[0]["slo_output_tokens_per_second"], 0.0);
    assert_eq!(report[0]["usd_per_million_slo_tokens"], Value::Null);
    assert_eq!(report[0]["slo_pareto"], false);
    assert_eq!(report[0]["pareto"], true);
}
