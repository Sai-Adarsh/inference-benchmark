use crate::{AnyError, Config, Endpoint, Point, SAMPLING_TEMPERATURE};
use ring::digest::{SHA256, digest};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const SCHEMA_VERSION: u32 = 2;
const FILE_NAME: &str = "run.json";

#[derive(Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Live,
    Comparison,
}

#[derive(Deserialize, PartialEq, Serialize)]
pub struct Workload {
    prompt_count: usize,
    prompts_sha256: String,
    concurrency: Vec<usize>,
    requests_per_point: usize,
    warmup_requests_per_point: usize,
    max_output_tokens: usize,
    timeout_seconds: u64,
    temperature: u8,
    model: String,
    model_revision: Option<String>,
    precision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slo: Option<crate::slo::Budget>,
}

impl Workload {
    pub fn from_config(config: &Config) -> Result<Self, AnyError> {
        let first = &config.endpoints[0];
        if config.endpoints.iter().any(|endpoint| {
            endpoint.model != first.model
                || endpoint.model_revision != first.model_revision
                || endpoint.precision != first.precision
        }) {
            return Err("Use the same model ID, revision and precision for every endpoint.".into());
        }
        let workload = Self {
            prompt_count: config.prompts.len(),
            prompts_sha256: sha256(&serde_json::to_vec(&config.prompts)?),
            concurrency: config.concurrency.clone(),
            requests_per_point: config.requests,
            warmup_requests_per_point: config.warmup_requests,
            max_output_tokens: config.max_tokens,
            timeout_seconds: config.timeout_seconds,
            temperature: SAMPLING_TEMPERATURE,
            model: first.model.clone(),
            model_revision: first.model_revision.clone(),
            precision: first.precision.clone(),
            slo: config.slo,
        };
        if config.endpoints.len() > 1 && !workload.has_model_metadata() {
            return Err("Set model_revision and precision when benchmarking multiple endpoints.".into());
        }
        Ok(workload)
    }

    fn has_model_metadata(&self) -> bool {
        [&self.model_revision, &self.precision]
            .iter()
            .all(|value| value.as_deref().is_some_and(|s| !s.trim().is_empty()))
    }

    pub fn notes(&self) -> String {
        let mut notes = format!(
            "{} requests per point, {} warmup requests per point, {} prompts, {} max output tokens. See [run.json](run.json) for the workload and endpoint metadata.",
            self.requests_per_point,
            self.warmup_requests_per_point,
            self.prompt_count,
            self.max_output_tokens,
        );
        if let Some(budget) = self.slo {
            notes.push(' ');
            notes.push_str(&budget.label());
        }
        notes
    }
}

#[derive(Deserialize, Serialize)]
struct Engine {
    name: String,
    version: String,
    hardware: String,
    hourly_usd: f64,
}

impl From<&Endpoint> for Engine {
    fn from(endpoint: &Endpoint) -> Self {
        Self {
            name: endpoint.engine.clone(),
            version: endpoint.version.clone(),
            hardware: endpoint.hardware.clone(),
            hourly_usd: endpoint.hourly_usd,
        }
    }
}

#[derive(Deserialize, Serialize)]
struct Manifest {
    schema_version: u32,
    kind: Kind,
    started_unix_seconds: u64,
    workload: Workload,
    engines: Vec<Engine>,
    summary_sha256: String,
    input_runs: Vec<String>,
}

fn sha256(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest(&SHA256, bytes).as_ref() {
        write!(&mut hex, "{byte:02x}").expect("writing to String");
    }
    hex
}

pub fn timestamp() -> Result<u64, AnyError> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

pub fn prepare_output(output: &Path) -> Result<(), AnyError> {
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    if let Err(error) = fs::create_dir(output) {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            return Err(format!("{} already exists. Use a new output directory.", output.display()).into());
        }
        return Err(error.into());
    }
    Ok(())
}

pub fn write_run(
    output: &Path,
    workload: Workload,
    endpoints: &[Endpoint],
    started_unix_seconds: u64,
) -> Result<(), AnyError> {
    write(
        output,
        Kind::Live,
        workload,
        endpoints.iter().map(Engine::from).collect(),
        started_unix_seconds,
        Vec::new(),
    )
}

fn write(
    output: &Path,
    kind: Kind,
    workload: Workload,
    engines: Vec<Engine>,
    started_unix_seconds: u64,
    input_runs: Vec<String>,
) -> Result<(), AnyError> {
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        kind,
        started_unix_seconds,
        workload,
        engines,
        summary_sha256: sha256(&fs::read(output.join("summary.json"))?),
        input_runs,
    };
    fs::write(output.join(FILE_NAME), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

pub struct Comparison {
    points: Vec<Point>,
    workload: Workload,
    engines: Vec<Engine>,
    input_runs: Vec<String>,
}

impl Comparison {
    pub fn load(inputs: &[&str]) -> Result<Self, AnyError> {
        let mut workload = None;
        let mut points = Vec::new();
        let mut engines = Vec::new();
        let mut input_runs = Vec::new();
        for input in inputs {
            let path = Path::new(input);
            let raw_manifest = fs::read(path.with_file_name(FILE_NAME)).map_err(|error| {
                format!("{input}: cannot read adjacent {FILE_NAME}: {error}. Supply summaries from completed live runs.")
            })?;
            let manifest: Manifest = serde_json::from_slice(&raw_manifest)?;
            if manifest.schema_version != SCHEMA_VERSION || manifest.kind != Kind::Live {
                return Err(format!("{input}: expected a version {SCHEMA_VERSION} live run.").into());
            }
            if !manifest.workload.has_model_metadata() {
                return Err(format!("{input}: set model_revision and precision when recording comparable runs.").into());
            }
            let run_id = sha256(&serde_json::to_vec(&manifest)?);
            if input_runs.contains(&run_id) {
                return Err(format!("{input}: the same live run was supplied more than once.").into());
            }
            let summary = fs::read(path)?;
            if sha256(&summary) != manifest.summary_sha256 {
                return Err(format!("{input}: summary checksum does not match its run manifest.").into());
            }
            let rows: Vec<Point> = serde_json::from_slice(&summary)?;
            if rows.is_empty() || rows.iter().any(|point| point.version == "fixture") {
                return Err(format!("{input}: expected nonempty live measurements.").into());
            }
            if workload.as_ref().is_some_and(|reference| reference != &manifest.workload) {
                return Err(format!("{input}: workload or model settings differ from the first run.").into());
            }
            workload.get_or_insert(manifest.workload);
            points.extend(rows);
            engines.extend(manifest.engines);
            input_runs.push(run_id);
        }
        Ok(Self {
            points,
            workload: workload.ok_or("No live runs supplied.")?,
            engines,
            input_runs,
        })
    }

    pub fn write(self, output: &Path) -> Result<(), AnyError> {
        prepare_output(output)?;
        let started = timestamp()?;
        let notes = format!(
            "Combined {} live runs with matching workloads. {}",
            self.input_runs.len(),
            self.workload.notes()
        );
        crate::report::write(output, self.points, false, &notes)?;
        write(output, Kind::Comparison, self.workload, self.engines, started, self.input_runs)
    }
}
