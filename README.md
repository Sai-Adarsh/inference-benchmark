# Inference benchmark teardown

This Rust client compares vLLM, SGLang, and TensorRT-LLM through streaming chat completions. Each engine receives the same prompts, token budget, warmup, and concurrency sweep. The client writes per-request CSV, summary CSV and JSON, a Markdown table, and a two-panel SVG.

## Preview the report

```sh
cargo run --release -- demo ./results/fixture
```

Open `results/fixture/REPORT.md` or `results/fixture/pareto.svg` to see the report with synthetic data.

## Collect live measurements

1. Copy `h100.example.json` and set actual hourly costs, engine versions, model ID, precision, and hardware. Set `model_revision` to the checkpoint revision and `api_key_env` for endpoints that require a bearer token.
2. Serve the same model at the same precision on equivalent, isolated GPUs. If you reuse one GPU, run each engine separately into its own output directory.
3. Put representative requests in the `prompts` array. Record context length, engine flags, driver, and prefix caching alongside the results.

```sh
cargo run --release -- run h100.json ./results/live
cargo run --release -- compare ./results/combined \
  ./results/vllm/summary.json ./results/sglang/summary.json ./results/trtllm/summary.json
```

Use a new output path for each command. The benchmark creates that directory and refuses to write into an existing one. A completed live run writes `run.json` next to `summary.json`. The manifest records the workload, a SHA-256 digest of the ordered prompts, engine versions, hardware, hourly price, and a checksum of the summary. It omits prompt text, endpoint URLs, and API keys.

Keep each summary next to its manifest. `compare` requires matching prompt order, concurrency sweep, request and warmup counts, token budget, timeout, and model settings. Set `model_revision` and `precision` to compare runs or benchmark multiple endpoints at once. `compare` also rejects missing manifests, duplicate runs, and summaries that fail the checksum. Rerun older summaries to include them in a comparison.

## Metrics

- **TTFT** is the time from sending the request to the first complete event with generated or reasoning text. The client supports SSE comments, CRLF, and multiline data. A successful response reports a consistent `usage.completion_tokens` count and ends with a complete `[DONE]` event. Lines and event data are capped at 1 MiB.
- **Estimated TPOT** divides the time between the first and last text events by completion tokens minus one. It requires at least two text events and two completion tokens. Frames may contain several tokens, so this is a client estimate of time per output token.
- **Max frame gap** is the longest interval between adjacent events with text in one request. The summary reports the p95 of that per-request value. It measures delivery stalls, including buffering outside the model process.
- **Throughput** is successful output tokens divided by the batch's wall time.
- **Cost per million output tokens** is total hourly deployment cost divided by millions of output tokens per hour.
- **Pareto frontier** includes fully successful points for which no other fully successful point is at least as good on all three objectives and better on at least one. The objectives are lower p95 TTFT, higher throughput, and lower cost.

The load generator uses fixed concurrency with no imposed arrival rate. Report failed requests and repeat the sweep at least three times for a public comparison.

Engine references: [vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/), [SGLang OpenAI APIs](https://docs.sglang.io/docs/basic_usage/openai_api_completions), and [TensorRT-LLM serving](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html).

## Measure SLO goodput

Add an optional budget to the benchmark config. Limits are in milliseconds per
request.

```json
"slo": {"ttft_ms": 250, "total_ms": 3000, "estimated_tpot_ms": 40}
```

TTFT and total latency are required. With an estimated TPOT limit, a
multi-token response needs enough text events to estimate TPOT. Single-token
responses only need to meet the TTFT and total latency limits.

The sample log records the decision for each request. The summary divides
qualified output tokens by the entire batch duration, including time spent on
failed or late requests. It reports request and token counts, goodput, and
the cost per million qualified tokens. Points with zero qualified tokens have
zero goodput and no cost per token.

`slo-pareto.svg` shows p95 TTFT across successful requests, goodput, and cost
per million qualified tokens. Points with positive goodput can appear when
some requests failed. `compare` requires identical SLO budgets across inputs.

## Tests

`cargo test --locked` uses loopback HTTP servers and temporary output directories.
The suite checks request framing and concurrency, SSE text and usage, failed
warmups, incomplete responses, and rejection of altered or incompatible
comparison inputs. It needs no model server or API key.

To rebuild and measure production coverage, install the Rust `llvm-tools-preview`
component and make `jq` available on your PATH. Set `llvm` to the directory with its `llvm-profdata` and `llvm-cov`
executables, or use a compatible system LLVM.

```sh
set -eu
run="$(mktemp -d)"
mkdir "$run/profiles"
export CARGO_TARGET_DIR="$run/target"
export RUSTFLAGS="-C instrument-coverage"
export LLVM_PROFILE_FILE="$run/profiles/%p-%m.profraw"
cargo test --locked --message-format=json > "$run/cargo.json"
sse_test="$(jq -Rr 'fromjson? | select(.reason == "compiler-artifact"
   and .target.kind == ["test"] and .target.name == "sse") | .executable' "$run/cargo.json")"
"$llvm/llvm-profdata" merge -sparse "$run"/profiles/*.profraw -o "$run/run.profdata"
"$llvm/llvm-cov" report "$CARGO_TARGET_DIR/debug/inference-benchmark" \
  -object "$sse_test" -instr-profile "$run/run.profdata" "$PWD/src"
```
