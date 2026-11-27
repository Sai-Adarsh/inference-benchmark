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

- **TTFT** is the time from sending the request to the first complete event with generated or reasoning text. The client supports SSE comments, CRLF, and multiline data. A successful response reports `usage.completion_tokens` and ends with a complete `[DONE]` event. Lines and event data are capped at 1 MiB.
- **Estimated TPOT** divides the time between the first and last text events by completion tokens minus one. It requires at least two text events and two completion tokens. Frames may contain several tokens, so this is a client estimate of time per output token.
- **Max frame gap** is the longest interval between adjacent events with text in one request. The summary reports the p95 of that per-request value. It measures delivery stalls, including buffering outside the model process.
- **Throughput** is successful output tokens divided by the batch's wall time.
- **Cost per million output tokens** is total hourly deployment cost divided by millions of output tokens per hour.
- **Pareto frontier** includes fully successful points for which no other fully successful point is at least as good on all three objectives and better on at least one. The objectives are lower p95 TTFT, higher throughput, and lower cost.

The load generator uses fixed concurrency with no imposed arrival rate. Report failed requests and repeat the sweep at least three times for a public comparison.

Engine references: [vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/), [SGLang OpenAI APIs](https://docs.sglang.io/docs/basic_usage/openai_api_completions), and [TensorRT-LLM serving](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html).
