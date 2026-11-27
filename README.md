# Inference benchmark teardown

This Rust client compares vLLM, SGLang, and TensorRT-LLM through streaming chat completions. Each engine receives the same prompts, token budget, warmup, and concurrency sweep. The client writes per-request CSV, summary CSV and JSON, a Markdown table, and a two-panel SVG.

## Preview the report

```sh
cargo run --release -- demo ./results/fixture
```

Open `results/fixture/REPORT.md` or `results/fixture/pareto.svg` to see the report with synthetic data.

## Collect live measurements

1. Copy `h100.example.json` and set actual hourly costs, engine versions, model IDs, and hardware. Set `api_key_env` for endpoints that require a bearer token.
2. Serve the same model at the same precision on equivalent, isolated GPUs. If you reuse one GPU, run each engine separately and combine its `summary.json` with `compare`.
3. Put representative requests in the `prompts` array. Record context length, engine flags, quantization, driver, prefix caching, warmup, and concurrency.

```sh
cargo run --release -- run h100.json ./results/live
cargo run --release -- compare ./results/combined \
  ./results/vllm/summary.json ./results/sglang/summary.json ./results/trtllm/summary.json
```

## Metrics

- **TTFT** is the time from sending the request to the first stream frame with generated or reasoning text. A successful response reports `usage.completion_tokens` and ends with `[DONE]`.
- **Throughput** is successful output tokens divided by the batch's wall time.
- **Cost per million output tokens** is total hourly deployment cost divided by millions of output tokens per hour.
- **Pareto frontier** includes fully successful points for which no other fully successful point is at least as good on all three objectives and better on at least one. The objectives are lower p95 TTFT, higher throughput, and lower cost.

The load generator uses fixed concurrency with no imposed arrival rate. Report failed requests and repeat the sweep at least three times for a public comparison.

Engine references: [vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/), [SGLang OpenAI APIs](https://docs.sglang.io/docs/basic_usage/openai_api_completions), and [TensorRT-LLM serving](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html).
