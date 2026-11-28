# Inference benchmark teardown

Compare vLLM, SGLang, and TensorRT-LLM on latency, throughput, and cost.

## Preview the report

Run from the repo root. Use a fresh output directory.

```sh
cargo run --release -- demo ./results/fixture
```

Open `results/fixture/REPORT.md` or `results/fixture/pareto.svg`. The demo uses synthetic data.

[Live runs, metrics, and tests](docs/reference.md)
