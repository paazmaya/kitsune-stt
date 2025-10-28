# kitsune-stt
An implementation in Rust of Voxtral speech recognition using candle

# candle-voxtral: speech recognition

An implementation of Voxtral speech recognition using candle.

## Running the example

Run with the `cuda` feature for GPU acceleration:
```bash
cargo run --features cuda --release
# you may also add the `cudnn` feature for extra performance
# cargo run --features cuda,cudnn --release
```

Remove the `cuda` feature to run on the CPU instead:
```bash
cargo run --release
# or pass the `--cpu` flag to force CPU usage
# cargo run --features cuda --release -- --cpu
```

## Command line options

- `--cpu`: Run on CPU rather than on GPU (default: false, uses GPU if available)
- `--input`: Audio file path in wav format. If not provided, a sample file is automatically downloaded from the hub.
- `--model-id`: Model to use (default: `mistralai/Voxtral-Mini-3B-2507`)
