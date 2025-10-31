# kitsune-stt

[![CI](https://github.com/paazmaya/kitsune-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/paazmaya/kitsune-stt/actions/workflows/ci.yml)

An Speech-to-Text implementation in Rust of Voxtral speech recognition using candle.

![Fox speaking to microphone and writing papers](./logo.png)

The model used in the conversion is https://huggingface.co/mistralai/Voxtral-Mini-3B-2507

## Features

- 🎤 **Speech-to-Text**: Convert audio to text using Voxtral-Mini-3B model
- 🚀 **GPU Acceleration**: CUDA and CUDNN support for faster inference
- 📦 **Audio Format and Codec Support**: WAV, MP3, FLAC, OGG, M4A, and more, see https://docs.rs/symphonia/latest/symphonia/index.html
- ⚡ **Performance**: F16 memory optimization, chunked processing

## Quick Start

### Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/paazmaya/kitsune-stt.git
cd kitsune-stt
```

There are two features, `cuda` and `cudnn`. They require additional libraries each:

* https://developer.nvidia.com/cuda-toolkit
* https://docs.nvidia.com/deeplearning/cudnn/installation/latest/backend.html

### Running

**GPU (Recommended):**
```bash
cargo run --all-features --release -- audio.wav
```

**CPU Only:**
```bash
cargo run --release -- --cpu audio.wav
```

### Transcribe Audio

The following example commands would create a `audio.txt` in the same folder as the source audio file.

```bash
# Transcribe an audio file
cargo run --release -- --input audio.wav

# Force CPU mode
cargo run --release --features cuda -- --cpu --input audio.wav
```
## Testing

Run the complete test suite:

```bash
# All tests
cargo test

# With all features
cargo test --all-features
```

### Code Quality

```bash
# Format code
cargo fmt --all

# Run clippy lints
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo install cargo-audit
cargo audit
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Requirements

- Rust 1.70+
- pkg-config (Linux)
- CUDA toolkit (optional, for GPU)
- ~7GB disk space (for model files)

## License

MIT License - see [LICENSE](LICENSE) file for details.
