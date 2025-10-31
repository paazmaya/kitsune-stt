# kitsune-stt

An implementation in Rust of Voxtral speech recognition using candle.

![Fox speaking to microphone and writing papers](./logo.png)

## Features

- 🎤 **Speech-to-Text**: Convert audio to text using Voxtral-Mini-3B model
- 🚀 **GPU Acceleration**: CUDA and CUDNN support for faster inference
- 📦 **Audio Format Support**: WAV, MP3, FLAC, OGG, M4A, and more
- ⚡ **Performance**: F16 memory optimization, chunked processing
- 🧪 **Well Tested**: Comprehensive unit and integration tests
- 📊 **CI/CD**: Automated testing and quality checks

## Quick Start

### Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/paazmaya/kitsune-stt.git
cd kitsune-stt
```

### Running

**GPU (Recommended):**
```bash
cargo run --features cuda,cudnn --release
```

**CPU Only:**
```bash
cargo run --release
```

### Transcribe Audio

```bash
# Transcribe an audio file
cargo run --release -- --input path/to/your/audio.wav

# Force CPU mode
cargo run --release --features cuda -- --cpu --input audio.wav
```

## Command Line Options

- `--cpu`: Force CPU mode (default: auto-detect GPU)
- `--input`: Path to audio file (supports WAV, MP3, FLAC, OGG, M4A)

## Testing

Run the complete test suite:

```bash
# All tests
cargo test

# Unit tests only
cargo test --lib

# Integration tests
cargo test --test integration_tests

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

### Project Structure

- `src/main.rs` - CLI interface and orchestration
- `src/audio.rs` - Audio decoding and resampling
- `src/model.rs` - Voxtral model loading and transcription
- `src/download.rs` - Model file management
- `tests/` - Integration tests

### CI Pipeline

Automated checks on every PR:
- ✅ Multi-version Rust testing (stable, beta)
- ✅ Code formatting verification
- ✅ Clippy linting
- ✅ Full test suite
- ✅ Security audit
- ✅ MSRV verification

## Requirements

- Rust 1.70+
- pkg-config (Linux)
- CUDA toolkit (optional, for GPU)
- ~7GB disk space (for model files)

## License

MIT License - see [LICENSE](LICENSE) file for details.
