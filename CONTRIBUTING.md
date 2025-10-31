# Contributing to Kitsune STT

Thank you for your interest in contributing to Kitsune STT! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- pkg-config (on Linux)
- CUDA toolkit (optional, for GPU support)
- cudnn (optional, for improved CUDA performance)

### Building the Project

```bash
# Build for CPU
cargo build --release

# Build with CUDA support
cargo build --release --features cuda,cudnn
```

### Running Tests

Run all tests including unit and integration tests:

```bash
cargo test
```

Run only unit tests for a specific module:

```bash
cargo test --lib audio
cargo test --lib download
cargo test --lib model
```

## Code Quality

We maintain high code quality through automated checks:

### Formatting

```bash
cargo fmt --all
```

### Linting

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### All Checks

The CI pipeline runs:
- `cargo fmt --all -- --check` - Format verification
- `cargo clippy --all-targets --all-features -- -D warnings` - Linting
- `cargo test --all-features` - Full test suite
- `cargo doc --no-deps --all-features` - Documentation build

## Development Guidelines

### Code Structure

- **main.rs**: CLI argument parsing and main application logic
- **audio.rs**: Audio file decoding and resampling
- **model.rs**: Voxtral model loading and transcription
- **download.rs**: Model file downloading from Hugging Face Hub

### Writing Tests

1. **Unit Tests**: Place in `src/{module}/tests.rs`
2. **Integration Tests**: Place in `tests/`
3. **Test Coverage**: Aim for comprehensive coverage of:
   - Error handling paths
   - Edge cases
   - Configuration validation
   - Data transformation logic

### Audio Module Tests

Test the audio processing pipeline:
- File format support
- Resampling quality
- Error handling for unsupported formats
- Channel conversion (multi-channel to mono)

### Model Module Tests

Test model configuration and data structures:
- Configuration parsing
- Parameter extraction
- Token generation logic
- Audio feature validation

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run all checks: `cargo fmt && cargo clippy && cargo test`
5. Commit with a clear message
6. Push to your fork
7. Open a pull request

### Commit Messages

Follow conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `test:` Test additions or fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `style:` Code style changes

Example:
```
feat: add unit tests for audio resampling
test: add integration test for chunking logic
```

## Performance Considerations

### GPU Memory

- The model uses F16 (half precision) for memory efficiency
- Memory-mapped loading is used for large model files
- Cache clearing is performed before/after model loading

### CPU Usage

- Multi-threaded audio decoding via Symphonia
- FFT-based resampling for quality
- Chunking with overlap for long audio files

## Common Issues

### CUDA Not Available

If CUDA is not detected:
```bash
# Force CPU mode
cargo run --release -- --cpu
```

### Missing Model Files

The first run will download model files (~7GB):
- Files are cached locally in `Voxtral-Mini-3B-2507/`
- Ensure sufficient disk space
- Check network connectivity

### Audio Format Support

Supported formats (via Symphonia):
- WAV, MP3, FLAC, OGG, M4A, and more
- See: https://docs.rs/symphonia/latest/symphonia/index.html

## Documentation

- All public APIs should have documentation comments
- Use `cargo doc --open` to build and view documentation locally
- Documentation is tested in CI via `cargo doc --no-deps`

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- General questions
- Discussion of implementation details

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
