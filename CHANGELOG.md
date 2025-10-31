# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive unit test suite for all modules
- Integration tests for end-to-end functionality
- GitHub Actions CI/CD pipeline with:
  - Multi-version Rust testing (stable, beta)
  - Code formatting checks (rustfmt)
  - Linting with clippy
  - Security auditing with cargo-audit
  - MSRV (Minimum Supported Rust Version) checking
- Code quality configuration:
  - rustfmt.toml for consistent formatting
  - Lint configuration in Cargo.toml
- Development dependencies and tools
- CONTRIBUTING.md with contribution guidelines
- Enhanced .gitignore with additional entries

### Changed
- Reorganized code structure with better module boundaries
- Added constants for important values (SAMPLE_RATE)
- Improved error handling documentation

### Improved
- Test coverage across audio, model, and download modules
- CI pipeline for automated quality checks
- Developer documentation and guidelines

## [0.1.0] - Initial Release

### Added
- Voxtral speech-to-text implementation using Candle
- CUDA and CUDNN GPU acceleration support
- Audio decoding via Symphonia (WAV, MP3, FLAC, OGG, etc.)
- Automatic model downloading from Hugging Face Hub
- Resampling to 16kHz for model compatibility
- Audio chunking for long files (15s chunks with 10% overlap)
- CLI interface with clap
- Multiple audio format support
- F16 memory optimization

### Features
- `--cpu` flag to force CPU mode
- `--input` flag for audio file path
- Automatic output to .txt file with same name as input
- Support for mistralai/Voxtral-Mini-3B-2507 model
- Memory-mapped model file loading

### Technical Details
- Uses Candle ML framework
- 128-mel filter bank audio features
- Token-based audio input handling
- Overlap-and-add for chunked transcription
- Configurable generation parameters (max_new_tokens, temperature)
