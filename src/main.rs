use anyhow::{Context, Result};
use clap::Parser;
use model::VoxtralModel;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

mod audio;
mod download;
mod model;

// Re-export SAMPLE_RATE for use in tests
pub use audio::SAMPLE_RATE;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input audio file to be processed (any format supported by Symphonia).
    input: Option<PathBuf>,

    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[cfg(feature = "cuda")]
/// Return whether the build's default runtime should use CPU when `cuda` feature is enabled.
///
/// This version is selected when the `cuda` feature is enabled at compile time.
fn use_cpu() -> bool {
    true
}

#[cfg(not(feature = "cuda"))]
/// Return whether the build's default runtime should use CPU when `cuda` is not enabled.
///
/// This version is selected when the `cuda` feature is not enabled at compile time.
fn use_cpu() -> bool {
    false
}

/// CLI entrypoint: parse arguments, load model, decode audio and run transcription.
///
/// The function returns a `Result` so failures in model loading, audio decoding
/// or transcription are propagated to the caller.
fn main() -> Result<()> {
    let args = Args::parse();

    let use_cpu = args.cpu || !use_cpu();

    let audio_file = if let Some(input) = args.input {
        input
    } else {
        println!("No audio file submitted");
        return Ok(());
    };

    // Create model - equivalent to loading the model and processor in Python
    let mut model = load_model(use_cpu).context("Failed to load Voxtral model")?;

    println!("Model loaded successfully on device: {:?}", model.device());

    let target_sr: u32 = 16_000;
    let prepared_audio =
        decode_and_prepare(&audio_file, target_sr).context("Failed to decode/prepare audio")?;

    transcribe_and_stream(&mut model, &prepared_audio, target_sr, &audio_file)
}

fn load_model(use_cpu: bool) -> Result<VoxtralModel> {
    let model = VoxtralModel::new(use_cpu).context("Failed to create VoxtralModel")?;
    Ok(model)
}

fn decode_and_prepare(path: &PathBuf, target_sr: u32) -> Result<Vec<f32>> {
    let (audio_data, sample_rate) = audio::pcm_decode(path)
        .context("Failed to decode audio file. Perhaps its not supported? See https://docs.rs/symphonia/latest/symphonia/index.html")?;

    let prepared = if sample_rate != target_sr {
        println!(
            "Resampling audio from {} Hz to {} Hz to match model expectations...",
            sample_rate, target_sr
        );
        audio::resample(&audio_data, sample_rate, target_sr)
            .context("Failed to resample audio to 16 kHz")?
    } else {
        audio_data
    };

    if prepared.is_empty() {
        anyhow::bail!("No audio samples after decoding/resampling.");
    }

    Ok(prepared)
}

fn transcribe_and_stream(
    model: &mut VoxtralModel,
    prepared_audio: &[f32],
    target_sr: u32,
    audio_file: &Path,
) -> Result<()> {
    // Chunking parameters
    let chunk_seconds = 15.0_f32; // model's approx max (derived from config)
    let overlap_ratio = 0.10_f32; // 10% overlap

    let chunk_samples = (chunk_seconds * target_sr as f32) as usize;
    let overlap_samples = (chunk_samples as f32 * overlap_ratio) as usize;
    let step = if chunk_samples > overlap_samples {
        chunk_samples - overlap_samples
    } else {
        chunk_samples
    };

    let mut all_tokens: Vec<u32> = Vec::new();

    // Prepare output file: same stem as input file with .txt extension
    let mut out_path = audio_file.to_path_buf();
    out_path.set_extension("txt");
    let out_file =
        File::create(&out_path).context("Failed to create output file for transcription")?;
    let mut writer = BufWriter::new(out_file);

    if prepared_audio.len() <= chunk_samples {
        let result = model
            .transcribe_audio(prepared_audio, target_sr)
            .context("Failed to transcribe audio with tokens")?;
        writeln!(writer, "{}", result.text).context("Failed to write transcription to file")?;
        writer.flush().ok();
        println!("Transcription written to {}", out_path.display());
        return Ok(());
    }

    // Iterate chunks
    let mut start = 0usize;
    let mut chunk_index = 0usize;
    while start < prepared_audio.len() {
        let end = std::cmp::min(start + chunk_samples, prepared_audio.len());
        let chunk = &prepared_audio[start..end];

        println!(
            "Transcribing chunk {}/{} (samples {}..{})...",
            chunk_index + 1,
            prepared_audio.len().div_ceil(step),
            start,
            end
        );

        let result = model
            .transcribe_audio(chunk, target_sr)
            .context("Failed to transcribe audio chunk")?;

        // Stream chunk text to output file immediately
        writeln!(writer, "{}", result.text)
            .context("Failed to write chunk transcription to file")?;
        writer.flush().ok();

        // Collect tokens for downstream use if needed
        all_tokens.extend(result.tokens);

        chunk_index += 1;
        if end == prepared_audio.len() {
            break;
        }
        start += step;
    }

    println!("Transcription written to {}", out_path.display());

    Ok(())
}
