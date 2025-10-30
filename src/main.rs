use anyhow::{Context, Result};
use clap::Parser;
use model::VoxtralModel;

mod audio;
mod download;
mod model;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// The input audio file to be processed (any format supported by Symphonia).
    /// Alternatively this can be set to sample:jfk, sample:gb1, ... to fetch a
    /// sample from: https://huggingface.co/datasets/Narsil/candle_demo/
    #[arg(long)]
    input: Option<String>,
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

    // Create model - equivalent to loading the model and processor in Python
    let mut model = VoxtralModel::new(use_cpu).context("Failed to load Voxtral model")?;

    println!("Model loaded successfully on device: {:?}", model.device());

    let audio_file = if let Some(input) = args.input {
        std::path::PathBuf::from(input)
    } else {
        println!("No audio file submitted");
        return Ok(());
    };

    let (audio_data, sample_rate) =
        audio::pcm_decode(audio_file).context("Failed to decode audio file. Perhaps its not supported? See https://docs.rs/symphonia/latest/symphonia/index.html")?;

    // Ensure audio matches model expectations: mono + 16 kHz sample rate.
    // `audio::pcm_decode` already mixes to mono. Resample here so the model
    // receives 16 kHz audio directly (the model will also resample if needed,
    // but doing it here avoids double work and makes preprocessing explicit).
    let target_sr: u32 = 16_000;
    let prepared_audio = if sample_rate != target_sr {
        println!(
            "Resampling audio from {} Hz to {} Hz to match model expectations...",
            sample_rate, target_sr
        );
        audio::resample(&audio_data, sample_rate, target_sr)
            .context("Failed to resample audio to 16 kHz")?
    } else {
        audio_data
    };

    // Check duration and warn if longer than recommended encoder window (~15s).
    let duration_s = prepared_audio.len() as f32 / target_sr as f32;
    if duration_s > 15.0 {
        println!(
            "Warning: input audio is {:.1} seconds long. The model's encoder is configured for ~15s of audio per example (you may want to chunk long recordings). Proceeding to transcribe the full audio.",
            duration_s
        );
    }

    // Chunking: split long audio into windows with 10% overlap and transcribe
    // each chunk separately. We use a default chunk length that matches the
    // model's encoder capacity (≈15s). Overlap helps maintain context at
    // boundaries.
    let chunk_seconds = 15.0_f32; // model's approx max (derived from config)
    let overlap_ratio = 0.10_f32; // 10% overlap

    let chunk_samples = (chunk_seconds * target_sr as f32) as usize;
    let overlap_samples = (chunk_samples as f32 * overlap_ratio) as usize;
    let step = if chunk_samples > overlap_samples {
        chunk_samples - overlap_samples
    } else {
        chunk_samples
    };

    let mut texts: Vec<String> = Vec::new();
    let mut all_tokens: Vec<u32> = Vec::new();

    if prepared_audio.is_empty() {
        println!("No audio samples after decoding/resampling.");
        return Ok(());
    }

    if prepared_audio.len() <= chunk_samples {
        // Short audio: transcribe once
        let result = model
            .transcribe_audio(&prepared_audio, target_sr)
            .context("Failed to transcribe audio with tokens")?;
        println!("\n===================================================\n");
        println!("{}", result.text);
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

        // Collect results
        texts.push(result.text.clone());
        all_tokens.extend(result.tokens);

        chunk_index += 1;
        if end == prepared_audio.len() {
            break;
        }
        start += step;
    }

    // Concatenate chunk texts with a space. Note: overlapping regions may
    // produce duplicated words at chunk boundaries; post-processing can be
    // added later to mitigate this.
    let joined_text = texts.join(" ");

    println!("\n===================================================\n");
    println!("{}", joined_text);

    Ok(())
}
