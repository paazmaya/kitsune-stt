use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
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

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/candle_demo/
    #[arg(long)]
    input: Option<String>,

    #[arg(long, default_value = "mistralai/Voxtral-Mini-3B-2507")]
    model_id: Option<String>,
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

    let model_id = args.model_id.unwrap();

    // Create model - equivalent to loading the model and processor in Python
    let mut model =
        VoxtralModel::new(&model_id, use_cpu).context("Failed to load Voxtral model")?;

    println!("Model loaded successfully on device: {:?}", model.device());

    // No longer used directly; keep creation if future code needs it. If you
    // prefer to remove entirely, delete this line.
    let _api = Api::new()?;

    let audio_file = if let Some(input) = args.input {
        std::path::PathBuf::from(input)
    } else {
        println!("No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/candle_demo/blob/main/samples_jfk.wav");
        return Ok(());
    };

    let (audio_data, sample_rate) =
        audio::pcm_decode(audio_file).context("Failed to decode audio file")?;

    // Transcribe audio with token output
    let result = model
        .transcribe_audio(&audio_data, sample_rate)
        .context("Failed to transcribe audio with tokens")?;

    println!("\n===================================================\n");
    println!("{}", result.text);

    Ok(())
}
