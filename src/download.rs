use std::path::PathBuf;

use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};

/// Download model artifacts from Hugging Face Hub for a given model id.
///
/// This function fetches the `config.json`, a set of `safetensors` weight files
/// (matching common Voxtral naming patterns) and a tokenizer file (tries
/// `tekken.json` and `tokenizer/tokenizer.json`). It returns a tuple with the
/// config, the list of safetensors files and the tokenizer path.
///
/// # Errors
///
/// Returns an error if any of the network requests or file retrievals fail.
pub fn model_files(model_id: &str) -> Result<((PathBuf, Vec<PathBuf>), PathBuf)> {
    let revision = "main";

    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config = repo.get("config.json")?;

    // Download model files - look for safetensors
    let mut model_files = Vec::new();

    // Common Voxtral/Ultravox safetensors file patterns
    let safetensors_files = match model_id {
        "mistralai/Voxtral-Mini-3B-2507" => vec![
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "mistralai/Voxtral-Small-24B-2507" => vec![
            "model-00001-of-00011.safetensors",
            "model-00001-of-00011.safetensors",
            "model-00002-of-00011.safetensors",
            "model-00003-of-00011.safetensors",
            "model-00004-of-00011.safetensors",
            "model-00005-of-00011.safetensors",
            "model-00006-of-00011.safetensors",
            "model-00007-of-00011.safetensors",
            "model-00008-of-00011.safetensors",
            "model-00009-of-00011.safetensors",
            "model-00010-of-00011.safetensors",
            "model-00011-of-00011.safetensors",
        ],
        _ => vec![
            "model.safetensors",
            "pytorch_model.safetensors",
            "model-00001-of-00001.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    };

    println!("Downloading safetensors files...");
    for filename in &safetensors_files {
        if let Ok(file) = repo.get(filename) {
            println!("{} downloaded", filename);
            model_files.push(file);
        }
    }

    if model_files.is_empty() {
        anyhow::bail!("No safetensors files found in model repository {model_id}",);
    }

    // Download tokenizer
    let tokenizer_file = repo
        .get("tekken.json")
        .or_else(|_| repo.get("tokenizer/tokenizer.json"))?;

    Ok(((config, model_files), tokenizer_file))
}
