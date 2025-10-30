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
pub fn model_files() -> Result<Vec<PathBuf>> {
    let revision = "main";

    // Local model folder name (same as repository name)
    let model_dir = PathBuf::from("Voxtral-Mini-3B-2507");

    // https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
    let model_files: Vec<&str> = vec![
        "config.json",
        "tekken.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let mut existing_files: Vec<&str> = Vec::new();

    // If the folder already exists and contains expected files, use them.
    if model_dir.exists() {
        // Which of the model files already exist and which not?
        existing_files = model_files
            .iter()
            .filter(|p| model_dir.join(p).exists())
            .cloned()
            .collect();

        if existing_files.len() == model_files.len() {
            println!("Using existing model files in {}", model_dir.display());
            return Ok(existing_files.iter().map(|p| model_dir.join(p)).collect());
        }
    } else {
        // Ensure local directory exists
        std::fs::create_dir_all(&model_dir)?;
    }

    // Otherwise download into the local folder.
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(
        "mistralai/Voxtral-Mini-3B-2507".to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Download model files - look for safetensors
    let needed_files: Vec<&str> = model_files
        .iter()
        .filter(|p| existing_files.contains(p))
        .cloned()
        .collect();
    let mut downloaded_files: Vec<PathBuf> = Vec::new();
    println!("Downloading model files...");
    for filename in &needed_files {
        match repo.get(filename) {
            Ok(tmp) => {
                let target = model_dir.join(filename);
                std::fs::copy(&tmp, &target)?;
                println!("{} downloaded -> {}", filename, target.display());
                downloaded_files.push(target);
            }
            Err(_) => continue,
        }
    }

    if downloaded_files.is_empty() {
        anyhow::bail!("No model files found in model repository");
    }

    Ok(model_files.iter().map(|p| model_dir.join(p)).collect())
}
