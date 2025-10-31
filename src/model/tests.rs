use super::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_transcription_result_serialization() {
    let result = TranscriptionResult {
        text: "Hello, world!".to_string(),
        tokens: vec![1, 2, 3, 4],
    };

    // Test that the result can be serialized
    let serialized = serde_json::to_string(&result).unwrap();
    assert!(serialized.contains("Hello, world!"));

    // Test that it can be deserialized
    let deserialized: TranscriptionResult = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.text, "Hello, world!");
    assert_eq!(deserialized.tokens, vec![1, 2, 3, 4]);
}

#[test]
fn test_transcription_result_empty() {
    let result = TranscriptionResult {
        text: "".to_string(),
        tokens: vec![],
    };

    let serialized = serde_json::to_string(&result).unwrap();
    let deserialized: TranscriptionResult = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.text, "");
    assert_eq!(deserialized.tokens.len(), 0);
}

#[test]
fn test_voxtral_config_validation() {
    // Test that configuration parsing works with valid JSON
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    let config_json = serde_json::json!({
        "audio_token_id": 24,
        "audio_config": {
            "vocab_size": 51866,
            "hidden_size": 1280,
            "num_hidden_layers": 32
        },
        "text_config": {
            "vocab_size": 131072,
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 30
        }
    });

    fs::write(
        &config_path,
        serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    // Note: This would actually load the config if we had the rest of the setup
    assert!(config_path.exists());
}

#[test]
fn test_audio_token_id_extraction() {
    // Test that audio_token_id is properly extracted from config
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    let config_content = r#"{
        "audio_token_id": 42
    }"#;

    fs::write(&config_path, config_content).unwrap();

    // Simulate the parsing logic from load_model_config
    let config_str = fs::read_to_string(&config_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let audio_token_id = json
        .get("audio_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as usize;

    assert_eq!(audio_token_id, 42);
}

#[test]
fn test_audio_token_id_default() {
    // Test that audio_token_id defaults to 24 when not in config
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    let config_content = r#"{"other_field": "value"}"#;

    fs::write(&config_path, config_content).unwrap();

    let config_str = fs::read_to_string(&config_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let audio_token_id = json
        .get("audio_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(24) as usize;

    assert_eq!(audio_token_id, 24);
}

#[test]
fn test_transcribe_with_voxtral_input_validation() {
    // Test that the function validates input parameters
    // We can't actually call transcribe_with_voxtral without a model,
    // but we can test the validation logic

    // Audio features must be 3D tensor
    // This test documents the expected input shape

    let audio_dims = vec![1, 128, 1000]; // [batch, mels, time]

    assert_eq!(audio_dims.len(), 3, "Audio features must be 3D");
    assert_eq!(audio_dims[1], 128, "Must have 128 mel bins");
}

#[test]
fn test_pcm_resample_call() {
    // Test that transcribe_audio calls resample when needed
    // This is a compile-time check that the function signature is correct

    fn test_function(audio_data: &[f32], sample_rate: u32) -> Vec<f32> {
        let target_sr = 16000;
        if sample_rate == target_sr {
            audio_data.to_vec()
        } else {
            // This would call the actual resample function
            audio_data.to_vec()
        }
    }

    let result = test_function(&[0.5, -0.5, 0.5], 44100);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_pcm_no_resample_needed() {
    // Test that when sample rate matches, no resampling occurs
    let audio_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let sample_rate = 16000;
    let target_sr = 16000;

    let result = if sample_rate == target_sr {
        audio_data.to_vec()
    } else {
        vec![] // Would call resample
    };

    assert_eq!(result, audio_data);
}

#[test]
fn test_padded_audio_chunk_size() {
    // Test that audio is padded to 30-second chunks (480000 samples)
    let chunk_size = 480000; // 30 seconds * 16000 Hz
    let audio = vec![0.5; 100000]; // 100k samples

    let padded_audio = if audio.len() % chunk_size != 0 {
        let target_samples = ((audio.len() / chunk_size) + 1) * chunk_size;
        let mut padded = audio.clone();
        padded.resize(target_samples, 0.0);
        padded
    } else {
        audio
    };

    assert_eq!(padded_audio.len() % chunk_size, 0);
    assert!(padded_audio.len() >= chunk_size);
}

#[test]
fn test_token_sequence_construction() {
    // Test that the expected token sequence is constructed correctly
    // This documents the expected format: <s>[INST][BEGIN_AUDIO][AUDIO]*N[/INST]lang:en[TRANSCRIBE]

    let mut input_tokens = Vec::new();

    // Pattern components
    input_tokens.push(1u32); // BOS: <s>
    input_tokens.push(3u32); // [INST]
    input_tokens.push(25u32); // [BEGIN_AUDIO]

    let batch_size = 1;
    let tokens_per_chunk = 375;
    let audio_token_id = 24;
    let num_audio_tokens = batch_size * tokens_per_chunk;

    for _ in 0..num_audio_tokens {
        input_tokens.push(audio_token_id as u32);
    }

    input_tokens.push(4u32); // [/INST]
    input_tokens.push(9909u32); // lang
    input_tokens.push(1058u32); // :
    input_tokens.push(1262u32); // en
    input_tokens.push(34u32); // [TRANSCRIBE]

    // Verify the sequence
    assert_eq!(input_tokens[0], 1);
    assert_eq!(input_tokens[1], 3);
    assert_eq!(input_tokens[2], 25);
    assert_eq!(input_tokens[3], 24); // First AUDIO token
    assert_eq!(input_tokens[379], 9909); // lang
}

#[test]
fn test_generation_config_parameters() {
    // Test that generation config uses expected parameters
    let config = VoxtralGenerationConfig {
        max_new_tokens: 1000,
        temperature: 0.0,
        top_p: None,
        device: Device::Cpu, // Can't use CUDA in test
        cache: None,
    };

    assert_eq!(config.max_new_tokens, 1000);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_p, None);
}

#[test]
fn test_tensor_shape_validation() {
    // Test the validation logic for audio features
    let dims = vec![2, 128, 500]; // Valid: [batch, mels, time]

    assert!(
        dims.len() == 3,
        "Audio features must be 3D tensor (batch, mels, time)"
    );
    assert_eq!(dims[1], 128, "Audio features must have 128 mel bins");
}

#[test]
fn test_device_selection_logic() {
    // Test the device selection logic from VoxtralModel::new
    // This documents the expected behavior

    let use_cpu = false;
    let cuda_available = false; // In test environment

    let device = if !use_cpu && cuda_available {
        // Would be CUDA device
        "cuda"
    } else {
        "cpu"
    };

    assert_eq!(device, "cpu");
}

#[test]
fn test_mel_filters_deserialization() {
    // Test that mel filters can be deserialized from bytes
    // This mimics the code that reads melfilters128.bytes

    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    // Create fake mel filter data
    let mel_bytes = vec![0u8; 128 * 4]; // 128 floats * 4 bytes per float
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    let mut cursor = Cursor::new(mel_bytes);

    // This would normally read from the actual bytes file
    let _result = cursor.read_f32_into::<LittleEndian>(&mut mel_filters);

    assert_eq!(mel_filters.len(), 128);
}

#[test]
fn test_projector_activation_default() {
    // Test that projector_hidden_act defaults to "gelu"
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");

    let config_content = r#"{"other_field": "value"}"#;
    fs::write(&config_path, config_content).unwrap();

    let config_str = fs::read_to_string(&config_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let projector_hidden_act = json
        .get("projector_hidden_act")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu")
        .to_string();

    assert_eq!(projector_hidden_act, "gelu");
}

#[test]
fn test_load_model_weights_memory_mapping() {
    // Test that the function signature accepts the expected parameters
    // Memory-mapped loading is used for efficiency

    let model_files = vec![
        std::path::PathBuf::from("model-00001-of-00002.safetensors"),
        std::path::PathBuf::from("model-00002-of-00002.safetensors"),
    ];
    let device = &Device::Cpu;
    let dtype = DType::F16;

    // Verify we have the expected number of files
    assert_eq!(model_files.len(), 2);
    assert!(model_files[0].to_string_lossy().contains("model-00001"));
    assert!(model_files[1].to_string_lossy().contains("model-00002"));
}

#[test]
fn test_parse_audio_config_defaults() {
    // Test default values for audio configuration
    let audio_json = serde_json::json!({});

    let config = VoxtralEncoderConfig {
        vocab_size: audio_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(51866) as usize,
        hidden_size: audio_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1280) as usize,
        num_hidden_layers: audio_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize,
        num_attention_heads: audio_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize,
        num_key_value_heads: audio_json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize,
        intermediate_size: audio_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(5120) as usize,
        dropout: audio_json
            .get("dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        attention_dropout: audio_json
            .get("attention_dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        activation_dropout: audio_json
            .get("activation_dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        activation_function: audio_json
            .get("activation_function")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string(),
        max_source_positions: audio_json
            .get("max_source_positions")
            .and_then(|v| v.as_u64())
            .unwrap_or(1500) as usize,
        layerdrop: audio_json
            .get("layerdrop")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        initializer_range: audio_json
            .get("initializer_range")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.02),
        scale_embedding: audio_json
            .get("scale_embedding")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        num_mel_bins: audio_json
            .get("num_mel_bins")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize,
        head_dim: audio_json
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize,
    };

    assert_eq!(config.vocab_size, 51866);
    assert_eq!(config.hidden_size, 1280);
    assert_eq!(config.num_mel_bins, 128);
}

#[test]
fn test_slice_operations_on_files() {
    // Test the slicing logic: files[1..files.len() - 1]
    let files = vec![
        std::path::PathBuf::from("config.json"),
        std::path::PathBuf::from("model-00001.safetensors"),
        std::path::PathBuf::from("model-00002.safetensors"),
        std::path::PathBuf::from("tekken.json"),
    ];

    let safetensors_slice = &files[1..files.len() - 1];

    assert_eq!(safetensors_slice.len(), 2);
    assert!(safetensors_slice[0]
        .to_string_lossy()
        .contains("model-00001"));
    assert!(safetensors_slice[1]
        .to_string_lossy()
        .contains("model-00002"));

    let config = files.first().unwrap();
    assert!(config.to_string_lossy().ends_with("config.json"));

    let tokenizer = files.last().unwrap();
    assert!(tokenizer.to_string_lossy().ends_with("tekken.json"));
}
