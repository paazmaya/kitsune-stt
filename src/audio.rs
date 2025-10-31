use candle_core::{Error, Result};

/// Sample rate used by the Voxtral model (16 kHz)
pub const SAMPLE_RATE: u32 = 16_000;

#[cfg(test)]
mod tests;

#[cfg(test)]
use tests::*;

/// Decode an audio file into a mono PCM float vector and its sample rate.
///
/// This function uses `symphonia` to probe and decode the given audio file
/// path. It selects the first decodable audio track and converts samples to
/// `f32` PCM samples in the range appropriate for the original sample format.
/// The returned audio is mono (first channel) as `Vec<f32>` together with the
/// sample rate (Hz).
///
/// Errors are returned via `candle::Error` on file/codec failures.
pub fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};

    // Open the media source.
    let src = std::fs::File::open(path.as_ref()).map_err(Error::wrap)?;

    // Create the media source stream.
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    // This helps Symphonia choose the correct format reader based on
    // the file extension when available.
    let mut hint = symphonia::core::probe::Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Use the default options for metadata and format readers.
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(Error::wrap)?;
    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::Msg("no supported audio tracks".to_string()))?;

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|_| Error::Msg("unsupported codec".to_string()))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    // The decode loop.
    while let Ok(packet) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        // Decode to an AudioBufferRef and copy samples into a SampleBuffer<f32>
        // which provides interleaved f32 samples regardless of the packet's
        // original sample type. Then average channels to produce mono.
        let decoded = decoder.decode(&packet).map_err(Error::wrap)?;
        let frames = decoded.frames();
        let spec = *decoded.spec();

        // Create a sample buffer of f32 and copy interleaved samples into it.
        let mut sample_buf = SampleBuffer::<f32>::new(frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let interleaved = sample_buf.samples();
        let channels = spec.channels.count();
        if channels == 0 {
            continue;
        }

        // Average channels into mono per frame.
        for frame in 0..frames {
            let base = frame * channels;
            let mut sum = 0f32;
            for ch in 0..channels {
                sum += interleaved[base + ch];
            }
            pcm_data.push(sum / channels as f32);
        }
    }
    Ok((pcm_data, sample_rate))
}

/// Resample a PCM buffer from `sr_in` to `sr_out` using a high-quality FFT resampler.
///
/// - `pcm_in`: input mono PCM samples (f32)
/// - `sr_in`: input sample rate in Hz
/// - `sr_out`: desired output sample rate in Hz
///
/// Returns a newly allocated `Vec<f32>` with the resampled audio.
pub fn resample(pcm_in: &[f32], sr_in: u32, sr_out: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let mut pcm_out =
        Vec::with_capacity((pcm_in.len() as f64 * sr_out as f64 / sr_in as f64) as usize + 1024);

    let mut resampler = rubato::FftFixedInOut::<f32>::new(sr_in as usize, sr_out as usize, 1024, 1)
        .map_err(candle_core::Error::wrap)?;
    let mut output_buffer = resampler.output_buffer_allocate(true);
    let mut pos_in = 0;
    while pos_in + resampler.input_frames_next() < pcm_in.len() {
        let (in_len, out_len) = resampler
            .process_into_buffer(&[&pcm_in[pos_in..]], &mut output_buffer, None)
            .map_err(candle_core::Error::wrap)?;
        pos_in += in_len;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    if pos_in < pcm_in.len() {
        let (_in_len, out_len) = resampler
            .process_partial_into_buffer(Some(&[&pcm_in[pos_in..]]), &mut output_buffer, None)
            .map_err(candle_core::Error::wrap)?;
        pcm_out.extend_from_slice(&output_buffer[0][..out_len]);
    }

    Ok(pcm_out)
}
