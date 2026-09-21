use std::env;
use std::error::Error;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::sleep;

static LOG_FILE: OnceLock<Option<Mutex<fs::File>>> = OnceLock::new();

fn init_logging(path: &Path) {
    let opened = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .ok()
        .map(Mutex::new);
    if opened.is_none() {
        eprintln!(
            "⚠️  Failed to open log file at {} — continuing without file logging.",
            path.display()
        );
    }
    let _ = LOG_FILE.set(opened);
}

fn log_line(stream: &str, message: &str) {
    let Some(Some(mutex)) = LOG_FILE.get() else {
        return;
    };
    if let Ok(mut file) = mutex.lock() {
        let ts = Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%z");
        let _ = writeln!(file, "{} [{}] {}", ts, stream, message);
    }
}

macro_rules! info {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        println!("{}", msg);
        $crate::log_line("info", &msg);
    }};
}

macro_rules! warn_ {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        eprintln!("{}", msg);
        $crate::log_line("warn", &msg);
    }};
}

#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Content,
}

#[derive(Deserialize, Debug)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Deserialize, Debug)]
struct Part {
    text: String,
}

#[derive(Deserialize, Debug, Serialize)]
struct StockMetadata {
    title: String,
    description: String,
    keywords: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct StockMetadataBatch {
    results: Vec<StockMetadata>,
}

fn strip_markdown_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let without_prefix = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed)
        .trim_start();
    without_prefix
        .strip_suffix("```")
        .unwrap_or(without_prefix)
        .trim()
}

// The title/description/keyword rules are identical whether we're tagging a
// still photo or a video's frames, so they live here and are composed into both
// prompts below.
const METADATA_RULES: &str = "1. A catchy, highly relevant Title of 5-7 words that reads as ONE coherent, grammatical sentence: the words must connect into a natural, readable phrase describing the scene, NOT a list of loosely related keywords. The title must never exceed 200 characters. Write it to drive sales: lead with the terms commercial buyers actually search for and the concept the image sells.\n\
     2. A detailed Description/Caption (1-2 sentences describing the scene), also written to drive sales: highlight the commercial concepts and use cases buyers search for, while staying factual to what is visible.\n\
     3. Up to 25 keywords strictly sorted in ORDER OF PRECEDENCE (the most important, visible subjects must come first, followed by broader categories, with abstract moods at the very end).\n\
     STRICT RULE FOR KEYWORDS: Only include elements that are directly visible or explicitly factual to the scene. Do not guess locations (e.g., 'Tokyo'), seasons, or industries unless there is undeniable visual proof in the image. Avoid fluff.\n\
     CRITICAL GETTY IMAGES CONSTRAINT: Every keyword must be a single, standalone word or a universally standard two-word term (e.g., 'digital tablet', 'golden retriever'). Avoid descriptive phrases, sentences, or action-statements in the keywords array. Keep them literal, concrete, and distinct.";

// Lowercase everything, then capitalize the first letter of each sentence, so a
// title or caption reads as sentence case regardless of how the model cased it.
// Note: proper nouns and acronyms are lowercased too (e.g. "SMPTE" -> "Smpte") —
// a deliberate trade-off to guarantee "only the first word is capitalized".
fn to_sentence_case(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut capitalize_next = true;
    for ch in text.trim().to_lowercase().chars() {
        if capitalize_next && ch.is_alphabetic() {
            result.extend(ch.to_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                capitalize_next = true;
            }
        }
    }
    result
}

fn build_prompt(count: usize) -> String {
    format!(
        "Analyze the following {count} image(s) for stock photography optimization. \
         The images are provided in order, each preceded by a text label like 'Image N:'. \
         For EACH image independently, provide:\n{rules}\n\
         You must return the response STRICTLY as a JSON object with a single key 'results', whose value is an array of exactly {count} object(s), one per image IN THE SAME ORDER as the images were provided. Each object must have keys: 'title', 'description', and 'keywords'.",
        count = count,
        rules = METADATA_RULES
    )
}

// Frames are sampled from a single clip and must be reasoned about together, so
// the video prompt asks for exactly ONE result describing the whole video.
fn build_video_prompt(frame_count: usize) -> String {
    format!(
        "You are given {frame_count} still frame(s) extracted in chronological order from a SINGLE short video clip, \
         each preceded by a text label like 'Frame N:'. Treat them together as ONE video (not separate images) and \
         analyze the clip as a whole for stock footage optimization. Provide:\n{rules}\n\
         You must return the response STRICTLY as a JSON object with a single key 'results', whose value is an array \
         containing EXACTLY ONE object with keys 'title', 'description', and 'keywords', describing the video as a whole.",
        frame_count = frame_count,
        rules = METADATA_RULES
    )
}

#[derive(Clone, Copy, Debug)]
enum Provider {
    Gemini,
    Groq,
}

struct LlmConfig {
    provider: Provider,
    api_key: String,
    model: String,
    rate_limit_ms: u64,
    batch_size: usize,
}

fn load_llm_config() -> Result<LlmConfig, Box<dyn Error>> {
    let provider_raw = env::var("PROVIDER").unwrap_or_else(|_| "gemini".to_string());
    let provider = match provider_raw.trim().to_lowercase().as_str() {
        "gemini" => Provider::Gemini,
        "groq" => Provider::Groq,
        other => {
            return Err(format!("Unknown PROVIDER '{}'. Use 'gemini' or 'groq'.", other).into());
        }
    };

    let (key_var, model_var, rate_var, default_model) = match provider {
        Provider::Gemini => (
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "GEMINI_RATE_LIMIT_MS",
            "gemini-3.5-flash-lite",
        ),
        Provider::Groq => (
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "GROQ_RATE_LIMIT_MS",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
    };

    let api_key = env::var(key_var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| format!("{} is not set (env var or .env file).", key_var))?;

    let model = env::var(model_var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_model.to_string());

    let rate_limit_ms: u64 = env::var(rate_var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let batch_size: usize = env::var("BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(10);

    Ok(LlmConfig {
        provider,
        api_key,
        model,
        rate_limit_ms,
        batch_size,
    })
}

// Read + base64-encode one image, retrying transient failures. macOS can return
// EDEADLK ("Resource deadlock avoided", os error 11) or a sharing violation when
// another process (Spotlight indexing, iCloud/Dropbox sync, antivirus) holds a
// lock on the file at that instant; a brief wait usually clears it.
async fn read_image_b64(path: &Path) -> Result<String, Box<dyn Error>> {
    const MAX_ATTEMPTS: u32 = 3;
    let mut last_err: Option<std::io::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        match tokio::fs::read(path).await {
            Ok(bytes) => return Ok(STANDARD.encode(&bytes)),
            Err(e) => {
                if attempt < MAX_ATTEMPTS {
                    warn_!(
                        "   ⏳ Read attempt {}/{} failed for [{}]: {} — retrying…",
                        attempt,
                        MAX_ATTEMPTS,
                        path.display(),
                        e
                    );
                    sleep(Duration::from_millis(300 * attempt as u64)).await;
                }
                last_err = Some(e);
            }
        }
    }
    Err(format!(
        "failed to read {} after {} attempts: {}",
        path.display(),
        MAX_ATTEMPTS,
        last_err.expect("loop runs at least once")
    )
    .into())
}

async fn query_vision(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    prompt: &str,
    label: &str,
    images: &[String],
) -> Result<Vec<StockMetadata>, Box<dyn Error>> {
    let raw_json = match cfg.provider {
        Provider::Gemini => {
            call_gemini(client, &cfg.api_key, &cfg.model, prompt, label, images).await?
        }
        Provider::Groq => {
            call_groq(client, &cfg.api_key, &cfg.model, prompt, label, images).await?
        }
    };

    let clean = strip_markdown_fence(&raw_json);
    let mut results = parse_batch(clean)?;

    // Normalize casing to stock conventions:
    //  • Title & description -> sentence case (only sentence-initial words are
    //    capitalized), flattening the model's occasional Title Case.
    //  • Keywords -> all lowercase, then de-duplicated case-insensitively while
    //    preserving order (duplicate keywords are rejected by some agencies).
    for result in &mut results {
        result.title = to_sentence_case(&result.title);
        result.description = to_sentence_case(&result.description);

        let mut seen = std::collections::HashSet::new();
        result.keywords = std::mem::take(&mut result.keywords)
            .into_iter()
            .map(|k| k.trim().to_lowercase())
            .filter(|k| !k.is_empty() && seen.insert(k.clone()))
            .collect();
    }

    Ok(results)
}

// The model is asked for `{"results": [...]}`, but tolerate a bare top-level
// array too in case it ignores the wrapper.
fn parse_batch(clean: &str) -> Result<Vec<StockMetadata>, Box<dyn Error>> {
    if let Ok(batch) = serde_json::from_str::<StockMetadataBatch>(clean) {
        return Ok(batch.results);
    }
    serde_json::from_str::<Vec<StockMetadata>>(clean)
        .map_err(|e| format!("Failed to parse model JSON: {} (payload: {})", e, clean).into())
}

async fn call_gemini(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    label: &str,
    base64_images: &[String],
) -> Result<String, Box<dyn Error>> {
    let mut parts: Vec<serde_json::Value> = Vec::with_capacity(base64_images.len() * 2 + 1);
    parts.push(json!({ "text": prompt }));
    for (i, base64_image) in base64_images.iter().enumerate() {
        parts.push(json!({ "text": format!("{} {}:", label, i + 1) }));
        parts.push(json!({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64_image
            }
        }));
    }

    let payload = json!({
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let response = client.post(&url).json(&payload).send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let err_text = response.text().await.unwrap_or_default();
        return Err(format!("Gemini API error ({}): {}", status, err_text).into());
    }

    let res: GeminiResponse = response.json().await?;
    let part = res
        .candidates
        .into_iter()
        .next()
        .ok_or("Gemini response contained no candidates")?
        .content
        .parts
        .into_iter()
        .next()
        .ok_or("Gemini response candidate contained no parts")?;
    Ok(part.text)
}

#[derive(Deserialize, Debug)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Deserialize, Debug)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Deserialize, Debug)]
struct GroqMessage {
    content: String,
}

async fn call_groq(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    label: &str,
    base64_images: &[String],
) -> Result<String, Box<dyn Error>> {
    let mut content: Vec<serde_json::Value> = Vec::with_capacity(base64_images.len() * 2 + 1);
    content.push(json!({ "type": "text", "text": prompt }));
    for (i, base64_image) in base64_images.iter().enumerate() {
        content.push(json!({ "type": "text", "text": format!("{} {}:", label, i + 1) }));
        content.push(json!({
            "type": "image_url",
            "image_url": { "url": format!("data:image/jpeg;base64,{}", base64_image) }
        }));
    }

    let payload = json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": content
        }],
        "response_format": { "type": "json_object" }
    });

    let response = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let err_text = response.text().await.unwrap_or_default();
        return Err(format!("Groq API error ({}): {}", status, err_text).into());
    }

    let res: GroqResponse = response.json().await?;
    let choice = res
        .choices
        .into_iter()
        .next()
        .ok_or("Groq response contained no choices")?;
    Ok(choice.message.content)
}

#[derive(Default)]
struct ExtraTags {
    country: Option<String>,
    camera_make: Option<String>,
    camera_model: Option<String>,
}

fn existing_exif_field(image_path: &Path, tag: &str) -> Option<String> {
    let out = Command::new("exiftool")
        .arg("-s3") // value only
        .arg(format!("-{}", tag))
        .arg(image_path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn run_exiftool(args: &[&std::ffi::OsStr]) -> Result<(), Box<dyn Error>> {
    let output = Command::new("exiftool")
        .args(args)
        .output()
        .map_err(|e| -> Box<dyn Error> {
            if e.kind() == std::io::ErrorKind::NotFound {
                "exiftool not found on PATH (install via `brew install exiftool`)".into()
            } else {
                format!("failed to invoke exiftool: {}", e).into()
            }
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("exiftool failed: {}", stderr.trim()).into());
    }
    Ok(())
}

fn write_iptc_headers(
    image_path: &Path,
    metadata: StockMetadata,
    extras: &ExtraTags,
) -> Result<(), Box<dyn Error>> {
    use std::ffi::OsString;

    // exiftool quirk: list-type tags (XMP-dc:Subject is a Bag, IPTC:Keywords a
    // list) cannot be both cleared and re-populated in the same invocation — the
    // clear is silently dropped. So we clear them in a separate first pass to
    // avoid keywords accumulating across runs. This is non-fatal: if it fails
    // (e.g. a malformed IRB), the write below will trigger the rebuild fallback.
    let clear_args: Vec<OsString> = vec![
        "-overwrite_original".into(),
        "-m".into(),
        "-XMP-dc:Subject=".into(),
        "-IPTC:Keywords=".into(),
        image_path.as_os_str().to_os_string(),
    ];
    let clear_refs: Vec<&std::ffi::OsStr> = clear_args.iter().map(|s| s.as_os_str()).collect();
    if let Err(clear_err) = run_exiftool(&clear_refs) {
        warn_!(
            "⚠️  Could not pre-clear keyword tags for [{}]: {}",
            image_path.display(),
            clear_err
        );
    }

    // Only fill camera fields if the source JPEG doesn't already carry them, so
    // we never clobber genuine EXIF from a real camera (an iPhone's own
    // "Apple" / "iPhone 15 Pro" tags are kept exactly as shot).
    let add_make = extras
        .camera_make
        .as_deref()
        .filter(|_| existing_exif_field(image_path, "EXIF:Make").is_none());
    let add_model = extras
        .camera_model
        .as_deref()
        .filter(|_| existing_exif_field(image_path, "EXIF:Model").is_none());

    // `rebuild_irb` toggles `-Photoshop:all=`, which wipes and regenerates the
    // Photoshop IRB. That destroys every *other* IPTC field (city, creator,
    // copyright, …), so we only use it as a fallback when a normal write fails —
    // some source JPEGs ship a malformed IRB that blocks IPTC writes until it is
    // regenerated.
    let build_args = |rebuild_irb: bool| -> Vec<OsString> {
        let mut args: Vec<OsString> = Vec::new();
        args.push("-overwrite_original".into());
        args.push("-m".into()); // tolerate minor errors
        args.push("-codedcharacterset=utf8".into());
        if rebuild_irb {
            args.push("-Photoshop:all=".into());
        }
        args.push(format!("-IPTC:ObjectName={}", metadata.title).into());
        args.push(format!("-IPTC:Caption-Abstract={}", metadata.description).into());
        args.push(format!("-XMP-dc:Title={}", metadata.title).into());
        args.push(format!("-XMP-dc:Description={}", metadata.description).into());

        for keyword in &metadata.keywords {
            let trimmed = keyword.trim();
            if trimmed.is_empty() {
                continue;
            }
            args.push(format!("-IPTC:Keywords+={}", trimmed).into());
            args.push(format!("-XMP-dc:Subject+={}", trimmed).into());
        }

        if let Some(country) = extras.country.as_deref() {
            args.push(format!("-IPTC:Country-PrimaryLocationName={}", country).into());
            args.push(format!("-XMP-photoshop:Country={}", country).into());
            args.push(format!("-XMP-iptcExt:LocationCreatedCountryName={}", country).into());
            args.push(format!("-XMP-iptcExt:LocationShownCountryName={}", country).into());
        }

        if let Some(make) = add_make {
            args.push(format!("-EXIF:Make={}", make).into());
        }
        if let Some(model) = add_model {
            args.push(format!("-EXIF:Model={}", model).into());
        }

        args.push(image_path.as_os_str().to_os_string());
        args
    };

    // First attempt is non-destructive: it preserves existing IPTC fields such
    // as city, sub-location, creator, and copyright.
    let write_args = build_args(false);
    let write_refs: Vec<&std::ffi::OsStr> = write_args.iter().map(|s| s.as_os_str()).collect();
    if run_exiftool(&write_refs).is_ok() {
        return Ok(());
    }

    // Fallback: the source JPEG likely has a malformed IRB. Rebuild it. This may
    // drop other IPTC fields, but that segment was unreadable anyway.
    warn_!(
        "⚠️  Standard IPTC write failed for [{}]; retrying with IRB rebuild (other IPTC fields may be lost).",
        image_path.display()
    );
    let rebuild_args = build_args(true);
    let rebuild_refs: Vec<&std::ffi::OsStr> = rebuild_args.iter().map(|s| s.as_os_str()).collect();
    run_exiftool(&rebuild_refs)
}

// Write stock metadata to a QuickTime (.mov) container. Videos carry no Photoshop
// IRB or EXIF, so this uses XMP (read by Adobe apps and most stock ingesters)
// plus QuickTime Keys (surfaced by Finder and QuickTime Player) instead of the
// IPTC/EXIF path used for JPEGs.
fn write_video_metadata(
    video_path: &Path,
    metadata: StockMetadata,
    extras: &ExtraTags,
) -> Result<(), Box<dyn Error>> {
    use std::ffi::OsString;

    // Clear the keyword tags first so they don't accumulate across runs (same
    // exiftool list-tag quirk as for JPEGs). Non-fatal if it fails.
    let clear_args: Vec<OsString> = vec![
        "-overwrite_original".into(),
        "-m".into(),
        "-XMP-dc:Subject=".into(),
        "-Keys:Keywords=".into(),
        video_path.as_os_str().to_os_string(),
    ];
    let clear_refs: Vec<&std::ffi::OsStr> = clear_args.iter().map(|s| s.as_os_str()).collect();
    if let Err(clear_err) = run_exiftool(&clear_refs) {
        warn_!(
            "⚠️  Could not pre-clear keyword tags for [{}]: {}",
            video_path.display(),
            clear_err
        );
    }

    // Preserve a genuine camera identity (an iPhone's Apple/iPhone Keys, or a
    // real camera's tags) and only inject defaults when the clip carries none.
    let add_make = extras
        .camera_make
        .as_deref()
        .filter(|_| existing_exif_field(video_path, "Make").is_none());
    let add_model = extras
        .camera_model
        .as_deref()
        .filter(|_| existing_exif_field(video_path, "Model").is_none());

    let mut args: Vec<OsString> = Vec::new();
    args.push("-overwrite_original".into());
    args.push("-m".into());
    args.push(format!("-XMP-dc:Title={}", metadata.title).into());
    args.push(format!("-XMP-dc:Description={}", metadata.description).into());
    args.push(format!("-Keys:Title={}", metadata.title).into());
    args.push(format!("-Keys:Description={}", metadata.description).into());

    // XMP-dc:Subject is a proper list; QuickTime's Keys:Keywords is a single
    // scalar, so keywords are also joined into one comma-separated string.
    let mut kept: Vec<&str> = Vec::new();
    for keyword in &metadata.keywords {
        let trimmed = keyword.trim();
        if trimmed.is_empty() {
            continue;
        }
        kept.push(trimmed);
        args.push(format!("-XMP-dc:Subject+={}", trimmed).into());
    }
    if !kept.is_empty() {
        args.push(format!("-Keys:Keywords={}", kept.join(", ")).into());
    }

    if let Some(country) = extras.country.as_deref() {
        args.push(format!("-XMP-photoshop:Country={}", country).into());
        args.push(format!("-XMP-iptcExt:LocationCreatedCountryName={}", country).into());
        args.push(format!("-XMP-iptcExt:LocationShownCountryName={}", country).into());
    }
    if let Some(make) = add_make {
        args.push(format!("-Keys:Make={}", make).into());
    }
    if let Some(model) = add_model {
        args.push(format!("-Keys:Model={}", model).into());
    }

    args.push(video_path.as_os_str().to_os_string());
    let refs: Vec<&std::ffi::OsStr> = args.iter().map(|s| s.as_os_str()).collect();
    run_exiftool(&refs)
}

// Pick up to `k` items spread evenly across the slice (first and last included).
fn pick_evenly(items: &[PathBuf], k: usize) -> Vec<PathBuf> {
    if k == 0 {
        return Vec::new();
    }
    if items.len() <= k {
        return items.to_vec();
    }
    let n = items.len();
    (0..k)
        .map(|i| items[i * (n - 1) / (k - 1)].clone())
        .collect()
}

// Sample frames from a video with ffmpeg into a fresh temp directory, downscaled
// so we never ship 4K/ProRes-sized stills to the model. Returns the temp dir (so
// the caller can delete it) and the frame paths actually selected.
fn extract_frames(
    video: &Path,
    fps: f64,
    max_dim: u32,
    max_frames: usize,
) -> Result<(PathBuf, Vec<PathBuf>), Box<dyn Error>> {
    let unique = format!(
        "phototag_frames_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = env::temp_dir().join(unique);
    fs::create_dir_all(&dir)?;

    // scale='min(max_dim,iw)':-2 preserves aspect ratio, never upscales, and
    // forces an even height (required by the encoder).
    let vf = format!("fps={},scale='min({},iw)':-2", fps, max_dim);
    let pattern = dir.join("frame_%04d.jpg");
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(video)
        .arg("-vf")
        .arg(&vf)
        .arg("-q:v")
        .arg("3")
        .arg(&pattern)
        .output()
        .map_err(|e| -> Box<dyn Error> {
            if e.kind() == std::io::ErrorKind::NotFound {
                "ffmpeg not found on PATH (install via `brew install ffmpeg`)".into()
            } else {
                format!("failed to invoke ffmpeg: {}", e).into()
            }
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_dir_all(&dir);
        return Err(format!("ffmpeg failed: {}", stderr.trim()).into());
    }

    let mut frames: Vec<PathBuf> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| has_jpeg_extension(p))
        .collect();
    frames.sort();

    if frames.is_empty() {
        let _ = fs::remove_dir_all(&dir);
        return Err("ffmpeg produced no frames".into());
    }
    if frames.len() > max_frames {
        frames = pick_evenly(&frames, max_frames);
    }
    Ok((dir, frames))
}

// Tag each video: extract frames, ask the model for one combined result, and
// embed it in the .mov. Each clip is its own API call (its frames fill the
// request), so videos are not batched with photos or with each other.
async fn process_videos(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    extras: &ExtraTags,
    videos: &[PathBuf],
    fps: f64,
) {
    // Downscale target for sampled frames, and a hard cap so a long clip can't
    // explode into hundreds of frames (or blow past provider image limits).
    const FRAME_MAX_DIM: u32 = 1024;
    const MAX_FRAMES: usize = 15;

    let total = videos.len();
    for (idx, video) in videos.iter().enumerate() {
        info!("🎬 Video {}/{} — {}", idx + 1, total, video.display());

        let (dir, frame_paths) = match extract_frames(video, fps, FRAME_MAX_DIM, MAX_FRAMES) {
            Ok(v) => v,
            Err(e) => {
                warn_!(
                    "❌ Frame extraction failed for [{}]: {}",
                    video.display(),
                    e
                );
                continue;
            }
        };
        info!("   🖼  Using {} frame(s) at {} fps.", frame_paths.len(), fps);

        let mut frames_b64: Vec<String> = Vec::with_capacity(frame_paths.len());
        for fp in &frame_paths {
            match read_image_b64(fp).await {
                Ok(b64) => frames_b64.push(b64),
                Err(e) => warn_!("   ⚠️  Skipping unreadable frame [{}]: {}", fp.display(), e),
            }
        }

        if frames_b64.is_empty() {
            warn_!("❌ No readable frames for [{}]; skipping.", video.display());
            let _ = fs::remove_dir_all(&dir);
            continue;
        }

        let prompt = build_video_prompt(frames_b64.len());
        match query_vision(client, cfg, &prompt, "Frame", &frames_b64).await {
            Ok(mut results) => match results.drain(..).next() {
                Some(metadata) => {
                    info!(
                        "   → title: {} | keywords: {}",
                        metadata.title,
                        metadata.keywords.len()
                    );
                    if let Err(e) = write_video_metadata(video, metadata, extras) {
                        warn_!("❌ Metadata write failed for [{}]: {}", video.display(), e);
                    } else {
                        info!("✅ Embedded video metadata.");
                    }
                }
                None => warn_!("❌ Model returned no metadata for [{}].", video.display()),
            },
            Err(e) => warn_!(
                "❌ {:?} video call failed for [{}]: {}",
                cfg.provider,
                video.display(),
                e
            ),
        }

        let _ = fs::remove_dir_all(&dir);

        if idx + 1 < total {
            sleep(Duration::from_millis(cfg.rate_limit_ms)).await;
        }
    }
}

fn has_jpeg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let lower = e.to_ascii_lowercase();
            lower == "jpg" || lower == "jpeg"
        })
        .unwrap_or(false)
}

fn has_mov_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("mov"))
        .unwrap_or(false)
}

fn is_supported_media(path: &Path) -> bool {
    has_jpeg_extension(path) || has_mov_extension(path)
}

fn collect_targets(input: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut targets = Vec::new();
    if input.is_file() {
        if is_supported_media(input) {
            targets.push(input.to_path_buf());
        }
    } else if input.is_dir() {
        for entry in fs::read_dir(input)? {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    eprintln!("⚠️  Skipping unreadable entry: {}", err);
                    continue;
                }
            };
            let path = entry.path();
            if path.is_file() && is_supported_media(&path) {
                targets.push(path);
            }
        }
        targets.sort();
    } else {
        return Err(format!(
            "Path does not exist as a file or folder: {}",
            input.display()
        )
        .into());
    }
    Ok(targets)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();

    let log_path = env::var("LOG_FILE").unwrap_or_else(|_| "photo_tagger.log".to_string());
    init_logging(Path::new(&log_path));

    let llm_cfg = match load_llm_config() {
        Ok(c) => c,
        Err(e) => {
            warn_!("❌ {}", e);
            std::process::exit(1);
        }
    };

    let extras = ExtraTags {
        country: Some(
            env::var("DEFAULT_COUNTRY")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "United Kingdom".to_string()),
        ),
        camera_make: Some(
            env::var("DEFAULT_CAMERA_MAKE")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "Panasonic".to_string()),
        ),
        camera_model: Some(
            env::var("DEFAULT_CAMERA_MODEL")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "DC-S5M2X".to_string()),
        ),
    };

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        warn_!("🚀 Automated Stock Photo Tagger");
        warn_!("Usage: {} <file_or_directory>", args[0]);
        std::process::exit(1);
    }

    let input_target = Path::new(&args[1]);
    let target_files = match collect_targets(input_target) {
        Ok(files) => files,
        Err(e) => {
            warn_!("❌ {}", e);
            std::process::exit(1);
        }
    };

    // .jpg/.jpeg go through the image pipeline; .mov go through the video
    // pipeline (frame extraction + QuickTime/XMP tags).
    let (photo_files, video_files): (Vec<PathBuf>, Vec<PathBuf>) = target_files
        .into_iter()
        .partition(|p| has_jpeg_extension(p));

    if photo_files.is_empty() && video_files.is_empty() {
        warn_!(
            "⚠️  No .jpg / .jpeg / .mov files found at {}",
            input_target.display()
        );
        return Ok(());
    }

    let video_fps: f64 = env::var("VIDEO_FPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&f| f > 0.0)
        .unwrap_or(2.0);

    info!(
        "⚙️  Found {} photo(s) and {} video(s). Provider: {:?} | Model: {} | Log: {}",
        photo_files.len(),
        video_files.len(),
        llm_cfg.provider,
        llm_cfg.model,
        log_path
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    if !photo_files.is_empty() {
        let batch_size = llm_cfg.batch_size;
        let total_batches = photo_files.len().div_ceil(batch_size);
        info!(
            "📸 Tagging {} photo(s) in {} batch(es) of up to {}.",
            photo_files.len(),
            total_batches,
            batch_size
        );

        for (batch_idx, chunk) in photo_files.chunks(batch_size).enumerate() {
            info!(
                "📦 Batch {}/{} — {} image(s):",
                batch_idx + 1,
                total_batches,
                chunk.len()
            );
            for path in chunk {
                info!("   • {}", path.display());
            }

            // Read the batch's files up front. An unreadable file (e.g. transiently
            // locked by Spotlight/cloud-sync) is skipped rather than aborting the
            // whole batch, so its readable siblings still get tagged.
            let mut readable_paths: Vec<&PathBuf> = Vec::with_capacity(chunk.len());
            let mut images: Vec<String> = Vec::with_capacity(chunk.len());
            for path in chunk {
                match read_image_b64(path).await {
                    Ok(b64) => {
                        readable_paths.push(path);
                        images.push(b64);
                    }
                    Err(read_err) => {
                        warn_!(
                            "❌ Skipping unreadable file [{}]: {}",
                            path.display(),
                            read_err
                        );
                    }
                }
            }

            if images.is_empty() {
                warn_!("⚠️  No readable images in this batch; skipping API call.");
                continue;
            }

            match query_vision(
                &client,
                &llm_cfg,
                &build_prompt(images.len()),
                "Image",
                &images,
            )
            .await
            {
                Ok(results) => {
                    if results.len() != readable_paths.len() {
                        warn_!(
                            "⚠️  Model returned {} result(s) for {} image(s); pairing by order.",
                            results.len(),
                            readable_paths.len()
                        );
                    }

                    let mut paired = 0usize;
                    for (target_path, metadata) in readable_paths.iter().zip(results.into_iter()) {
                        paired += 1;
                        info!(
                            "   → [{}] title: {} | keywords: {}",
                            target_path.display(),
                            metadata.title,
                            metadata.keywords.len()
                        );
                        if let Err(iptc_err) =
                            write_iptc_headers(target_path.as_path(), metadata, &extras)
                        {
                            warn_!(
                                "❌ IPTC write failed for [{}]: {}",
                                target_path.display(),
                                iptc_err
                            );
                        } else {
                            info!("✅ Embedded IPTC metadata.");
                        }
                    }

                    for unpaired in readable_paths.iter().skip(paired) {
                        warn_!(
                            "❌ No metadata returned for [{}] (model returned too few results).",
                            unpaired.display()
                        );
                    }
                }
                Err(api_err) => {
                    warn_!(
                        "❌ {:?} batch call failed ({} image(s)): {}",
                        llm_cfg.provider,
                        images.len(),
                        api_err
                    );
                }
            }

            if batch_idx + 1 < total_batches {
                sleep(Duration::from_millis(llm_cfg.rate_limit_ms)).await;
            }
        }
    }

    if !video_files.is_empty() {
        info!(
            "🎬 Tagging {} video(s) at {} fps.",
            video_files.len(),
            video_fps
        );
        process_videos(&client, &llm_cfg, &extras, &video_files, video_fps).await;
    }

    info!("🎉 Done.");
    Ok(())
}
