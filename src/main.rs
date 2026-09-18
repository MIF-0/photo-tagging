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

fn build_prompt(count: usize) -> String {
    format!(
        "Analyze the following {count} image(s) for stock photography optimization. \
         The images are provided in order, each preceded by a text label like 'Image N:'. \
         For EACH image independently, provide:\n\
         1. A catchy, highly relevant Title of 5-7 words that reads as ONE coherent, grammatical sentence: the words must connect into a natural, readable phrase describing the scene, NOT a list of loosely related keywords. The title must never exceed 200 characters. Write it to drive sales: lead with the terms commercial buyers actually search for and the concept the image sells.\n\
         2. A detailed Description/Caption (1-2 sentences describing the scene), also written to drive sales: highlight the commercial concepts and use cases buyers search for, while staying factual to what is visible.\n\
         3. Up to 25 keywords strictly sorted in ORDER OF PRECEDENCE (the most important, visible subjects must come first, followed by broader categories, with abstract moods at the very end).\n\
         STRICT RULE FOR KEYWORDS: Only include elements that are directly visible or explicitly factual to the scene. Do not guess locations (e.g., 'Tokyo'), seasons, or industries unless there is undeniable visual proof in the image. Avoid fluff.\n\
         CRITICAL GETTY IMAGES CONSTRAINT: Every keyword must be a single, standalone word or a universally standard two-word term (e.g., 'digital tablet', 'golden retriever'). Avoid descriptive phrases, sentences, or action-statements in the keywords array. Keep them literal, concrete, and distinct.\n\
         You must return the response STRICTLY as a JSON object with a single key 'results', whose value is an array of exactly {count} object(s), one per image IN THE SAME ORDER as the images were provided. Each object must have keys: 'title', 'description', and 'keywords'.",
        count = count
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

async fn query_vision_batch(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    images: &[String],
) -> Result<Vec<StockMetadata>, Box<dyn Error>> {
    let raw_json = match cfg.provider {
        Provider::Gemini => call_gemini(client, &cfg.api_key, &cfg.model, images).await?,
        Provider::Groq => call_groq(client, &cfg.api_key, &cfg.model, images).await?,
    };

    let clean = strip_markdown_fence(&raw_json);
    parse_batch(clean)
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
    base64_images: &[String],
) -> Result<String, Box<dyn Error>> {
    let mut parts: Vec<serde_json::Value> = Vec::with_capacity(base64_images.len() * 2 + 1);
    parts.push(json!({ "text": build_prompt(base64_images.len()) }));
    for (i, base64_image) in base64_images.iter().enumerate() {
        parts.push(json!({ "text": format!("Image {}:", i + 1) }));
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
    base64_images: &[String],
) -> Result<String, Box<dyn Error>> {
    let mut content: Vec<serde_json::Value> = Vec::with_capacity(base64_images.len() * 2 + 1);
    content.push(json!({ "type": "text", "text": build_prompt(base64_images.len()) }));
    for (i, base64_image) in base64_images.iter().enumerate() {
        content.push(json!({ "type": "text", "text": format!("Image {}:", i + 1) }));
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

fn has_jpeg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let lower = e.to_ascii_lowercase();
            lower == "jpg" || lower == "jpeg"
        })
        .unwrap_or(false)
}

fn collect_targets(input: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut targets = Vec::new();
    if input.is_file() {
        if has_jpeg_extension(input) {
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
            if path.is_file() && has_jpeg_extension(&path) {
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

    let total = target_files.len();
    if total == 0 {
        warn_!(
            "⚠️  No .jpg / .jpeg files found at {}",
            input_target.display()
        );
        return Ok(());
    }
    let batch_size = llm_cfg.batch_size;
    let total_batches = total.div_ceil(batch_size);
    info!(
        "⚙️  Found {} image target(s) to process in {} batch(es) of up to {}. Provider: {:?} | Model: {} | Log: {}",
        total, total_batches, batch_size, llm_cfg.provider, llm_cfg.model, log_path
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    for (batch_idx, chunk) in target_files.chunks(batch_size).enumerate() {
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

        match query_vision_batch(&client, &llm_cfg, &images).await {
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

    info!("🎉 Done.");
    Ok(())
}
