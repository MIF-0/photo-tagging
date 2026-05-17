use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::sleep;

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

async fn query_gemini_vision(
    client: &reqwest::Client,
    api_key: &str,
    image_path: &Path,
) -> Result<StockMetadata, Box<dyn Error>> {
    let image_bytes = fs::read(image_path)?;
    let base64_image = STANDARD.encode(&image_bytes);

    // let prompt = "Analyze this image for stock photography optimization. Provide:\n\
    //               1. A catchy, highly relevant Title (max 5-7 words).\n\
    //               2. A detailed Description/Caption (1-2 sentences describing the scene).\n\
    //               3. Up to 25 keywords strictly sorted in ORDER OF PRECEDECE (the most important, visible subjects must come first, followed by broader categories, with abstract moods at the very end).\n\
    //               STRICT RULE FOR KEYWORDS: Only include elements that are directly visible or explicitly factual to the scene. Do not guess locations (e.g., 'Tokyo'), seasons, or industries unless there is undeniable visual proof in the image. Avoid fluff.\n\
    //               You must return the response strictly as a JSON object with keys: 'title', 'description', and 'keywords'.";

    let getty_images_improved_prompt = "Analyze this image for stock photography optimization. Provide:\n\
                  1. A catchy, highly relevant Title (max 5-7 words).\n\
                  2. A detailed Description/Caption (1-2 sentences describing the scene).\n\
                  3. Up to 25 keywords strictly sorted in ORDER OF PRECEDECE (the most important, visible subjects must come first, followed by broader categories, with abstract moods at the very end).\n\
                  STRICT RULE FOR KEYWORDS: Only include elements that are directly visible or explicitly factual to the scene. Do not guess locations (e.g., 'Tokyo'), seasons, or industries unless there is undeniable visual proof in the image. Avoid fluff.\n\
                  You must return the response strictly as a JSON object with keys: 'title', 'description', and 'keywords'.
                  CRITICAL GETTY IMAGES CONSTRAINT: Every keyword must be a single, standalone word or a universally standard two-word term (e.g., 'digital tablet', 'golden retriever'). Avoid descriptive phrases, sentences, or action-statements in the keywords array. Keep them literal, concrete, and distinct.\n\
                  You must return the response strictly as a JSON object with keys: 'title', 'description', and 'keywords'.";

    let payload = json!({
        "contents": [{
            "parts": [
                { "text": getty_images_improved_prompt },
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={}",
        api_key
    );

    let response = client.post(&url).json(&payload).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let err_text = response.text().await.unwrap_or_default();
        return Err(format!("Gemini API error ({}): {}", status, err_text).into());
    }

    let gemini_res: GeminiResponse = response.json().await?;
    let candidate = gemini_res
        .candidates
        .into_iter()
        .next()
        .ok_or("Gemini response contained no candidates")?;
    let part = candidate
        .content
        .parts
        .into_iter()
        .next()
        .ok_or("Gemini response candidate contained no parts")?;

    let clean_json = strip_markdown_fence(&part.text);
    let metadata: StockMetadata = serde_json::from_str(clean_json).map_err(|e| {
        format!(
            "Failed to parse Gemini JSON: {} (payload: {})",
            e, clean_json
        )
    })?;
    Ok(metadata)
}

fn write_iptc_headers(image_path: &Path, metadata: StockMetadata) -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::new("exiftool");
    cmd.arg("-overwrite_original")
        .arg("-m") // tolerate minor errors
        .arg("-codedcharacterset=utf8")
        // Rebuild the Photoshop IRB from scratch. Some source JPEGs (e.g. the
        // bundled example) ship with a malformed IRB that blocks any IPTC write
        // until the segment is regenerated.
        .arg("-Photoshop:all=")
        .arg(format!("-IPTC:ObjectName={}", metadata.title))
        .arg(format!("-IPTC:Caption-Abstract={}", metadata.description))
        .arg(format!("-XMP-dc:Title={}", metadata.title))
        .arg(format!("-XMP-dc:Description={}", metadata.description))
        .arg("-XMP-dc:Subject=");

    for keyword in &metadata.keywords {
        let trimmed = keyword.trim();
        if trimmed.is_empty() {
            continue;
        }
        cmd.arg(format!("-IPTC:Keywords+={}", trimmed));
        cmd.arg(format!("-XMP-dc:Subject+={}", trimmed));
    }

    cmd.arg(image_path);

    let output = cmd.output().map_err(|e| -> Box<dyn Error> {
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

    let api_key = match env::var("GEMINI_API_KEY") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            eprintln!("❌ GEMINI_API_KEY is not set (env var or .env file).");
            std::process::exit(1);
        }
    };

    let rate_limit_ms: u64 = env::var("GEMINI_RATE_LIMIT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("🚀 Automated Stock Photo Tagger");
        eprintln!("Usage: {} <file_or_directory>", args[0]);
        std::process::exit(1);
    }

    let input_target = Path::new(&args[1]);
    let target_files = match collect_targets(input_target) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("❌ {}", e);
            std::process::exit(1);
        }
    };

    let total = target_files.len();
    if total == 0 {
        eprintln!(
            "⚠️  No .jpg / .jpeg files found at {}",
            input_target.display()
        );
        return Ok(());
    }
    println!("⚙️  Found {} image target(s) to process.", total);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    for (idx, target_path) in target_files.iter().enumerate() {
        println!(
            "[{}/{}] Processing: {}",
            idx + 1,
            total,
            target_path.display()
        );

        match query_gemini_vision(&client, &api_key, target_path).await {
            Ok(metadata) => {
                println!(
                    "   → title: {} | keywords: {}",
                    metadata.title,
                    metadata.keywords.len()
                );
                if let Err(iptc_err) = write_iptc_headers(target_path, metadata) {
                    eprintln!(
                        "❌ IPTC write failed for [{}]: {}",
                        target_path.display(),
                        iptc_err
                    );
                } else {
                    println!("✅ Embedded IPTC metadata.");
                }
            }
            Err(api_err) => {
                eprintln!(
                    "❌ Gemini call failed for [{}]: {}",
                    target_path.display(),
                    api_err
                );
            }
        }

        if idx + 1 < total {
            sleep(Duration::from_millis(rate_limit_ms)).await;
        }
    }

    println!("🎉 Done.");
    Ok(())
}
