use std::collections::{BTreeMap, HashMap, HashSet};
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
use image::{imageops, GrayImage};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::sleep;

static LOG_FILE: OnceLock<Option<Mutex<fs::File>>> = OnceLock::new();

// How long a digiKam queue's log stays open for its next image (see
// init_logging).
const DIGIKAM_LOG_SESSION: Duration = Duration::from_secs(10 * 60);

// Start the run's log: a fresh file, except that digiKam runs one process per
// image, so a digiKam run adds to a log written within DIGIKAM_LOG_SESSION
// (the same queue run) instead of wiping the images before it.
fn init_logging(path: &Path, digikam: bool) {
    let recent = || {
        fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age < DIGIKAM_LOG_SESSION)
    };
    let append = digikam && recent();
    let opened = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(path);
    let file = match opened {
        Ok(file) => Some(Mutex::new(file)),
        Err(e) => {
            eprintln!(
                "⚠️  Failed to open log file at {}: {} — continuing without file logging.",
                path.display(),
                e
            );
            None
        }
    };
    let _ = LOG_FILE.set(file);
}

// The log goes next to the photos (the tagged folder, like the grades report)
// rather than into the working directory, which apps such as digiKam set to
// `/`. LOG_FILE overrides it: relative paths are taken inside the tagged
// folder, and a leading `~/` means the home folder.
fn run_log_path(input: &Path) -> PathBuf {
    resolve_log_path(
        env::var("LOG_FILE").unwrap_or_default().trim(),
        input,
        env::var_os("HOME").as_deref(),
    )
}

fn resolve_log_path(configured: &str, input: &Path, home: Option<&std::ffi::OsStr>) -> PathBuf {
    if configured.is_empty() {
        return report_dir(input).join("photo_tagger.log");
    }
    if let (Some(rest), Some(home)) = (configured.strip_prefix("~/"), home) {
        return Path::new(home).join(rest);
    }
    let path = Path::new(configured);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        report_dir(input).join(path)
    }
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

// Console writes ignore errors: println! would panic when the output pipe is
// closed (e.g. `| head`, or a script host that stopped reading).
macro_rules! info {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let msg = format!($($arg)*);
        let _ = writeln!(std::io::stdout(), "{}", msg);
        $crate::log_line("info", &msg);
    }};
}

macro_rules! warn_ {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let msg = format!($($arg)*);
        let _ = writeln!(std::io::stderr(), "{}", msg);
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
    #[serde(default)]
    text: Option<String>,
    // Thinking models only return thought summaries when asked to, but never
    // treat one as the answer if it does show up.
    #[serde(default)]
    thought: bool,
}

#[derive(Deserialize, Debug, Serialize)]
struct StockMetadata {
    title: String,
    description: String,
    keywords: Vec<String>,
    // Only photo calls ask for grades (see GRADING_RULES). Both fields are
    // parsed leniently so a missing or malformed grade never costs a file its
    // tags.
    #[serde(default, deserialize_with = "lenient_grades")]
    grades: Option<Grades>,
    #[serde(default, deserialize_with = "lenient_notes")]
    grade_notes: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StockMetadataBatch {
    results: Vec<StockMetadata>,
}

// 1–10 grades from the photo prompt. `None` means the model didn't return a
// usable value for that dimension.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize)]
struct Grades {
    editing_quality: Option<u8>,
    technical_quality: Option<u8>,
    commercial_cleanliness: Option<u8>,
    market_demand: Option<u8>,
    overall: Option<u8>,
}

// The grade keys, in the order the model is asked for them and the report
// lists them.
const GRADE_KEYS: [&str; 5] = [
    "editing_quality",
    "technical_quality",
    "commercial_cleanliness",
    "market_demand",
    "overall",
];

impl Grades {
    // Build from a lookup by grade key.
    fn from_fn(mut grade: impl FnMut(&str) -> Option<u8>) -> Self {
        let [editing_quality, technical_quality, commercial_cleanliness, market_demand, overall] =
            GRADE_KEYS.map(&mut grade);
        Grades {
            editing_quality,
            technical_quality,
            commercial_cleanliness,
            market_demand,
            overall,
        }
    }

    // The grades in GRADE_KEYS order.
    fn values(&self) -> [Option<u8>; 5] {
        [
            self.editing_quality,
            self.technical_quality,
            self.commercial_cleanliness,
            self.market_demand,
            self.overall,
        ]
    }
}

// Parse `grades` field by field, so one malformed value (e.g. "n/a") only
// drops that grade instead of failing the whole batch's JSON.
fn lenient_grades<'de, D>(deserializer: D) -> Result<Option<Grades>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .and_then(|v| v.as_object())
        .map(|obj| Grades::from_fn(|key| obj.get(key).and_then(grade_from_value))))
}

// Accept a string, or a list of strings (joined with "; "); anything else
// counts as no notes rather than failing the whole batch's JSON.
fn lenient_notes<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(
        match Option::<serde_json::Value>::deserialize(deserializer)? {
            Some(serde_json::Value::String(notes)) => Some(notes),
            Some(serde_json::Value::Array(items)) => {
                let parts: Vec<&str> = items.iter().filter_map(|v| v.as_str()).collect();
                (!parts.is_empty()).then(|| parts.join("; "))
            }
            _ => None,
        },
    )
}

// Accept 7, 7.4 or "7"; round to a whole grade and clamp to 1–10. Anything
// non-numeric counts as missing.
fn grade_from_value(value: &serde_json::Value) -> Option<u8> {
    match value {
        serde_json::Value::Number(n) => normalize_grade(n.as_f64()?),
        serde_json::Value::String(s) => grade_from_str(s),
        _ => None,
    }
}

fn grade_from_str(text: &str) -> Option<u8> {
    normalize_grade(text.trim().parse().ok()?)
}

fn normalize_grade(n: f64) -> Option<u8> {
    n.is_finite().then(|| n.round().clamp(1.0, 10.0) as u8)
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
const METADATA_RULES: &str = "1. Title / Caption: A search-optimized commercial title (maximum 150-200 characters) structured as a single, grammatically coherent sentence. The words must be strictly arranged in descending order of commercial search demand—front-loading the highest-intent buyer keywords and primary subjects first, followed by contextual modifiers. It must satisfy strict search relevance thresholds and eliminate 'poor discovery' flags while remaining readable.\n\
     2. Description / Additional Info: A comprehensive, factual description of 1-2 sentences detailing the scene's context, physical attributes, materials, and explicit commercial use cases or editorial applications. It must provide sufficient semantic depth to maximize indexing weight and meet platform metadata quality standards.\n\
     3. Keywords (25 to 40 items): Strictly structured in descending ORDER OF PRECEDENCE — starting with exact, highly specific visible subjects and materials, moving to broader categorical descriptors, and ending with abstract conceptual attributes.\n\
     - STRICT VISIBILITY RULE: Strictly limit tags to indisputably visible elements. Do not extrapolate unverified locations, brands, temporal seasons, or industries.\n\
     - GETTY / ALAMY COMPLIANCE: Every keyword must be an individual standalone noun/adjective or a standardized, industry-accepted two-word compound term (e.g., 'wooden shelf', 'interior design'). Prohibit descriptive phrases, action clauses, or subjective modifiers.";

// Rule 3's keyword cap, applied after parsing (the model may return more). Its
// minimum is only asked for: a shorter list is still written.
const MAX_KEYWORDS: usize = 40;

// Lowercase everything, then capitalize the first letter of each sentence, so a
// title or caption reads as sentence case regardless of how the model cased it.
// A '.', '!' or '?' only ends a sentence when whitespace follows it (optionally
// after a closing quote or bracket) and, for '.', when the word before it is not
// an abbreviation. So decimals ("4.5 inch"), times ("3.30 pm"), ellipses
// ("event...crowd") and abbreviations ("e.g.", "dr.") do not capitalize the next
// word, and a sentence that opens with a digit ("4 tips") keeps its next word
// lowercase.
// Note: proper nouns and acronyms are lowercased too (e.g. "SMPTE" -> "Smpte") —
// a deliberate trade-off to guarantee "only the first word is capitalized".
fn to_sentence_case(text: &str) -> String {
    const ABBREVIATIONS: &[&str] = &["mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr", "approx"];

    let lower = text.trim().to_lowercase();
    let mut result = String::with_capacity(lower.len());
    let mut capitalize_next = true;
    // A terminator was just seen; it becomes a sentence break once whitespace follows.
    let mut pending_break = false;
    // The current word so far (since the last whitespace), for abbreviation checks.
    let mut word = String::new();

    for ch in lower.chars() {
        if ch.is_whitespace() {
            if pending_break {
                capitalize_next = true;
                pending_break = false;
            }
            word.clear();
            result.push(ch);
            continue;
        }

        if matches!(ch, '.' | '!' | '?') {
            // "e.g" / "u.s" / "event.." already contain a '.', so the next '.'
            // belongs to an abbreviation or an ellipsis, not a sentence end.
            let stem = word.trim_start_matches(|c: char| !c.is_alphanumeric());
            let abbreviation = ch == '.' && (stem.contains('.') || ABBREVIATIONS.contains(&stem));
            pending_break = !abbreviation;
        } else if !(pending_break && matches!(ch, '"' | '\'' | ')' | ']' | '”' | '’')) {
            // Anything but closing punctuation right after a terminator cancels
            // the break ("4.5", "3.30", "e.g").
            pending_break = false;
            if capitalize_next && ch.is_alphanumeric() {
                capitalize_next = false;
                if ch.is_alphabetic() {
                    result.extend(ch.to_uppercase());
                    word.push(ch);
                    continue;
                }
            }
        }

        word.push(ch);
        result.push(ch);
    }
    result
}

// Photo-only grading rubric, appended after METADATA_RULES. The fixed scale
// anchors matter: without them models give 7-8 to anything decent, which would
// make the "overall > BEST_MIN_GRADE" cut meaningless.
const GRADING_RULES: &str = "4. Grades: integers from 1 to 10, judged against the acceptance standards of professional stock agencies (Adobe Stock, Shutterstock, Getty Images). Grade every image on its own, in absolute terms; never rank or compare it against the other images in this request.\n\
     - editing_quality: post-processing craft: composition and crop, straight horizons and verticals, retouching (dust spots and distractions removed), tasteful colour grading. Penalize over-processing: halos, oversharpening, an HDR look, oversaturation, banding.\n\
     - technical_quality: capture-level technical standards: light and exposure (clipped highlights, crushed shadows), noise and grain, colour correction and white balance, focus and sharpness, chromatic aberration, compression artifacts.\n\
     - commercial_cleanliness: how safe the image is for COMMERCIAL licensing. 10 means there are no recognizable people, logos, brands, trademarks, readable text, identifiable private property or copyrighted artwork; lower it for each such element, since each needs a release or restricts the image to editorial use.\n\
     - market_demand: how much stock buyers need this content as of today's date: demand for its concept and commercial use cases versus how saturated the subject already is on stock sites.\n\
     - overall: expected sellability on major stock agencies, weighing all of the above. A serious technical flaw or a commercial-use blocker must keep it low.\n\
     Scale: 1-3 = likely rejected or unsellable; 4-6 = acceptable but generic, low expected sales; 7-8 = strong and clearly marketable; 9-10 = exceptional and rare. Be strict: most competent images belong in the 4-6 range.\n\
     5. grade_notes: one short line (at most 20 words) with the main reasons behind the grades, e.g. 'slight shadow noise; logo on mug; strong remote-work concept'.";

fn build_prompt(count: usize) -> String {
    format!(
        "Today's date is {today}. Analyze the following {count} image(s) for stock photography optimization. \
         The images are provided in order, each preceded by a text label like 'Image N:'. \
         For EACH image independently, provide:\n{rules}\n{grading}\n\
         You must return the response STRICTLY as a JSON object with a single key 'results', whose value is an array of exactly {count} object(s), one per image IN THE SAME ORDER as the images were provided. Each object must have keys: 'title', 'description', 'keywords', 'grades' (an object with integer keys 'editing_quality', 'technical_quality', 'commercial_cleanliness', 'market_demand' and 'overall') and 'grade_notes'.",
        today = Local::now().format("%Y-%m-%d"),
        count = count,
        rules = METADATA_RULES,
        grading = GRADING_RULES
    )
}

// Gemini structured-output schema for a photo batch: it enforces the result
// shape and integer 1–10 grades, so parsing rarely has to fall back. Array
// lengths are left unbounded on purpose: bounded (and especially nested)
// arrays multiply the schema's states until Gemini rejects it as too complex
// ("too many states for serving"), which would fail every batch. The result
// count and the keyword cap are checked after parsing instead.
fn photo_response_schema() -> serde_json::Value {
    let grade_properties: serde_json::Map<String, serde_json::Value> = GRADE_KEYS
        .iter()
        .map(|key| {
            (
                key.to_string(),
                json!({ "type": "INTEGER", "minimum": 1, "maximum": 10 }),
            )
        })
        .collect();

    json!({
        "type": "OBJECT",
        "properties": {
            "results": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": { "type": "STRING" },
                        "description": { "type": "STRING" },
                        "keywords": { "type": "ARRAY", "items": { "type": "STRING" } },
                        "grades": {
                            "type": "OBJECT",
                            "properties": grade_properties,
                            "required": GRADE_KEYS,
                            "propertyOrdering": GRADE_KEYS
                        },
                        "grade_notes": { "type": "STRING" }
                    },
                    "required": ["title", "description", "keywords", "grades", "grade_notes"],
                    // Describe the image first, then grade it.
                    "propertyOrdering": ["title", "description", "keywords", "grades", "grade_notes"]
                }
            }
        },
        "required": ["results"]
    })
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

struct LlmConfig {
    api_key: String,
    model: String,
    rate_limit_ms: u64,
    batch_size: usize,
    // Total time one API call may take, retries included (None = only the
    // client's per-request timeout applies).
    time_budget: Option<Duration>,
}

// Per-request timeout of the HTTP client.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
// Attempts per API call; see send_with_retries.
const MAX_ATTEMPTS: u32 = 4;

fn load_llm_config() -> Result<LlmConfig, Box<dyn Error>> {
    let api_key = env::var("GEMINI_API_KEY")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .ok_or("GEMINI_API_KEY is not set (env var or .env file).")?;

    let model = env::var("GEMINI_MODEL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "gemini-3.8-flash".to_string());

    let rate_limit_ms: u64 = env::var("GEMINI_RATE_LIMIT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let batch_size: usize = env::var("BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(10);

    Ok(LlmConfig {
        api_key,
        model,
        rate_limit_ms,
        batch_size,
        time_budget: None,
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
    schema: Option<&serde_json::Value>,
) -> Result<Vec<StockMetadata>, Box<dyn Error>> {
    let raw_json = call_gemini(client, cfg, prompt, label, images, schema).await?;

    let clean = strip_markdown_fence(&raw_json);
    let mut results = parse_batch(clean)?;

    // Normalize casing to stock conventions:
    //  • Title & description -> sentence case (only sentence-initial words are
    //    capitalized), flattening the model's occasional Title Case.
    //  • Keywords -> all lowercase, then de-duplicated case-insensitively while
    //    preserving order (duplicate keywords are rejected by some agencies),
    //    and capped at MAX_KEYWORDS.
    for result in &mut results {
        result.title = to_sentence_case(&result.title);
        result.description = to_sentence_case(&result.description);

        let mut seen = std::collections::HashSet::new();
        result.keywords = std::mem::take(&mut result.keywords)
            .into_iter()
            .map(|k| k.trim().to_lowercase())
            .filter(|k| !k.is_empty() && seen.insert(k.clone()))
            .take(MAX_KEYWORDS)
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
    cfg: &LlmConfig,
    prompt: &str,
    label: &str,
    base64_images: &[String],
    schema: Option<&serde_json::Value>,
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

    let mut generation_config = json!({ "responseMimeType": "application/json" });
    if let Some(schema) = schema {
        generation_config["responseSchema"] = schema.clone();
    }
    let payload = json!({
        "contents": [{
            "parts": parts
        }],
        "generationConfig": generation_config
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
        cfg.model
    );
    let response = send_with_retries(client, cfg, &url, &payload).await?;

    let res: GeminiResponse = response.json().await?;
    let candidate = res
        .candidates
        .into_iter()
        .next()
        .ok_or("Gemini response contained no candidates")?;
    // Thinking models (e.g. the gemini-3.x flash line) may split the answer
    // across several parts, so join every non-thought text part.
    let text: String = candidate
        .content
        .parts
        .into_iter()
        .filter(|p| !p.thought)
        .filter_map(|p| p.text)
        .collect();
    if text.trim().is_empty() {
        return Err("Gemini response candidate contained no text".into());
    }
    Ok(text)
}

// POST to Gemini, retrying its transient failures — rate limits (429), "high
// demand" and gateway errors (500/502/503/504), dropped connections and
// timeouts — up to MAX_ATTEMPTS times with backoff, honouring the delay the
// API suggests and cfg.time_budget. Other errors (a bad key, a rejected
// request) fail at once. The key goes in a header, not the query string:
// reqwest includes the URL in its errors, which are printed and logged.
async fn send_with_retries(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    url: &str,
    payload: &serde_json::Value,
) -> Result<reqwest::Response, Box<dyn Error>> {
    let started = std::time::Instant::now();
    let mut attempt = 1;
    loop {
        let mut request = client
            .post(url)
            .header("x-goog-api-key", &cfg.api_key)
            .json(payload);
        if let Some(budget) = cfg.time_budget {
            request = request.timeout(budget.saturating_sub(started.elapsed()));
        }
        let (reason, error, suggested_wait) = match request.send().await {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) => {
                let status = response.status();
                let header_wait = retry_after(&response);
                let body = response.text().await.unwrap_or_default();
                let error = format!("Gemini API error ({}): {}", status, body.trim());
                if !is_transient(status) {
                    return Err(error.into());
                }
                let wait = header_wait.or_else(|| retry_delay_in_body(&body));
                (format!("Gemini returned {}", status), error, wait)
            }
            Err(e) if e.is_timeout() || e.is_connect() || e.is_request() => {
                (e.to_string(), e.to_string(), None)
            }
            Err(e) => return Err(e.into()),
        };
        let wait = suggested_wait
            .unwrap_or(Duration::from_secs(1 << attempt))
            .clamp(Duration::from_secs(1), Duration::from_secs(30));
        let out_of_time = cfg
            .time_budget
            .is_some_and(|budget| started.elapsed() + wait >= budget);
        if attempt == MAX_ATTEMPTS || out_of_time {
            return Err(match attempt {
                1 => error,
                n => format!("{} (gave up after {} attempts)", error, n),
            }
            .into());
        }
        warn_!(
            "   ⏳ {} — retrying in {} s (attempt {} of {})…",
            reason,
            wait.as_secs(),
            attempt + 1,
            MAX_ATTEMPTS
        );
        sleep(wait).await;
        attempt += 1;
    }
}

// Rate limits, overload ("high demand") and gateway errors pass with time.
fn is_transient(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
}

// A `Retry-After: <seconds>` header.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    let seconds = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds))
}

// The `retryDelay` (e.g. "36s") Gemini puts in the details of a rate-limit
// error.
fn retry_delay_in_body(body: &str) -> Option<Duration> {
    let error: serde_json::Value = serde_json::from_str(body).ok()?;
    error["error"]["details"]
        .as_array()?
        .iter()
        .find_map(|detail| {
            let seconds: f64 = detail["retryDelay"]
                .as_str()?
                .strip_suffix('s')?
                .parse()
                .ok()?;
            Duration::try_from_secs_f64(seconds).ok()
        })
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
// request), so videos are not batched with photos or with each other. Returns
// how many videos couldn't be tagged.
async fn process_videos(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    extras: &ExtraTags,
    videos: &[PathBuf],
    fps: f64,
) -> usize {
    // Downscale target for sampled frames, and a hard cap so a long clip can't
    // explode into hundreds of frames (or blow past provider image limits).
    const FRAME_MAX_DIM: u32 = 1024;
    const MAX_FRAMES: usize = 15;

    let mut failures = 0;
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
                failures += 1;
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
            failures += 1;
            continue;
        }

        let prompt = build_video_prompt(frames_b64.len());
        match query_vision(client, cfg, &prompt, "Frame", &frames_b64, None).await {
            Ok(mut results) => match results.drain(..).next() {
                Some(metadata) => {
                    info!(
                        "   → title: {} | keywords: {}",
                        metadata.title,
                        metadata.keywords.len()
                    );
                    if let Err(e) = write_video_metadata(video, metadata, extras) {
                        warn_!("❌ Metadata write failed for [{}]: {}", video.display(), e);
                        failures += 1;
                    } else {
                        info!("✅ Embedded video metadata.");
                    }
                }
                None => {
                    warn_!("❌ Model returned no metadata for [{}].", video.display());
                    failures += 1;
                }
            },
            Err(e) => {
                warn_!(
                    "❌ Gemini video call failed for [{}]: {}",
                    video.display(),
                    e
                );
                failures += 1;
            }
        }

        let _ = fs::remove_dir_all(&dir);

        if idx + 1 < total {
            sleep(Duration::from_millis(cfg.rate_limit_ms)).await;
        }
    }
    failures
}

// Tag photos in batches (one API call per batch). After each batch, the grade
// rows of the photos whose tags were written go to `on_graded`, so they can be
// saved before the next batch starts. Returns how many photos couldn't be
// tagged.
async fn process_photos(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    extras: &ExtraTags,
    photos: &[PathBuf],
    mut on_graded: impl FnMut(Vec<GradeRow>),
) -> usize {
    let mut failures = 0;
    let batch_size = cfg.batch_size;
    let total_batches = photos.len().div_ceil(batch_size);
    info!(
        "📸 Tagging {} photo(s) in {} batch(es) of up to {}.",
        photos.len(),
        total_batches,
        batch_size
    );
    let schema = photo_response_schema();

    for (batch_idx, chunk) in photos.chunks(batch_size).enumerate() {
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
                    failures += 1;
                }
            }
        }

        if images.is_empty() {
            warn_!("⚠️  No readable images in this batch; skipping API call.");
            continue;
        }

        let mut rows = Vec::with_capacity(readable_paths.len());
        let prompt = build_prompt(images.len());
        match query_vision(client, cfg, &prompt, "Image", &images, Some(&schema)).await {
            Ok(results) => {
                if results.len() != readable_paths.len() {
                    warn_!(
                        "⚠️  Model returned {} result(s) for {} image(s); pairing by order.",
                        results.len(),
                        readable_paths.len()
                    );
                }

                let mut paired = 0usize;
                for (target_path, mut metadata) in readable_paths.iter().zip(results.into_iter()) {
                    paired += 1;
                    info!(
                        "   → [{}] title: {} | keywords: {}",
                        target_path.display(),
                        metadata.title,
                        metadata.keywords.len()
                    );
                    let grades = metadata.grades.take();
                    let notes = metadata.grade_notes.take().unwrap_or_default();
                    if let Err(iptc_err) =
                        write_iptc_headers(target_path.as_path(), metadata, extras)
                    {
                        warn_!(
                            "❌ IPTC write failed for [{}]: {}",
                            target_path.display(),
                            iptc_err
                        );
                        failures += 1;
                        continue;
                    }
                    info!("✅ Embedded IPTC metadata.");
                    match grades {
                        Some(g) => info!(
                            "   ⭐ overall {} · editing {} · technical {} · commercial {} · market {}",
                            fmt_grade(g.overall),
                            fmt_grade(g.editing_quality),
                            fmt_grade(g.technical_quality),
                            fmt_grade(g.commercial_cleanliness),
                            fmt_grade(g.market_demand)
                        ),
                        None => warn_!(
                            "   ⚠️  No grades returned for [{}]; it can't be picked as a best photo.",
                            target_path.display()
                        ),
                    }
                    rows.push(GradeRow::new(
                        target_path.as_path(),
                        grades.unwrap_or_default(),
                        notes,
                    ));
                }

                for unpaired in readable_paths.iter().skip(paired) {
                    warn_!(
                        "❌ No metadata returned for [{}] (model returned too few results).",
                        unpaired.display()
                    );
                    failures += 1;
                }
            }
            Err(api_err) => {
                warn_!(
                    "❌ Gemini batch call failed ({} image(s)): {}",
                    images.len(),
                    api_err
                );
                failures += images.len();
            }
        }

        if !rows.is_empty() {
            on_graded(rows);
        }

        if batch_idx + 1 < total_batches {
            sleep(Duration::from_millis(cfg.rate_limit_ms)).await;
        }
    }
    failures
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

// ---------------------------------------------------------------------------
// Grades report (stock_grades.csv) and best-pick copies (best_for_stock/)
// ---------------------------------------------------------------------------

struct GradingConfig {
    report_file: String,
    best_dir: String,
    // A photo qualifies for the best folder when overall > min_grade.
    min_grade: u8,
    // Max fingerprint distance (bits out of 64) at which two photos count as
    // near-duplicates.
    max_distance: u32,
}

fn load_grading_config() -> GradingConfig {
    GradingConfig {
        report_file: folder_entry_env("GRADES_FILE", "stock_grades.csv"),
        best_dir: folder_entry_env("BEST_DIR", "best_for_stock"),
        min_grade: env::var("BEST_MIN_GRADE")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(6),
        max_distance: env::var("SIMILARITY_MAX_DISTANCE")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(10),
    }
}

// GRADES_FILE and BEST_DIR name an entry inside the tagged folder. The report
// is keyed by bare file names, so a location shared by several folders (an
// absolute path, `..`, `~`, a nested path) would mix their rows and copies up.
fn folder_entry_env(key: &str, default: &str) -> String {
    let value = env::var(key).unwrap_or_default().trim().to_string();
    if value.is_empty() {
        return default.to_string();
    }
    if is_plain_name(&value) {
        return value;
    }
    warn_!(
        "⚠️  {}={:?} must be a plain name inside the tagged folder; using {:?}.",
        key,
        value,
        default
    );
    default.to_string()
}

fn is_plain_name(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
    ) && !value.starts_with('~')
}

// digiKam's Batch Queue Manager hands custom scripts a temporary copy named
// like `BatchTool-XXXXXX.digikamtempfile.JPG` and renames it once the script
// exits, so a report row or best copy under that name would be meaningless.
fn is_digikam_temp_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.contains(".digikamtempfile."))
}

// One photo's row in the grades report.
#[derive(Debug, Clone, PartialEq)]
struct GradeRow {
    name: String,
    grades: Grades,
    notes: String,
    // Cached near-duplicate fingerprint (None = not computed yet).
    fingerprint: Option<Fingerprint>,
    // When the photo was last copied to the best folder ("" = never).
    copied: String,
    // Values of the report columns this tool doesn't know (added by the user),
    // in `Report::extra_headers` order.
    extra: Vec<String>,
}

impl GradeRow {
    fn new(path: &Path, grades: Grades, notes: String) -> Self {
        GradeRow {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            grades,
            notes: notes.trim().to_string(),
            fingerprint: None,
            copied: String::new(),
            extra: Vec::new(),
        }
    }

    // Sum of the four sub-grades: the tie-breaker between equal overall grades.
    fn subtotal(&self) -> u16 {
        let g = &self.grades;
        [
            g.editing_quality,
            g.technical_quality,
            g.commercial_cleanliness,
            g.market_demand,
        ]
        .into_iter()
        .flatten()
        .map(u16::from)
        .sum()
    }
}

fn fmt_grade(grade: Option<u8>) -> String {
    grade.map_or_else(|| "–".to_string(), |g| g.to_string())
}

// The report lives next to the photos: in the tagged folder, or in the file's
// folder when a single photo was passed.
fn report_dir(input: &Path) -> PathBuf {
    if input.is_dir() {
        return input.to_path_buf();
    }
    input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[derive(Debug, Default, PartialEq)]
struct Report {
    // Headers of the columns this tool doesn't know, kept after its own.
    extra_headers: Vec<String>,
    rows: Vec<GradeRow>,
}

// The report's own columns, in the order they're written. `index` is the row
// number, recomputed on every write.
fn report_columns() -> impl Iterator<Item = &'static str> {
    ["name", "index"]
        .into_iter()
        .chain(GRADE_KEYS)
        .chain(["notes", "fingerprint", "copied"])
}

// Headers are matched ignoring case, surrounding spaces, and spaces or dashes
// in place of underscores, so a spreadsheet's "Overall" still counts.
fn column_key(header: &str) -> String {
    header.trim().to_lowercase().replace([' ', '-'], "_")
}

// Parse a report leniently: columns are matched by name (so they can be
// reordered or re-cased), short rows are padded, unknown columns are kept, and
// unreadable grade cells count as missing. Whatever rewriting the file would
// lose or change is returned as a problem, so the caller can back it up first.
fn parse_report(text: &str) -> (Report, Vec<String>) {
    let mut records = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(text.as_bytes())
        .into_records();
    let mut problems = Vec::new();
    let header = match records.next() {
        None => return (Report::default(), problems),
        Some(Ok(header)) => header,
        Some(Err(e)) => {
            problems.push(format!("its header is unreadable ({})", e));
            return (Report::default(), problems);
        }
    };
    let keys: Vec<String> = header.iter().map(column_key).collect();
    let column = |key: &str| keys.iter().position(|k| k == key);
    let Some(name_col) = column("name") else {
        problems.push("it has no `name` column".to_string());
        return (Report::default(), problems);
    };
    for key in GRADE_KEYS.into_iter().chain(["notes"]) {
        if column(key).is_none() {
            problems.push(format!("it has no `{}` column", key));
        }
    }
    let extra_cols: Vec<usize> = (0..keys.len())
        .filter(|&i| !report_columns().any(|c| c == keys[i]))
        .collect();
    let mut report = Report {
        extra_headers: extra_cols
            .iter()
            .map(|&i| header[i].trim().to_string())
            .collect(),
        rows: Vec::new(),
    };

    // Spreadsheet row numbers: the header is row 1.
    for (row_number, record) in (2..).zip(records) {
        let record = match record {
            Ok(record) => record,
            Err(e) => {
                problems.push(format!("row {} is unreadable ({})", row_number, e));
                continue;
            }
        };
        let cell = |col: Option<usize>| col.and_then(|i| record.get(i)).unwrap_or("").trim();
        let name = cell(Some(name_col));
        if name.is_empty() {
            if record.iter().any(|v| !v.trim().is_empty()) {
                problems.push(format!("row {} has no name", row_number));
            }
            continue;
        }
        let grades = Grades::from_fn(|key| {
            let raw = cell(column(key));
            let grade = grade_from_str(raw);
            if grade.is_none() && !raw.is_empty() {
                problems.push(format!("{}'s {} {:?} is unreadable", name, key, raw));
            }
            grade
        });
        report.rows.push(GradeRow {
            name: name.to_string(),
            grades,
            notes: cell(column("notes")).to_string(),
            fingerprint: Fingerprint::parse(cell(column("fingerprint"))),
            copied: cell(column("copied")).to_string(),
            extra: extra_cols
                .iter()
                .map(|&i| cell(Some(i)).to_string())
                .collect(),
        });
    }
    (report, problems)
}

fn render_report(report: &Report) -> csv::Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    let mut header: Vec<&str> = report_columns().collect();
    header.extend(report.extra_headers.iter().map(String::as_str));
    writer.write_record(&header)?;
    for (index, row) in (1..).zip(&report.rows) {
        let index = u32::to_string(&index);
        let grades = row
            .grades
            .values()
            .map(|g| g.map(|g| g.to_string()).unwrap_or_default());
        let fingerprint = row.fingerprint.map(|f| f.to_string()).unwrap_or_default();
        let extra =
            (0..report.extra_headers.len()).map(|i| row.extra.get(i).map_or("", String::as_str));
        writer.write_record(
            [row.name.as_str(), &index]
                .into_iter()
                .chain(grades.iter().map(String::as_str))
                .chain([row.notes.as_str(), &fingerprint, &row.copied])
                .chain(extra),
        )?;
    }
    writer
        .into_inner()
        .map_err(|e| csv::Error::from(e.into_error()))
}

// Merge this run's rows into the report, keyed by each photo's file name as
// spelled on disk (a case- or normalization-insensitive volume reaches one
// file by several spellings). Fresh grades replace the old ones, while the
// copy record and user-added columns carry over; rows whose photo is gone are
// dropped. Returns the fresh rows' names.
fn merge_rows(
    report: &mut Report,
    fresh: Vec<GradeRow>,
    on_disk: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    let mut by_name = BTreeMap::new();
    for mut row in std::mem::take(&mut report.rows) {
        if let Some(name) = on_disk(&row.name) {
            row.name.clone_from(&name);
            by_name.entry(name).or_insert(row);
        }
    }
    let mut names = Vec::with_capacity(fresh.len());
    for mut row in fresh {
        if let Some(name) = on_disk(&row.name) {
            row.name = name;
        }
        if let Some(old) = by_name.remove(&row.name) {
            row.copied = old.copied;
            row.extra = old.extra;
        }
        names.push(row.name.clone());
        by_name.insert(row.name.clone(), row);
    }
    report.rows = by_name.into_values().collect();
    names
}

// The files in a folder, to resolve a report name to its spelling on disk:
// macOS volumes are usually case- and normalization-insensitive, so
// `p1000123.jpg` and `P1000123.JPG` can name the same photo.
struct DirIndex {
    dir: PathBuf,
    names: HashSet<String>,
    #[cfg(unix)]
    by_inode: HashMap<(u64, u64), String>,
}

impl DirIndex {
    fn scan(dir: &Path) -> std::io::Result<Self> {
        let mut index = DirIndex {
            dir: dir.to_path_buf(),
            names: HashSet::new(),
            #[cfg(unix)]
            by_inode: HashMap::new(),
        };
        for entry in fs::read_dir(dir)?.flatten() {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            match fs::metadata(entry.path()) {
                Ok(meta) if meta.is_file() => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        index
                            .by_inode
                            .insert((meta.dev(), meta.ino()), name.clone());
                    }
                    index.names.insert(name);
                }
                _ => {}
            }
        }
        Ok(index)
    }

    // The on-disk spelling of `name`, or None when it isn't a file here.
    fn resolve(&self, name: &str) -> Option<String> {
        if self.names.contains(name) {
            return Some(name.to_string());
        }
        let meta = fs::metadata(self.dir.join(name))
            .ok()
            .filter(|m| m.is_file())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            self.by_inode.get(&(meta.dev(), meta.ino())).cloned()
        }
        #[cfg(not(unix))]
        {
            let _ = meta;
            let lower = name.to_lowercase();
            self.names
                .iter()
                .find(|n| n.to_lowercase() == lower)
                .cloned()
        }
    }
}

// The grades report on disk. Every change is a locked read → change → write,
// so runs that overlap in one folder don't drop each other's rows, and the
// write goes through a temp file, so an interrupted run can't truncate it.
struct ReportStore {
    dir: PathBuf,
    path: PathBuf,
    lock_path: PathBuf,
}

impl ReportStore {
    fn new(dir: &Path, file_name: &str) -> Self {
        ReportStore {
            dir: dir.to_path_buf(),
            path: dir.join(file_name),
            lock_path: dir.join(format!(".{}.lock", file_name)),
        }
    }

    // Apply `change` to the report and save it; None (after a warning) when
    // it couldn't be saved.
    fn update<T>(&mut self, change: impl FnOnce(&mut Report, &DirIndex) -> T) -> Option<T> {
        match self.try_update(change) {
            Ok(result) => Some(result),
            Err(e) => {
                warn_!(
                    "❌ Could not update grades report [{}]: {}",
                    self.path.display(),
                    e
                );
                None
            }
        }
    }

    fn try_update<T>(
        &mut self,
        change: impl FnOnce(&mut Report, &DirIndex) -> T,
    ) -> Result<T, Box<dyn Error>> {
        let _lock = self.lock()?;
        let mut report = self.load();
        let index = DirIndex::scan(&self.dir)?;
        let result = change(&mut report, &index);
        write_atomically(&self.path, &render_report(&report)?)?;
        Ok(result)
    }

    // Held until the returned file is dropped; None on volumes without file
    // locks (some network shares), where overlapping runs just aren't guarded.
    // The lock file stays in the folder: deleting it would race with a run
    // waiting on it.
    fn lock(&self) -> std::io::Result<Option<fs::File>> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&self.lock_path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(file)),
            Err(std::fs::TryLockError::WouldBlock) => {
                info!(
                    "⏳ Waiting for another run to finish updating [{}]…",
                    self.path.display()
                );
                file.lock()?;
                Ok(Some(file))
            }
            Err(std::fs::TryLockError::Error(_)) => Ok(None),
        }
    }

    // The report as it is on disk, repaired leniently (see parse_report). A
    // report that can't be read, or needs repairs and can't be backed up
    // first, is left untouched and this run's grades go to a new file instead.
    fn load(&mut self) -> Report {
        let bytes = match read_retrying(&self.path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Report::default(),
            Err(e) => return self.divert(&format!("it can't be read ({})", e)),
        };
        let text = String::from_utf8_lossy(&bytes);
        let (report, mut problems) = parse_report(&text);
        if matches!(text, std::borrow::Cow::Owned(_)) {
            problems.insert(
                0,
                "it isn't UTF-8 (re-saved in another encoding?), so some characters were replaced"
                    .to_string(),
            );
        }
        if problems.is_empty() {
            return report;
        }
        warn_!(
            "⚠️  Repairing grades report [{}]: {}.",
            self.path.display(),
            problems.join("; ")
        );
        match write_new(&timestamped_sibling(&self.path, "", ".bak"), &bytes) {
            Ok(backup) => {
                warn_!("   The original is kept as [{}].", backup.display());
                report
            }
            Err(e) => self.divert(&format!(
                "it needs repairs and couldn't be backed up ({})",
                e
            )),
        }
    }

    // Leave the report untouched and send this run's grades to a new file.
    fn divert(&mut self, reason: &str) -> Report {
        let diverted = timestamped_sibling(&self.path, "unmerged-", ".csv");
        warn_!(
            "⚠️  Leaving grades report [{}] untouched because {}; this run's grades go to [{}] instead.",
            self.path.display(),
            reason,
            diverted.display()
        );
        self.path = diverted;
        Report::default()
    }
}

// Read a whole file, retrying the transient lock errors described at
// read_image_b64.
fn read_retrying(path: &Path) -> std::io::Result<Vec<u8>> {
    const MAX_ATTEMPTS: u64 = 3;
    let mut attempt = 1;
    loop {
        match fs::read(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound && attempt < MAX_ATTEMPTS => {
                std::thread::sleep(Duration::from_millis(300 * attempt));
                attempt += 1;
            }
            result => return result,
        }
    }
}

// Write through a temp file + rename, so an interrupted write can't leave a
// truncated file behind.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_sibling(path);
    let written = fs::File::create(&tmp)
        .and_then(|mut file| {
            file.write_all(bytes)?;
            file.sync_all()
        })
        .and_then(|()| fs::rename(&tmp, path));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written
}

// Write a file that must not exist yet; returns its path.
fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(path.to_path_buf())
}

// A hidden temp name next to `path`, unique to this process so concurrent runs
// never share one.
fn temp_sibling(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(".{}.{}.tmp", name, std::process::id()))
}

// `<name>.<prefix><timestamp><suffix>` next to `path`, never an existing file.
fn timestamped_sibling(path: &Path, prefix: &str, suffix: &str) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stamp = Local::now().format("%Y%m%d-%H%M%S").to_string();
    (1..)
        .map(|n: u32| match n {
            1 => format!("{}.{}{}{}", name, prefix, stamp, suffix),
            n => format!("{}.{}{}-{}{}", name, prefix, stamp, n, suffix),
        })
        .map(|file| path.with_file_name(file))
        .find(|candidate| !candidate.exists())
        .expect("some suffix is unused")
}

// Near-duplicate fingerprint: perceptual hashes of the whole frame and of the
// subject (see fingerprint_image). Two photos are as far apart as their most
// different view, so both the framing and the subject have to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fingerprint {
    frame: u64,
    subject: u64,
}

impl Fingerprint {
    // Prefix of the stored form. Change it whenever the hashing changes, so
    // fingerprints cached in reports are recomputed instead of compared.
    const VERSION: &'static str = "p1:";

    fn distance(&self, other: &Fingerprint) -> u32 {
        (self.frame ^ other.frame)
            .count_ones()
            .max((self.subject ^ other.subject).count_ones())
    }

    fn parse(text: &str) -> Option<Self> {
        let hex = text.strip_prefix(Self::VERSION)?;
        if hex.len() != 32 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        Some(Fingerprint {
            frame: u64::from_str_radix(&hex[..16], 16).ok()?,
            subject: u64::from_str_radix(&hex[16..], 16).ok()?,
        })
    }
}

impl std::fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{:016x}{:016x}",
            Self::VERSION,
            self.frame,
            self.subject
        )
    }
}

fn fingerprint(path: &Path) -> Result<Fingerprint, Box<dyn Error>> {
    // Decode by sniffing the file's magic bytes rather than trusting its
    // extension (cameras write upper-case ".JPG", and misnamed files still
    // decode).
    let img = image::load_from_memory(&read_retrying(path)?)?;
    Ok(fingerprint_image(&img))
}

// Hash a 512 px grayscale copy: fast to scan, and averaging evens out noise.
// On a plain backdrop the subject also gets its own hash, because seen whole,
// distinct products on the same seamless background look alike: the backdrop
// dominates the frame.
fn fingerprint_image(img: &image::DynamicImage) -> Fingerprint {
    let gray = img.thumbnail(512, 512).to_luma8();
    let frame = phash(&gray, 0.0);
    // The crop is centred on the subject, so a symmetric product (a ball, a
    // bottle) cancels out many of its DCT coefficients; a small dead zone keeps
    // those near-zero ones from flipping bits between the frames of a burst.
    let subject = subject_square(&gray).map_or(frame, |square| phash(&square, 0.01));
    Fingerprint { frame, subject }
}

// 64-bit perceptual hash: shrink to 32×32, take the 8×8 lowest frequencies of
// its DCT and record which are above their median (plus `dead_zone` times the
// largest one). Bursts and small edits of one shot land within a few bits of
// each other.
fn phash(gray: &GrayImage, dead_zone: f32) -> u64 {
    const N: usize = 32;
    let small = imageops::resize(gray, N as u32, N as u32, imageops::FilterType::Triangle);
    let cos: [[f32; N]; 8] = std::array::from_fn(|k| {
        std::array::from_fn(|n| {
            (std::f32::consts::PI * (2 * n + 1) as f32 * k as f32 / (2 * N) as f32).cos()
        })
    });
    // Separable 2-D DCT-II, computing only the 8 lowest frequencies per axis.
    let rows: [[f32; 8]; N] = std::array::from_fn(|y| {
        std::array::from_fn(|u| {
            (0..N)
                .map(|x| f32::from(small.get_pixel(x as u32, y as u32)[0]) * cos[u][x])
                .sum()
        })
    });
    let coefficients: [f32; 64] =
        std::array::from_fn(|i| (0..N).map(|y| rows[y][i % 8] * cos[i / 8][y]).sum());
    // The first (DC) coefficient only encodes overall brightness.
    let mut ac = coefficients[1..].to_vec();
    ac.sort_by(f32::total_cmp);
    let largest = ac.iter().fold(0.0f32, |max, c| max.max(c.abs()));
    let threshold = ac[ac.len() / 2] + dead_zone * largest;
    coefficients
        .iter()
        .fold(0, |hash, &c| (hash << 1) | u64::from(c > threshold))
}

// Luma levels within which a pixel still counts as the plain backdrop.
const BACKDROP_TOLERANCE: u8 = 16;

// On a plain backdrop (most of the frame's edge within BACKDROP_TOLERANCE of
// its median, e.g. seamless white or black), a square window around the
// subject with a 10% margin, padded with the backdrop so the subject's shape
// and proportions survive; None for ordinary scenes.
fn subject_square(gray: &GrayImage) -> Option<GrayImage> {
    let (w, h) = gray.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let mut edge: Vec<u8> = (0..w)
        .flat_map(|x| [gray.get_pixel(x, 0)[0], gray.get_pixel(x, h - 1)[0]])
        .chain((1..h - 1).flat_map(|y| [gray.get_pixel(0, y)[0], gray.get_pixel(w - 1, y)[0]]))
        .collect();
    edge.sort_unstable();
    let backdrop = edge[edge.len() / 2];
    let is_backdrop = |luma: u8| luma.abs_diff(backdrop) <= BACKDROP_TOLERANCE;
    if edge.iter().filter(|&&luma| is_backdrop(luma)).count() * 10 < edge.len() * 9 {
        return None;
    }

    let mut cols = vec![0u32; w as usize];
    let mut rows = vec![0u32; h as usize];
    for (x, y, pixel) in gray.enumerate_pixels() {
        if !is_backdrop(pixel[0]) {
            cols[x as usize] += 1;
            rows[y as usize] += 1;
        }
    }
    // First and last column/row holding more than a speck of subject.
    let span = |counts: &[u32]| {
        let first = counts.iter().position(|&c| c >= 2)?;
        let last = counts.iter().rposition(|&c| c >= 2)?;
        Some((first as i64, (last - first + 1) as i64))
    };
    let (x, sw) = span(&cols)?;
    let (y, sh) = span(&rows)?;
    if sw < 8 || sh < 8 || (sw == i64::from(w) && sh == i64::from(h)) {
        return None;
    }

    let side = (sw.max(sh) as f32 * 1.2).ceil() as i64;
    let left = x + sw / 2 - side / 2;
    let top = y + sh / 2 - side / 2;
    Some(GrayImage::from_fn(side as u32, side as u32, |sx, sy| {
        let (gx, gy) = (left + i64::from(sx), top + i64::from(sy));
        if (0..i64::from(w)).contains(&gx) && (0..i64::from(h)).contains(&gy) {
            *gray.get_pixel(gx as u32, gy as u32)
        } else {
            image::Luma([backdrop])
        }
    }))
}

struct BestCandidate {
    name: String,
    overall: u8,
    subtotal: u16,
    // None when the photo couldn't be decoded; it is then treated as unique.
    fingerprint: Option<Fingerprint>,
}

struct Skipped {
    candidate: BestCandidate,
    duplicate_of: String,
    distance: u32,
}

// Keep the best photo of each near-duplicate group: walk candidates from best to
// worst (overall, then sub-grade total, then name) and skip any within
// `max_distance` of a photo already kept.
fn select_unique(
    mut candidates: Vec<BestCandidate>,
    max_distance: u32,
) -> (Vec<BestCandidate>, Vec<Skipped>) {
    candidates.sort_by(|a, b| {
        b.overall
            .cmp(&a.overall)
            .then(b.subtotal.cmp(&a.subtotal))
            .then_with(|| a.name.cmp(&b.name))
    });
    let mut kept: Vec<BestCandidate> = Vec::new();
    let mut skipped = Vec::new();
    for candidate in candidates {
        let duplicate = candidate.fingerprint.and_then(|fp| {
            kept.iter()
                .filter_map(|k| k.fingerprint.map(|kf| (k, fp.distance(&kf))))
                .filter(|&(_, distance)| distance <= max_distance)
                .min_by_key(|&(_, distance)| distance)
                .map(|(k, distance)| (k.name.clone(), distance))
        });
        match duplicate {
            Some((duplicate_of, distance)) => skipped.push(Skipped {
                candidate,
                duplicate_of,
                distance,
            }),
            None => kept.push(candidate),
        }
    }
    (kept, skipped)
}

// Copy via a temp name + rename: an existing copy is replaced atomically, and on
// APFS the copy stays a zero-cost clone (clonefile can't target an existing file).
fn copy_into(src: &Path, dest_dir: &Path) -> std::io::Result<()> {
    let name = src.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
    })?;
    let dest = dest_dir.join(name);
    let tmp = temp_sibling(&dest);
    let _ = fs::remove_file(&tmp);
    let copied = fs::copy(src, &tmp).and_then(|_| fs::rename(&tmp, &dest));
    if copied.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    copied
}

// What pick_best did, by photo name.
#[derive(Debug, Default)]
struct BestPicks {
    copied: Vec<String>,
    // Picks whose copy the user removed after an earlier run made it.
    not_recopied: Vec<String>,
    // Copies in the best folder of photos that are no longer selected.
    stale: Vec<String>,
}

// Pick the best photo of each near-duplicate group among every photo in the
// report with overall > min_grade, and copy the picks into the best folder —
// after tagging, so the copies carry their tags. A pick graded in this run
// (`graded_now`) gets a fresh copy; one graded earlier is only copied if it
// never was (e.g. the run that graded it was interrupted), so an existing copy
// is never replaced on the strength of an old grade. A copy the user removed
// is not re-created.
fn pick_best(
    report: &mut Report,
    dir: &Path,
    graded_now: &HashSet<String>,
    cfg: &GradingConfig,
) -> BestPicks {
    let best_dir = dir.join(&cfg.best_dir);
    let mut candidates = Vec::new();
    for row in &mut report.rows {
        let Some(overall) = row.grades.overall.filter(|&g| g > cfg.min_grade) else {
            continue;
        };
        if row.fingerprint.is_none() {
            let path = dir.join(&row.name);
            row.fingerprint = match fingerprint(&path) {
                Ok(fp) => Some(fp),
                Err(e) => {
                    warn_!(
                        "⚠️  Could not fingerprint [{}] for the duplicate check: {} — treating it as unique.",
                        path.display(),
                        e
                    );
                    None
                }
            };
        }
        candidates.push(BestCandidate {
            name: row.name.clone(),
            overall,
            subtotal: row.subtotal(),
            fingerprint: row.fingerprint,
        });
    }
    let (kept, skipped) = select_unique(candidates, cfg.max_distance);

    info!(
        "🏆 Best picks (overall > {}) → {}",
        cfg.min_grade,
        best_dir.display()
    );
    let mut picks = BestPicks::default();
    let mut to_copy = Vec::new();
    for pick in &kept {
        let Some(row) = report.rows.iter().find(|r| r.name == pick.name) else {
            continue;
        };
        let fresh = graded_now.contains(&pick.name);
        let has_copy = best_dir.join(&pick.name).exists();
        if !row.copied.is_empty() && !has_copy {
            if fresh {
                info!(
                    "   • {} (overall {}) not copied again: it was removed from {} after an earlier copy (clear its `copied` cell in the report to copy it again).",
                    pick.name,
                    pick.overall,
                    best_dir.display()
                );
                picks.not_recopied.push(pick.name.clone());
            }
        } else if fresh || (row.copied.is_empty() && !has_copy) {
            to_copy.push(pick);
        }
    }
    let can_copy = to_copy.is_empty()
        || match fs::create_dir_all(&best_dir) {
            Ok(()) => true,
            Err(e) => {
                warn_!("❌ Could not create [{}]: {}", best_dir.display(), e);
                false
            }
        };
    let copied_at = Local::now().format("%Y-%m-%d %H:%M").to_string();
    for pick in to_copy.into_iter().filter(|_| can_copy) {
        match copy_into(&dir.join(&pick.name), &best_dir) {
            Ok(()) => {
                let earlier = if graded_now.contains(&pick.name) {
                    ""
                } else {
                    ", graded earlier"
                };
                info!("   • {} (overall {}{})", pick.name, pick.overall, earlier);
                if let Some(row) = report.rows.iter_mut().find(|r| r.name == pick.name) {
                    row.copied.clone_from(&copied_at);
                }
                picks.copied.push(pick.name.clone());
            }
            Err(e) => warn_!("❌ Could not copy [{}]: {}", pick.name, e),
        }
    }
    let fresh_skips: Vec<&Skipped> = skipped
        .iter()
        .filter(|s| graded_now.contains(&s.candidate.name))
        .collect();
    for s in &fresh_skips {
        info!(
            "   ↪ {} (overall {}) skipped: near-duplicate of {} (distance {})",
            s.candidate.name, s.candidate.overall, s.duplicate_of, s.distance
        );
    }

    // Never delete from the best folder; just point out copies of this folder's
    // photos that are no longer selected.
    let kept_names: HashSet<&str> = kept.iter().map(|k| k.name.as_str()).collect();
    picks.stale = report
        .rows
        .iter()
        .map(|r| &r.name)
        .filter(|name| !kept_names.contains(name.as_str()) && best_dir.join(name).is_file())
        .cloned()
        .collect();
    if !picks.stale.is_empty() {
        info!(
            "ℹ️  No longer selected but still in {} (remove manually if needed): {}",
            best_dir.display(),
            picks.stale.join(", ")
        );
    }

    info!(
        "🏆 Copied {} photo(s) to {}; skipped {} near-duplicate(s){}.",
        picks.copied.len(),
        best_dir.display(),
        fresh_skips.len(),
        match picks.not_recopied.len() {
            0 => String::new(),
            n => format!("; {} removed earlier and not copied again", n),
        }
    );
    picks
}

// The settings this run uses, at the top of its log. `key_from_environment`:
// GEMINI_API_KEY was already set before the .env file was read (dotenvy never
// overrides a set variable).
fn log_settings(
    input: &Path,
    log_path: &Path,
    env_file: Option<&Path>,
    key_from_environment: bool,
    llm: &LlmConfig,
    grading: &GradingConfig,
    video_fps: f64,
) {
    let absolute = |p: &Path| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let dir = report_dir(input);
    let key_source = match env_file {
        Some(file) if !key_from_environment => format!("from {}", file.display()),
        _ => "from the environment".to_string(),
    };

    info!("   Input:          {}", absolute(input).display());
    if is_digikam_temp_file(input) {
        info!("   Mode:           digiKam temp file — tagged only, not added to the grades report or best picks");
    }
    info!("   Model:          {}", llm.model);
    info!(
        "   API key:        {} ({} characters, {})",
        mask_key(&llm.api_key),
        llm.api_key.chars().count(),
        key_source
    );
    info!("   Video frames:   {} fps", video_fps);
    info!(
        "   Grades report:  {}",
        absolute(&dir.join(&grading.report_file)).display()
    );
    info!(
        "   Best picks:     overall > {} → {} (near-duplicates within {} bits)",
        grading.min_grade,
        absolute(&dir.join(&grading.best_dir)).display(),
        grading.max_distance
    );
    info!(
        "   Settings from:  {}",
        env_file.map_or_else(
            || "the environment only (no .env found)".to_string(),
            |p| p.display().to_string()
        )
    );
    info!(
        "   Working dir:    {}",
        env::current_dir().map_or_else(|e| format!("unknown ({})", e), |d| d.display().to_string())
    );
    info!("   Log:            {}", absolute(log_path).display());
}

// Enough of a key to tell which one a run used, without writing the secret
// into log files that live next to the photos.
fn mask_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 8 {
        return "•".repeat(chars.len());
    }
    let head: String = chars[..4].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{}…{}", head, tail)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let key_from_environment = env::var_os("GEMINI_API_KEY").is_some();
    let env_file = dotenvy::dotenv().ok();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        warn_!("🚀 Automated Stock Photo Tagger");
        warn_!("Usage: {} <file_or_directory>", args[0]);
        std::process::exit(1);
    }
    let input_target = Path::new(&args[1]);
    let digikam = is_digikam_temp_file(input_target);

    let log_path = run_log_path(input_target);
    init_logging(&log_path, digikam);
    // A panic reaches the log too, not only the console.
    let console_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        log_line("panic", &panic.to_string());
        console_hook(panic);
    }));
    info!(
        "🚀 photo_tagger {} · {}",
        env!("CARGO_PKG_VERSION"),
        Local::now().format("%Y-%m-%d %H:%M:%S")
    );

    let mut llm_cfg = match load_llm_config() {
        Ok(c) => c,
        Err(e) => {
            warn_!("❌ {}", e);
            std::process::exit(1);
        }
    };
    // digiKam kills a custom script after 60 s and then keeps the untagged
    // copy, so each API call, retries included, has to give up well before.
    if digikam {
        llm_cfg.time_budget = Some(Duration::from_secs(50));
    }

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

    let video_fps: f64 = env::var("VIDEO_FPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&f| f > 0.0)
        .unwrap_or(2.0);

    let grading = load_grading_config();
    log_settings(
        input_target,
        &log_path,
        env_file.as_deref(),
        key_from_environment,
        &llm_cfg,
        &grading,
        video_fps,
    );

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

    info!(
        "⚙️  Found {} photo(s) and {} video(s).",
        photo_files.len(),
        video_files.len()
    );

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;

    let mut failures = 0;
    if !photo_files.is_empty() {
        let mut store = ReportStore::new(&report_dir(input_target), &grading.report_file);
        let mut graded_now = HashSet::new();
        failures += process_photos(&client, &llm_cfg, &extras, &photo_files, |mut rows| {
            rows.retain(|row| {
                let temp = is_digikam_temp_file(Path::new(&row.name));
                if temp {
                    info!(
                        "   ℹ️  {} is a digiKam temp file — not added to the grades report or best picks.",
                        row.name
                    );
                }
                !temp
            });
            // Saved after every batch, so an interrupted run keeps what it graded.
            if !rows.is_empty() {
                if let Some(names) = store
                    .update(|report, disk| merge_rows(report, rows, |name| disk.resolve(name)))
                {
                    graded_now.extend(names);
                }
            }
        })
        .await;

        // Before the videos: best picks only depend on the photos' grades.
        if !graded_now.is_empty() {
            store.update(|report, disk| pick_best(report, &disk.dir, &graded_now, &grading));
            info!("📊 Grades report: {}", store.path.display());
        }
    }

    // Only photos are graded; videos are tagged but never reported or copied.
    if !video_files.is_empty() {
        info!(
            "🎬 Tagging {} video(s) at {} fps.",
            video_files.len(),
            video_fps
        );
        failures += process_videos(&client, &llm_cfg, &extras, &video_files, video_fps).await;
    }

    if failures > 0 {
        // A non-zero exit lets callers such as the digiKam wrapper notice.
        warn_!(
            "⚠️  Done, but {} file(s) could not be tagged — see the messages above.",
            failures
        );
        std::process::exit(1);
    }
    info!("🎉 Done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sentence_case_flattens_title_case() {
        assert_eq!(
            to_sentence_case("  Golden Retriever On A Beach  "),
            "Golden retriever on a beach"
        );
        assert_eq!(
            to_sentence_case("SUNSET OVER THE SEA. WAVES CRASH! IS IT COLD? YES."),
            "Sunset over the sea. Waves crash! Is it cold? Yes."
        );
        assert_eq!(to_sentence_case(""), "");
    }

    #[test]
    fn sentence_case_ignores_decimals_times_and_ellipses() {
        assert_eq!(
            to_sentence_case("Close-up of a 4.5 inch smartphone display"),
            "Close-up of a 4.5 inch smartphone display"
        );
        assert_eq!(
            to_sentence_case("Woman pours a 1.5 liter bottle. She smiles at 3.30 pm."),
            "Woman pours a 1.5 liter bottle. She smiles at 3.30 pm."
        );
        assert_eq!(
            to_sentence_case("Portrait with an f/1.8 lens"),
            "Portrait with an f/1.8 lens"
        );
        assert_eq!(
            to_sentence_case("Version 2.0 launch event...crowd cheers"),
            "Version 2.0 launch event...crowd cheers"
        );
    }

    #[test]
    fn sentence_case_ignores_abbreviations() {
        assert_eq!(
            to_sentence_case("Shallow depth of field, e.g. a blurred park"),
            "Shallow depth of field, e.g. a blurred park"
        );
        assert_eq!(
            to_sentence_case("A flag waves in the U.S. capital"),
            "A flag waves in the u.s. capital"
        );
        assert_eq!(
            to_sentence_case("Nurse talks to Dr. Lee in a clinic"),
            "Nurse talks to dr. lee in a clinic"
        );
    }

    #[test]
    fn sentence_case_handles_digits_and_closing_quotes() {
        assert_eq!(
            to_sentence_case("4 tips for better photos"),
            "4 tips for better photos"
        );
        assert_eq!(
            to_sentence_case("Two women laugh. 3 kids play nearby."),
            "Two women laugh. 3 kids play nearby."
        );
        assert_eq!(
            to_sentence_case("A sign reads \"open.\" A man walks in."),
            "A sign reads \"open.\" A man walks in."
        );
    }

    #[test]
    fn prompt_asks_for_the_keyword_cap() {
        assert!(METADATA_RULES.contains(&format!("to {MAX_KEYWORDS} items")));
    }

    #[test]
    fn grades_accept_numbers_and_strings_and_clamp_to_scale() {
        assert_eq!(grade_from_value(&json!(7)), Some(7));
        assert_eq!(grade_from_value(&json!(7.4)), Some(7));
        assert_eq!(grade_from_value(&json!(7.6)), Some(8));
        assert_eq!(grade_from_value(&json!(" 8 ")), Some(8));
        assert_eq!(grade_from_value(&json!(0)), Some(1));
        assert_eq!(grade_from_value(&json!(14)), Some(10));
        assert_eq!(grade_from_value(&json!("n/a")), None);
        assert_eq!(grade_from_value(&json!(null)), None);
        assert_eq!(grade_from_value(&json!(true)), None);
    }

    #[test]
    fn missing_or_malformed_grades_never_fail_the_batch() {
        let batch = parse_batch(
            r#"{"results": [
                {"title": "A", "description": "B", "keywords": ["x"],
                 "grades": {"editing_quality": 6, "technical_quality": "7",
                            "commercial_cleanliness": 9.2, "market_demand": "high", "overall": 7},
                 "grade_notes": "sharp; generic subject"},
                {"title": "C", "description": "D", "keywords": [], "grades": "excellent",
                 "grade_notes": ["slight noise", "logo on mug"]},
                {"title": "E", "description": "F", "keywords": [], "grade_notes": {"noise": 1}},
                {"title": "G", "description": "H", "keywords": [], "grade_notes": 5}
            ]}"#,
        )
        .expect("batch parses despite bad grades");
        assert_eq!(
            batch[0].grades,
            Some(Grades {
                editing_quality: Some(6),
                technical_quality: Some(7),
                commercial_cleanliness: Some(9),
                market_demand: None,
                overall: Some(7),
            })
        );
        assert_eq!(
            batch[0].grade_notes.as_deref(),
            Some("sharp; generic subject")
        );
        assert_eq!(batch[1].grades, None);
        assert_eq!(
            batch[1].grade_notes.as_deref(),
            Some("slight noise; logo on mug")
        );
        assert_eq!(batch[2].grades, None);
        assert_eq!(batch[2].grade_notes, None);
        assert_eq!(batch[3].grade_notes, None);
    }

    fn row(name: &str, overall: Option<u8>) -> GradeRow {
        GradeRow::new(
            Path::new(name),
            Grades {
                overall,
                ..Grades::default()
            },
            String::new(),
        )
    }

    // A fresh, empty directory for one test.
    fn test_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("phototag_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn merge_rows_keys_by_on_disk_name_and_keeps_user_columns() {
        let mut b = row("b.jpg", Some(6));
        b.copied = "2026-09-01 10:00".to_string();
        b.extra = vec!["yes".to_string()];
        let mut report = Report {
            extra_headers: vec!["uploaded".to_string()],
            rows: vec![
                row("a.jpg", Some(5)),
                b,
                row("gone.jpg", Some(9)),
                row("A.JPG", Some(3)),
            ],
        };
        // A case-insensitive volume: every spelling resolves to the file's name on disk.
        let on_disk = |name: &str| {
            ["a.jpg", "b.jpg", "c.jpg"]
                .into_iter()
                .find(|n| n.eq_ignore_ascii_case(name))
                .map(String::from)
        };
        let fresh = merge_rows(
            &mut report,
            vec![row("c.jpg", None), row("B.JPG", Some(8))],
            on_disk,
        );
        assert_eq!(fresh, vec!["c.jpg", "b.jpg"]);
        let summary: Vec<(&str, Option<u8>, &str, &[String])> = report
            .rows
            .iter()
            .map(|r| {
                (
                    r.name.as_str(),
                    r.grades.overall,
                    r.copied.as_str(),
                    r.extra.as_slice(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("a.jpg", Some(5), "", &[][..]),
                (
                    "b.jpg",
                    Some(8),
                    "2026-09-01 10:00",
                    &["yes".to_string()][..]
                ),
                ("c.jpg", None, "", &[][..]),
            ]
        );
    }

    #[test]
    fn subtotal_sums_the_available_subgrades() {
        let r = GradeRow::new(
            Path::new("a.jpg"),
            Grades {
                editing_quality: Some(6),
                technical_quality: Some(7),
                commercial_cleanliness: None,
                market_demand: Some(5),
                overall: Some(9),
            },
            String::new(),
        );
        assert_eq!(r.subtotal(), 18);
    }

    #[test]
    fn report_round_trips_through_csv_even_with_a_bom() {
        let mut first = GradeRow::new(
            Path::new("shoot/a, b.jpg"),
            Grades {
                editing_quality: Some(6),
                overall: Some(7),
                ..Grades::default()
            },
            "shadow noise; \"logo\" on mug".to_string(),
        );
        first.fingerprint = Some(Fingerprint {
            frame: 0x0123_4567_89ab_cdef,
            subject: u64::MAX,
        });
        first.copied = "2026-09-28 16:30".to_string();
        first.extra = vec!["yes".to_string()];
        let mut second = row("c.jpg", None);
        second.extra = vec![String::new()];
        let report = Report {
            extra_headers: vec!["Uploaded to Adobe".to_string()],
            rows: vec![first, second],
        };

        let text = String::from_utf8(render_report(&report).unwrap()).unwrap();
        assert!(text.starts_with(
            "name,index,editing_quality,technical_quality,commercial_cleanliness,market_demand,overall,notes,fingerprint,copied,Uploaded to Adobe\n"
        ));
        let (parsed, problems) = parse_report(&format!("\u{feff}{}", text));
        assert!(problems.is_empty(), "{:?}", problems);
        assert_eq!(parsed, report);
    }

    #[test]
    fn report_parsing_tolerates_spreadsheet_edits() {
        // Re-cased and padded headers, reordered columns, a user-added column, a
        // row with its trailing cells trimmed, and a grade typed by hand.
        let (report, problems) = parse_report(
            " Name ,Overall,Status,Editing Quality,technical_quality,commercial_cleanliness,market_demand,notes\n\
             a.jpg,8,uploaded,6,7,8,9,sharp\n\
             b.jpg,7\n\
             c.jpg,8+,,,,,,\n",
        );
        assert_eq!(report.extra_headers, vec!["Status"]);
        let grades: Vec<(&str, Option<u8>, Option<u8>)> = report
            .rows
            .iter()
            .map(|r| (r.name.as_str(), r.grades.overall, r.grades.editing_quality))
            .collect();
        assert_eq!(
            grades,
            vec![
                ("a.jpg", Some(8), Some(6)),
                ("b.jpg", Some(7), None),
                ("c.jpg", None, None),
            ]
        );
        assert_eq!(report.rows[0].notes, "sharp");
        assert_eq!(report.rows[0].extra, vec!["uploaded"]);
        assert_eq!(report.rows[1].extra, vec![""]);
        assert_eq!(problems, vec!["c.jpg's overall \"8+\" is unreadable"]);
    }

    #[test]
    fn report_missing_its_columns_is_flagged() {
        let (report, problems) = parse_report("name;index;overall\na.jpg;1;7\n");
        assert!(report.rows.is_empty());
        assert_eq!(problems, vec!["it has no `name` column"]);

        let (report, problems) = parse_report("name,Overall grade\na.jpg,7\n");
        assert_eq!(report.extra_headers, vec!["Overall grade"]);
        assert_eq!(report.rows[0].extra, vec!["7"]);
        assert!(problems.contains(&"it has no `overall` column".to_string()));
    }

    #[test]
    fn report_store_repairs_a_non_utf8_report_and_keeps_a_backup() {
        let dir = test_dir("repair");
        let mut original = b"name,index,overall,notes\na.jpg,1,8,caf".to_vec();
        original.push(0xE9); // "é" as Excel's legacy CSV encodings write it
        original.extend(b"\nb.jpg,2,5,\n");
        fs::write(dir.join("stock_grades.csv"), &original).unwrap();

        let mut store = ReportStore::new(&dir, "stock_grades.csv");
        let names = store.update(|report, _| {
            report
                .rows
                .iter()
                .map(|r| r.name.clone())
                .collect::<Vec<_>>()
        });
        assert_eq!(names, Some(vec!["a.jpg".to_string(), "b.jpg".to_string()]));
        let rewritten = fs::read_to_string(dir.join("stock_grades.csv")).expect("now UTF-8");
        assert!(
            rewritten.contains("a.jpg,1,,,,,8,caf\u{fffd},,\n"),
            "{}",
            rewritten
        );
        let backups: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "bak"))
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(fs::read(&backups[0]).unwrap(), original);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_store_leaves_an_unreadable_report_untouched() {
        let dir = test_dir("unreadable");
        fs::create_dir(dir.join("stock_grades.csv")).unwrap(); // can't be read as a file
        fs::write(dir.join("a.jpg"), b"jpeg").unwrap();

        let mut store = ReportStore::new(&dir, "stock_grades.csv");
        let merged = store.update(|report, disk| {
            merge_rows(report, vec![row("a.jpg", Some(8))], |n| disk.resolve(n))
        });
        assert_eq!(merged, Some(vec!["a.jpg".to_string()]));
        assert!(dir.join("stock_grades.csv").is_dir());
        let diverted: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("stock_grades.csv.unmerged-"))
            .collect();
        assert_eq!(diverted.len(), 1);
        let text = fs::read_to_string(dir.join(&diverted[0])).unwrap();
        assert!(text.contains("a.jpg,1,,,,,8,,,\n"), "{}", text);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_index_resolves_only_files_in_the_folder() {
        let dir = test_dir("index");
        fs::write(dir.join("P1000123.JPG"), b"jpeg").unwrap();
        fs::create_dir(dir.join("best_for_stock")).unwrap();
        let index = DirIndex::scan(&dir).unwrap();
        assert_eq!(
            index.resolve("P1000123.JPG").as_deref(),
            Some("P1000123.JPG")
        );
        assert_eq!(index.resolve("missing.jpg"), None);
        assert_eq!(index.resolve("best_for_stock"), None);
        // Another spelling reaches the file only on a case-insensitive volume.
        if dir.join("p1000123.jpg").is_file() {
            assert_eq!(
                index.resolve("p1000123.jpg").as_deref(),
                Some("P1000123.JPG")
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_dir_is_the_folder_of_a_single_file() {
        assert_eq!(
            report_dir(Path::new("shoot/IMG_1.jpg")),
            PathBuf::from("shoot")
        );
        assert_eq!(report_dir(Path::new("IMG_1.jpg")), PathBuf::from("."));
    }

    #[test]
    fn grading_folders_must_be_plain_names() {
        for ok in [
            "best_for_stock",
            "best picks",
            "best_for_stock/",
            "grades.csv",
        ] {
            assert!(is_plain_name(ok), "{}", ok);
        }
        for bad in [
            "/Users/me/Stock",
            "../best",
            "shoot/best",
            "~/best",
            "~",
            ".",
            "..",
        ] {
            assert!(!is_plain_name(bad), "{}", bad);
        }
    }

    #[test]
    fn log_goes_next_to_the_photos_unless_log_file_says_otherwise() {
        let dir = test_dir("logpath");
        let home = std::ffi::OsStr::new("/Users/me");
        let photo = dir.join("IMG_1.jpg");
        assert_eq!(
            resolve_log_path("", &dir, Some(home)),
            dir.join("photo_tagger.log")
        );
        assert_eq!(
            resolve_log_path("", &photo, Some(home)),
            dir.join("photo_tagger.log")
        );
        assert_eq!(
            resolve_log_path("logs/run.log", &dir, Some(home)),
            dir.join("logs/run.log")
        );
        assert_eq!(
            resolve_log_path("/var/tmp/run.log", &dir, Some(home)),
            PathBuf::from("/var/tmp/run.log")
        );
        assert_eq!(
            resolve_log_path("~/Library/Logs/photo_tagger.log", &dir, Some(home)),
            PathBuf::from("/Users/me/Library/Logs/photo_tagger.log")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn api_key_is_logged_masked() {
        assert_eq!(
            mask_key("AIzaSyD-EXAMPLE-not-a-real-key-0123Xk3Q"),
            "AIza…Xk3Q"
        );
        assert_eq!(mask_key("short"), "•••••");
        assert_eq!(mask_key(""), "");
    }

    #[test]
    fn only_transient_gemini_errors_are_retried() {
        for code in [429, 500, 502, 503, 504] {
            assert!(
                is_transient(reqwest::StatusCode::from_u16(code).unwrap()),
                "{}",
                code
            );
        }
        for code in [400, 401, 403, 404] {
            assert!(
                !is_transient(reqwest::StatusCode::from_u16(code).unwrap()),
                "{}",
                code
            );
        }
    }

    #[test]
    fn retry_delay_comes_from_the_error_details() {
        let body = r#"{"error": {"code": 429, "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36.5s"}
        ]}}"#;
        assert_eq!(
            retry_delay_in_body(body),
            Some(Duration::from_millis(36_500))
        );
        assert_eq!(
            retry_delay_in_body(r#"{"error": {"details": [{"retryDelay": "-3s"}]}}"#),
            None
        );
        assert_eq!(retry_delay_in_body("Service Unavailable"), None);
    }

    #[test]
    fn digikam_temp_files_are_recognised() {
        assert!(is_digikam_temp_file(Path::new(
            "/album/BatchTool-EpEjEz-9e1c7a12.digikamtempfile.JPG"
        )));
        assert!(!is_digikam_temp_file(Path::new("/album/P1000123.JPG")));
    }

    fn pick(
        name: &str,
        overall: u8,
        subtotal: u16,
        fingerprint: Option<Fingerprint>,
    ) -> BestCandidate {
        BestCandidate {
            name: name.to_string(),
            overall,
            subtotal,
            fingerprint,
        }
    }

    fn fp(bits: u64) -> Option<Fingerprint> {
        Some(Fingerprint {
            frame: bits,
            subject: bits,
        })
    }

    #[test]
    fn select_unique_keeps_the_best_of_each_near_duplicate_group() {
        let base = 0xF0F0_F0F0_F0F0_F0F0u64;
        let (kept, skipped) = select_unique(
            vec![
                pick("burst_2.jpg", 7, 28, fp(base ^ 0b111)), // 3 bits from burst_1
                pick("burst_1.jpg", 8, 30, fp(base)),
                pick("other.jpg", 7, 25, fp(!base)), // 64 bits away
                pick("undecodable.jpg", 7, 20, None), // no fingerprint: kept
            ],
            10,
        );
        let kept: Vec<&str> = kept.iter().map(|k| k.name.as_str()).collect();
        assert_eq!(kept, vec!["burst_1.jpg", "other.jpg", "undecodable.jpg"]);
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].candidate.name, "burst_2.jpg");
        assert_eq!(skipped[0].duplicate_of, "burst_1.jpg");
        assert_eq!(skipped[0].distance, 3);
    }

    #[test]
    fn select_unique_breaks_ties_by_subgrades_then_name() {
        let same = fp(42);
        let names = |picks: &[BestCandidate]| -> Vec<String> {
            picks.iter().map(|p| p.name.clone()).collect()
        };

        let (kept, _) = select_unique(
            vec![pick("a.jpg", 7, 20, same), pick("z.jpg", 7, 24, same)],
            0,
        );
        assert_eq!(names(&kept), vec!["z.jpg"]);

        let (kept, _) = select_unique(
            vec![pick("b.jpg", 7, 20, same), pick("a.jpg", 7, 20, same)],
            0,
        );
        assert_eq!(names(&kept), vec!["a.jpg"]);
    }

    #[test]
    fn near_duplicates_need_both_views_to_match() {
        let a = Fingerprint {
            frame: 0,
            subject: 0,
        };
        let b = Fingerprint {
            frame: 0b11,
            subject: 0xFFFF,
        };
        assert_eq!(a.distance(&b), 16);
    }

    #[test]
    fn fingerprint_round_trips_through_its_stored_form() {
        let fp = Fingerprint {
            frame: 0x0123_4567_89ab_cdef,
            subject: 42,
        };
        assert_eq!(fp.to_string(), "p1:0123456789abcdef000000000000002a");
        assert_eq!(Fingerprint::parse(&fp.to_string()), Some(fp));
        assert_eq!(Fingerprint::parse("0123456789abcdef000000000000002a"), None);
        assert_eq!(Fingerprint::parse(&format!("p1:{}", "é".repeat(16))), None);
        assert_eq!(Fingerprint::parse(""), None);
    }

    type Shape = fn(f32, f32) -> bool;

    // A textured grey product lit from the left, a little off-centre on a
    // seamless white backdrop. (Perfectly clean, symmetric renders are
    // degenerate for a DCT hash: most of their coefficients are zero.)
    fn studio_shot(shape: Shape, shift: f32, light: f32) -> image::DynamicImage {
        use image::{DynamicImage, Rgb, RgbImage};
        let (w, h) = (600u32, 400u32);
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            let u = (x as f32 / w as f32 - 0.46 - shift) * 1.5;
            let v = y as f32 / h as f32 - 0.53 - shift / 2.0;
            if shape(u, v) {
                let texture = 25.0 * (17.0 * u + 5.0 * v).sin() * (11.0 * v).cos();
                let luma = (90.0 + 80.0 * (u + 0.5) + texture) * light;
                Rgb([luma.clamp(0.0, 255.0) as u8; 3])
            } else {
                Rgb([255, 255, 255])
            }
        }))
    }

    fn ball(u: f32, v: f32) -> bool {
        u * u + v * v < 0.04
    }
    fn bottle(u: f32, v: f32) -> bool {
        u.abs() < 0.07 && v.abs() < 0.35
    }
    fn slab(u: f32, v: f32) -> bool {
        u.abs() < 0.3 && v.abs() < 0.12
    }
    fn egg(u: f32, v: f32) -> bool {
        let (du, dv) = ((u + 0.25) / 0.1, v / 0.16);
        du * du + dv * dv < 1.0
    }

    #[test]
    fn fingerprint_separates_different_products_on_one_backdrop() {
        let shapes: [Shape; 4] = [ball, bottle, slab, egg];
        let shots: Vec<Fingerprint> = shapes
            .iter()
            .map(|&shape| fingerprint_image(&studio_shot(shape, 0.0, 1.0)))
            .collect();
        for i in 0..shots.len() {
            for j in i + 1..shots.len() {
                let distance = shots[i].distance(&shots[j]);
                assert!(
                    distance > 10,
                    "products {} and {} are {} apart",
                    i,
                    j,
                    distance
                );
            }
            // The next frame of a burst: nudged and a touch brighter.
            let burst = fingerprint_image(&studio_shot(shapes[i], 0.01, 1.03));
            let distance = shots[i].distance(&burst);
            assert!(
                distance <= 10,
                "burst of product {} is {} apart",
                i,
                distance
            );
        }
    }

    #[test]
    fn fingerprint_matches_near_duplicates_and_separates_different_images() {
        use image::{DynamicImage, Rgb, RgbImage};

        // Smooth random terrain, for the broad spectrum of a real photo (a few
        // sinusoids or a ramp are degenerate cases for a DCT hash).
        let scene = |mirror: bool| {
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 256) as f32
            };
            let coarse: Vec<f32> = (0..9 * 7).map(|_| next()).collect();
            let fine: Vec<f32> = (0..33 * 25).map(|_| next()).collect();
            RgbImage::from_fn(320, 240, move |x, y| {
                let x = if mirror { 319 - x } else { x };
                let sample = |grid: &[f32], width: usize, cell: f32| {
                    let (fx, fy) = (x as f32 / cell, y as f32 / cell);
                    let (x0, y0) = (fx as usize, fy as usize);
                    let at = |i: usize, j: usize| grid[j * width + i];
                    let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx.fract();
                    let bottom =
                        at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx.fract();
                    top + (bottom - top) * fy.fract()
                };
                let luma = 0.7 * sample(&coarse, 9, 40.0) + 0.3 * sample(&fine, 33, 10.0);
                Rgb([luma as u8; 3])
            })
        };
        let base = scene(false);
        let mut touched = base.clone();
        for y in 100..116 {
            for x in 150..166 {
                touched.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }

        let base_fp = fingerprint_image(&DynamicImage::ImageRgb8(base));
        let touched_fp = fingerprint_image(&DynamicImage::ImageRgb8(touched));
        let mirrored_fp = fingerprint_image(&DynamicImage::ImageRgb8(scene(true)));
        let near = base_fp.distance(&touched_fp);
        let far = base_fp.distance(&mirrored_fp);
        assert!(
            near <= 10,
            "a small local change stays a near-duplicate ({})",
            near
        );
        assert!(
            far > 10,
            "a different image is not a near-duplicate ({})",
            far
        );
    }

    #[test]
    fn fingerprint_reads_upper_case_camera_extensions() {
        use image::{ImageFormat, Rgb, RgbImage};

        let dir = test_dir("fp");
        let path = dir.join("P1000123.JPG");
        let img = RgbImage::from_fn(64, 48, |x, y| Rgb([(x * 4) as u8, (y * 5) as u8, 128]));
        img.save_with_format(&path, ImageFormat::Jpeg).unwrap();

        let result = fingerprint(&path);
        let _ = fs::remove_dir_all(&dir);
        assert!(result.is_ok(), "fingerprint failed: {:?}", result.err());
    }

    fn grading(min_grade: u8) -> GradingConfig {
        GradingConfig {
            report_file: "stock_grades.csv".to_string(),
            best_dir: "best".to_string(),
            min_grade,
            max_distance: 10,
        }
    }

    #[test]
    fn pick_best_copies_new_picks_and_leaves_existing_or_removed_copies() {
        use image::ImageFormat;

        let dir = test_dir("pick");
        let best = dir.join("best");
        fs::create_dir_all(&best).unwrap();
        let shapes: [(&str, Shape); 5] = [
            ("new.jpg", ball),
            ("old.jpg", bottle),
            ("removed.jpg", slab),
            ("earlier.jpg", egg),
            ("low.jpg", egg),
        ];
        for (name, shape) in shapes {
            studio_shot(shape, 0.0, 1.0)
                .save_with_format(dir.join(name), ImageFormat::Jpeg)
                .unwrap();
        }
        // Graded in an earlier run; its copy has since been removed.
        let mut old = row("old.jpg", Some(9));
        old.copied = "2026-09-01 10:00".to_string();
        // Graded again now, but the user removed its copy after the last run.
        let mut removed = row("removed.jpg", Some(8));
        removed.copied = "2026-09-01 10:00".to_string();
        // Graded earlier (say by an interrupted run) and never copied.
        let earlier = row("earlier.jpg", Some(7));
        // No longer qualifies, but its old copy is still there.
        fs::copy(dir.join("low.jpg"), best.join("low.jpg")).unwrap();
        let mut report = Report {
            extra_headers: Vec::new(),
            rows: vec![
                row("new.jpg", Some(8)),
                old,
                removed,
                earlier,
                row("low.jpg", Some(3)),
            ],
        };
        let graded_now: HashSet<String> = ["new.jpg", "removed.jpg", "low.jpg"]
            .into_iter()
            .map(String::from)
            .collect();

        let picks = pick_best(&mut report, &dir, &graded_now, &grading(6));
        assert_eq!(picks.copied, vec!["new.jpg", "earlier.jpg"]);
        assert_eq!(picks.not_recopied, vec!["removed.jpg"]);
        assert_eq!(picks.stale, vec!["low.jpg"]);
        assert!(best.join("new.jpg").is_file() && best.join("earlier.jpg").is_file());
        assert!(!best.join("old.jpg").exists() && !best.join("removed.jpg").exists());
        let copied: Vec<bool> = report
            .rows
            .iter()
            .map(|r| r.copied.starts_with("20"))
            .collect();
        assert_eq!(copied, vec![true, true, true, true, false]);
        assert!(report.rows[..4].iter().all(|r| r.fingerprint.is_some()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_copies_are_reported_even_when_nothing_qualifies() {
        let dir = test_dir("stale");
        fs::create_dir_all(dir.join("best")).unwrap();
        fs::write(dir.join("a.jpg"), b"jpeg").unwrap();
        fs::write(dir.join("best").join("a.jpg"), b"jpeg").unwrap();
        let mut report = Report {
            extra_headers: Vec::new(),
            rows: vec![row("a.jpg", Some(7))],
        };
        let graded_now: HashSet<String> = HashSet::from(["a.jpg".to_string()]);

        let picks = pick_best(&mut report, &dir, &graded_now, &grading(8));
        assert!(picks.copied.is_empty());
        assert_eq!(picks.stale, vec!["a.jpg"]);
        let _ = fs::remove_dir_all(&dir);
    }
}
