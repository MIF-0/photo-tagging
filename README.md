# photo-tagging

A Rust CLI that iterates over JPEG photos and `.mov` videos and uses a Google Gemini vision model to embed a title, caption, and up to 25 keywords — optimized for stock uploads (Shutterstock, Adobe Stock, Pixta, Getty, Pond5). Photos get IPTC + XMP tags; videos get XMP + QuickTime tags (frames are sampled with `ffmpeg` and analyzed as one clip). Photos are also graded for stock potential: the grades go to `stock_grades.csv`, and the best unique photos are copied to `best_for_stock/`.

## Requirements

- Rust (stable)
- A Gemini API key
- **`exiftool`** on your `PATH` — used to write the metadata back into the file. The tool will fail with a clear error if it is missing.
- **`ffmpeg`** on your `PATH` — only required for tagging `.mov` videos (used to sample frames). Install with `brew install ffmpeg`. Not needed if you only tag photos.

### Installing exiftool on macOS

The easiest way is via [Homebrew](https://brew.sh):

```sh
brew install exiftool
```

Verify the installation:

```sh
exiftool -ver
```

## Configuration

Create a `.env` file in the project root:

```
# --- Gemini ---
GEMINI_API_KEY=your-gemini-key
GEMINI_RATE_LIMIT_MS=2000
# Optional — Gemini model name. Defaults to "gemini-3.8-flash".
GEMINI_MODEL=gemini-3.8-flash

# Optional — defaults to photo_tagger.log in the tagged folder (next to stock_grades.csv).
# A relative path is taken inside the tagged folder; use an absolute path (or ~/…) for one shared log.
LOG_FILE=/path/to/photo_tagger.log

# Optional — frames per second sampled from .mov videos (requires ffmpeg).
# Defaults to 2. The total number of frames per clip is capped internally.
VIDEO_FPS=2

# Optional — photo grading & best picks (see "Grades & best picks" below).
# Photos with overall grade > BEST_MIN_GRADE are copied to BEST_DIR.
# GRADES_FILE and BEST_DIR are names inside the tagged folder, not paths.
GRADES_FILE=stock_grades.csv
BEST_DIR=best_for_stock
BEST_MIN_GRADE=6
# Max fingerprint distance (bits out of 64) for two photos to count as near-duplicates.
SIMILARITY_MAX_DISTANCE=10

# Optional — extra fields some stock sites (e.g. Pond5) require.
# Defaults (when unset): country="United Kingdom", make="Panasonic", model="DC-S5M2X" (Lumix S5IIx).
# Country is written to IPTC, XMP-photoshop and XMP-iptcExt schemas.
DEFAULT_COUNTRY=United Kingdom
# Camera make/model are only written if the source file has no camera identity
# yet — existing real camera data (photo EXIF, or a video's QuickTime tags such
# as an iPhone's Apple / iPhone model) is never overwritten.
DEFAULT_CAMERA_MAKE=Panasonic
DEFAULT_CAMERA_MODEL=DC-S5M2X
```

## Supported Gemini models

Any model exposed by the Gemini `generateContent` REST endpoint will work — set the model id with `GEMINI_MODEL`. Common choices, roughly cheapest → most capable:

| Model id                    | Notes                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `gemini-3.5-flash-lite`     | Latest low-cost, high-volume tier. Fast and cheap; fine for stock keywording.          |
| `gemini-2.5-flash-lite`     | Previous-generation lite tier. Still available if you prefer it.                      |
| `gemini-2.5-flash`          | Previous-generation flash. A good fallback when the 3.x models return 503 "high demand". |
| `gemini-3.5-flash`          | Better captions, keyword precision and grading judgment than the lite tier, at higher cost/latency. |
| `gemini-3.8-flash`          | **Default.** Newest full-flash model; highest quality of the flash line, at more cost/latency. |

Transient errors — `429` rate limits and `500`/`502`/`503`/`504`, such as `503 UNAVAILABLE` ("model is currently experiencing high demand"), as well as timeouts and dropped connections — are retried up to 3 times with backoff (2, 4 and 8 s, or the delay Gemini asks for) before a batch counts as failed; each retry is logged. If a `429 RESOURCE_EXHAUSTED` persists, you've usually hit the **per-day** free-tier cap on the chosen model — switch to a lighter model (e.g. `gemini-2.5-flash-lite`) or enable billing on the Google AI Studio project. A `503` that persists is a capacity problem on Google's side — try again later or switch models.

## Usage

Build the release binary, then point it at a single file or a directory (JPEGs and/or `.mov` videos):

```sh
cargo build --release
./target/release/photo_tagger path/to/photo.jpg
./target/release/photo_tagger path/to/clip.mov
./target/release/photo_tagger path/to/folder   # mixed photos + videos
```

The metadata is written in-place. Both IPTC Core (`ObjectName`, `Caption-Abstract`, `Keywords`) and XMP Dublin Core (`dc:Title`, `dc:Description`, `dc:Subject`) fields are populated, which covers every major stock agency's parser.

If `DEFAULT_COUNTRY` is set, it is also written to `IPTC:Country-PrimaryLocationName`, `XMP-photoshop:Country`, and `XMP-iptcExt:LocationCreated/LocationShown CountryName`. If `DEFAULT_CAMERA_MAKE` / `DEFAULT_CAMERA_MODEL` are set, they are written to `EXIF:Make` / `EXIF:Model` **only when the source file does not already have them** — genuine camera EXIF is never overwritten.

### Videos (`.mov`)

Each `.mov` is sampled into stills with `ffmpeg` (`VIDEO_FPS` frames/second, default 2, capped internally) and the frames are analyzed together as a single clip, yielding one title, description, and keyword set. Because QuickTime has no IPTC/EXIF, the metadata is written as **XMP Dublin Core** (`dc:Title`, `dc:Description`, `dc:Subject`) plus **QuickTime `Keys`** tags (`Title`, `Description`, `Keywords`) — read by Adobe apps, Finder, and QuickTime Player. Country and camera make/model are written the same way, and an existing camera identity (e.g. an iPhone's `Apple` / `iPhone` tags) is preserved rather than overwritten. Each clip is its own API call, so `.mov` files are not batched with photos.

### Grades & best picks (photos)

The same Gemini call that tags each photo also grades it from 1 to 10 against stock-agency standards:

| Column | What it measures |
| --- | --- |
| `editing_quality` | Post-processing craft: composition and crop, straight horizons, retouching, tasteful grading; penalizes over-processing. |
| `technical_quality` | Light and exposure, noise, colour correction / white balance, focus and sharpness. |
| `commercial_cleanliness` | How safe the image is for commercial licensing (10 = no recognizable people, logos, readable text, private property or artwork). |
| `market_demand` | How much buyers need the subject right now, versus how saturated it already is. |
| `overall` | Expected sellability; a serious flaw or a commercial blocker keeps it low. |

After every batch, the grades are saved to `stock_grades.csv` in the tagged folder (or the photo's own folder when a single file is passed), so an interrupted run keeps what it graded. Besides the grades, each row has `name`, `index`, a short `notes` line explaining the grades, the photo's near-duplicate `fingerprint`, and when it was last `copied` to the best folder. The report is updated by file name, so re-running on part of a folder keeps the other rows; rows for deleted photos are dropped. You can edit it in a spreadsheet: columns are matched by name, and columns you add are kept. If the file can't be read cleanly (say, it was re-saved in another encoding), the original is kept as `stock_grades.csv.<timestamp>.bak` before it's repaired; if it can't be read at all, it's left alone and the run's grades go to `stock_grades.csv.unmerged-<timestamp>.csv`.

Once the photos are tagged (before any videos), every photo with `overall > BEST_MIN_GRADE` (default 6, so 7 and up) is **copied** — never moved — into `best_for_stock/`. Among near-duplicates (bursts, tiny variations) only the best-graded photo is copied, since agencies reject similar submissions; photos graded in earlier runs take part in that choice. Near-duplicates are found locally with perceptual fingerprints of the whole frame and, on a plain backdrop such as seamless white, of the subject itself, so different products shot on the same backdrop aren't mistaken for each other; tune `SIMILARITY_MAX_DISTANCE` if it groups too much or too little.

A photo's copy is refreshed whenever the photo is tagged again. Copies of photos you didn't re-tag are left as they are, and a copy you delete is not re-created (clear the photo's `copied` cell in the report to have it copied again). Nothing is ever deleted from `best_for_stock/`: copies that are no longer selected are listed in the log. On APFS the copies are instant clones that take no extra disk space. `GRADES_FILE` and `BEST_DIR` must be plain names inside the tagged folder, since the report is keyed by file name. Videos are tagged but not graded.

Treat the grades as a screen, not final quality control: Gemini sees a downscaled image, so fine noise or slight misfocus at 100% can slip through; `market_demand` is the model's estimate, not live sales data; and `commercial_cleanliness` can't know whether you hold a model or property release. Check how the grades spread out on your first real folder, and raise `BEST_MIN_GRADE` if too many photos qualify.

### digiKam

`digicam_photo_tagget_wrap.sh` is a template for digiKam's Batch Queue Manager *Custom Script* tool (replace the placeholder paths). digiKam hands the script a temporary copy of each image and renames it afterwards, so these runs tag the photo but don't add it to the grades report or the best picks — run the tool on the album folder for those. If tagging fails, the wrapper removes the temporary output, so digiKam reports the item as failed instead of keeping an untagged copy. digiKam runs the script once per queued image and kills any single run that takes longer than 60 seconds (a limit built into digiKam; the queue as a whole can run as long as it needs). So when the tool is given a digiKam temp file, it keeps that image's Gemini call, retries included, under 50 seconds. Command-line runs on a folder have no such limit; each request just has a 120-second timeout. digiKam only keeps a script's output in its debug log, so look at `photo_tagger.log` in the album folder instead (see below).

## Logs

Every line printed to the console is also written (with an ISO-8601 timestamp) to `photo_tagger.log` in the tagged folder — the same folder as `stock_grades.csv`, whatever the working directory — or to `$LOG_FILE` if set. Each run starts a fresh log that opens with the settings it uses: input, model, the API key (masked to its first and last 4 characters, with its length and whether it came from `.env` or the environment), video frame rate, grades report, best-pick folder and threshold, which `.env` was loaded, working directory and log path. digiKam runs one process per image, so its runs add to a log written in the last 10 minutes instead, and one queue run ends up in one log. The tool exits with status 1 if any file couldn't be tagged. Useful for batch runs:

```sh
tail -f path/to/folder/photo_tagger.log
```
