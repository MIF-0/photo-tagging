# photo-tagging

A Rust CLI that iterates over JPEG photos and uses Gemini Vision to embed an IPTC/XMP title, caption, and up to 25 keywords — optimized for stock photography uploads (Shutterstock, Adobe Stock, Pixta, Getty).

## Requirements

- Rust (stable)
- A Gemini API key
- **`exiftool`** on your `PATH` — used to write the IPTC and XMP fields back into the JPEG. The tool will fail with a clear error if it is missing.

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
GEMINI_API_KEY=your-key-here
GEMINI_RATE_LIMIT_MS=2000
# Optional — defaults to ./photo_tagger.log in the current working directory.
LOG_FILE=/path/to/photo_tagger.log

# Optional — extra fields some stock sites (e.g. Pond5) require.
# Defaults (when unset): country="United Kingdom", make="Panasonic", model="DC-S5M2X" (Lumix S5IIx).
# Country is written to IPTC, XMP-photoshop and XMP-iptcExt schemas.
DEFAULT_COUNTRY=United Kingdom
# Camera make/model are only written if the source JPEG has no EXIF Make/Model
# yet — existing real camera data is never overwritten.
DEFAULT_CAMERA_MAKE=Panasonic
DEFAULT_CAMERA_MODEL=DC-S5M2X
```

## Usage

Build the release binary, then point it at a single JPEG or a directory of JPEGs:

```sh
cargo build --release
./target/release/photo_tagger path/to/photo.jpg
./target/release/photo_tagger path/to/folder
```

The metadata is written in-place. Both IPTC Core (`ObjectName`, `Caption-Abstract`, `Keywords`) and XMP Dublin Core (`dc:Title`, `dc:Description`, `dc:Subject`) fields are populated, which covers every major stock agency's parser.

If `DEFAULT_COUNTRY` is set, it is also written to `IPTC:Country-PrimaryLocationName`, `XMP-photoshop:Country`, and `XMP-iptcExt:LocationCreated/LocationShown CountryName`. If `DEFAULT_CAMERA_MAKE` / `DEFAULT_CAMERA_MODEL` are set, they are written to `EXIF:Make` / `EXIF:Model` **only when the source file does not already have them** — genuine camera EXIF is never overwritten.

## Logs

Every line printed to the console is also written (with an ISO-8601 timestamp) to the log file at `$LOG_FILE` (default: `photo_tagger.log` in the current working directory). The file is truncated on each run, so it always reflects the most recent invocation. Useful for batch runs:

```sh
tail -f photo_tagger.log
```
