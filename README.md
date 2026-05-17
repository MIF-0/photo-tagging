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
```

## Usage

Build the release binary, then point it at a single JPEG or a directory of JPEGs:

```sh
cargo build --release
./target/release/photo_tagger path/to/photo.jpg
./target/release/photo_tagger path/to/folder
```

The metadata is written in-place. Both IPTC Core (`ObjectName`, `Caption-Abstract`, `Keywords`) and XMP Dublin Core (`dc:Title`, `dc:Description`, `dc:Subject`) fields are populated, which covers every major stock agency's parser.
