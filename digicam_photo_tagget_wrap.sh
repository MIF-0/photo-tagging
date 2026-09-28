#!/usr/bin/env bash
cp "$INPUT" "$OUTPUT"
set -e
set -a; source "/absolute/path/to/your/photo-tagging/.env"; set +a
# digiKam keeps whatever is left in $OUTPUT as the processed image: remove it
# when tagging fails, so the item is reported as failed instead of kept untagged.
/absolute/path/to/your/photo-tagging/target/release/photo_tagger "$OUTPUT" || { rm -f "$OUTPUT"; exit 1; }