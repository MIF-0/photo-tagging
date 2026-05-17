#!/usr/bin/env bash
cp "$INPUT" "$OUTPUT"
set -e
set -a; source "/absolute/path/to/your/photo-tagging/.env"; set +a
/absolute/path/to/your/photo-tagging/target/release/photo_tagger "$OUTPUT"