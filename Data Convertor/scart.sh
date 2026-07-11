#!/bin/bash

INPUT_DIR="/path/to/mseed/files"
OUTPUT_DIR="SeisComP"

for file in "$INPUT_DIR"/*.MSEED; do
    # Skip if no files match
    [ -e "$file" ] || continue

    echo "Importing $(basename "$file")..."
    scart -I "$file" "$OUTPUT_DIR"
done
