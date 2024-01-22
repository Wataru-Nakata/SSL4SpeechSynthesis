#!/bin/bash

root_directory=$1

declare -A flac_files

# Find all FLAC files under the specified directory
while IFS= read -r -d '' file; do
	if [[ "$file" == *.flac ]]; then
		filename=$(basename "$file")
		echo $file $filename
	fi
done < <(find "$root_directory" -type f -name "*.flac" -print0)

# Print the keys and paths for each FLAC file
