#!/bin/bash
# Download No-Intro ROM sets from Archive.org
# Usage: ./download_roms.sh [output_directory]
# Default output: ~/roms/no-intro/

set -e

OUTPUT_DIR="${1:-$HOME/roms/no-intro}"
COLLECTION_ID="No-IntroCollection_2016-01-03_Fixed"

# Platform mappings (Archive.org folder names)
declare -A PLATFORMS=(
    ["Nes"]="Nintendo - Nintendo Entertainment System (NES)"
    ["Genesis"]="Sega - Mega Drive - Genesis"
    ["Snes"]="Nintendo - Super Nintendo Entertainment System"
    ["Sms"]="Sega - Master System - Mark III"
    ["GameBoy"]="Nintendo - Game Boy"
    ["Atari2600"]="Atari - 2600"
)

echo "=========================================="
echo "Downloading No-Intro ROMs from Archive.org"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if internetarchive tool is available
# First try to find it in the current PATH
if ! command -v ia &> /dev/null; then
    # Try to find it in conda environments
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/ia" ]; then
        # Already in a conda environment, use the ia from there
        export PATH="$CONDA_PREFIX/bin:$PATH"
    elif [ -d "$HOME/anaconda3/envs/retro_datagen/bin" ] && [ -f "$HOME/anaconda3/envs/retro_datagen/bin/ia" ]; then
        # Found in retro_datagen environment, add to PATH
        export PATH="$HOME/anaconda3/envs/retro_datagen/bin:$PATH"
    elif [ -d "$HOME/miniconda3/envs/retro_datagen/bin" ] && [ -f "$HOME/miniconda3/envs/retro_datagen/bin/ia" ]; then
        # Found in retro_datagen environment (miniconda), add to PATH
        export PATH="$HOME/miniconda3/envs/retro_datagen/bin:$PATH"
    fi
    
    # Check again after trying conda paths
    if ! command -v ia &> /dev/null; then
        echo "Error: 'ia' (internetarchive) tool is required for downloading from Archive.org"
        echo ""
        echo "Please install it:"
        echo "  conda activate retro_datagen"
        echo "  pip install internetarchive"
        echo ""
        echo "Or run this script from within the retro_datagen environment:"
        echo "  conda activate retro_datagen"
        echo "  bash data_generation/external/download_roms.sh ~/roms/no-intro/"
        echo ""
        echo "Note: wget/curl cannot download from Archive.org collections directly."
        echo "      You must use the 'ia' tool or download manually from:"
        echo "      https://archive.org/details/${COLLECTION_ID}"
        exit 1
    fi
fi

echo "Using 'ia' (internetarchive) tool..."
echo "  Found at: $(which ia)"
echo ""

# Verify the collection exists and has items
echo "Verifying collection access..."
collection_items=$(ia search "collection:${COLLECTION_ID}" --itemlist 2>/dev/null | wc -l)
if [ "$collection_items" -eq 0 ]; then
    echo "Warning: Collection '${COLLECTION_ID}' appears to be empty or doesn't exist."
    echo "  This might be a collection name, not an item identifier."
    echo "  The script will try to search for individual items instead."
    echo ""
else
    echo "  Found $collection_items items in collection"
    echo ""
fi

download_with_ia() {
    local platform_name="$1"
    local output_path="$2"
    local platform_key="$3"
    
    echo ""
    echo "Downloading: $platform_name"
    echo "  This may take a while (files can be several GB)..."
    echo "  Output: $output_path"
    
    local downloaded=false
    
    # Search for items containing the platform name in No-Intro collections
    echo "  Searching for No-Intro items for: $platform_name"
    
    # Try different search queries to find the right item
    local search_queries=(
        "collection:No-IntroCollection_2016-01-03_Fixed AND title:*${platform_key}*"
        "collection:No-IntroCollection_2016-01-03_Fixed AND title:*${platform_name}*"
        "identifier:*No-Intro*${platform_key}*"
        "identifier:*No-Intro*${platform_name}*"
        "title:No-Intro*${platform_key}*"
        "title:No-Intro*${platform_name}*"
    )
    
    local found_item=""
    for query in "${search_queries[@]}"; do
        echo "  Trying search: $query"
        found_item=$(ia search "$query" --itemlist 2>/dev/null | head -1)
        if [ -n "$found_item" ] && [ "$found_item" != "" ]; then
            echo "  ✓ Found item: $found_item"
            break
        fi
    done
    
    # If we found an item, try to download it
    if [ -n "$found_item" ] && [ "$found_item" != "" ]; then
        echo "  Downloading from item: $found_item"
        if ia download "$found_item" \
            --destdir="$output_path" \
            --no-directories \
            --retries=3 \
            --checksum 2>&1 | tee /tmp/ia_download_${platform_key}.log; then
            local file_count=$(find "$output_path" -type f 2>/dev/null | wc -l)
            if [ "$file_count" -gt 0 ]; then
                echo "  ✓ Downloaded $file_count files"
                downloaded=true
            else
                echo "  Warning: Download completed but no files found"
            fi
        fi
    fi
    
    # If that didn't work, try the collection identifier directly with glob patterns
    # (in case it's actually an item, not a collection)
    if [ "$downloaded" = false ]; then
        echo "  Trying direct download from collection identifier..."
        local glob_patterns=(
            "${platform_name}/*"
            "*${platform_name}*/*"
            "*${platform_name}*.zip"
            "${platform_key}/*"
            "*${platform_key}*/*"
        )
        
        for glob_pattern in "${glob_patterns[@]}"; do
            echo "  Trying glob pattern: $glob_pattern"
            # Use --dry-run first to check if files exist
            if ia download "$COLLECTION_ID" \
                --glob="$glob_pattern" \
                --dry-run 2>&1 | grep -q "item\|file"; then
                echo "  Pattern matches! Downloading..."
                if ia download "$COLLECTION_ID" \
                    --glob="$glob_pattern" \
                    --destdir="$output_path" \
                    --no-directories \
                    --retries=2 \
                    2>&1 | tee /tmp/ia_download_${platform_key}.log; then
                    local file_count=$(find "$output_path" -type f 2>/dev/null | wc -l)
                    if [ "$file_count" -gt 0 ]; then
                        echo "  ✓ Downloaded $file_count files"
                        downloaded=true
                        break
                    fi
                fi
            fi
        done
    fi
    
    if [ "$downloaded" = false ]; then
        echo ""
        echo "Warning: Failed to download $platform_name automatically"
        echo "  The collection/item identifier may be incorrect or the structure has changed."
        echo ""
        echo "  Manual download options:"
        echo "  1. Visit: https://archive.org/details/${COLLECTION_ID}"
        echo "  2. Search Archive.org for: No-Intro ${platform_name}"
        echo "  3. Download the ROM set manually and place in: $output_path"
        return 1
    fi
    
    echo "  ✓ Download completed for $platform_name"
    return 0
}

# Download each platform
failed_platforms=()
for platform_key in "${!PLATFORMS[@]}"; do
    platform_name="${PLATFORMS[$platform_key]}"
    platform_dir="$OUTPUT_DIR/$platform_key"
    mkdir -p "$platform_dir"
    
    if ! download_with_ia "$platform_name" "$platform_dir" "$platform_key"; then
        failed_platforms+=("$platform_name")
    fi
done

# Report results
echo ""
if [ ${#failed_platforms[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Some platforms failed to download:"
    echo "=========================================="
    for platform in "${failed_platforms[@]}"; do
        echo "  - $platform"
    done
    echo ""
    echo "You can download them manually from:"
    echo "  https://archive.org/details/${COLLECTION_ID}"
    echo ""
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo "Downloaded ROMs are in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Process nested ZIPs:"
echo "   bash data_generation/external/process_no_intro_roms.sh $OUTPUT_DIR"
echo "2. Import ROMs:"
echo "   bash data_generation/external/import_roms.sh $OUTPUT_DIR/roms"

