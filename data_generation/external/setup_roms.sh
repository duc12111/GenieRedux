#!/bin/bash
# Main orchestration script: Download → Process → Import ROMs
# Usage: ./setup_roms.sh [output_directory] [--skip-download] [--skip-process] [--skip-import]
# Example: ./setup_roms.sh ~/roms/no-intro/

set -e

# Default values
OUTPUT_DIR="${1:-$HOME/roms/no-intro}"
SKIP_DOWNLOAD=false
SKIP_PROCESS=false
SKIP_IMPORT=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-process)
            SKIP_PROCESS=true
            shift
            ;;
        --skip-import)
            SKIP_IMPORT=true
            shift
            ;;
        -*)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_roms.sh"
PROCESS_SCRIPT="$SCRIPT_DIR/process_no_intro_roms.sh"
IMPORT_SCRIPT="$SCRIPT_DIR/import_roms.sh"

echo "=========================================="
echo "Stable-Retro ROM Setup"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if scripts exist
for script in "$DOWNLOAD_SCRIPT" "$PROCESS_SCRIPT" "$IMPORT_SCRIPT"; do
    if [ ! -f "$script" ]; then
        echo "Error: Script not found: $script"
        exit 1
    fi
    if [ ! -x "$script" ]; then
        chmod +x "$script"
    fi
done

# Step 1: Download ROMs
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=========================================="
    echo "Step 1: Downloading ROMs from Archive.org"
    echo "=========================================="
    echo ""
    
    if "$DOWNLOAD_SCRIPT" "$OUTPUT_DIR"; then
        echo ""
        echo "✓ Download completed successfully"
    else
        echo ""
        echo "✗ Download failed or was interrupted"
        echo "  You can resume by running:"
        echo "    $0 $OUTPUT_DIR --skip-download"
        exit 1
    fi
else
    echo "Skipping download (--skip-download flag set)"
fi

# Step 2: Process ROMs
if [ "$SKIP_PROCESS" = false ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Processing ROM Collections"
    echo "=========================================="
    echo ""
    
    if "$PROCESS_SCRIPT" "$OUTPUT_DIR"; then
        echo ""
        echo "✓ Processing completed successfully"
    else
        echo ""
        echo "✗ Processing failed"
        echo "  You can retry by running:"
        echo "    $0 $OUTPUT_DIR --skip-download --skip-process"
        exit 1
    fi
else
    echo "Skipping processing (--skip-process flag set)"
fi

# Step 3: Import ROMs
if [ "$SKIP_IMPORT" = false ]; then
    PROCESSED_DIR="$OUTPUT_DIR/roms"
    
    if [ ! -d "$PROCESSED_DIR" ]; then
        echo ""
        echo "Error: Processed ROM directory not found: $PROCESSED_DIR"
        echo "  Please run processing step first"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Step 3: Importing ROMs into Stable-Retro"
    echo "=========================================="
    echo ""
    
    if "$IMPORT_SCRIPT" "$PROCESSED_DIR"; then
        echo ""
        echo "✓ Import completed successfully"
    else
        echo ""
        echo "✗ Import failed"
        echo "  Check the error messages above"
        exit 1
    fi
else
    echo "Skipping import (--skip-import flag set)"
fi

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "ROMs have been downloaded, processed, and imported."
echo ""
echo "You can now generate data:"
echo "  conda activate retro_datagen"
echo "  python run.py generate config=retro_act/pretrain"
echo ""
echo "To run individual steps:"
echo "  Download:  bash $DOWNLOAD_SCRIPT $OUTPUT_DIR"
echo "  Process:   bash $PROCESS_SCRIPT $OUTPUT_DIR"
echo "  Import:    bash $IMPORT_SCRIPT $OUTPUT_DIR/roms"

