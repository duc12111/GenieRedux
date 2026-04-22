#!/bin/bash
# Import processed ROMs into stable-retro
# Usage: ./import_roms.sh <processed_rom_directory>
# Example: ./import_roms.sh ~/roms/no-intro/roms

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <processed_rom_directory>"
    echo "Example: $0 ~/roms/no-intro/roms"
    exit 1
fi

ROM_DIR="$1"

if [ ! -d "$ROM_DIR" ]; then
    echo "Error: Directory does not exist: $ROM_DIR"
    exit 1
fi

echo "=========================================="
echo "Importing ROMs into Stable-Retro"
echo "=========================================="
echo "ROM directory: $ROM_DIR"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available. Please activate the retro_datagen environment manually."
    exit 1
fi

# Activate conda environment
echo "Activating retro_datagen environment..."
# Source conda initialization if needed
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate retro_datagen || {
    echo "Error: Failed to activate retro_datagen environment"
    echo "Please ensure the environment exists: conda env list"
    exit 1
}

# Check if retro is available
if ! python -c "import retro" 2>/dev/null; then
    echo "Error: stable-retro is not installed in retro_datagen environment"
    exit 1
fi

# Count ROM files before import
rom_count_before=$(find "$ROM_DIR" -type f \( -name "*.nes" -o -name "*.smd" -o -name "*.md" -o -name "*.sfc" -o -name "*.gb" -o -name "*.gbc" -o -name "*.bin" -o -name "*.rom" \) 2>/dev/null | wc -l)
echo "Found $rom_count_before ROM files to process"
echo ""

# Import ROMs
echo "Importing ROMs (this may take a while)..."
echo "Running: python -m retro.import $ROM_DIR"
echo ""

python -m retro.import "$ROM_DIR" 2>&1 | tee /tmp/retro_import.log

import_exit_code=${PIPESTATUS[0]}

if [ $import_exit_code -ne 0 ]; then
    echo ""
    echo "Warning: Import process exited with code $import_exit_code"
    echo "Check /tmp/retro_import.log for details"
else
    echo ""
    echo "Import process completed"
fi

# Verify import by testing a sample game
echo ""
echo "=========================================="
echo "Verifying Import"
echo "=========================================="

# Get list of expected games from annotation file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANNOTATION_FILE="$SCRIPT_DIR/../annotations/RetroAct_v0.1.csv"

if [ -f "$ANNOTATION_FILE" ]; then
    # Extract platformer games (matching the config)
    test_games=(
        "3NinjasKickBack-Genesis"
        "MarioBros-Nes"
        "SonicTheHedgehog3-Genesis"
        "SuperMarioBros-Nes"
    )
    
    success_count=0
    fail_count=0
    
    for game in "${test_games[@]}"; do
        echo -n "Testing $game... "
        if python << EOF 2>/dev/null
import retro
env = retro.make('$game')
env.close()
print("OK")
EOF
        then
            echo "✓ OK"
            success_count=$((success_count + 1))
        else
            echo "✗ Failed"
            fail_count=$((fail_count + 1))
        fi
    done
    
    echo ""
    echo "Verification results:"
    echo "  Successful: $success_count"
    echo "  Failed: $fail_count"
else
    echo "Annotation file not found, skipping verification"
fi

# Count imported games by checking retro data directory
echo ""
echo "Checking imported games..."
python << 'PYTHON_EOF'
import retro
import os

data_path = os.path.join(os.path.dirname(retro.__file__), 'data', 'stable')
if os.path.exists(data_path):
    games = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    # Count games with ROM files
    games_with_roms = 0
    for game in games:
        game_path = os.path.join(data_path, game)
        rom_files = [f for f in os.listdir(game_path) 
                     if f.startswith('rom.') or f.endswith(('.md', '.smd', '.nes', '.sfc', '.gb', '.bin', '.rom'))]
        if rom_files:
            games_with_roms += 1
    
    print(f"Total game directories: {len(games)}")
    print(f"Games with ROM files: {games_with_roms}")
else:
    print("Retro data directory not found")
PYTHON_EOF

echo ""
echo "=========================================="
echo "Import Complete!"
echo "=========================================="
echo ""
echo "You can now use the imported ROMs for data generation:"
echo "  conda activate retro_datagen"
echo "  python run.py generate config=retro_act/pretrain"

