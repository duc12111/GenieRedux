# Downloading and Importing ROMs from Archive.org No-Intro Collection

## Overview

Stable-Retro requires ROM files to be obtained separately. The recommended source is the **No-Intro Collection on Archive.org**, which provides legally available ROM sets for research purposes.

## Step 1: Download ROMs from Archive.org

### Required Platforms (based on your config):
- **NES**: ~160 games
- **Genesis/Mega Drive**: ~150 games  
- **SNES**: ~108 games
- **Sega Master System (SMS)**: ~47 games
- **Game Boy**: ~14 games
- **Atari 2600**: ~4 games

### Archive.org No-Intro Collections:

1. **Nintendo Entertainment System (NES)**
   - Search: `No-Intro Nintendo Entertainment System`
   - URL: https://archive.org/details/No-IntroCollection_2016-01-03_Fixed

2. **Sega Genesis / Mega Drive**
   - Search: `No-Intro Sega Genesis`
   - URL: https://archive.org/details/No-IntroCollection_2016-01-03_Fixed

3. **Super Nintendo Entertainment System (SNES)**
   - Search: `No-Intro Super Nintendo`
   - URL: https://archive.org/details/No-IntroCollection_2016-01-03_Fixed

4. **Sega Master System**
   - Search: `No-Intro Sega Master System`

5. **Game Boy**
   - Search: `No-Intro Game Boy`

6. **Atari 2600**
   - Search: `No-Intro Atari 2600`

### Download Instructions:

1. Visit Archive.org and search for "No-Intro Collection"
2. Download the complete ROM sets for each platform you need
3. Extract the ZIP files to a directory (e.g., `~/roms/no-intro/`)

## Step 2: Process and Import ROMs

Once downloaded, use the provided script to import ROMs:

```bash
conda activate retro_datagen

# Process No-Intro ROMs (extracts nested ZIPs)
bash data_generation/external/process_no_intro_roms.sh /path/to/downloaded/roms

# Import ROMs into stable-retro
python -m retro.import /path/to/downloaded/roms/roms
```

## Step 3: Verify Import

Test that ROMs are imported correctly:

```bash
conda activate retro_datagen
python3 << 'EOF'
import retro
# Try to create an environment for a game
env = retro.make('3NinjasKickBack-Genesis')
print("✓ ROM imported successfully!")
env.close()
EOF
```

## Notes

- The import process matches ROMs by SHA-1 hash from `rom.sha` files
- Only ROMs matching the expected hashes will be imported
- You don't need all ROMs - only the ones matching games in your annotation file
- For testing, you can start with just a few games

