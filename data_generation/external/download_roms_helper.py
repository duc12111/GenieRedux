#!/usr/bin/env python3
"""
Helper script to identify which ROMs are needed and provide download links.
"""
import pandas as pd
import os
from pathlib import Path

def get_required_games():
    """Get list of required games based on config."""
    annotation_path = Path(__file__).parent.parent / "annotations" / "RetroAct_v0.1.csv"
    df = pd.read_csv(annotation_path)
    df[['view', 'motion', 'genre']] = df['tags'].str.split(' ', expand=True)[[0, 1, 2]]
    
    # Filter for platformer games (matching the pretrain config)
    platformer_games = df[df['genre'] == 'pl']['game'].tolist()
    return platformer_games, df[df['genre'] == 'pl']

def get_platform_distribution(games_df):
    """Get distribution of games by platform."""
    platforms = games_df['game'].str.split('-').str[1].value_counts()
    return platforms

def get_rom_hashes():
    """Get ROM SHA hashes for required games."""
    import retro
    import os
    
    retro_path = os.path.dirname(retro.__file__)
    data_path = os.path.join(retro_path, 'data', 'stable')
    
    hashes = {}
    games, _ = get_required_games()
    
    for game in games:
        game_path = os.path.join(data_path, game)
        rom_sha_path = os.path.join(game_path, 'rom.sha')
        if os.path.exists(rom_sha_path):
            with open(rom_sha_path, 'r') as f:
                sha = f.read().strip()
                platform = game.split('-')[1]
                hashes[game] = {
                    'sha': sha,
                    'platform': platform
                }
    
    return hashes

def print_download_info():
    """Print information about required ROMs and download sources."""
    games, games_df = get_required_games()
    platforms = get_platform_distribution(games_df)
    hashes = get_rom_hashes()
    
    print("=" * 70)
    print("ROM Download Information")
    print("=" * 70)
    print(f"\nTotal games needed: {len(games)}")
    print(f"\nPlatform distribution:")
    for platform, count in platforms.items():
        print(f"  {platform:15s}: {count:3d} games")
    
    print(f"\n{'=' * 70}")
    print("Archive.org No-Intro Collection Links")
    print("=" * 70)
    print("\n1. Nintendo Entertainment System (NES)")
    print("   https://archive.org/details/No-IntroCollection_2016-01-03_Fixed")
    print("   Look for: Nintendo - Nintendo Entertainment System (NES)")
    
    print("\n2. Sega Genesis / Mega Drive")
    print("   https://archive.org/details/No-IntroCollection_2016-01-03_Fixed")
    print("   Look for: Sega - Mega Drive - Genesis")
    
    print("\n3. Super Nintendo Entertainment System (SNES)")
    print("   https://archive.org/details/No-IntroCollection_2016-01-03_Fixed")
    print("   Look for: Nintendo - Super Nintendo Entertainment System")
    
    print("\n4. Sega Master System (SMS)")
    print("   https://archive.org/details/No-IntroCollection_2016-01-03_Fixed")
    print("   Look for: Sega - Master System - Mark III")
    
    print("\n5. Game Boy")
    print("   https://archive.org/details/No-IntroCollection_2016-01-03_Fixed")
    print("   Look for: Nintendo - Game Boy")
    
    print("\n6. Atari 2600")
    print("   https://archive.org/details/No-IntroCollection_2016-01-03_Fixed")
    print("   Look for: Atari - 2600")
    
    print(f"\n{'=' * 70}")
    print("Import Instructions")
    print("=" * 70)
    print("\n1. Download the No-Intro ROM sets from Archive.org")
    print("2. Extract them to a directory (e.g., ~/roms/no-intro/)")
    print("3. Process nested ZIPs:")
    print("   bash data_generation/external/process_no_intro_roms.sh ~/roms/no-intro/")
    print("4. Import ROMs:")
    print("   conda activate retro_datagen")
    print("   python -m retro.import ~/roms/no-intro/roms")
    
    print(f"\n{'=' * 70}")
    print("Sample ROM Hashes (first 10 games)")
    print("=" * 70)
    for i, (game, info) in enumerate(list(hashes.items())[:10]):
        print(f"{game:40s} SHA: {info['sha']}")

if __name__ == "__main__":
    print_download_info()

