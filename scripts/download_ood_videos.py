"""
Download out-of-distribution gameplay videos from YouTube and convert them
into the dataset directory structure expected by MultiEnvironmentDataset.

Usage
-----
# Download a single video
python scripts/download_ood_videos.py \
    --urls "https://www.youtube.com/watch?v=XXXX" \
    --game-names "Celeste" \
    --output-dir data/ood_test/ood_pixelart \
    --resolution 64

# Download several videos at once
python scripts/download_ood_videos.py \
    --urls "URL1" "URL2" "URL3" \
    --game-names "Celeste" "ShovelKnight" "DeadCells" \
    --output-dir data/ood_test/ood_pixelart \
    --resolution 64 \
    --max-frames 1024 \
    --frame-skip 4

# Search YouTube instead of providing URLs
python scripts/download_ood_videos.py \
    --search "Celeste gameplay no commentary" "Shovel Knight gameplay" \
    --game-names "Celeste" "ShovelKnight" \
    --output-dir data/ood_test/ood_pixelart \
    --resolution 64
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_and_resize(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Pad frame to square aspect ratio, then resize to (target_size, target_size).

    This mirrors the TransformsGenerator logic in data/data.py.
    """
    h, w = frame.shape[:2]
    if h != w:
        side = max(h, w)
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        padded[y_off : y_off + h, x_off : x_off + w] = frame
        frame = padded
    resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def _transcode_to_h264(src: Path, dst: Path) -> None:
    """Re-encode a video to H.264 using ffmpeg so OpenCV can read it."""
    log.info(f"Transcoding to H.264: {src.name} -> {dst.name}")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-an",  # drop audio, we only need video
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ffmpeg transcode failed: {result.stderr[-500:]}")
        raise RuntimeError("ffmpeg transcode failed")


def download_video(url: str, output_path: Path) -> Path:
    """Download a video from YouTube using yt-dlp. Returns path to H.264 mp4."""
    log.info(f"Downloading {url} ...")
    # Prefer h264 streams; fall back to anything available
    cmd = [
        sys.executable, "-m", "yt_dlp",
        url,
        "-f", "bestvideo[height<=480][vcodec^=avc1]+bestaudio/bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path / "%(title)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"yt-dlp failed: {result.stderr}")
        raise RuntimeError(f"yt-dlp failed for {url}")

    # Find the downloaded file
    all_files = sorted(
        [f for f in output_path.iterdir() if f.is_file() and f.suffix in (".mp4", ".mkv", ".webm")],
        key=lambda p: p.stat().st_mtime,
    )
    if not all_files:
        raise FileNotFoundError(f"No downloaded file found in {output_path}")
    raw_path = all_files[-1]

    # Check if OpenCV can open it; if not, transcode with ffmpeg
    cap = cv2.VideoCapture(str(raw_path))
    ok, _ = cap.read()
    cap.release()
    if ok:
        return raw_path

    log.warning("OpenCV cannot decode the video directly (likely AV1). Transcoding via ffmpeg...")
    h264_path = raw_path.with_stem(raw_path.stem + "_h264").with_suffix(".mp4")
    _transcode_to_h264(raw_path, h264_path)
    return h264_path


def search_and_download(query: str, output_path: Path) -> Path:
    """Search YouTube for query, download the first result."""
    log.info(f"Searching YouTube for: {query}")
    search_url = f"ytsearch1:{query}"
    return download_video(search_url, output_path)


def extract_frames(
    video_path: Path,
    target_size: int,
    frame_skip: int,
    max_frames: int,
    start_sec: float = 0.0,
) -> list[np.ndarray]:
    """Extract frames from video, pad to square, resize, and return as list of BGR arrays."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Video: {video_path.name}  |  {total} frames @ {fps:.1f} fps")

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:
            # OpenCV reads BGR, convert to RGB for saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = pad_and_resize(frame_rgb, target_size)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    log.info(f"Extracted {len(frames)} frames (skip={frame_skip})")
    return frames


def write_dataset_env(
    frames: list[np.ndarray],
    env_dir: Path,
    game_name: str,
    session_length: int,
    n_actions: int = 5,
):
    """Write frames into the MultiEnvironmentDataset directory structure.

    Layout:
        env_dir/
          info.json
          000000/            (instance)
            000000/          (session 0)
              frames/
                000000.jpg
                000001.jpg
                ...
              actions.json
            000001/          (session 1)
              ...
    """
    # info.json
    info = {
        "action_space": [n_actions],
        "version": "1.1.1",
        "name": game_name,
        "generator_config": {
            "output_mode": "frame",
        },
    }
    env_dir.mkdir(parents=True, exist_ok=True)
    with open(env_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Split frames into sessions
    n_sessions = max(1, len(frames) // session_length)
    instance_id = 0
    instance_dir = env_dir / f"{instance_id:06d}"

    for sess_idx in range(n_sessions):
        start = sess_idx * session_length
        end = min(start + session_length, len(frames))
        session_frames = frames[start:end]
        if len(session_frames) < 2:
            continue

        sess_dir = instance_dir / f"{sess_idx:06d}"
        frames_dir = sess_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Write frames as JPEG
        for i, frame in enumerate(session_frames):
            fpath = frames_dir / f"{i:06d}.jpg"
            # Convert RGB back to BGR for cv2.imwrite
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write actions.json with dummy zero-actions
        actions_list = []
        for i in range(len(session_frames) - 1):
            actions_list.append(
                {
                    "src_frame_id": i,
                    "tgt_frame_id": i + 1,
                    "action": [0] * n_actions,
                }
            )
        with open(sess_dir / "actions.json", "w") as f:
            json.dump({"actions": actions_list}, f)

        log.info(f"  Session {sess_idx:03d}: {len(session_frames)} frames -> {frames_dir}")

    # Handle remaining frames as an extra session
    remaining_start = n_sessions * session_length
    if remaining_start < len(frames) and len(frames) - remaining_start >= 2:
        session_frames = frames[remaining_start:]
        sess_dir = instance_dir / f"{n_sessions:06d}"
        frames_dir = sess_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(session_frames):
            fpath = frames_dir / f"{i:06d}.jpg"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        actions_list = []
        for i in range(len(session_frames) - 1):
            actions_list.append(
                {
                    "src_frame_id": i,
                    "tgt_frame_id": i + 1,
                    "action": [0] * n_actions,
                }
            )
        with open(sess_dir / "actions.json", "w") as f:
            json.dump({"actions": actions_list}, f)
        log.info(f"  Session {n_sessions:03d} (remainder): {len(session_frames)} frames")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download OOD gameplay videos and create dataset for tokenizer testing"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", nargs="+", help="YouTube video URLs")
    group.add_argument("--search", nargs="+", help="YouTube search queries (downloads first result each)")

    parser.add_argument("--game-names", nargs="+", required=True,
                        help="Game name for each URL/search (used as env directory name)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output dataset directory (e.g. data/ood_test/ood_pixelart)")
    parser.add_argument("--resolution", type=int, default=64, choices=[64, 256],
                        help="Target resolution (64 or 256)")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Take every N-th frame (default: 4, matching training frameskip)")
    parser.add_argument("--max-frames", type=int, default=1024,
                        help="Max frames to extract per video")
    parser.add_argument("--session-length", type=int, default=256,
                        help="Frames per session (default: 256)")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Start extracting from this timestamp in seconds")
    args = parser.parse_args()

    sources = args.urls if args.urls else args.search
    assert len(sources) == len(args.game_names), (
        f"Number of URLs/searches ({len(sources)}) must match game names ({len(args.game_names)})"
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for source, game_name in zip(sources, args.game_names):
        log.info(f"\n{'='*60}")
        log.info(f"Processing: {game_name}")
        log.info(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if args.urls:
                video_path = download_video(source, tmpdir)
            else:
                video_path = search_and_download(source, tmpdir)

            frames = extract_frames(
                video_path,
                target_size=args.resolution,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                start_sec=args.start_sec,
            )

            if len(frames) < 2:
                log.warning(f"Skipping {game_name}: only {len(frames)} frames extracted")
                continue

            # Env dir name follows the convention: {index}_{GameName}
            existing = sorted(output_root.glob("*"))
            idx = len(existing)
            env_name = f"{idx:03d}_{game_name}"
            env_dir = output_root / env_name

            write_dataset_env(
                frames,
                env_dir=env_dir,
                game_name=game_name,
                session_length=args.session_length,
                n_actions=5,
            )

    log.info(f"\nDataset written to: {output_root}")
    log.info("Run tokenizer evaluation with:")
    log.info(
        f"  python eval_genie_redux.py config=tokenizer "
        f"tokenizer_fpath=<checkpoint> "
        f"eval.dataset_root_dpath={output_root.parent} "
        f"eval.dataset_name={output_root.name}"
    )


if __name__ == "__main__":
    main()
