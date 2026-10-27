from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FFMPEG = Path(r"D:\github_cangku\lubo\AutoSlides\resources\ffmpeg-static\ffmpeg.exe")

VIDEOS_DIR = ROOT / "static" / "videos"
BACKGROUND_DIR = ROOT / "static" / "background-videos"
POSTERS_DIR = ROOT / "static" / "video-posters"
BACKGROUND_PREVIEWS_DIR = ROOT / "static" / "background-previews"

HERO_BACKGROUND_FILES = [
    "portrait-source-01-h264.mp4",
    "video4-h256.mp4",
    "portrait-source-02-h264.mp4",
    "square-source-02-h264.mp4",
    "portrait-source-03-h264.mp4",
    "square-source-03-h264.mp4",
    "reference-h264.mp4",
    "video3-h264.mp4",
]


def find_ffmpeg() -> str:
    env_path = os.environ.get("FFMPEG_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        DEFAULT_FFMPEG,
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "FFmpeg was not found. Set FFMPEG_PATH or install ffmpeg before generating video assets."
    )


def run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_dirs() -> None:
    for directory in (POSTERS_DIR, BACKGROUND_PREVIEWS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def source_videos() -> list[Path]:
    return sorted(VIDEOS_DIR.glob("*.mp4"))


def background_sources() -> list[Path]:
    return [BACKGROUND_DIR / name for name in HERO_BACKGROUND_FILES if (BACKGROUND_DIR / name).exists()]


def create_poster(ffmpeg: str, source: Path, output: Path) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-ss",
            "0.3",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(960,iw)':-2",
            "-compression_level",
            "5",
            "-quality",
            "75",
            str(output),
        ],
    )


def create_background_preview(ffmpeg: str, source: Path, output: Path) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-i",
            str(source),
            "-an",
            "-vf",
            "scale='min(640,iw)':-2,fps=24",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
    )


def main() -> int:
    ensure_dirs()
    ffmpeg = find_ffmpeg()
    demo_sources = source_videos()
    hero_sources = background_sources()

    if not demo_sources:
        raise RuntimeError(f"No demo videos found in {VIDEOS_DIR}")

    if not hero_sources:
        raise RuntimeError(f"No background videos found in {BACKGROUND_DIR}")

    print(f"Using FFmpeg: {ffmpeg}")

    for source in demo_sources:
        stem = source.stem
        print(f"Demo poster: {source.name}")
        create_poster(ffmpeg, source, POSTERS_DIR / f"{stem}.webp")

    for source in hero_sources:
        stem = source.stem
        print(f"Background poster: {source.name}")
        create_poster(ffmpeg, source, POSTERS_DIR / f"{stem}.webp")
        print(f"Background preview: {source.name}")
        create_background_preview(ffmpeg, source, BACKGROUND_PREVIEWS_DIR / f"{stem}.mp4")

    print("Video posters and background previews generated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"FFmpeg failed with exit code {error.returncode}", file=sys.stderr)
        raise SystemExit(error.returncode)
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
