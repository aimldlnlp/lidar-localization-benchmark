"""Video export helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from massive_lidar_benchmark.core.io import ensure_dir


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def export_mp4(frame_dir: str | Path, output_path: str | Path, fps: int) -> Path:
    frame_root = Path(frame_dir)
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_root / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def export_gif(frame_dir: str | Path, output_path: str | Path, fps: int, scale_width: int) -> Path:
    frame_root = Path(frame_dir)
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    palette_path = frame_root / "palette.png"

    palette_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_root / "frame_%05d.png"),
        "-vf",
        f"scale={scale_width}:-1:flags=lanczos,palettegen",
        str(palette_path),
    ]
    gif_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_root / "frame_%05d.png"),
        "-i",
        str(palette_path),
        "-lavfi",
        f"scale={scale_width}:-1:flags=lanczos,paletteuse",
        str(out_path),
    ]
    subprocess.run(palette_cmd, check=True, capture_output=True)
    subprocess.run(gif_cmd, check=True, capture_output=True)
    return out_path


def export_video_bundle(
    frame_dir: str | Path,
    output_root: str | Path,
    video_name: str,
    fps: int,
    gif_scale_width: int,
) -> dict[str, Path | None]:
    frame_root = Path(frame_dir)
    root = Path(output_root)
    result: dict[str, Path | None] = {
        "frame_dir": frame_root,
        "mp4": None,
        "gif": None,
    }
    if not ffmpeg_available():
        return result

    mp4_path = ensure_dir(root / "videos") / f"{video_name}.mp4"
    gif_path = ensure_dir(root / "gifs") / f"{video_name}.gif"
    result["mp4"] = export_mp4(frame_root, mp4_path, fps=fps)
    result["gif"] = export_gif(frame_root, gif_path, fps=min(fps, 12), scale_width=gif_scale_width)
    return result


def cleanup_frame_dir(frame_dir: str | Path) -> None:
    root = Path(frame_dir)
    if not root.exists():
        return
    for png_path in root.glob("*.png"):
        png_path.unlink()
