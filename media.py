"""Media conversion helpers shared by local and API backends."""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE frame (HWC or BHWC) into an RGB PIL image."""
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"IMAGE must be HWC or BHWC, got shape {tuple(frame.shape)}")
    array = (frame.detach().float().clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"IMAGE needs 3 RGB channels, got shape {array.shape}")
    return Image.fromarray(array, mode="RGB")


def image_batch_frames(batch: torch.Tensor | None) -> list[torch.Tensor]:
    """Return every frame from a ComfyUI IMAGE batch without resizing."""
    if batch is None:
        return []
    if batch.ndim == 3:
        return [batch]
    if batch.ndim != 4:
        raise ValueError(f"Expected IMAGE batch in BHWC format, got {tuple(batch.shape)}")
    return [batch[i] for i in range(int(batch.shape[0]))]


def sample_image_batch(batch: torch.Tensor | None, count: int) -> list[torch.Tensor]:
    """Evenly sample a ComfyUI IMAGE batch without materializing a copy."""
    if batch is None:
        return []
    if batch.ndim == 3:
        return [batch]
    if batch.ndim != 4:
        raise ValueError(f"Expected IMAGE batch in BHWC format, got {tuple(batch.shape)}")
    total = int(batch.shape[0])
    count = max(1, int(count))
    if total <= count:
        return [batch[i] for i in range(total)]
    indices = np.linspace(0, total - 1, count, dtype=int)
    return [batch[int(i)] for i in indices]


def pil_to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def get_video_file_path(video: Any) -> str | None:
    """Best-effort path lookup for ComfyUI's VIDEO input object."""
    private_path = getattr(video, "_VideoFromFile__file", None)
    if isinstance(private_path, (str, os.PathLike)) and os.path.exists(private_path):
        return os.fspath(private_path)

    get_source = getattr(video, "get_stream_source", None)
    if callable(get_source):
        try:
            source = get_source()
            if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
                return os.fspath(source)
        except Exception:
            pass

    for attr in ("path", "file"):
        path = getattr(video, attr, None)
        if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
            return os.fspath(path)
    return None


def _video_source(video: Any) -> str | io.BytesIO | None:
    path = get_video_file_path(video)
    if path:
        return path
    get_source = getattr(video, "get_stream_source", None)
    if callable(get_source):
        try:
            source = get_source()
            if isinstance(source, (str, os.PathLike, io.BytesIO)):
                if isinstance(source, io.BytesIO):
                    source.seek(0)
                return source
        except Exception:
            pass
    return None


def encode_video_bytes(video: Any, max_bytes: int = 256 * 1024 * 1024) -> bytes:
    """Read a VIDEO object without transcoding for an API request."""
    source = _video_source(video)
    if isinstance(source, str):
        size = os.path.getsize(source)
        if size > max_bytes:
            raise ValueError(
                f"Video is {size / 1024 / 1024:.1f} MiB; API upload limit is "
                f"{max_bytes / 1024 / 1024:.0f} MiB"
            )
        return Path(source).read_bytes()
    if isinstance(source, io.BytesIO):
        data = source.getvalue()
        if len(data) > max_bytes:
            raise ValueError("VIDEO input exceeds the API upload limit")
        return data

    save_to = getattr(video, "save_to", None)
    if not callable(save_to):
        raise ValueError(f"Unable to read video data from object type: {type(video)}")

    fd, temp_path = tempfile.mkstemp(prefix="comfy_mllm_", suffix=".mp4")
    os.close(fd)
    try:
        save_to(temp_path)
        size = os.path.getsize(temp_path)
        if size > max_bytes:
            raise ValueError("VIDEO input exceeds the API upload limit")
        return Path(temp_path).read_bytes()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def encode_video_data_url(video: Any, max_bytes: int = 256 * 1024 * 1024) -> str:
    data = encode_video_bytes(video, max_bytes=max_bytes)
    # Keep the original container: a Matroska/WebM file is not MP4 merely
    # because the API accepts a video_url. Unknown formats fall back to frames
    # in auto mode, or give an explicit error in video_url mode.
    import av

    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.video:
            raise ValueError("VIDEO input has no video stream")
        formats = set(container.format.name.split(","))
    if "mp4" in formats:
        mime = "video/quicktime" if data[8:12] == b"qt  " else "video/mp4"
    elif "matroska" in formats:
        mime = "video/webm" if b"\x42\x82\x84webm" in data[:4096] else "video/x-matroska"
    elif "avi" in formats:
        mime = "video/x-msvideo"
    elif "mpegts" in formats:
        mime = "video/mp2t"
    else:
        raise ValueError("Native API video format is unsupported; select frames transport")
    payload = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{payload}"


def extract_video_frames(video: Any, max_frames: int) -> list[Image.Image]:
    """Sample the full video, retaining at most max_frames decoded images.

    Containers such as Matroska may not expose a frame count. Count them in
    a first pass, then reopen for exact sampling instead of taking the intro.
    """
    source = _video_source(video)
    if source is None:
        raise ValueError("Cannot locate a readable source for the VIDEO input")

    try:
        import av
    except Exception as exc:  # pragma: no cover - ComfyUI ships PyAV
        raise RuntimeError("PyAV is required to convert VIDEO input to frames") from exc

    def open_video():
        if isinstance(source, io.BytesIO):
            source.seek(0)
        return av.open(source, mode="r")

    def video_stream(container):
        if not container.streams.video:
            raise ValueError("VIDEO input has no video stream")
        return container.streams.video[0]

    with open_video() as container:
        stream = video_stream(container)
        total = int(stream.frames or 0)
        if total <= 0:
            total = sum(1 for _ in container.decode(stream))
    if total <= 0:
        raise ValueError("VIDEO input contains no decodable frames")

    limit = max(1, int(max_frames))
    wanted = set(int(x) for x in np.linspace(0, total - 1, min(total, limit), dtype=int))
    with open_video() as container:
        stream = video_stream(container)
        frames: list[Image.Image] = []
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            frames.append(frame.to_image().convert("RGB"))
            if len(frames) >= len(wanted):
                break
        if not frames:
            raise ValueError("VIDEO input contains no decodable frames")

    return frames
