import io
import base64
from types import SimpleNamespace

import av
import numpy as np
import pytest


def make_video(path, frames=20):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1" if path.suffix == ".mkv" else "mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
@pytest.mark.parametrize("in_memory", [False, True])
def test_video_samples_full_duration(module, tmp_path, suffix, in_memory):
    path = tmp_path / f"video{suffix}"
    make_video(path)
    if suffix == ".mkv":
        with av.open(str(path)) as container:
            assert container.streams.video[0].frames == 0
    source = io.BytesIO(path.read_bytes()) if in_memory else str(path)
    video = SimpleNamespace(get_stream_source=lambda: source)
    images = module("media").extract_video_frames(video, 4)
    brightness = [round(np.asarray(image).mean()) for image in images]
    assert len(images) == 4
    assert brightness[0] < 5 and brightness[-1] > 180
    assert brightness[1] > 50 and brightness[2] > 110


def test_native_video_stats(module, tmp_path):
    from test_nodes import Backend
    path = tmp_path / "video.mkv"
    make_video(path)
    video = SimpleNamespace(get_stream_source=lambda: str(path))
    backend = Backend()
    result = module("chat")._run_chat(backend, prompt="test", system_prompt="", max_tokens=32, temperature=.6,
        top_p=.95, top_k=40, min_p=.05, repeat_penalty=1.05, seed=1, max_video_frames=5,
        video_transport="frames", thinking_mode="off", image=None, video_frames=None, video=video, tools_json=None)
    assert "video_frames=5" in result.stats
    assert len([part for part in backend.request["messages"][0]["content"] if part["type"] == "image_url"]) == 5


@pytest.mark.parametrize("suffix,mime", [(".mp4", "video/mp4"), (".mkv", "video/x-matroska")])
def test_native_video_preserves_container(module, tmp_path, suffix, mime):
    path = tmp_path / f"video{suffix}"
    make_video(path)
    video = SimpleNamespace(get_stream_source=lambda: io.BytesIO(path.read_bytes()))
    url = module("media").encode_video_data_url(video)
    header, payload = url.split(",", 1)
    assert header == f"data:{mime};base64"
    assert base64.b64decode(payload) == path.read_bytes()
