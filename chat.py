"""Shared multimodal request assembly and response handling for TankNodes."""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from .backends import BackendError
from .media import (encode_video_data_url, extract_video_frames, image_batch_frames,
                    pil_to_data_url, sample_image_batch, tensor_frame_to_pil)


@dataclass(frozen=True)
class ChatResult:
    response: str
    reasoning: str
    raw_response: str
    usage: dict[str, Any]
    stats: str

def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain(model_dump())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_plain(to_dict())
        except Exception:
            pass
    return value

def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)

def _extract_completion(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _to_plain(result)
    if not isinstance(data, dict):
        raise BackendError(f"The backend returned an unsupported result type: {type(result).__name__}.")
    choices = data.get("choices") or []
    if not choices:
        raise BackendError("The backend response did not contain any choices.")
    choice = choices[0] if isinstance(choices[0], dict) else _to_plain(choices[0])
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        # A few completion-style compatible endpoints return `text` instead.
        message = {"content": choice.get("text", "") if isinstance(choice, dict) else ""}
    message = dict(message)
    if "content" in message:
        message["content"] = _content_text(message["content"])
    return message, data.get("usage") or {}

def _merge_tool_call_delta(
    tool_calls: dict[int, dict[str, Any]], delta_calls: Any
) -> None:
    """Merge streamed OpenAI tool-call fragments by their index."""
    for position, raw_call in enumerate(_to_plain(delta_calls) or []):
        if not isinstance(raw_call, dict):
            continue
        index = int(raw_call.get("index", position))
        target = tool_calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if raw_call.get("id"):
            target["id"] = str(raw_call["id"])
        if raw_call.get("type"):
            target["type"] = str(raw_call["type"])
        function = raw_call.get("function") or {}
        if isinstance(function, dict):
            target_function = target.setdefault("function", {"name": "", "arguments": ""})
            if function.get("name"):
                target_function["name"] += str(function["name"])
            if function.get("arguments"):
                target_function["arguments"] += str(function["arguments"])

def _collect_stream(result: Any, progress_callback=None) -> Any:
    """Turn a local/API completion stream into a regular response dictionary."""
    plain = _to_plain(result)
    if isinstance(plain, dict):
        if callable(progress_callback):
            usage = plain.get("usage") or {}
            progress_callback(int(usage.get("completion_tokens") or 0))
        return plain

    if isinstance(result, (str, bytes, bytearray)) or not isinstance(result, Iterable):
        return result

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    usage: dict[str, Any] = {}
    chunk_count = 0

    for raw_chunk in result:
        chunk = _to_plain(raw_chunk)
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else _to_plain(choices[0])
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            continue
        content = _content_text(delta.get("content"))
        if content:
            content_parts.append(content)
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("thinking_content")
            or ""
        )
        if reasoning:
            reasoning_parts.append(str(reasoning))
        if delta.get("tool_calls"):
            _merge_tool_call_delta(tool_calls, delta["tool_calls"])
        finish_reason = choice.get("finish_reason") or finish_reason
        chunk_count += 1
        if callable(progress_callback):
            progress_callback(chunk_count)

    if not chunk_count:
        raise BackendError("The backend stream did not return any response choices.")
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }

def _split_message(message: dict[str, Any]) -> tuple[str, str, str]:
    content = str(message.get("content") or "")
    reasoning = str(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking_content")
        or ""
    )
    raw = content
    if not reasoning and "<think>" in content:
        before, _, tail = content.partition("<think>")
        thought, marker, after = tail.partition("</think>")
        if marker:
            reasoning = thought.strip()
            content = (before + after).strip()
    elif not reasoning and "</think>" in content:
        # llama.cpp/Qwen can prefill the opening <think> token in the assistant
        # template.  Generated text then contains only the closing marker.  In
        # that representation everything before </think> is still reasoning,
        # not part of the response consumed by downstream ComfyUI nodes.
        thought, _, after = content.partition("</think>")
        reasoning = thought.strip()
        content = after.strip()
    content = content.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return content, reasoning.strip(), raw

def _parse_tools(tools_json: str | None) -> list[dict[str, Any]] | None:
    if not tools_json or not tools_json.strip():
        return None
    try:
        parsed = json.loads(tools_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tools_json is not valid JSON: {exc}") from exc
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError("tools_json must be a tool object or an array of tool objects.")
    return parsed

def _build_content(
    prompt: str,
    image: torch.Tensor | None,
    video_frames: torch.Tensor | None,
    video: Any | None,
    max_video_frames: int,
    video_transport: str,
    api_backend: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    content: list[dict[str, Any]] = []
    counts = {"images": 0, "video_frames": 0, "videos": 0}

    if image is not None:
        frames = image_batch_frames(image)
        counts["images"] = len(frames)
        for index, frame in enumerate(frames, 1):
            if len(frames) > 1:
                content.append({"type": "text", "text": f"Image {index}/{len(frames)}:"})
            pil = tensor_frame_to_pil(frame)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil)}})

    # A VIDEO object can be sent as a native data URL to APIs that implement
    # the OpenAI-compatible `video_url` convention.  Local llama.cpp handlers instead
    # receive decoded frames, because they consume image parts.
    use_native_video = video is not None and api_backend and video_transport in {"auto", "video_url"}
    if use_native_video:
        try:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": encode_video_data_url(video)},
                }
            )
            counts["videos"] = 1
        except Exception:
            if video_transport == "video_url":
                raise
            use_native_video = False

    if video_frames is not None:
        frames = sample_image_batch(video_frames, max_video_frames)
        counts["video_frames"] = len(frames)
        for index, frame in enumerate(frames, 1):
            content.append({"type": "text", "text": f"Video frame {index}/{len(frames)}:"})
            pil = tensor_frame_to_pil(frame)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil, 85)}})
    elif video is not None and not use_native_video:
        frames = extract_video_frames(video, max_video_frames)
        counts["video_frames"] = len(frames)
        for index, pil in enumerate(frames, 1):
            content.append({"type": "text", "text": f"Video frame {index}/{len(frames)}:"})
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil, 85)}})

    content.append({"type": "text", "text": prompt.strip()})
    return content, counts

def _run_chat(
    backend: Any,
    *,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repeat_penalty: float,
    seed: int,
    max_video_frames: int,
    video_transport: str,
    thinking_mode: str,
    image: torch.Tensor | None,
    video_frames: torch.Tensor | None,
    video: Any | None,
    tools_json: str | None,
    progress_callback=None,
):
    is_api = getattr(backend, "backend_kind", "") == "openai_compatible"
    content, media_counts = _build_content(
        prompt,
        image,
        video_frames,
        video,
        int(max_video_frames),
        video_transport,
        is_api,
    )
    # Some image/video chat templates (including Qwen3.8's MTMD template)
    # insert their own default system message when none is supplied.  Passing
    # an empty system message leaves it in second position after that default
    # and the template rejects the prompt with "System message must be at the
    # beginning".  Omit empty optional messages instead of serializing them.
    messages = []
    cleaned_system_prompt = system_prompt.strip()
    if cleaned_system_prompt:
        messages.append({"role": "system", "content": cleaned_system_prompt})
    messages.append({"role": "user", "content": content})
    tools = _parse_tools(tools_json)
    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "repeat_penalty": float(repeat_penalty),
        "seed": int(seed),
        "stream": callable(progress_callback),
    }
    if is_api:
        kwargs["thinking_mode"] = thinking_mode
    if tools:
        kwargs["tools"] = tools

    started = time.perf_counter()
    stream = backend.complete(**kwargs)
    try:
        result = _collect_stream(stream, progress_callback)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    elapsed = time.perf_counter() - started
    message, usage = _extract_completion(result)
    response, reasoning, raw = _split_message(message)
    tool_calls = message.get("tool_calls")
    if tool_calls and not response:
        response = json.dumps(tool_calls, ensure_ascii=False, indent=2)
    if tool_calls:
        raw = json.dumps(message, ensure_ascii=False, indent=2)
    completion_tokens = usage.get("completion_tokens")
    speed = (
        f"{int(completion_tokens) / elapsed:.2f} tok/s"
        if completion_tokens is not None and elapsed > 0 else "n/a"
    )
    stats = (
        f"time={elapsed:.2f}s\n"
        f"prompt_tokens={usage.get('prompt_tokens', 'n/a')}\n"
        f"completion_tokens={usage.get('completion_tokens', 'n/a')}\n"
        f"speed={speed}\n"
        f"images={media_counts['images']}\nvideo_frames={media_counts['video_frames']}\n"
        f"backend={getattr(backend, 'backend_kind', type(backend).__name__)}"
    )
    model_path = getattr(getattr(backend, "settings", None), "model_path", None)
    model = model_path.name if model_path is not None else getattr(backend, "model", "")
    stats += f"\nvideos={media_counts['videos']}\nmodel={model}"
    reasoning_info = getattr(backend, "reasoning_info", None)
    if callable(reasoning_info):
        detail = str(reasoning_info()).strip()
        if detail:
            stats = f"{stats}\n{detail}"
    return ChatResult(response, reasoning, raw, usage, stats)
