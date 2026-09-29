from __future__ import annotations

"""
Summary generation for the reboot plugin.

Ported from the Auto Delete Session plugin (ADS v2.1.4) `summarizer.py`, with the
keep_recent_turns / anchor machinery removed (this plugin always clears the whole
context, so there is no "kept window" to align against).

Design notes:
- Anything that can fail (no model, timeout, empty result, exception) returns
  ``None``. Callers degrade to a plain clear - the summary must never block or
  break the reset itself.
- ``SUMMARY_MARKER`` is intentionally byte-identical to the one used by ADS so
  that both plugins recognise each other's summary head. See compat.py.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agent.message import OpenAIMessage
from core.provider import LLMRequest

DEFAULT_SUMMARIZE_PROMPT = (
    "你是聊天记忆压缩器。下面这段较早的聊天记录即将被清理,请把它压缩成一段简短摘要（200字内）,"
    "用于衔接后续对话,让你能像没有失忆一样自然地继续聊下去。\n"
    "请优先保留：\n"
    "- 正在进行或未完成的事情（话题、任务、约定、待办）\n"
    "- 重要事实（人物、关系、时间、地点、已做的决定）\n"
    "- 对方的偏好、称呼、语气和你们之间的相处方式\n"
    "时间一律写绝对日期,直接输出摘要正文,不要任何开场白、标题或解释。"
)

DEFAULT_MERGE_PROMPT = (
    "你是聊天记忆压缩器。下面有两段材料：【旧摘要】是更早对话的已有摘要，"
    "【新增记录】是刚刚被清理的对话片段。请把它们合并成一段新的摘要（600字内），"
    "用于衔接后续对话，让你能像没有失忆一样自然地继续聊下去。\n"
    "请优先保留：\n"
    "- 正在进行或未完成的事情（话题、任务、约定、待办）\n"
    "- 重要事实（人物、关系、时间、地点、已做的决定）\n"
    "- 对方的偏好、称呼、语气和你们之间的相处方式\n"
    "若新旧信息冲突，以新增记录为准。时间一律写绝对日期。"
    "直接输出合并后的摘要正文，不要任何开场白、标题或解释。\n\n"
    "【旧摘要】\n{old_summary}\n\n【新增记录】\n{new_summary}"
)

DEFAULT_SELF_COMPRESS_PROMPT = (
    "你是聊天记忆压缩器。下面这段对话摘要过长，请在不丢失关键信息的前提下进一步压缩（600字内），"
    "用于衔接后续对话，让你能像没有失忆一样自然地继续聊下去。\n"
    "优先保留：未完成的事情、重要事实（人物/关系/时间/地点/已做的决定）、对方的偏好与称呼。\n"
    "直接输出压缩后的摘要正文，不要任何开场白或解释。\n\n{summary}"
)

# Summary head marker.
#
# IMPORTANT: this string is a cross-plugin contract. It is kept byte-identical to
# Auto Delete Session's marker so both plugins recognise the same summary head and
# neither re-injects a stale one. Do NOT change it without updating compat.py.
SUMMARY_MARKER = "[前情摘要|系统注入]"


def _safe_format(template: str, mapping: Dict[str, Any]) -> str:
    """Explicit placeholder replacement.

    ``str.format`` would explode on ``{`` / ``}`` appearing in model output or in
    a chat transcript pasted by the user, so placeholders are replaced by hand.
    """
    text = template
    for key, value in mapping.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def _msg_get(msg: Any, key: str, default: Any = None) -> Any:
    """Read a field from a message that may be a dict or a pydantic object."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_text(content: Any) -> str:
    """Flatten message content (str, or multimodal part list) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def build_summary_chunk(summary_text: str) -> list:
    """Wrap summary text into a framework memory chunk (starts with a user message)."""
    return [
        {
            "role": "user",
            "content": (
                f"{SUMMARY_MARKER} 以下是更早对话被清理前的自动摘要，"
                f"仅供延续上下文参考：\n{summary_text}"
            ),
        }
    ]


def is_summary_chunk(chunk) -> bool:
    """True when a memory chunk is already a summary chunk (avoids double injection)."""
    if not isinstance(chunk, list) or not chunk:
        return False
    first = chunk[0]
    content = _msg_get(first, "content")
    return isinstance(content, str) and content.startswith(SUMMARY_MARKER)


def extract_summary_text(content: str) -> str:
    """Strip the marker line and lead-in from a summary message content."""
    if not isinstance(content, str) or not content.startswith(SUMMARY_MARKER):
        return ""
    lines = content.split("\n", 1)
    return lines[1].strip() if len(lines) > 1 else ""


def dropped_fingerprint(dropped_flat: List[dict]) -> str:
    """Fingerprint of the dropped range (count + hash of the last text message).

    Used by tests to assert the summary input really covers the dropped range.
    """
    if not dropped_flat:
        return ""
    last_text = ""
    for m in reversed(dropped_flat):
        text = _msg_text(_msg_get(m, "content"))
        if text.strip():
            last_text = text.strip()
            break
    h = hashlib.sha256(last_text[:200].encode("utf-8")).hexdigest()[:12]
    return f"{len(dropped_flat)}:{h}"


def extract_dropped_text(dropped_flat: List[Any], max_input_chars: int) -> str:
    """Flatten history into plain text for the summary model.

    - user / assistant take their text content
    - assistant tool calls are collapsed into a single line
    - tool results are included
    - system and other roles are skipped

    When ``max_input_chars > 0`` the blob is truncated from the TAIL, because the
    most recent content matters most. ``<= 0`` means no limit.
    """
    lines: List[str] = []
    for m in dropped_flat:
        if m is None:
            continue
        role = _msg_get(m, "role")
        text = _msg_text(_msg_get(m, "content"))
        if role == "user" and text.strip():
            lines.append(f"用户: {text.strip()}")
        elif role == "assistant":
            tool_calls = _msg_get(m, "tool_calls") or []
            if text.strip():
                lines.append(f"助手: {text.strip()}")
            elif tool_calls:
                names = []
                for tc in tool_calls:
                    name = _msg_get(_msg_get(tc, "function", {}) or {}, "name")
                    if name:
                        names.append(str(name))
                if names:
                    lines.append(f"助手: [调用工具 {', '.join(names)}]")
        elif role == "tool" and text.strip():
            lines.append(f"工具结果: {text.strip()}")

    blob = "\n".join(lines)
    if max_input_chars > 0 and len(blob) > max_input_chars:
        blob = blob[-max_input_chars:]
    return blob


def _resolve_client(ctx, model_id: str, logger=None):
    """Resolve the summary model client: explicit model -> fast -> default -> None."""
    model_id = (model_id or "").strip()
    if model_id:
        try:
            client = ctx.get_llm_client(model_id)
            if client:
                return client
        except Exception as e:
            if logger:
                logger.warning(f"[reboot] get_llm_client({model_id}) failed: {e}, fallback")
    for getter in ("get_default_fast_llm_client", "get_default_llm_client"):
        try:
            fn = getattr(ctx, getter, None)
            if fn:
                client = fn()
                if client:
                    return client
        except Exception:
            continue
    return None


async def _chat_text(
    client,
    sid: str,
    prompt: str,
    timeout_sec: float,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """One LLM round-trip returning plain text. Any failure returns None."""
    if client is None:
        return None
    req = LLMRequest(messages=[OpenAIMessage(role="user", content=prompt)])
    try:
        timeout = max(0.5, float(timeout_sec or 30.0))
        resp = await asyncio.wait_for(client.chat(req), timeout)
    except asyncio.TimeoutError:
        if logger:
            logger.warning(f"[reboot] summary LLM timeout ({timeout_sec}s) for {sid}")
        return None
    except Exception as e:
        if logger:
            logger.warning(f"[reboot] summary LLM call failed for {sid}: {e}")
        return None
    text = (getattr(resp, "text_response", None) or "").strip()
    if not text and enable_detail_log and logger:
        logger.info("[reboot][summary] LLM returned empty result")
    return text or None


async def summarize_history(
    ctx,
    sid: str,
    dropped_flat: List[Any],
    model_id: str = "",
    prompt_template: str = "",
    timeout_sec: float = 60.0,
    max_input_chars: int = 10000,
    max_output_chars: int = 5000,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """Summarize history that is about to be cleared. None on any failure."""
    if not dropped_flat:
        if enable_detail_log and logger:
            logger.info("[reboot][summary] nothing to summarize (empty history)")
        return None

    client = _resolve_client(ctx, model_id, logger=logger)
    if client is None:
        if logger:
            logger.warning(f"[reboot] no LLM client available, skip summary for {sid}")
        return None

    text = extract_dropped_text(dropped_flat, max_input_chars)
    if not text.strip():
        if enable_detail_log and logger:
            logger.info("[reboot][summary] extracted text is empty, skip summary")
        return None
    if enable_detail_log and logger:
        logger.info(f"[reboot][summary] input ({len(text)} chars):\n{text[:500]}...")

    prompt = (prompt_template or "").strip() or DEFAULT_SUMMARIZE_PROMPT
    if "{text}" in prompt:
        prompt_text = _safe_format(prompt, {"text": text})
    else:
        prompt_text = f"{prompt}\n\n{text}"

    summary = await _chat_text(
        client,
        sid,
        prompt_text,
        timeout_sec=timeout_sec,
        logger=logger,
        enable_detail_log=enable_detail_log,
    )
    if not summary:
        return None
    if max_output_chars > 0 and len(summary) > max_output_chars:
        summary = summary[:max_output_chars] + "…"
    if logger:
        logger.info(f"[reboot] summary ok for {sid} ({len(summary)} chars)")
    if enable_detail_log and logger:
        logger.info(f"[reboot][summary] generated:\n{summary}")
    return summary


async def merge_summaries(
    ctx,
    sid: str,
    old_summary: str,
    new_summary: str,
    model_id: str = "",
    prompt_template: str = "",
    timeout_sec: float = 120.0,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """Merge the previous cumulative summary with the new delta. None on failure."""
    old_summary = (old_summary or "").strip()
    new_summary = (new_summary or "").strip()
    if not old_summary:
        return new_summary or None
    if not new_summary:
        return old_summary or None
    client = _resolve_client(ctx, model_id, logger=logger)
    prompt = (prompt_template or "").strip() or DEFAULT_MERGE_PROMPT
    prompt_text = _safe_format(
        prompt, {"old_summary": old_summary, "new_summary": new_summary}
    )
    merged = await _chat_text(
        client,
        sid,
        prompt_text,
        timeout_sec=timeout_sec,
        logger=logger,
        enable_detail_log=enable_detail_log,
    )
    if merged and enable_detail_log and logger:
        logger.info(f"[reboot][summary] merged:\n{merged}")
    return merged


async def self_compress_summary(
    ctx,
    sid: str,
    summary: str,
    model_id: str = "",
    prompt_template: str = "",
    timeout_sec: float = 120.0,
    logger=None,
    enable_detail_log: bool = False,
) -> Optional[str]:
    """Fallback compression when the cumulative summary exceeds the output cap."""
    summary = (summary or "").strip()
    if not summary:
        return None
    client = _resolve_client(ctx, model_id, logger=logger)
    prompt = (prompt_template or "").strip() or DEFAULT_SELF_COMPRESS_PROMPT
    prompt_text = _safe_format(prompt, {"summary": summary})
    return await _chat_text(
        client,
        sid,
        prompt_text,
        timeout_sec=timeout_sec,
        logger=logger,
        enable_detail_log=enable_detail_log,
    )


# Upper bound on persisted store entries. One entry is kept per session and is only
# dropped when that session is cleared or deleted, so a long running bot with many
# sessions would otherwise grow the file forever. Oldest entries are evicted first.
MAX_STORE_ENTRIES = 500


class CumulativeSummaryStore:
    """Per-session cumulative summary persisted in the plugin data directory.

    Invariant: the stored summary must stay consistent with the summary head of
    the session memory. ``sync_with_head`` reconciles the two and is the only
    supported way to obtain the authoritative base summary.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._data: Dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        self._loaded = True
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {
                        str(k): v for k, v in raw.items() if isinstance(v, dict)
                    }
        except Exception:
            self._data = {}

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, sid: str) -> str:
        self._ensure_loaded()
        entry = self._data.get(sid) or {}
        return str(entry.get("summary") or "")

    def set(self, sid: str, summary: str) -> None:
        self._ensure_loaded()
        self._data[sid] = {"summary": summary, "updated_at": int(time.time())}

    def pop(self, sid: str) -> None:
        self._ensure_loaded()
        self._data.pop(sid, None)

    def sync_with_head(
        self, sid: str, head_summary: str, session_has_messages: bool = True
    ) -> str:
        """Reconcile the store against the summary head and return the authority.

        - head present and equal to store  -> use store (normal path)
        - head present but different       -> trust the head, write it back
          (another writer, e.g. ADS, updated it)
        - head absent:
            * session still has other messages -> the head was very likely evicted
              by the framework window; keep the store as the authority
            * session is empty (user cleared it) -> drop the entry and return ""
        """
        self._ensure_loaded()
        stored = self.get(sid)
        head_summary = (head_summary or "").strip()
        if head_summary:
            if stored != head_summary:
                self.set(sid, head_summary)
            return head_summary
        if stored and not session_has_messages:
            self.pop(sid)
            return ""
        return stored or ""

    def prune(self, max_entries: int = MAX_STORE_ENTRIES) -> int:
        """Drop the oldest entries when the store grows past the cap."""
        self._ensure_loaded()
        overflow = len(self._data) - max(1, int(max_entries))
        if overflow <= 0:
            return 0
        def _ts(entry):
            try:
                return int((entry or {}).get("updated_at") or 0)
            except Exception:
                return 0
        # Timestamps have one second resolution, so ties are broken by insertion
        # order: reverse first, then stable-sort by timestamp, so the most recently
        # inserted entries survive a same-second batch.
        items = list(self._data.items())
        items.reverse()
        items.sort(key=lambda kv: _ts(kv[1]), reverse=True)
        self._data = dict(items[: max(1, int(max_entries))])
        return overflow

    def save(self) -> None:
        """Atomically persist the store (best effort, never raises)."""
        self._ensure_loaded()
        self.prune()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:
            pass
