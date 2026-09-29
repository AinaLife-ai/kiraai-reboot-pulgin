from __future__ import annotations

"""
Compatibility layer with Auto Delete Session (ADS) and other memory-rewriting plugins.

There are two independent things to keep apart:

1. THE SUMMARY HEAD (data). Both plugins store their summary as the first memory
   chunk, prefixed with the same marker string. Sharing the marker is what stops
   ADS' `_bridge_truncated_summary` from re-injecting its own stale cumulative
   summary right after we cleared the context. If we used a private marker, ADS
   would look at the head, fail to recognise it as a summary, and push its old
   summary back into the request - the user would see "I just reset it, why does
   it still remember?". See `SUMMARY_MARKER` in summarizer.py.

2. THE COMMANDS (control). `/reboot` and `/resume` belong to this plugin, `/resum`
   is ADS'. Defaults deliberately differ by one character, so a default install
   never collides. `yield_overlapping_commands` covers the case where a user edits
   the lists so they DO overlap: both plugins intercept on `on.im_message` at
   Priority.HIGH and call `event.stop()`, and same-priority handlers keep their
   insertion order - i.e. the winner would depend on plugin load order and could
   change across restarts. We resolve that deterministically by removing the
   overlapping keywords from OUR side and logging a warning.
"""

from typing import Any, Iterable, List, Optional

# Candidate plugin ids for Auto Delete Session. The directory name may carry a
# suffix (e.g. "-main") so an exact match is tried first and a normalized fuzzy
# scan afterwards.
ADS_PLUGIN_IDS = (
    "auto_delete_session",
    "KiraAI_auto_delete_session_plugin",
    "kiraai-auto-delete-session-plugin",
)

# Other plugins known to rewrite session memory. We only warn about these.
MEMORY_PLUGIN_HINTS = {
    "auto_delete_session": "Auto Delete Session",
    "context_condensation": "Context Condensation (CCS)",
    "kiraai_context_condensation": "Context Condensation (CCS)",
    "ksm": "Kira Session Merger (KSM)",
}


def _normalize(plugin_id: Any) -> str:
    return str(plugin_id).lower().replace("-", "").replace("_", "")


def _iter_plugin_ids(plugin_mgr) -> List[str]:
    ids: List[str] = []
    try:
        for attr in ("plugin_instances", "plugins", "_plugins", "plugin_configs"):
            container = getattr(plugin_mgr, attr, None)
            if isinstance(container, dict):
                for key in container:
                    if key not in ids:
                        ids.append(str(key))
    except Exception:
        pass
    return ids


def find_ads_plugin(plugin_mgr) -> Optional[str]:
    """Return the plugin id of Auto Delete Session, or None if not installed."""
    if not plugin_mgr:
        return None
    try:
        for pid in ADS_PLUGIN_IDS:
            try:
                if plugin_mgr.has_plugin(pid):
                    return pid
            except Exception:
                continue
        for pid in _iter_plugin_ids(plugin_mgr):
            if "autodeletesession" in _normalize(pid):
                return pid
    except Exception:
        pass
    return None


def is_plugin_active(plugin_mgr, plugin_id: str) -> bool:
    """Best-effort check that a plugin is installed AND enabled."""
    if not plugin_mgr or not plugin_id:
        return False
    try:
        if not plugin_mgr.has_plugin(plugin_id):
            return False
    except Exception:
        return False
    try:
        return bool(plugin_mgr.is_plugin_enabled(plugin_id))
    except Exception:
        return True


def read_ads_summary_marker(plugin_mgr, plugin_id: str) -> Optional[str]:
    """Read the marker ADS actually uses, so a mismatch can be reported.

    Returns None when it cannot be determined (older/newer ADS, private module).
    """
    if not plugin_mgr or not plugin_id:
        return None
    try:
        inst = plugin_mgr.get_plugin_inst(plugin_id)
    except Exception:
        inst = None
    if inst is None:
        return None
    try:
        import sys

        module = sys.modules.get(type(inst).__module__)
        marker = getattr(module, "SUMMARY_MARKER", None)
        if isinstance(marker, str) and marker:
            return marker
    except Exception:
        pass
    # Some versions keep it as a class attribute.
    marker = getattr(type(inst), "SUMMARY_MARKER", None)
    if isinstance(marker, str) and marker:
        return marker
    return None


def read_ads_reset_commands(plugin_mgr, plugin_id: str) -> List[str]:
    """Read ADS' configured reset keywords, from the instance or its saved config."""
    if not plugin_mgr or not plugin_id:
        return []
    try:
        inst = plugin_mgr.get_plugin_inst(plugin_id)
        raw = getattr(inst, "reset_commands", None)
        commands = _as_command_list(raw)
        if commands:
            return commands
    except Exception:
        pass
    try:
        cfg = (getattr(plugin_mgr, "plugin_configs", None) or {}).get(plugin_id) or {}
        section = cfg.get("section_command")
        if isinstance(section, dict):
            commands = _as_command_list(section.get("reset_commands"))
            if commands:
                return commands
        # Legacy flat layout.
        return _as_command_list(cfg.get("reset_commands"))
    except Exception:
        return []


def _as_command_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [x for x in raw.split(",")]
    if not isinstance(raw, Iterable):
        return []
    out: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def yield_overlapping_commands(
    own_commands: Iterable[str],
    ads_commands: Iterable[str],
    logger=None,
) -> List[str]:
    """Drop keywords that ADS also claims, and warn about each removal.

    This plugin yields to ADS on purpose: `/resum` is ADS' established keyword and
    existing ADS users expect it to keep working after installing reboot.
    """
    ads_set = {str(c).strip().lower() for c in (ads_commands or []) if str(c).strip()}
    kept: List[str] = []
    yielded: List[str] = []
    for command in own_commands or []:
        text = str(command).strip()
        if not text:
            continue
        if text.lower() in ads_set:
            yielded.append(text)
            continue
        if text not in kept:
            kept.append(text)
    if yielded and logger:
        logger.warning(
            "[reboot] 指令 %s 已让位给 Auto Delete Session（ADS 也注册了同一关键词）。"
            "如需本插件的摘要重开，请改用其它关键词。",
            "/".join(yielded),
        )
    return kept


def verify_ads_marker(plugin_mgr, ads_plugin_id: str, own_marker: str, logger=None) -> bool:
    """Check that ADS uses the same summary marker as us.

    Returns True when the contract holds (or cannot be checked), False on a
    confirmed mismatch. A mismatch means both plugins may write their own summary
    head and fight over it.
    """
    marker = read_ads_summary_marker(plugin_mgr, ads_plugin_id)
    if marker is None:
        return True
    if marker == own_marker:
        return True
    if logger:
        logger.warning(
            "[reboot] 检测到 Auto Delete Session 的摘要标记已变更（ADS=%r，本插件=%r）："
            "两个插件的摘要将无法互相认领，可能出现双摘要头。"
            "可在插件设置中关闭「与 Auto Delete Session 兼容」以隔离。",
            marker,
            own_marker,
        )
    return False


def report_memory_plugins(plugin_mgr, enabled: bool, logger=None) -> List[str]:
    """Log which other memory-rewriting plugins are present (advisory only)."""
    if not enabled or not plugin_mgr:
        return []
    found: List[str] = []
    try:
        for pid in _iter_plugin_ids(plugin_mgr):
            norm = _normalize(pid)
            for hint, label in MEMORY_PLUGIN_HINTS.items():
                target = _normalize(hint)
                # Guard the reverse containment: a very short plugin id would
                # otherwise "contain" every hint and produce bogus reports.
                matched = target in norm or (len(norm) >= 4 and norm in target)
                if matched and label not in found:
                    found.append(label)
    except Exception:
        return []
    if found and logger:
        logger.info(
            "[reboot] 检测到同样会改写会话记忆的插件：%s。"
            "本插件重置时会清空上下文，请确认它们的行为符合预期。",
            "、".join(found),
        )
    return found
