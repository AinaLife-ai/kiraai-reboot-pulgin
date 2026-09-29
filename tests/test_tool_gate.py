"""The LLM tool: registration, the allow_llm_reboot gate and its behaviour."""

from harness import (
    bootstrap,
    check,
    check_eq,
    make_batch_event,
    make_fixture,
    make_llm_request,
)

SID = "qq:dm:10001"
TOOL = "reset_context"


def registered_tools():
    """Read the framework registry the decorator writes into."""
    bootstrap()
    from core.plugin.plugin_registry import _plugin_components

    return dict(_plugin_components.get("reboot_plugin").tools)


# ── registration ─────────────────────────────────────────────────────────────


async def test_tool_is_registered_under_the_new_name_only():
    tools = registered_tools()
    check(TOOL in tools, f"{TOOL} must be registered, got {sorted(tools)}")
    check("clear_context" not in tools, "the pre-2.0 tool name must be gone")
    meta = tools[TOOL]
    check_eq(meta["parameters"]["type"], "object", "parameters must be an object schema")
    check("上下文" in meta["description"], "a description is required for the model")


async def test_plugin_id_and_manifest_are_consistent():
    bootstrap()
    from core.plugin.plugin_registry import _plugin_manifests

    manifest = _plugin_manifests.get("reboot_plugin")
    check(manifest is not None, "manifest must be discovered as reboot_plugin")
    check_eq(manifest["plugin_id"], "reboot_plugin", "plugin_id must never change")


# ── the gate ─────────────────────────────────────────────────────────────────


async def test_tool_visible_when_allowed():
    f = await make_fixture({"section_llm_tool.allow_llm_reboot": True})
    req = make_llm_request(tool_names=(TOOL, "other"))
    await f.plugin.on_llm_request(make_batch_event(SID), req, None)
    check(TOOL in req.tool_set, "the tool must stay visible")


async def test_tool_hidden_from_the_model_when_disabled():
    """The request hook runs before the framework serialises the tool list, so
    removing it here is authoritative."""
    f = await make_fixture({"section_llm_tool.allow_llm_reboot": False})
    req = make_llm_request(tool_names=(TOOL, "other"))
    await f.plugin.on_llm_request(make_batch_event(SID), req, None)
    check(TOOL not in req.tool_set, "the tool must be removed for this request")
    check("other" in req.tool_set, "unrelated tools must be left alone")


async def test_gate_is_reversible_per_request():
    f = await make_fixture({"section_llm_tool.allow_llm_reboot": False})
    first = make_llm_request(tool_names=(TOOL,))
    await f.plugin.on_llm_request(make_batch_event(SID), first, None)
    f.plugin.allow_llm_reboot = True
    second = make_llm_request(tool_names=(TOOL,))
    await f.plugin.on_llm_request(make_batch_event(SID), second, None)
    check(TOOL in second.tool_set, "the gate follows the live switch")
    check(TOOL not in first.tool_set, "and does not corrupt earlier requests")


async def test_on_loaded_unregisters_the_tool_when_disabled():
    f = await make_fixture({"section_llm_tool.allow_llm_reboot": False})
    await f.plugin.drop_llm_tool_if_disabled()
    check_eq(f.ctx.tool_mgr.unregistered, [TOOL], "tool unregistered globally")


async def test_on_loaded_keeps_the_tool_when_enabled():
    f = await make_fixture({"section_llm_tool.allow_llm_reboot": True})
    await f.plugin.drop_llm_tool_if_disabled()
    check_eq(f.ctx.tool_mgr.unregistered, [], "nothing unregistered")


async def test_tool_call_is_refused_when_disabled():
    f = await make_fixture({"section_llm_tool.allow_llm_reboot": False})
    f.seed_session(SID, turns=2)
    result = await f.plugin.reset_context(make_batch_event(SID))
    check("关闭" in result, f"expected a refusal, got {result!r}")
    check_eq(len(f.memory(SID)), 2, "context untouched")


# ── behaviour ────────────────────────────────────────────────────────────────


async def test_tool_call_resets_the_context():
    f = await make_fixture({})
    f.seed_session(SID, turns=3)
    result = await f.plugin.reset_context(make_batch_event(SID))
    check("已重置" in result, f"expected success, got {result!r}")
    check_eq(f.memory(SID), [], "context cleared")
    check_eq(f.meta(SID)["title"], "T", "metadata preserved on the tool path too")


async def test_tool_call_on_an_empty_session_reports_it():
    f = await make_fixture({})
    f.seed_session(SID, turns=0)
    f.sm.write_memory(SID, [])
    result = await f.plugin.reset_context(make_batch_event(SID))
    check("没有可重置" in result, f"expected the empty notice, got {result!r}")


async def test_tool_call_respects_the_permission_whitelist():
    f = await make_fixture({"section_command.reboot_enable_permission": True,
                            "section_command.reboot_allowed_users": ["10001"]})
    f.seed_session(SID, turns=1)
    denied = await f.plugin.reset_context(make_batch_event(SID, user_id="5"))
    check("权限不足" in denied, "outsider denied")
    check_eq(len(f.memory(SID)), 1, "context untouched")

    allowed = await f.plugin.reset_context(make_batch_event(SID, user_id="10001"))
    check("已重置" in allowed, "whitelisted user allowed")


async def test_tool_call_uses_the_same_cooldown_as_the_command_path():
    f = await make_fixture({})
    f.seed_session(SID, turns=2)
    first = await f.plugin.reset_context(make_batch_event(SID))
    check("已重置" in first, "first call resets")
    f.seed_session(SID, turns=2)
    second = await f.plugin.reset_context(make_batch_event(SID))
    check("刚刚" in second, f"second call inside the window must be short-circuited, got {second!r}")
    check_eq(len(f.memory(SID)), 2, "the second call must not have cleared anything")


async def test_tool_call_without_a_session_id_fails_gracefully():
    f = await make_fixture({})
    from core.chat.message_utils import KiraMessageBatchEvent

    event = KiraMessageBatchEvent(message_types=["dm"], timestamp=0, session=None)
    result = await f.plugin.reset_context(event)
    check("失败" in result or "无法" in result or "error" in result.lower(),
          f"expected a graceful error, got {result!r}")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
