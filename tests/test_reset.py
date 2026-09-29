"""Core reset semantics: what a reset must and must not destroy."""

import asyncio

from harness import (
    check,
    check_eq,
    make_batch_event,
    make_fixture,
    make_message_event,
)

SID = "qq:dm:10001"


# ── the fix: clearing must keep the session metadata ─────────────────────────


async def test_reset_keeps_title_description_timestamp_and_capabilities():
    f = await make_fixture({"section_command.enable_reboot_command": True},
                           capability_overrides={"tts": {"enabled": False},
                                                 "image_generation": {"enabled": False}})
    s = f.seed_session(SID, turns=3)
    f.sm.update_session_capabilities(s, {"tts": {"enabled": False},
                                         "image_generation": {"enabled": False}})
    before = f.meta(SID)
    check_eq(before["title"], "T", "precondition title")
    check_eq(before["description"], "D", "precondition description")

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    after = f.meta(SID)
    check_eq(after["title"], "T", "title must survive")
    check_eq(after["description"], "D", "description must survive")
    check_eq(after["timestamp"], 1700000000, "timestamp must survive")
    check_eq(after["capabilities"], before["capabilities"],
             "per-session capability overrides must survive")
    check_eq(f.memory(SID), [], "memory must be emptied")


async def test_old_delete_session_implementation_loses_metadata():
    """Reverse verification: the pre-2.0 code path really did destroy metadata.

    This is the behaviour the 2.0 rewrite exists to fix; if `delete_session` ever
    stops dropping these fields the assertion below turns red and this test must be
    revisited.
    """
    f = await make_fixture(with_capabilities=False)
    s = f.seed_session(SID, turns=2)
    f.sm.update_session_capabilities(s, {"tts": {"enabled": False}})
    check_eq(f.meta(SID)["capabilities"], {"tts": {"enabled": False}}, "precondition")

    # exactly what the old plugin did
    f.sm.delete_session(s)
    f.sm.get_session_info(s)  # rebuild the empty shell

    after = f.meta(SID)
    check_eq(after["title"], "", "old path wiped the title")
    check_eq(after["description"], "", "old path wiped the description")
    check_eq(after["timestamp"], None, "old path wiped the timestamp")
    check("capabilities" not in after,
          "old path dropped the capability overrides and never restored them")


async def test_delete_session_mode_is_still_available():
    f = await make_fixture({"section_reset.clear_mode": "delete_session",
                            "section_command.enable_reboot_command": True},
                           with_capabilities=False)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    after = f.meta(SID)
    check_eq(after["memory"], [], "memory cleared")
    check_eq(after["title"], "", "escape hatch keeps the legacy destructive behaviour")


# ── empty history ────────────────────────────────────────────────────────────


async def test_empty_history_replies_but_changes_nothing():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    s = f.seed_session(SID, turns=0)
    f.sm.write_memory(s, [])

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    check("没有可重置的历史" in f.last_reply(), f"expected empty notice, got {f.last_reply()!r}")
    check_eq(f.memory(SID), [], "memory untouched")
    check("qq:dm:10001" not in f.plugin._store._data, "store must stay untouched")


async def test_missing_session_is_treated_as_empty():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check("没有可重置的历史" in f.last_reply(), "unknown session must not crash")


# ── pending buffer ───────────────────────────────────────────────────────────


async def test_pending_buffer_is_dropped_before_clearing():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    s = f.seed_session(SID, turns=1)
    buf = f.ctx.get_buffer(s)
    for i in range(3):
        buf.add(f"pending {i}")
    check_eq(buf.get_length(), 3, "precondition")

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(buf.get_length(), 0, "buffered messages must not survive the reset")


async def test_pending_buffer_can_be_kept():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_reset.drop_pending_buffer": False})
    s = f.seed_session(SID, turns=1)
    buf = f.ctx.get_buffer(s)
    buf.add("pending")
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(buf.get_length(), 1, "opt-out keeps the buffer")


# ── locking / cooldown ───────────────────────────────────────────────────────


async def test_concurrent_resets_are_serialized():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=4)

    results = await asyncio.gather(
        f.plugin._reset(SID, force_summary=False, reason="t1"),
        f.plugin._reset(SID, force_summary=False, reason="t2"),
    )
    statuses = sorted(r[0] for r in results)
    check_eq(statuses, ["empty", "ok"],
             "the second reset must observe the already-cleared session")


async def test_tool_path_has_a_cooldown_but_commands_do_not():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_llm_tool.allow_llm_reboot": True})
    f.seed_session(SID, turns=2)
    first = await f.plugin._reset(SID, force_summary=False, reason="tool", apply_cooldown=True)
    check_eq(first[0], "ok", "first tool reset runs")

    f.seed_session(SID, turns=2)  # the user talks again
    second = await f.plugin._reset(SID, force_summary=False, reason="tool", apply_cooldown=True)
    check_eq(second[0], "cooldown", "second tool reset inside the window is ignored")

    f.seed_session(SID, turns=2)
    third = await f.plugin._reset(SID, force_summary=False, reason="command")
    check_eq(third[0], "ok", "explicit commands are never rate limited")


# ── store hygiene on external clears ─────────────────────────────────────────


async def test_session_deleted_event_drops_the_stored_summary():
    f = await make_fixture({})
    f.seed_session(SID, turns=2)
    f.plugin._store_set(SID, "old summary")
    check_eq(f.plugin._store.get(SID), "old summary", "precondition")

    f.sm.delete_session(SID)
    await f.settle()
    check_eq(f.plugin._store.get(SID), "", "store entry must be dropped")


async def test_empty_memory_write_drops_the_stored_summary():
    f = await make_fixture({})
    s = f.seed_session(SID, turns=2)
    f.plugin._store_set(SID, "old summary")
    f.sm.write_memory(s, [])
    await f.settle()
    check_eq(f.plugin._store.get(SID), "", "external clear must drop the store entry")


async def test_non_empty_memory_write_keeps_the_store():
    """Our own summary write publishes session_memory_written too - it must not
    wipe the store it just wrote."""
    f = await make_fixture({})
    s = f.seed_session(SID, turns=2)
    f.plugin._store_set(SID, "summary")
    f.sm.write_memory(s, [[{"role": "user", "content": "[前情摘要|系统注入] x\ny"}]])
    await f.settle()
    check_eq(f.plugin._store.get(SID), "summary", "non-empty write must be ignored")




# ── lifecycle ────────────────────────────────────────────────────────────────


async def test_terminate_is_reentrant():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.plugin.terminate()
    await f.plugin.terminate()  # must not raise
    check_eq(f.plugin._async_tasks, {})
    check_eq(f.plugin._locks, {})
    check_eq(f.plugin._own_writes, {})
    check_eq(f.plugin._command_map, {})


async def test_external_clear_cancels_a_pending_async_summary():
    from harness import FakeLLMClient

    llm = FakeLLMClient(["LATE"], delay=0.3)
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "async"},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    task = f.plugin._async_tasks.get(SID)
    check(task is not None, "a summary task must be pending")

    f.sm.write_memory(SID, [])  # somebody else clears the session
    await f.wait_for(lambda: task.done(), timeout=2.0)
    check(task.done(), "the stale summary task must be cancelled")
    check(task.cancelled(), "cancelled, not completed")
    check_eq(f.plugin._store.get(SID), "", "and the store entry is dropped")


async def test_missing_session_manager_disables_the_plugin_gracefully():
    from harness import FakeCtx, default_cfg, plugin_class

    ctx = FakeCtx(None)
    plugin = plugin_class()(ctx, default_cfg())
    await plugin.initialize()
    check(plugin.session_mgr is None, "plugin stays disabled")
    await plugin.handle_command(make_message_event("/reboot"))  # no-op, no crash
    result = await plugin.reset_context(make_batch_event(SID))
    check("失败" in result, f"the tool must report the failure, got {result!r}")
    await plugin.terminate()


async def test_session_manager_missing_methods_disables_the_plugin():
    from harness import FakeCtx, default_cfg, plugin_class

    class HalfManager:
        def write_memory(self, sid, chunks):
            pass

    ctx = FakeCtx(HalfManager())
    plugin = plugin_class()(ctx, default_cfg())
    await plugin.initialize()
    check(plugin.session_mgr is None, "an incomplete SessionManager must disable the plugin")




async def test_buffer_is_dropped_at_write_time_not_before_the_summary_call():
    """Dropping the buffer before an awaited summary call would leave a window in
    which a debounce flush re-fills it."""
    from harness import FakeLLMClient

    llm = FakeLLMClient(["S"], delay=0.15)
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=2)
    buf = f.ctx.get_buffer(SID)
    buf.add("before")

    task = asyncio.ensure_future(
        f.plugin._reset(SID, force_summary=True, reason="t")
    )
    await asyncio.sleep(0.05)          # inside the summary await
    check(buf.get_length() >= 1, "still buffered mid-reset (precondition)")
    buf.add("during")
    await task

    check_eq(buf.get_length(), 0,
             "messages buffered during the summary call must be dropped too")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
