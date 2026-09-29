"""Summary generation, cumulative merging and the async write discipline."""

import asyncio

from harness import (
    FakeLLMClient,
    check,
    check_eq,
    make_fixture,
    make_message_event,
)

SID = "qq:dm:10001"

CMD = {
    "section_command.enable_reboot_command": True,
    "section_command.enable_resum_command": True,
}

MARKER = "[前情摘要|系统注入]"


def is_summary(message) -> bool:
    """Accepts either a memory chunk or a single message."""
    if isinstance(message, list):
        message = message[0] if message else {}
    return str(message.get("content", "")).startswith(MARKER)


def check_chunk_shape(f, sid=SID):
    """Structural invariant: memory is a list of chunks, each chunk a list of messages."""
    mem = f.memory(sid)
    check(isinstance(mem, list), "memory must be a list")
    for i, chunk in enumerate(mem):
        check(isinstance(chunk, list),
              f"chunk #{i} must be a list of messages, got {type(chunk).__name__}")
        for msg in chunk:
            check(isinstance(msg, dict), f"chunk #{i} holds a non-dict message")
    return mem


def flat_memory(f, sid=SID):
    """read_memory() returns chunks; flatten them for content assertions."""
    out = []
    for chunk in f.memory(sid):
        out.extend(chunk)
    return out


def head_text(f):
    """Full content of the summary head chunk (read_memory returns chunks)."""
    mem = f.memory(SID)
    if mem and is_summary(mem[0]):
        return str(mem[0][0].get("content", ""))
    return ""


# ── sync mode ────────────────────────────────────────────────────────────────


async def test_sync_summary_is_written_to_the_memory_head():
    llm = FakeLLMClient(["SUMMARY-A"])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=3)

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    mem = check_chunk_shape(f)
    check_eq(len(mem), 1, "memory must hold exactly the summary chunk")
    check(is_summary(mem[0]), "first message must be the summary head")
    check("SUMMARY-A" in head_text(f), "summary text must be stored")
    check_eq(f.plugin._store.get(SID), "SUMMARY-A", "store must mirror the head")
    check_eq(len(llm.prompts), 1, "one summarise call")
    check("user message 0" in llm.prompts[0], "the history must be the summary input")
    check_eq(f.meta(SID)["title"], "T", "metadata still preserved")


async def test_summary_disabled_makes_no_llm_call():
    llm = FakeLLMClient(["SHOULD-NOT-BE-USED"])
    f = await make_fixture(CMD, llm=llm)
    f.seed_session(SID, turns=3)

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    check_eq(f.memory(SID), [], "plain clear")
    check_eq(llm.prompts, [], "no summary must be requested")


async def test_resume_forces_summary_even_when_the_switch_is_off():
    llm = FakeLLMClient(["FORCED"])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": False,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=2)

    await f.plugin.handle_command(make_message_event("/resume"))
    await f.settle()

    check("FORCED" in head_text(f), "/resume must summarise regardless of the switch")
    check_eq(len(llm.prompts), 1, "one summarise call")


async def test_reboot_keeps_summary_off_when_the_switch_is_off():
    llm = FakeLLMClient(["NOPE"])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": False,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=2)

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    check_eq(f.memory(SID), [], "/reboot must stay a hard reset")
    check_eq(len(llm.prompts), 0, "no summarise call")


# ── cumulative ───────────────────────────────────────────────────────────────


async def test_cumulative_summary_survives_a_second_reset():
    llm = FakeLLMClient(["DELTA-1", "DELTA-2", "MERGED-2"])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)

    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.plugin._store.get(SID), "DELTA-1",
             "with no previous summary the delta is the first cumulative summary")

    # the user keeps talking, then resets again
    f.sm.update_memory(SID, [{"role": "user", "content": "later"},
                             {"role": "assistant", "content": "reply"}])
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    check_eq(f.plugin._store.get(SID), "MERGED-2", "second cumulative summary")
    merge_prompt = llm.prompts[-1]
    check("DELTA-1" in merge_prompt, "the previous summary must be the merge base")
    check("DELTA-2" in merge_prompt, "the fresh delta must be the merge delta")
    check("later" in llm.prompts[-2], "the raw history goes to the summarise call")


async def test_cumulative_disabled_drops_the_previous_summary():
    llm = FakeLLMClient(["FIRST", "SECOND"])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync",
                            "section_summary.cumulative_summary": False},
                           llm=llm)

    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.plugin._store.get(SID), "FIRST", "non cumulative still stores the head")

    f.sm.update_memory(SID, [{"role": "user", "content": "later"},
                             {"role": "assistant", "content": "reply"}])
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    # without cumulative mode the head is still adopted as the base, so the second
    # reset summarises the new history and then merges it onto the previous summary
    check("later" in llm.prompts[-2], "the new history must be summarised")
    check("FIRST" in llm.prompts[-1], "the previous head must be the merge base")


# ── failure handling ─────────────────────────────────────────────────────────


async def test_summary_failure_degrades_to_a_plain_clear():
    llm = FakeLLMClient([])
    llm.default_reply = ""
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=2)

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    check_eq(f.memory(SID), [], "memory cleared even when the summary fails")
    check_eq(f.plugin._store.get(SID), "", "nothing stored")
    check_eq(f.meta(SID)["title"], "T", "session metadata untouched")
    check("已重置" in f.last_reply(), "the user still gets a success reply")


async def test_llm_exception_never_breaks_the_reset():
    llm = FakeLLMClient(["x"], raise_on_call=RuntimeError("boom"))
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=2)

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.memory(SID), [], "reset still happened")


async def test_output_cap_falls_back_to_hard_truncation():
    """Self-compression first; when it returns nothing, truncate with an ellipsis."""
    llm = FakeLLMClient(["x" * 500, ""])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync",
                            "section_summary.summarize_max_output_chars": 50,
                            "section_summary.merge_timeout_sec": 1.0},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    stored = f.plugin._store.get(SID)
    check(len(stored) <= 51, f"summary must be capped, got {len(stored)}")
    check(stored.endswith("…"), "hard truncation marker expected")
    check_eq(len(llm.prompts), 2, "one summarise + one self-compress call")


async def test_output_cap_uses_the_self_compressed_result():
    llm = FakeLLMClient(["x" * 500, "SHORT"])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync",
                            "section_summary.summarize_max_output_chars": 50,
                            "section_summary.merge_timeout_sec": 1.0},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.plugin._store.get(SID), "SHORT", "self-compressed summary wins")


async def test_zero_output_cap_means_unlimited():
    llm = FakeLLMClient(["y" * 300])
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync",
                            "section_summary.summarize_max_output_chars": 0},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(len(f.plugin._store.get(SID)), 300, "0 disables the cap")
    check_eq(len(llm.prompts), 1, "no self-compress call needed")


async def test_llm_timeout_does_not_block_forever():
    llm = FakeLLMClient(["late"], delay=0.5)
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync",
                            "section_summary.summarize_timeout_sec": 0.05},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.memory(SID), [], "timeout degrades to a plain clear")


# ── async mode ───────────────────────────────────────────────────────────────


async def test_async_mode_replies_immediately_and_writes_the_summary_later():
    llm = FakeLLMClient(["ASYNC-SUM"], delay=0.05)
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "async"},
                           llm=llm)
    f.seed_session(SID, turns=2)

    await f.plugin.handle_command(make_message_event("/reboot"))
    check("后台生成" in f.last_reply(), "async mode must announce a pending summary")
    check_eq(f.memory(SID), [], "placeholder is empty when there is no previous summary")

    ok = await f.wait_for(lambda: "ASYNC-SUM" in head_text(f), timeout=2.0)
    check(ok, "the background task must write the summary head")
    check_eq(f.plugin._store.get(SID), "ASYNC-SUM", "store updated after the write")


async def test_async_mode_writes_the_previous_summary_as_a_placeholder():
    llm = FakeLLMClient(["OLD-SUM", "NEW-SUM", "MERGED-SUM"], delay=0.05)
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "async"},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/resume"))
    ok = await f.wait_for(lambda: "OLD-SUM" in head_text(f), timeout=2.0)
    check(ok, "first async summary")

    f.sm.update_memory(SID, [{"role": "user", "content": "later"},
                             {"role": "assistant", "content": "reply"}])
    await f.plugin.handle_command(make_message_event("/resume"))

    check("OLD-SUM" in head_text(f), "the old summary must be carried over immediately")
    ok = await f.wait_for(lambda: "MERGED-SUM" in head_text(f), timeout=2.0)
    check(ok, "the merged summary must replace the placeholder")
    merge_prompt = llm.prompts[-1]
    check("OLD-SUM" in merge_prompt and "NEW-SUM" in merge_prompt,
          "the merge must see both the old summary and the fresh delta")


async def test_async_summary_does_not_lose_a_turn_written_meanwhile():
    """The core concurrency guarantee: read_memory -> write_memory has no await."""
    llm = FakeLLMClient(["ASYNC-SUM"], delay=0.05)
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "async"},
                           llm=llm)
    f.seed_session(SID, turns=2)

    await f.plugin.handle_command(make_message_event("/reboot"))
    # the framework appends a turn while the summary is still being generated
    f.sm.update_memory(SID, [{"role": "user", "content": "meanwhile"},
                             {"role": "assistant", "content": "ok"}])

    ok = await f.wait_for(lambda: "ASYNC-SUM" in head_text(f), timeout=2.0)
    check(ok, "summary written")

    flat = flat_memory(f)
    blob = "\n".join(str(m.get("content", "")) for m in flat)
    check("meanwhile" in blob, "the concurrent turn must NOT be lost")
    check("ASYNC-SUM" in blob, "the summary must be present too")
    check(is_summary(f.memory(SID)[0]),
          "the summary must sit at the head of the first chunk")


async def test_reverse_awaited_write_ordering_would_lose_the_turn():
    """Reverse check: proves the previous test can actually catch the bug.

    Mimics a naive implementation (ADS style) that awaits between read_memory and
    write_memory. A turn appended during that await is overwritten by the stale
    snapshot - exactly what _apply_summary avoids by being synchronous.
    """
    f = await make_fixture({})
    s = f.seed_session(SID, turns=1)

    async def naive_apply():
        chunks = [list(c) for c in (f.sm.read_memory(s) or [])]
        await asyncio.sleep(0.02)              # <-- the forbidden suspension point
        chunks[0] = [{"role": "user", "content": MARKER + " x\nSUM"}] + list(chunks[0])
        f.sm.write_memory(s, chunks)

    task = asyncio.ensure_future(naive_apply())
    await asyncio.sleep(0.005)
    f.sm.update_memory(s, [{"role": "user", "content": "meanwhile"}])
    await task

    blob = "\n".join(str(m.get("content", "")) for m in flat_memory(f))
    check("meanwhile" not in blob,
          "the naive ordering is expected to lose the concurrent turn")


async def test_cancelling_terminate_cleans_up_background_tasks():
    llm = FakeLLMClient(["S"], delay=0.2)
    f = await make_fixture({**CMD,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "async"},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    check(SID in f.plugin._async_tasks, "task scheduled")
    await f.plugin.terminate()
    await f.settle()
    check_eq(f.plugin._async_tasks, {}, "terminate must cancel and clear tasks")


# ── summary head bridge ──────────────────────────────────────────────────────


async def test_bridge_reinjects_an_evicted_summary_head_for_one_request():
    from harness import make_llm_request, make_batch_event

    f = await make_fixture({"section_summary.keep_summary_alive": True})
    f.plugin._store_set(SID, "LONG-LOST-SUMMARY")

    req = make_llm_request(messages=[{"role": "user", "content": "hi"}])
    await f.plugin.on_llm_request(make_batch_event(SID), req, None)
    check_eq(len(req.messages), 2, "the bridge must prepend one message")
    check(str(req.messages[0].content).startswith(MARKER), "bridged head uses the marker")
    check_eq(f.memory(SID), [], "the bridge must not touch the stored memory")


async def test_bridge_is_skipped_when_the_head_already_has_a_summary():
    from harness import make_llm_request, make_batch_event

    f = await make_fixture({"section_summary.keep_summary_alive": True})
    f.plugin._store_set(SID, "STORED")
    req = make_llm_request(messages=[
        {"role": "user", "content": MARKER + " x\nHEAD"},
        {"role": "user", "content": "hi"},
    ])
    await f.plugin.on_llm_request(make_batch_event(SID), req, None)
    check_eq(len(req.messages), 2, "no double injection")


async def test_bridge_can_be_disabled():
    from harness import make_llm_request, make_batch_event

    f = await make_fixture({"section_summary.keep_summary_alive": False})
    f.plugin._store_set(SID, "STORED")
    req = make_llm_request(messages=[{"role": "user", "content": "hi"}])
    await f.plugin.on_llm_request(make_batch_event(SID), req, None)
    check_eq(len(req.messages), 1, "switch off -> no bridge")




async def test_store_is_capped_and_evicts_the_oldest():
    """One entry per session is only dropped on clear/delete, so without a cap the
    store file would grow forever on a long running bot."""
    import tempfile
    from pathlib import Path as _Path
    from reboot_plugin.summarizer import CumulativeSummaryStore, MAX_STORE_ENTRIES

    tmp = _Path(tempfile.mkdtemp(prefix="store_"))
    store = CumulativeSummaryStore(tmp / "s.json")
    total = MAX_STORE_ENTRIES + 25
    for i in range(total):
        store.set(f"sid-{i}", f"summary-{i}")
    check_eq(len(store._data), total, "nothing pruned before save")
    store.save()

    check_eq(len(store._data), MAX_STORE_ENTRIES, "capped after save")
    check("sid-%d" % (total - 1) in store._data, "the newest entry survives")
    check("sid-0" not in store._data, "the oldest entry is evicted")

    reloaded = CumulativeSummaryStore(tmp / "s.json")
    reloaded.load()
    check_eq(len(reloaded._data), MAX_STORE_ENTRIES, "the cap is persisted")
    check_eq(reloaded.get(f"sid-{total - 1}"), f"summary-{total - 1}", "content intact")




async def test_resume_on_an_already_summarized_session_is_idempotent():
    """A second /resume on a session whose memory is only the summary must not call
    the model again nor change the head text (that would churn the prompt prefix)."""
    from harness import make_message_event

    llm = FakeLLMClient(["SUM"])
    f = await make_fixture({"section_command.enable_resum_command": True,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=llm)
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/resume"))
    await f.settle()
    first = head_text(f)
    check("SUM" in first, "first summary written")
    check_eq(len(llm.prompts), 1, "one summarise call")

    await f.plugin.handle_command(make_message_event("/resume"))
    await f.settle()
    check_eq(head_text(f), first, "the head must be byte identical")
    check_eq(len(llm.prompts), 1, "no extra model call for an empty delta")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
