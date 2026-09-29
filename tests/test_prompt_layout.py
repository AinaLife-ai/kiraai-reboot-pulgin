"""Prompt layout / prefix-cache guards.

The whole point of putting the summary at the HEAD of the session memory is that
the resulting message list keeps a long, stable prefix: everything up to and
including the last completed turn never moves, and only the tail grows. These
tests assemble a real LLMRequest exactly like core/message_manager.py does and
assert the resulting layout, so a change that silently moves the summary (and
therefore invalidates the provider prefix cache every turn) fails loudly.
"""

from harness import (
    bootstrap,
    check,
    check_eq,
    make_fixture,
    make_message_event,
)
from harness import make_batch_event

SID = "qq:dm:10001"


def build_request(sm, sid=SID, user_text="hello"):
    """Mirror of MessageProcessor.handle_im_batch_message's request construction."""
    bootstrap()
    from core.provider import LLMRequest
    from core.prompt_manager import Prompt

    req = LLMRequest(messages=sm.fetch_memory(sid))
    # system prompt: a stable part plus the blocks the framework relocates
    req.system_prompt.extend([
        Prompt("STABLE-SYSTEM", name="format"),
        Prompt("TIME-NOW", name="time"),
        Prompt("SESSIONS", name="sessions"),
        Prompt("LONG-TERM-MEMORY", name="memory"),
    ])
    req.user_prompt.append(Prompt(user_text, name="message"))
    req.assemble_prompt(dynamic_position="latest_user", memory_position="latest_user")
    return req


def layout(req):
    return [(m.role, str(m.content)) for m in req.messages]


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ── where does the summary end up? ───────────────────────────────────────────


async def test_summary_is_the_first_message_after_the_system_prompt():
    from harness import FakeLLMClient

    f = await make_fixture({"section_command.enable_resum_command": True,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=FakeLLMClient(["SUMMARY"]))
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/resume"))
    await f.settle()

    req = build_request(f.sm)
    rows = layout(req)
    check_eq(rows[0][0], "system", "the static system prompt stays first")
    check("STABLE-SYSTEM" in rows[0][1], "and it is the stable part")
    check_eq(rows[1][0], "user", "the summary is a user message")
    check(rows[1][1].startswith("[前情摘要|系统注入]"),
          f"the summary must sit directly after the system prompt, got {rows[1][1][:40]!r}")
    check("SUMMARY" in rows[1][1], "with its content")
    check("STABLE-SYSTEM" not in rows[1][1], "and no relocated block leaked in front of it")


async def test_relocated_dynamic_blocks_go_after_the_whole_history():
    from harness import FakeLLMClient

    f = await make_fixture({"section_command.enable_resum_command": True,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=FakeLLMClient(["SUMMARY"]))
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/resume"))
    await f.settle()

    req = build_request(f.sm)
    rows = layout(req)
    joined = "\n".join(content for _, content in rows)
    check("TIME-NOW" in rows[-1][1], "time is relocated into the latest user message")
    check("SESSIONS" in rows[-1][1], "so are the sessions")
    check("LONG-TERM-MEMORY" in rows[-1][1], "and the memory block")
    check("system_reminder" in rows[-1][1], "wrapped in a reminder")
    check(joined.index("[前情摘要|系统注入]") < joined.index("TIME-NOW"),
          "the summary must come before the volatile blocks")
    check("STABLE-SYSTEM" not in rows[-1][1], "the stable part is not relocated")


# ── prefix stability across turns ────────────────────────────────────────────


async def test_prefix_stays_stable_while_the_conversation_grows():
    from harness import FakeLLMClient

    f = await make_fixture({"section_command.enable_resum_command": True,
                            "section_summary.enable_summary": True,
                            "section_summary.summarize_mode": "sync"},
                           llm=FakeLLMClient(["SUMMARY"]))
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/resume"))
    await f.settle()

    first = layout(build_request(f.sm, user_text="turn-A"))
    before = common_prefix_len(first, first)

    # the framework appends one chunk per completed turn
    f.sm.update_memory(SID, [{"role": "user", "content": "turn-B"},
                             {"role": "assistant", "content": "reply-B"}])
    second = layout(build_request(f.sm, user_text="turn-C"))

    # everything except the freshly appended turn + the new user prompt is reused
    shared = common_prefix_len(first, second)
    check_eq(shared, before - 1,
             f"only the trailing user prompt may differ; prefix {shared} of {before}")
    check(shared >= len(first) - 1, "the previous request is a prefix of the next one")
    check_eq(second[1][1], first[1][1], "the summary itself does not move or change")
    check_eq(second[:len(first) - 1], first[:len(first) - 1],
             "system + summary + all completed turns are byte identical")

    # ... and it keeps holding after another turn
    f.sm.update_memory(SID, [{"role": "user", "content": "turn-D"},
                             {"role": "assistant", "content": "reply-D"}])
    third = layout(build_request(f.sm, user_text="turn-E"))
    check_eq(common_prefix_len(second, third), len(second) - 1,
             "stable prefix is preserved on every subsequent turn")


async def test_a_plain_reset_leaves_no_summary_and_no_extra_message():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()

    f.sm.update_memory(SID, [{"role": "user", "content": "fresh"},
                             {"role": "assistant", "content": "start"}])
    req = build_request(f.sm)
    rows = layout(req)
    check_eq([r[0] for r in rows], ["system", "user", "assistant", "user"],
             f"system + one turn + user prompt, got {rows}")
    check_eq(rows[1][1], "fresh", "the new turn follows the system prompt directly")
    check(not any(r[1].startswith("[前情摘要") for r in rows),
          "a plain reset must not leave a summary behind")


# ── the bridge must reproduce exactly the same layout ────────────────────────


async def test_bridge_reproduces_the_stored_head_layout():
    from harness import make_llm_request

    f = await make_fixture({"section_summary.keep_summary_alive": True})
    f.seed_session(SID, turns=2)

    # (a) head present in memory -> index 1 after assembly
    from reboot_plugin.summarizer import build_summary_chunk

    f.sm.write_memory(SID, [build_summary_chunk("STORED")] + f.sm.read_memory(SID))
    with_head = layout(build_request(f.sm))

    # (b) head evicted, bridged in for the request
    chunks = f.sm.read_memory(SID)
    f.sm.write_memory(SID, chunks[1:])
    f.plugin._store_set(SID, "STORED")
    req = make_llm_request(messages=[m for m in f.sm.fetch_memory(SID)])
    await f.plugin.on_llm_request(make_batch_event(SID), req, None)
    from core.prompt_manager import Prompt

    req.system_prompt.extend([Prompt("STABLE-SYSTEM", name="format")])
    req.user_prompt.append(Prompt("hello", name="message"))
    req.assemble_prompt("latest_user", "latest_user")
    bridged = layout(req)

    check_eq(bridged[1][1], with_head[1][1],
             "the bridged request must produce the same summary message")
    check_eq(bridged[1][0], "user", "in the same role/position")


async def test_bridge_does_not_persist_into_memory():
    from harness import make_llm_request

    f = await make_fixture({"section_summary.keep_summary_alive": True})
    f.seed_session(SID, turns=1)
    f.plugin._store_set(SID, "STORED")
    before = f.sm.read_memory(SID)

    await f.plugin.on_llm_request(make_batch_event(SID),
                                  make_llm_request(messages=[{"role": "user", "content": "x"}]),
                                  None)
    check_eq(f.sm.read_memory(SID), before, "the bridge must never write to memory")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
