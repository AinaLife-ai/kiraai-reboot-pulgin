"""Command interception: matching, interception guarantees and permissions."""

from harness import check, check_eq, make_fixture, make_message_event

SID = "qq:dm:10001"

BOTH = {
    "section_command.enable_reboot_command": True,
    "section_command.enable_resum_command": True,
}


def assert_intercepted(event):
    check_eq(event.process_strategy, "discard",
             "the command message must not be buffered or forwarded to the LLM")
    check(event.is_stopped, "the command message must stop further handlers")


# ── matching ─────────────────────────────────────────────────────────────────


async def test_commands_are_disabled_by_default():
    f = await make_fixture({})
    f.seed_session(SID, turns=2)
    event = make_message_event("/reboot")
    await f.plugin.handle_command(event)
    await f.settle()
    check_eq(f.ctx.message_processor.sent, [], "nothing happens while disabled")
    check_eq(event.process_strategy, "discard", "untouched event keeps the default strategy")
    check_eq(len(f.memory(SID)), 2, "memory untouched")


async def test_reboot_command_is_intercepted_and_clears_the_context():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=3)
    event = make_message_event("/reboot")
    await f.plugin.handle_command(event)
    await f.settle()
    assert_intercepted(event)
    check_eq(f.memory(SID), [], "context cleared")
    check("已重置" in f.last_reply(), "the user is told")


async def test_case_insensitive_and_whitespace_tolerant():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("  /REBOOT  "))
    await f.settle()
    check_eq(f.memory(SID), [], "trim + lowercase must match")


async def test_substring_does_not_trigger():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1)
    for text in ("please /reboot now", "/reboot2", "/reboots", "reboot"):
        await f.plugin.handle_command(make_message_event(text))
    await f.settle()
    check_eq(len(f.memory(SID)), 1, f"only an exact match may trigger, got {f.memory(SID)}")
    check_eq(f.ctx.message_processor.sent, [], "no replies either")


async def test_multiple_keywords_and_custom_prefix():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_commands": ["/reboot", "重开", "/rs"]})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("重开"))
    await f.settle()
    check_eq(f.memory(SID), [], "'重开' must trigger")


async def test_resum_command_uses_its_own_keyword_list():
    f = await make_fixture({**BOTH, "section_command.resum_commands": ["/resume", "/newstart"]})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/newstart"))
    await f.settle()
    check("已重置" in f.last_reply(), "custom resum keyword must work")


async def test_a_keyword_shared_by_both_lists_prefers_the_summary_variant():
    f = await make_fixture({**BOTH,
                            "section_command.reboot_commands": ["/same"],
                            "section_command.resum_commands": ["/same"]})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/same"))
    await f.settle()
    check_eq(f.plugin._command_map.get("/same"), True, "resum semantics win")


async def test_commands_are_ignored_for_non_text_messages():
    from harness import bootstrap
    bootstrap()
    from core.chat.message_utils import KiraMessageEvent, KiraIMMessage
    from core.chat import MessageChain, User
    from core.chat.message_elements import Image
    from core.adapter.adapter_info import AdapterInfo

    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1)
    adapter = AdapterInfo(enabled=True, adapter_id="qq", name="qq", platform="QQ")
    event = KiraMessageEvent(
        message_types=["dm"], timestamp=0, adapter=adapter,
        message=KiraIMMessage(message_id="m", self_id="999",
                              chain=MessageChain([Image("http://x/y.png")]),
                              timestamp=0, sender=User(user_id="10001", nickname="n")),
    )
    await f.plugin.handle_command(event)
    await f.settle()
    check_eq(len(f.memory(SID)), 1, "an image-only message must not match")


# ── permissions ──────────────────────────────────────────────────────────────


async def test_permission_off_allows_everyone():
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1, )
    await f.plugin.handle_command(make_message_event("/reboot", user_id="424242"))
    await f.settle()
    check_eq(f.memory(SID), [], "allowed")


async def test_permission_on_with_empty_whitelist_denies_everyone():
    """Fail-closed: an empty allow-list must not silently mean 'everyone'."""
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_enable_permission": True})
    f.seed_session(SID, turns=2)
    event = make_message_event("/reboot", user_id="10001")
    await f.plugin.handle_command(event)
    await f.settle()
    check_eq(len(f.memory(SID)), 2, "context untouched")
    check("权限不足" in f.last_reply(), "denied with feedback")
    assert_intercepted(event)


async def test_permission_on_respects_the_whitelist():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_enable_permission": True,
                            "section_command.reboot_allowed_users": ["10001"]})
    f.seed_session(SID, turns=2)
    await f.plugin.handle_command(make_message_event("/reboot", user_id="99999"))
    await f.settle()
    check_eq(len(f.memory(SID)), 2, "outsider denied")
    check("权限不足" in f.last_reply(), "denied with feedback")

    await f.plugin.handle_command(make_message_event("/reboot", user_id="10001"))
    await f.settle()
    check_eq(f.memory(SID), [], "whitelisted user allowed")


async def test_permission_covers_both_commands():
    f = await make_fixture({**BOTH,
                            "section_command.reboot_enable_permission": True,
                            "section_command.reboot_allowed_users": ["10001"]})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/resume", user_id="7"))
    await f.settle()
    check("权限不足" in f.last_reply(), "/resume must obey the same permission")


# ── replies ──────────────────────────────────────────────────────────────────


async def test_reply_templates_are_used():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_success_message": "OK{summary}!",
                            "section_command.reboot_empty_message": "EMPTY!"})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.last_reply(), "OK!", "success template")

    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.last_reply(), "EMPTY!", "empty template")


async def test_reply_template_braces_do_not_break_formatting():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_success_message": 'JSON {"a":1} {summary}'})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/reboot"))
    await f.settle()
    check_eq(f.last_reply(), 'JSON {"a":1} ', "literal braces must survive")




async def test_notice_messages_cannot_trigger_a_reset():
    """Notices are synthesized by the bot/plugins (cross-session sends,
    publish_notice). Text that merely equals a keyword must not be able to reset a
    session from the outside."""
    f = await make_fixture({"section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=2)
    event = make_message_event("/reboot")
    event.message.is_notice = True
    check(event.is_notice, "precondition")

    await f.plugin.handle_command(event)
    await f.settle()

    check_eq(len(f.memory(SID)), 2, "a notice must not clear the context")
    check_eq(f.ctx.message_processor.sent, [], "and must not produce a reply")
    check(not event.is_stopped, "and must not be swallowed")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
