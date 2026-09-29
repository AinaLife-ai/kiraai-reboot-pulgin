"""Config surface: schema validity and migration from the 1.0 flat layout."""

from harness import bootstrap, check, check_eq, default_cfg, make_fixture, make_message_event

SID = "qq:dm:10001"


# ── schema ───────────────────────────────────────────────────────────────────


async def test_schema_parses_with_the_framework_parser():
    """Runs our schema.json through core.config.config_field, exactly like the
    plugin manager does on load."""
    bootstrap()
    import json
    from harness import _STATE
    from core.config.config_field import build_fields, SectionField

    raw = json.loads((_STATE["plugin_dir"] / "schema.json").read_text(encoding="utf-8"))
    fields = build_fields(raw)
    check(len(fields) >= 5, f"expected the 5 sections, got {len(fields)}")

    total = 0
    for field in fields:
        if isinstance(field, SectionField):
            check(bool(field.fields), f"section {field.key} must not be empty")
            total += len(field.fields)
    check_eq(total, 28, "the documented number of configuration items")


async def test_every_config_item_has_a_display_name_and_a_type():
    bootstrap()
    import json
    from harness import _STATE

    raw = json.loads((_STATE["plugin_dir"] / "schema.json").read_text(encoding="utf-8"))
    for section_key, section in raw.items():
        check_eq(section.get("type"), "section", f"{section_key} must be a section")
        check(bool(section.get("name")), f"{section_key} needs a name")
        for key, item in (section.get("fields") or {}).items():
            check(bool(item.get("type")), f"{section_key}.{key} needs a type")
            check(bool(item.get("name")), f"{section_key}.{key} needs a name")
            check("default" in item, f"{section_key}.{key} should declare a default")
            if item["type"] == "enum":
                check(item.get("default") in item.get("options", []),
                      f"{section_key}.{key}: default must be one of the options")


async def test_security_and_behaviour_defaults_are_as_designed():
    cfg = default_cfg()
    check_eq(cfg["section_reset"]["clear_mode"], "memory_only", "keep the session by default")
    check_eq(cfg["section_command"]["enable_reboot_command"], False, "commands off by default")
    check_eq(cfg["section_command"]["enable_resum_command"], False, "commands off by default")
    check_eq(cfg["section_summary"]["enable_summary"], False, "summary off by default")
    check_eq(cfg["section_summary"]["summarize_mode"], "async", "non blocking by default")
    check_eq(cfg["section_llm_tool"]["allow_llm_reboot"], True, "the tool stays on by default")
    check_eq(cfg["section_reset"]["drop_pending_buffer"], True, "drop the buffer by default")
    check_eq(cfg["section_compat"]["ads_interop"], True, "interop on by default")
    check_eq(cfg["section_command"]["reboot_commands"], ["/reboot"], "default keyword")
    check_eq(cfg["section_command"]["resum_commands"], ["/resume"], "default keyword")
    check_eq(cfg["section_command"]["reboot_allowed_users"], [], "empty whitelist")


# ── 1.0 -> 2.0 migration ─────────────────────────────────────────────────────


async def test_legacy_command_prefix_is_migrated():
    """The framework fills in section defaults before the plugin is constructed, so
    an upgrade must honour the old flat keys explicitly."""
    f = await make_fixture({"command_prefix": "/重开"})
    check_eq(f.plugin.reboot_commands, ["/重开"], "legacy prefix must win while the new key is default")
    check_eq(f.plugin._command_map, {}, "but the command stays disabled until enabled")


async def test_explicit_new_value_beats_the_legacy_one():
    f = await make_fixture({"command_prefix": "/legacy",
                            "section_command.reboot_commands": ["/new"],
                            "section_command.enable_reboot_command": True})
    check_eq(f.plugin.reboot_commands, ["/new"], "a user-set new value is authoritative")
    check_eq(f.plugin._command_map, {"/new": False}, "and it is the live keyword")


async def test_legacy_permission_settings_are_migrated():
    f = await make_fixture({"enable_permission": True, "allowed_users": ["1", "2"]})
    check_eq(f.plugin.enable_permission, True, "legacy switch honoured")
    check_eq(f.plugin.allowed_users, ["1", "2"], "legacy list honoured")


async def test_legacy_message_templates_are_migrated():
    f = await make_fixture({"success_message": "OLD-OK{summary}",
                            "permission_denied_message": "OLD-DENIED",
                            "error_message": "OLD-ERR {error}"})
    check_eq(f.plugin.success_message, "OLD-OK{summary}", "legacy success text")
    check_eq(f.plugin.permission_denied_message, "OLD-DENIED", "legacy denied text")
    check_eq(f.plugin.error_message, "OLD-ERR {error}", "legacy error text")


async def test_legacy_settings_are_actually_used_at_runtime():
    f = await make_fixture({"command_prefix": "/old",
                            "success_message": "LEGACY-OK",
                            "section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/old"))
    await f.settle()
    check_eq(f.memory(SID), [], "the legacy keyword works in the live command map")
    check_eq(f.last_reply(), "LEGACY-OK", "the legacy template is used")


async def test_legacy_allowed_users_accepts_multiline_strings():
    f = await make_fixture({"enable_permission": True, "allowed_users": "1\n2\n3"})
    check_eq(f.plugin.allowed_users, ["1", "2", "3"], "newline separated ids")


# ── value handling ───────────────────────────────────────────────────────────


async def test_zero_is_a_valid_limit_value():
    """0 means 'unlimited' for the char limits, so `or default` must not be used."""
    f = await make_fixture({"section_summary.summarize_max_input_chars": 0,
                            "section_summary.summarize_max_output_chars": 0})
    check_eq(f.plugin.summarize_max_input_chars, 0, "0 must survive")
    check_eq(f.plugin.summarize_max_output_chars, 0, "0 must survive")


async def test_invalid_enum_values_fall_back_to_the_safe_default():
    f = await make_fixture({"section_reset.clear_mode": "wat",
                            "section_summary.summarize_mode": "wat"})
    check_eq(f.plugin.clear_mode, "memory_only", "unknown clear mode is not destructive")
    check_eq(f.plugin.summarize_mode, "async", "unknown mode is non blocking")


async def test_command_lists_accept_strings_and_deduplicate():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_commands": "/a, /b, /a"})
    check_eq(f.plugin.reboot_commands, ["/a", "/b"], "comma separated string + dedup")
    check_eq(sorted(f.plugin._command_map), ["/a", "/b"], "both keywords live")


async def test_blank_command_list_falls_back_to_the_default():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_commands": []})
    check_eq(f.plugin.reboot_commands, ["/reboot"], "an empty list is not a valid config")




async def test_plugin_survives_without_a_data_directory():
    """PluginContext returns None when it cannot map the module to a plugin id;
    the reset must still work, only the cumulative store degrades."""
    from harness import FakeEventBus, FakeCtx, default_cfg, make_session_manager, plugin_class

    bus = FakeEventBus()
    sm = make_session_manager(event_bus=bus)
    ctx = FakeCtx(sm, data_dir=None, event_bus=bus)
    plugin = plugin_class()(ctx, default_cfg())
    ctx.get_plugin_data_dir = lambda: None  # simulate the unmappable module
    await plugin.initialize()
    check(plugin.session_mgr is not None, "the plugin must stay enabled")
    check_eq(plugin._store, None, "store degraded")

    plugin.session_mgr.get_session_info(SID)
    plugin.session_mgr.write_memory(SID, [[{"role": "user", "content": "hi"},
                                           {"role": "assistant", "content": "yo"}]])
    status, _ = await plugin._reset(SID, force_summary=False, reason="test")
    check_eq(status, "ok", "a reset still works without the store")
    check_eq(plugin.session_mgr.read_memory(SID), [], "context cleared")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
