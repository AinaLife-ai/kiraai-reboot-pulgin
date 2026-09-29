"""Interop with Auto Delete Session: shared summary head + command yielding."""

import logging
import sys
import types

from harness import bootstrap, check, check_eq, default_cfg, make_fixture

ADS_ID = "auto_delete_session"
SID = "qq:dm:10001"


class FakePluginMgr:
    def __init__(self, instances=None, configs=None, disabled=()):
        self.plugin_instances = dict(instances or {})
        self.plugin_configs = dict(configs or {})
        self._disabled = set(disabled)

    def has_plugin(self, plugin_id):
        return plugin_id in self.plugin_instances or plugin_id in self.plugin_configs

    def is_plugin_enabled(self, plugin_id):
        return plugin_id not in self._disabled

    def get_plugin_inst(self, plugin_id):
        return self.plugin_instances.get(plugin_id)


def make_ads_instance(marker=None, commands=None, module_suffix="1"):
    """Build a stand-in ADS instance whose class module exposes SUMMARY_MARKER."""
    module_name = f"fake_ads_module_{module_suffix}"
    module = types.ModuleType(module_name)
    if marker is not None:
        module.SUMMARY_MARKER = marker

    class FakeADS:
        def __init__(self):
            self.reset_commands = list(commands or ["/resum"])

    FakeADS.__module__ = module_name
    module.FakeADS = FakeADS
    sys.modules[module_name] = module
    return FakeADS()


class capture_logs:
    def __enter__(self):
        bootstrap()
        from core.plugin import logger as plugin_logger

        self.records = []
        handler = logging.Handler()
        handler.emit = lambda record: self.records.append(record.getMessage())
        self.logger = logging.getLogger(plugin_logger.name)
        self.logger.addHandler(handler)
        # attach to the root too so a logger passed explicitly by a test is captured
        self.logger.propagate = True
        self._handler = handler
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self._handler)
        return False

    def joined(self):
        return "\n".join(self.records)


# ── the marker contract ──────────────────────────────────────────────────────


async def test_summary_marker_is_byte_identical_to_ads():
    """This string is a cross-plugin contract: ADS recognises our summary head by
    prefix. If it ever drifts the two plugins start fighting over the head."""
    bootstrap()
    from reboot_plugin.summarizer import SUMMARY_MARKER

    check_eq(SUMMARY_MARKER, "[前情摘要|系统注入]",
             "the marker must match Auto Delete Session v2.1.4 exactly")


# ── plugin discovery ─────────────────────────────────────────────────────────


async def test_find_ads_plugin_exact_and_fuzzy():
    from reboot_plugin.compat import find_ads_plugin

    check_eq(find_ads_plugin(None), None, "no manager -> None")
    check_eq(find_ads_plugin(FakePluginMgr({ADS_ID: object()})), ADS_ID, "exact id")
    check_eq(find_ads_plugin(FakePluginMgr({"KiraAI_auto_delete_session_plugin": object()})),
             "KiraAI_auto_delete_session_plugin", "known alias")
    check_eq(find_ads_plugin(FakePluginMgr({"Auto-Delete-Session-main": object()})),
             "Auto-Delete-Session-main", "normalized fuzzy scan")
    check_eq(find_ads_plugin(FakePluginMgr({"unrelated": object()})), None, "unrelated plugin")
    check_eq(find_ads_plugin(FakePluginMgr({})), None, "empty manager")


async def test_is_plugin_active_honours_the_enabled_flag():
    from reboot_plugin.compat import is_plugin_active

    mgr = FakePluginMgr({ADS_ID: object()}, disabled=[ADS_ID])
    check_eq(is_plugin_active(mgr, ADS_ID), False, "disabled plugin is not active")
    check_eq(is_plugin_active(FakePluginMgr({ADS_ID: object()}), ADS_ID), True, "enabled plugin")
    check_eq(is_plugin_active(FakePluginMgr({}), ADS_ID), False, "not installed")


async def test_read_reset_commands_from_instance_and_config():
    from reboot_plugin.compat import read_ads_reset_commands

    mgr = FakePluginMgr({ADS_ID: make_ads_instance(commands=["/resum", "/reset2"])})
    check_eq(read_ads_reset_commands(mgr, ADS_ID), ["/resum", "/reset2"], "from instance")

    cfg_mgr = FakePluginMgr({}, {ADS_ID: {"section_command": {"reset_commands": ["/resum"]}}})
    check_eq(read_ads_reset_commands(cfg_mgr, ADS_ID), ["/resum"], "from section config")

    legacy_mgr = FakePluginMgr({}, {ADS_ID: {"reset_commands": "/resum, /old"}})
    check_eq(read_ads_reset_commands(legacy_mgr, ADS_ID), ["/resum", "/old"],
             "legacy flat config (comma separated)")

    check_eq(read_ads_reset_commands(FakePluginMgr({}), ADS_ID), [], "unknown plugin")


async def test_read_summary_marker_from_module_and_class():
    from reboot_plugin.compat import read_ads_summary_marker

    mgr = FakePluginMgr({ADS_ID: make_ads_instance(marker="[M]", module_suffix="2")})
    check_eq(read_ads_summary_marker(mgr, ADS_ID), "[M]", "read from the module")

    check_eq(read_ads_summary_marker(FakePluginMgr({ADS_ID: make_ads_instance(module_suffix="3")}),
                                     ADS_ID),
             None, "unknown marker -> None")
    check_eq(read_ads_summary_marker(FakePluginMgr({}), ADS_ID), None, "no instance -> None")


async def test_verify_marker_reports_a_mismatch():
    from reboot_plugin.compat import verify_ads_marker

    same = FakePluginMgr({ADS_ID: make_ads_instance(marker="[前情摘要|系统注入]",
                                                    module_suffix="4")})
    with capture_logs() as logs:
        check_eq(verify_ads_marker(same, ADS_ID, "[前情摘要|系统注入]", logs.logger),
                 True, "matching marker")
    check_eq(logs.records, [], "no warning when the contract holds")

    other = FakePluginMgr({ADS_ID: make_ads_instance(marker="[CHANGED]", module_suffix="5")})
    with capture_logs() as logs:
        check_eq(verify_ads_marker(other, ADS_ID, "[前情摘要|系统注入]", logs.logger),
                 False, "mismatch detected")
    check("摘要标记已变更" in logs.joined(), f"a warning is required, got {logs.records}")

    unknown = FakePluginMgr({ADS_ID: make_ads_instance(module_suffix="6")})
    with capture_logs() as logs:
        check_eq(verify_ads_marker(unknown, ADS_ID, "[前情摘要|系统注入]", logs.logger),
                 True, "unreadable marker is not an error")
    check_eq(logs.records, [], "no warning when it cannot be checked")


# ── command yielding ─────────────────────────────────────────────────────────


async def test_yield_overlapping_commands():
    from reboot_plugin.compat import yield_overlapping_commands

    with capture_logs() as logs:
        kept = yield_overlapping_commands(["/reboot", "/resum", "/RESUM"], ["/resum"], logs.logger)
    check_eq(kept, ["/reboot"], "overlapping keywords are dropped (case insensitive)")
    # /Resume (reboot's keyword) must survive an ADS /resum list
    check_eq(yield_overlapping_commands(["/Resume"], ["/resum"]), ["/Resume"],
             "the extra 'e' must keep them apart")
    check("让位" in logs.joined(), "the user must be told")

    check_eq(yield_overlapping_commands(["/reboot"], ["/resum"]), ["/reboot"], "no overlap")
    check_eq(yield_overlapping_commands([], ["/resum"]), [], "empty input")
    check_eq(yield_overlapping_commands(["/a", "/a"], []), ["/a"], "deduplicated")


# ── integration through the plugin ───────────────────────────────────────────


ADS_CFG = {
    "section_command.enable_reboot_command": True,
    "section_command.enable_resum_command": True,
    "section_compat.ads_interop": True,
}


async def _fixture_with_ads(mgr):
    f = await make_fixture(ADS_CFG, plugin_mgr=mgr)
    return f


async def test_default_keywords_do_not_collide():
    cfg = default_cfg()
    reboot_cmds = [c.lower() for c in cfg["section_command"]["reboot_commands"]]
    resum_cmds = [c.lower() for c in cfg["section_command"]["resum_commands"]]
    ads_cmds = {"/resum"}
    check(not (set(reboot_cmds) & set(resum_cmds)), "the two own lists must not overlap")
    check(not (set(reboot_cmds) & ads_cmds), "/reboot must not clash with ADS")
    check(not (set(resum_cmds) & ads_cmds), "/resume must not clash with ADS")


async def test_overlapping_keywords_yield_to_ads():
    mgr = FakePluginMgr({ADS_ID: make_ads_instance(marker="[前情摘要|系统注入]",
                                                   commands=["/resum"],
                                                   module_suffix="7")},
                        {ADS_ID: {}})
    with capture_logs() as logs:
        f = await make_fixture({**ADS_CFG,
                                "section_command.resum_commands": ["/resume", "/resum"]},
                               plugin_mgr=mgr)
    check_eq(f.plugin.resum_commands, ["/resume"], "the ADS keyword is removed from our side")
    check("/resume" in f.plugin._command_map, "our own keyword still works")
    check("/resum" not in f.plugin._command_map, "ADS keeps its keyword")
    check("让位" in logs.joined(), "a warning must be logged")


async def test_no_yield_when_ads_is_installed_but_disabled():
    mgr = FakePluginMgr({ADS_ID: make_ads_instance(commands=["/resum"], module_suffix="8")},
                        {ADS_ID: {}}, disabled=[ADS_ID])
    f = await make_fixture({**ADS_CFG, "section_command.resum_commands": ["/resume", "/resum"]},
                           plugin_mgr=mgr)
    check_eq(f.plugin.resum_commands, ["/resume", "/resum"],
             "a disabled ADS must not take anything away")


async def test_interop_can_be_turned_off():
    mgr = FakePluginMgr({ADS_ID: make_ads_instance(commands=["/resum"], module_suffix="9")},
                        {ADS_ID: {}})
    f = await make_fixture({**ADS_CFG,
                            "section_compat.ads_interop": False,
                            "section_command.resum_commands": ["/resume", "/resum"]},
                           plugin_mgr=mgr)
    check_eq(f.plugin.resum_commands, ["/resume", "/resum"], "isolation mode keeps everything")


async def test_marker_mismatch_is_surfaced_but_not_fatal():
    mgr = FakePluginMgr({ADS_ID: make_ads_instance(marker="[CHANGED]", module_suffix="10")},
                        {ADS_ID: {}})
    with capture_logs() as logs:
        f = await make_fixture(ADS_CFG, plugin_mgr=mgr)
    check_eq(f.plugin._ads_marker_ok, False, "mismatch recorded")
    check("摘要标记已变更" in logs.joined(), "warned")
    check(f.plugin.session_mgr is not None, "the plugin still works")


async def test_other_memory_plugins_are_only_reported():
    from reboot_plugin.compat import report_memory_plugins

    mgr = FakePluginMgr({"ksm_plugin": object(), "context_condensation": object()})
    with capture_logs() as logs:
        found = report_memory_plugins(mgr, True, logs.logger)
    check(any("KSM" in item for item in found), f"KSM must be reported, got {found}")
    check(any("CCS" in item for item in found), f"CCS must be reported, got {found}")
    check("检测到同样会改写会话记忆的插件" in logs.joined(), "logged")

    with capture_logs() as logs:
        check_eq(report_memory_plugins(mgr, False, logs.logger), [], "switch off -> silent")
    check_eq(logs.records, [], "no log when disabled")




async def test_short_plugin_ids_do_not_produce_bogus_conflict_reports():
    """A very short plugin id must not substring-match every known hint."""
    from reboot_plugin.compat import report_memory_plugins

    check_eq(report_memory_plugins(FakePluginMgr({"a": object(), "sm": object()}), True, None),
             [], "no false positives from short ids")
    check_eq(report_memory_plugins(FakePluginMgr({"cc": object()}), True, None),
             [], "nor from two-letter ids")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
