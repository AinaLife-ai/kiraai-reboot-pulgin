"""Release guard: the manifest, the schema and the changelog must stay in sync."""

import json
import re

from harness import _STATE, bootstrap, check, check_eq

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
CHANGELOG_RE = re.compile(r"^### v(\d+\.\d+\.\d+)\s*$", re.MULTILINE)


def plugin_dir():
    bootstrap()
    return _STATE["plugin_dir"]


def manifest():
    return json.loads((plugin_dir() / "manifest.json").read_text(encoding="utf-8"))


def readme():
    return (plugin_dir() / "README.md").read_text(encoding="utf-8")


def as_tuple(version: str):
    return tuple(int(part) for part in version.split("."))


# ── manifest ─────────────────────────────────────────────────────────────────


async def test_manifest_version_is_semver():
    version = manifest()["version"]
    check(VERSION_RE.match(version) is not None, f"'{version}' is not semver")


async def test_plugin_id_is_unchanged():
    """config path is data/config/plugins/<plugin_id>.json: changing it would
    silently reset every existing installation."""
    check_eq(manifest()["plugin_id"], "reboot_plugin", "plugin_id is frozen")


async def test_manifest_declares_the_framework_requirement():
    core_version = manifest().get("core_version")
    check(bool(core_version), "core_version is required by the 2.0 layout")
    from packaging.specifiers import SpecifierSet

    SpecifierSet(core_version)  # raises on a malformed specifier
    for feature_version in ("2.33",):
        check(feature_version in core_version,
              "sections/model_select/locales need core >= 2.33")


async def test_manifest_metadata_is_present():
    data = manifest()
    for key in ("display_name", "description", "author", "repo", "tags", "locales", "icon"):
        check(data.get(key), f"manifest.{key} is missing")
    check("znq19" in data["repo"], "repo must point at the maintained fork")
    check("zh" in data["locales"], "a Chinese display name is expected")


# ── schema ───────────────────────────────────────────────────────────────────


async def test_schema_is_valid_json_with_sections():
    schema = json.loads((plugin_dir() / "schema.json").read_text(encoding="utf-8"))
    check(len(schema) >= 5, "expected at least five sections")
    for key, value in schema.items():
        check_eq(value.get("type"), "section", f"{key} must be a section")
        check(isinstance(value.get("fields"), dict), f"{key} must declare fields")


# ── changelog ────────────────────────────────────────────────────────────────


async def test_readme_documents_the_current_version():
    version = manifest()["version"]
    versions = CHANGELOG_RE.findall(readme())
    check(bool(versions), "README must have a '### vX.Y.Z' changelog section")
    check(version in versions, f"README has no section for v{version} (found {versions})")


async def test_changelog_is_ordered_newest_first():
    versions = CHANGELOG_RE.findall(readme())
    ordered = [as_tuple(v) for v in versions]
    check(ordered == sorted(ordered, reverse=True),
          f"changelog must be newest first, got {versions}")
    check_eq(versions[0], manifest()["version"], "the top section is the current version")


async def test_readme_documents_both_commands_and_the_switch():
    text = readme()
    for needle in ("/reboot", "/resume", "/resum", "memory_only", "delete_session"):
        check(needle in text, f"README must mention {needle}")




# ── consistency guards ───────────────────────────────────────────────────────


async def test_no_dead_config_items():
    """Every declared setting must be referenced by the implementation source."""
    schema = json.loads((plugin_dir() / "schema.json").read_text(encoding="utf-8"))
    source = "\n".join(
        (plugin_dir() / name).read_text(encoding="utf-8")
        for name in ("main.py", "summarizer.py", "compat.py")
    )
    dead = [
        f"{section}.{key}"
        for section, body in schema.items()
        for key in body["fields"]
        if key not in source
    ]
    check_eq(dead, [], "declared but never read")


async def test_readme_documents_every_config_item():
    schema = json.loads((plugin_dir() / "schema.json").read_text(encoding="utf-8"))
    text = readme()
    missing = [
        f"{key} ({item['name']})"
        for section, body in schema.items()
        for key, item in body["fields"].items()
        if item["name"] not in text
    ]
    check_eq(missing, [], "README is out of date")


async def test_every_config_item_has_a_hint():
    schema = json.loads((plugin_dir() / "schema.json").read_text(encoding="utf-8"))
    missing = [
        f"{section}.{key}"
        for section, body in schema.items()
        for key, item in body["fields"].items()
        if not item.get("hint")
    ]
    check_eq(missing, [], "user facing settings need an explanation")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
