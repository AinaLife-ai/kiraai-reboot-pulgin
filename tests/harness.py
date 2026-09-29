from __future__ import annotations

"""
Test harness for the reboot plugin.

The tests run against the REAL KiraAI framework (core.*), not against hand written
stubs, so the plugin is exercised with the same SessionManager, the same event
objects and the same schema parser it will meet in production. Only the pieces that
would need a running bot are faked: the plugin context, the LLM client, the message
sender and the event bus.

Environment:
    KIRA_CORE_PATH  path to the KiraAI checkout (containing the `core` package).
                    Falls back to a small list of well known locations.

Layout created under a temp sandbox:
    <sandbox>/data/...            -> framework data dir (cwd based)
    <sandbox>/pkg/reboot_plugin/  -> a copy of the plugin files
"""

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(os.environ.get("REBOOT_PLUGIN_ROOT") or Path(__file__).resolve().parent.parent).resolve()

# Framework locations tried when KIRA_CORE_PATH is not set.
FALLBACK_CORE_PATHS = (
    "/var/minis/shared/kira_fw",
    str(REPO_ROOT.parent / "KiraAI"),
    "/app",
    "/KiraAI",
)

_STATE: Dict[str, Any] = {"booted": False, "sandbox": None, "pkg_parent": None}

PLUGIN_FILES = ("__init__.py", "main.py", "summarizer.py", "compat.py",
                "manifest.json", "schema.json", "icon.png", "README.md")


class FrameworkNotFound(RuntimeError):
    pass


def _detect_core_path() -> str:
    candidates = []
    env = os.environ.get("KIRA_CORE_PATH")
    if env:
        candidates.append(env)
    candidates.extend(FALLBACK_CORE_PATHS)
    for candidate in candidates:
        if candidate and (Path(candidate) / "core" / "plugin" / "plugin_registry.py").is_file():
            return str(Path(candidate).resolve())
    raise FrameworkNotFound(
        "KiraAI framework not found. Set KIRA_CORE_PATH to the KiraAI checkout "
        "(the directory that contains the `core` package)."
    )


def bootstrap() -> Dict[str, Any]:
    """Prepare cwd + sys.path so `core.*` and `reboot_plugin` can be imported."""
    if _STATE["booted"]:
        return _STATE

    core_path = _detect_core_path()
    sandbox = Path(tempfile.mkdtemp(prefix="reboot_tests_"))
    for sub in ("data/memory", "data/config", "data/plugin_data"):
        (sandbox / "data" / sub.split("/")[-1]).mkdir(parents=True, exist_ok=True)
    (sandbox / "data" / "log.log").touch()

    pkg_parent = sandbox / "pkg"
    plugin_dir = pkg_parent / "reboot_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_FILES:
        src = REPO_ROOT / name
        if src.is_file():
            shutil.copy2(src, plugin_dir / name)

    os.chdir(sandbox)
    sys.path.insert(0, core_path)
    sys.path.insert(0, str(pkg_parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    _STATE.update(
        booted=True,
        sandbox=sandbox,
        pkg_parent=pkg_parent,
        plugin_dir=plugin_dir,
        core_path=core_path,
    )
    return _STATE


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeLLMClient:
    """Minimal stand-in for LLMModelClient."""

    def __init__(self, replies: Optional[List[Optional[str]]] = None,
                 delay: float = 0.0, raise_on_call: Optional[Exception] = None):
        self.replies = list(replies or [])
        self.default_reply = "MERGED-SUMMARY"
        self.delay = delay
        self.raise_on_call = raise_on_call
        self.prompts: List[str] = []

    async def chat(self, req):
        prompt = ""
        try:
            messages = getattr(req, "messages", []) or []
            prompt = str(getattr(messages[0], "content", "") if messages else "")
        except Exception:
            prompt = ""
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.replies:
            reply = self.replies.pop(0)
        else:
            reply = self.default_reply
        return SimpleNamespace(text_response=reply, input_tokens=1, output_tokens=1)


class FakeMessageProcessor:
    def __init__(self):
        self.sent: List[Dict[str, Any]] = []

    async def send_message_chain(self, session, chain):
        texts = []
        for element in chain:
            texts.append(getattr(element, "text", ""))
        self.sent.append({"session": session, "text": "".join(texts)})
        return SimpleNamespace(ok=True)

    def last_text(self) -> str:
        return self.sent[-1]["text"] if self.sent else ""


class FakeBuffer:
    def __init__(self):
        self.items: List[Any] = []

    def add(self, item):
        self.items.append(item)

    def pop(self, count: int = 1):
        popped = self.items[:count]
        del self.items[:count]
        return popped

    def flush(self, count: int = None):
        return self.pop(len(self.items) if count is None else count)

    def get_length(self):
        return len(self.items)


class FakeToolMgr:
    def __init__(self, names=()):
        self.names = list(names)
        self.unregistered: List[str] = []

    def unregister_tool(self, name: str):
        self.unregistered.append(name)
        if name in self.names:
            self.names.remove(name)

    def register_tool(self, name: str, description: str = "", parameters=None, func=None):
        if name not in self.names:
            self.names.append(name)


class FakeToolSet:
    """Mirrors core.agent.tool.ToolSet (name based, .remove(*names))."""

    def __init__(self, names):
        self.tools = [SimpleNamespace(name=n) for n in names]

    def __contains__(self, item):
        return any(t.name == item for t in self.tools)

    @property
    def names(self):
        return [t.name for t in self.tools]

    def remove(self, *tool_names: str):
        wanted = set(tool_names)
        self.tools = [t for t in self.tools if t.name not in wanted]

    def add(self, *tools):
        for tool in tools:
            self.tools = [t for t in self.tools if t.name != tool.name]
            self.tools.append(tool)


class FakeEventBus:
    """Directly awaited dispatch (the real bus drains an asyncio queue)."""

    def __init__(self):
        self.subscribers = defaultdict(list)
        self.published: List[Any] = []

    def subscribe(self, event_type, handler):
        self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type, handler):
        if handler in self.subscribers.get(event_type, []):
            self.subscribers[event_type].remove(handler)

    async def publish(self, event):
        self.published.append(event)
        event_type = getattr(event, "event_type", None) or type(event)
        for handler in tuple(self.subscribers.get(event_type, ())):
            await handler(event)


class FakeCtx:
    def __init__(self, session_mgr, llm=None, plugin_mgr=None, data_dir=None, event_bus=None):
        self.session_mgr = session_mgr
        self.plugin_mgr = plugin_mgr
        self.event_bus = event_bus if event_bus is not None else FakeEventBus()
        self.message_processor = FakeMessageProcessor()
        self.tool_mgr = FakeToolMgr()
        self._llm = llm
        self._data_dir = Path(data_dir) if data_dir else None
        self._buffers: Dict[str, FakeBuffer] = {}

    # -- llm access (used by summarizer) --
    def get_llm_client(self, model_uuid: Optional[str] = None, llm_type: Optional[str] = None):
        return self._llm

    def get_default_llm_client(self):
        return self._llm

    def get_default_fast_llm_client(self):
        return self._llm

    # -- misc --
    def get_plugin_data_dir(self):
        return self._data_dir

    def get_buffer(self, sid: str) -> FakeBuffer:
        return self._buffers.setdefault(sid, FakeBuffer())

    def get_session_capabilities(self, sid: str) -> dict:
        return {}


# ── config helpers ───────────────────────────────────────────────────────────


def default_cfg() -> Dict[str, Any]:
    """Build the config exactly like the framework's _ensure_plugin_config does,
    using the schema parser from core.config.config_field."""
    bootstrap()
    from core.config.config_field import build_fields

    schema_path = _STATE["plugin_dir"] / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    fields = build_fields(schema)
    cfg: Dict[str, Any] = {}
    for field in fields:
        children = getattr(field, "fields", None)
        if children is not None:
            cfg[field.key] = {}
            for child in children:
                cfg[field.key][child.key] = child.default
        else:
            cfg[field.key] = field.default
    return cfg


def set_cfg(cfg: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return cfg


# ── plugin fixture ───────────────────────────────────────────────────────────


def plugin_class():
    bootstrap()
    module = importlib.import_module("reboot_plugin")
    return module.RebootPlugin


def make_session_manager(max_memory_length: int = 50, event_bus=None, tmp_dir=None):
    from core.chat.session_manager import SessionManager

    kira_config = {
        "bot_config": {
            "bot": {
                "max_memory_length": max_memory_length,
                "memory_overflow_discard_count": 1,
            }
        }
    }
    sm = SessionManager(db=None, kira_config=kira_config, event_bus=event_bus)
    tmp_dir = Path(tmp_dir or _STATE["sandbox"] / "memory")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sm.chat_memory_path = str(tmp_dir / f"chat_memory_{abs(hash(str(tmp_dir))) % 10 ** 8}.json")
    sm.chat_memory = {}
    return sm


class Fixture:
    def __init__(self, plugin, ctx, sm, llm, tmp):
        self.plugin = plugin
        self.ctx = ctx
        self.sm = sm
        self.llm = llm
        self.tmp = tmp

    # -- convenience ---------------------------------------------------------
    @property
    def sent(self):
        return self.ctx.message_processor.sent

    def last_reply(self) -> str:
        return self.ctx.message_processor.last_text()

    async def settle(self, rounds: int = 4):
        """Let scheduled tasks (event publishing, async summary) run."""
        for _ in range(rounds):
            await asyncio.sleep(0)

    async def wait_for(self, predicate, timeout: float = 2.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return predicate()

    def seed_session(self, sid="qq:dm:10001", turns=3, title="T", description="D",
                     capabilities=None):
        """Create a session with the metadata that must survive a reset."""
        self.sm.get_session_info(sid)
        self.sm.update_session_info(sid, title=title, description=description)
        if capabilities is not None:
            self.sm.update_session_capabilities(sid, capabilities)
        chunks = []
        for i in range(turns):
            chunks.append([
                {"role": "user", "content": f"user message {i}"},
                {"role": "assistant", "content": f"assistant reply {i}"},
            ])
        self.sm.write_memory(sid, chunks)
        self.sm.chat_memory[sid]["timestamp"] = 1700000000
        return sid

    def memory(self, sid="qq:dm:10001"):
        return self.sm.read_memory(sid)

    def meta(self, sid="qq:dm:10001"):
        return dict(self.sm.chat_memory.get(sid) or {})


async def make_fixture(cfg_overrides: Optional[Dict[str, Any]] = None,
                       llm=None, plugin_mgr=None, max_memory_length=50,
                       with_capabilities=True, capability_overrides=None):
    bootstrap()
    tmp = Path(tempfile.mkdtemp(prefix="reboot_case_", dir=str(_STATE["sandbox"])))

    cfg = default_cfg()
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            set_cfg(cfg, key, value)

    bus = FakeEventBus()
    sm = make_session_manager(max_memory_length=max_memory_length, event_bus=bus, tmp_dir=tmp)
    ctx = FakeCtx(sm, llm=llm, plugin_mgr=plugin_mgr, data_dir=tmp / "plugin_data",
                  event_bus=bus)
    ctx.get_plugin_data_dir().mkdir(parents=True, exist_ok=True)

    plugin = plugin_class()(ctx, cfg)
    await plugin.initialize()

    fixture = Fixture(plugin, ctx, sm, llm, tmp)
    if with_capabilities:
        fixture.capabilities = capability_overrides or {"tts": {"enabled": False}}
    return fixture


# ── event helpers ────────────────────────────────────────────────────────────


def make_message_event(text: str, sid="qq:dm:10001", user_id="10001",
                       session_type="dm", user_name="tester"):
    """Build a REAL KiraMessageEvent."""
    bootstrap()
    from core.chat.message_utils import KiraMessageEvent, KiraIMMessage
    from core.chat import MessageChain, User, Group
    from core.chat.message_elements import Text
    from core.adapter.adapter_info import AdapterInfo

    adapter = AdapterInfo(enabled=True, adapter_id="qq", name="qq", platform="QQ",
                          description="", config={"self_id": "999"})
    group = Group(group_id=sid.split(":")[-1], group_name="test group") \
        if session_type == "gm" else None
    event = KiraMessageEvent(
        message_types=["gm", "dm"],
        timestamp=0,
        adapter=adapter,
        message=KiraIMMessage(
            message_id="m1",
            self_id="999",
            chain=MessageChain([Text(text)]),
            timestamp=0,
            sender=User(user_id=user_id, nickname=user_name),
            group=group,
            is_mentioned=True,
        ),
    )
    event.session.session_id = sid.split(":")[-1]
    return event


def make_batch_event(sid="qq:dm:10001", user_id="10001"):
    from core.chat.message_utils import KiraMessageBatchEvent, KiraIMMessage
    from core.chat import MessageChain, Session, User
    from core.chat.message_elements import Text

    return KiraMessageBatchEvent(
        message_types=["dm"],
        timestamp=0,
        session=Session(adapter_name="qq", session_type="dm", session_id="10001"),
        messages=[
            KiraIMMessage(
                message_id="m2",
                self_id="999",
                chain=MessageChain([Text("please forget")]),
                timestamp=0,
                sender=User(user_id=user_id, nickname="tester"),
            )
        ],
    )


def make_llm_request(messages=None, tool_names=("reset_context", "other_tool")):
    from core.provider import LLMRequest
    from core.agent.message import OpenAIMessage

    req = LLMRequest(messages=[OpenAIMessage(**m) for m in (messages or [])])
    req.tool_set = FakeToolSet(tool_names)
    return req


# ── assertions ───────────────────────────────────────────────────────────────


class CheckFailed(AssertionError):
    pass


def check(condition, message: str):
    if not condition:
        raise CheckFailed(message)


def check_eq(actual, expected, message: str = ""):
    if actual != expected:
        raise CheckFailed(f"{message}: expected {expected!r}, got {actual!r}")
