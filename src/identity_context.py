"""Request-local AI cabinet identity with a legacy-safe default."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

DEFAULT_AI_IDENTITY = "susu"
SUPPORTED_AI_IDENTITIES = frozenset({"susu", "shenyan"})
_current_ai_identity: ContextVar[str] = ContextVar(
    "ombre_ai_identity", default=DEFAULT_AI_IDENTITY
)
_cabinet_writes_allowed: ContextVar[bool] = ContextVar(
    "ombre_cabinet_writes_allowed", default=True
)


def normalize_ai_identity(value: object, *, legacy_default: bool = False) -> str:
    text = str(value or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {"susu": "susu", "苏苏": "susu", "shenyan": "shenyan", "沈砚": "shenyan"}
    if not text and legacy_default:
        return DEFAULT_AI_IDENTITY
    identity = aliases.get(text)
    if identity is None:
        raise ValueError("cabinet must be susu or shenyan")
    return identity


def current_ai_identity() -> str:
    return _current_ai_identity.get()


def cabinet_writes_allowed() -> bool:
    return _cabinet_writes_allowed.get()


@contextmanager
def use_ai_identity(value: object, *, read_only: bool = False):
    identity = normalize_ai_identity(value)
    identity_token = _current_ai_identity.set(identity)
    writes_token = _cabinet_writes_allowed.set(not read_only)
    try:
        yield identity
    finally:
        _cabinet_writes_allowed.reset(writes_token)
        _current_ai_identity.reset(identity_token)


@contextmanager
def use_read_identity(value: object = ""):
    source = current_ai_identity()
    target = normalize_ai_identity(value) if value else source
    with use_ai_identity(target, read_only=target != source):
        yield target
