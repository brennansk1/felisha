from __future__ import annotations

from pathlib import Path

from causalrag.llm.cassette import CassetteStore, _stable_key


def test_key_is_deterministic() -> None:
    args = dict(
        model="qwen3:14b-q4_K_M",
        system="you are an investigator",
        prompt="describe column 'age'",
        fmt="json",
        options={"seed": 0, "temperature": 0.0, "num_ctx": 8192},
    )
    k1 = _stable_key(**args)
    k2 = _stable_key(**args)
    assert k1 == k2
    assert len(k1) == 32


def test_key_changes_with_option_order_is_stable() -> None:
    """Same options in different dict-insertion order must hash to the same key."""
    a = _stable_key(
        model="m",
        system="",
        prompt="p",
        fmt="json",
        options={"seed": 1, "temperature": 0.0, "num_ctx": 8192},
    )
    b = _stable_key(
        model="m",
        system="",
        prompt="p",
        fmt="json",
        options={"num_ctx": 8192, "temperature": 0.0, "seed": 1},
    )
    assert a == b


def test_key_changes_with_seed() -> None:
    base = dict(model="m", system="", prompt="p", fmt="json")
    a = _stable_key(**base, options={"seed": 0})  # type: ignore[arg-type]
    b = _stable_key(**base, options={"seed": 1})  # type: ignore[arg-type]
    assert a != b


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    key = "abcd1234"
    payload = {"response": '{"hello": "world"}', "model": "m"}
    store.save(key, payload)
    loaded = store.load(key)
    assert loaded == payload


def test_load_returns_none_on_miss(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    assert store.load("nonexistent") is None
