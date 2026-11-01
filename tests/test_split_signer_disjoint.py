import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("split_tool", ROOT / "tools/split_signer_disjoint.py")
split_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_tool)


def test_assign_is_signer_disjoint_and_deterministic():
    users = [f"s{i}" for i in range(20) for _ in range(i + 1)]
    fracs = {"train": 0.7, "val": 0.1, "test": 0.2}
    a = split_tool.assign(users, fracs, seed=42)
    b = split_tool.assign(users, fracs, seed=42)
    assert a == b, "assignment must be deterministic for a fixed seed"
    assert set(a) == {f"s{i}" for i in range(20)}
    assert set(a.values()) <= {"train", "val", "test"}
    # every clip lands in exactly one split by construction
    assert sum(1 for u in users if a[u] == "train") > 0
    assert sum(1 for u in users if a[u] == "test") > 0


def test_assign_respects_clip_share_with_skew():
    # one dominant signer must not be able to flip the test share alone
    users = ["big"] * 70 + ["m1"] * 10 + ["m2"] * 10 + ["s1"] * 5 + ["s2"] * 5
    fracs = {"train": 0.7, "val": 0.1, "test": 0.2}
    a = split_tool.assign(users, fracs, seed=1)
    share = {k: sum(1 for u in users if a[u] == k) / len(users) for k in fracs}
    assert abs(share["train"] - 0.7) <= 0.15
    assert abs(share["val"] - 0.1) <= 0.15
    assert abs(share["test"] - 0.2) <= 0.15


def test_coverage_counts_are_per_split():
    ids = ["c1", "c2", "c3", "c4"]
    texts = ["a", "a", "b", "no_event"]
    users = ["s1", "s2", "s1", "s2"]
    split_of = {"s1": "train", "s2": "test"}
    cov = split_tool.coverage(ids, texts, users, split_of, "test")
    assert cov["clips"] == 2
    assert cov["signers"] == 1
    assert cov["classes"] == 2
    assert cov["no_event_clips"] == 1
    assert cov["clip_ids"] == ["c2", "c4"]
