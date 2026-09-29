"""POST /meeting-schedule 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _post(body: dict) -> tuple[int, dict]:
    resp = client.post("/meeting-schedule", json=body)
    return resp.status_code, resp.json()


def _assign(manuscripts: dict[str, list[str]]) -> list[dict]:
    return [
        {"manuscript_id": mid, "reviewer_ids": rids}
        for mid, rids in manuscripts.items()
    ]


def _slots(specs: dict[str, int]) -> list[dict]:
    return [{"id": sid, "capacity": cap} for sid, cap in specs.items()]


def _avail(specs: dict[str, list[str]]) -> list[dict]:
    return [
        {"reviewer_id": rid, "slot_ids": sids} for rid, sids in specs.items()
    ]


def _schedule(body: dict) -> list[tuple[str, str]]:
    status, data = _post(body)
    assert status == 200, data
    assert data["status"] == "feasible", data
    return [(item["manuscript_id"], item["slot_id"]) for item in data["schedule"]]


# ---------- 基本可行与约束 ----------


def test_simple_feasible_schedule():
    body = {
        "assignments": _assign({"m1": ["r1", "r2"], "m2": ["r1"]}),
        "slots": _slots({"s1": 1, "s2": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s1", "s2"], "r2": ["s1"]}
        ),
    }
    # m1 全部评审人（r1、r2）同时可出席的最小时段为 s1；m2 随后只能 s2
    assert _schedule(body) == [("m1", "s1"), ("m2", "s2")]


def test_all_reviewers_must_attend_jointly():
    # r2 不能出席 s1：m1（需 r1、r2 同时在场）只能 s2，故 m2 占 s1
    body = {
        "assignments": _assign({"m1": ["r1", "r2"], "m2": ["r1"]}),
        "slots": _slots({"s1": 1, "s2": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s1", "s2"], "r2": ["s2"]}
        ),
    }
    assert _schedule(body) == [("m1", "s2"), ("m2", "s1")]


def test_reviewer_conflict_same_slot_forces_spread():
    # 两稿都需要 r1；每个时段容量足够但 r1 同一时段只能讨论一稿
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r1"]}),
        "slots": _slots({"s1": 2}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    status, data = _post(body)
    assert status == 200
    assert data == {"status": "infeasible", "schedule": None}


def test_capacity_not_exceeded():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r2"], "m3": ["r3"]}),
        "slots": _slots({"s1": 2, "s2": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s1"], "r2": ["s1"], "r3": ["s1"]}
        ),
    }
    # 三篇都只能在 s1 讨论，但 s1 容量仅 2
    status, data = _post(body)
    assert status == 200
    assert data == {"status": "infeasible", "schedule": None}


def test_capacity_zero_slot_is_unusable():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": _slots({"s0": 0, "s1": 1}),
        "reviewer_availability": _avail({"r1": ["s0"]}),
    }
    status, data = _post(body)
    assert status == 200
    assert data == {"status": "infeasible", "schedule": None}


def test_capacity_allows_sharing_a_slot():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r2"]}),
        "slots": _slots({"s1": 2}),
        "reviewer_availability": _avail({"r1": ["s1"], "r2": ["s1"]}),
    }
    # 字典序最小：两稿都排 s1
    assert _schedule(body) == [("m1", "s1"), ("m2", "s1")]


def test_manuscript_without_reviewers_can_be_scheduled():
    # 空评审人名单的稿件仍占一个时段与容量
    body = {
        "assignments": _assign({"m1": [], "m2": ["r1"]}),
        "slots": _slots({"s1": 1}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    status, data = _post(body)
    assert status == 200
    assert data == {"status": "infeasible", "schedule": None}

    body["slots"] = _slots({"s1": 2})
    assert _schedule(body) == [("m1", "s1"), ("m2", "s1")]


# ---------- 字典序最小 ----------


def test_lexicographically_minimal_compared_by_manuscript_id_order():
    # m1 可在 s1/s2、m2 只能 s1：字典序最小要求 m1 退到 s2 让 m2 占 s1
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r2"]}),
        "slots": _slots({"s1": 1, "s2": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s1", "s2"], "r2": ["s1"]}
        ),
    }
    assert _schedule(body) == [("m1", "s2"), ("m2", "s1")]


def test_schedule_sorted_by_manuscript_id_regardless_of_input_order():
    body = {
        "assignments": list(reversed(_assign(
            {"m1": ["r1"], "m2": ["r1"], "m3": ["r1"]}
        ))),
        "slots": list(reversed(_slots({"s1": 1, "s2": 1, "s3": 1}))),
        "reviewer_availability": list(reversed(_avail(
            {"r1": ["s3", "s1", "s2"]}
        ))),
    }
    assert _schedule(body) == [("m1", "s1"), ("m2", "s2"), ("m3", "s3")]


def test_lex_min_prefers_earliest_slot_for_first_manuscript():
    # m1 可 s1/s2、m2 可 s1/s2（不同评审人、容量 1）：m1 取最小可行的 s1
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r2"]}),
        "slots": _slots({"s2": 1, "s1": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s2", "s1"], "r2": ["s1", "s2"]}
        ),
    }
    assert _schedule(body) == [("m1", "s1"), ("m2", "s2")]


def test_string_ids_lexicographic_not_numeric():
    body = {
        "assignments": _assign({"m1": ["r1"], "m10": ["r2"], "m2": ["r3"]}),
        "slots": _slots({"s1": 1, "s2": 1, "s10": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s1", "s2", "s10"],
             "r2": ["s1", "s2", "s10"],
             "r3": ["s1", "s2", "s10"]}
        ),
    }
    # 字符串字典序：稿件 m1 < m10 < m2；时段 s1 < s10 < s2
    assert _schedule(body) == [
        ("m1", "s1"), ("m10", "s10"), ("m2", "s2"),
    ]


def test_input_reordering_does_not_change_output():
    base = {
        "assignments": _assign({"m1": ["r1", "r2"], "m2": ["r1"]}),
        "slots": _slots({"s1": 1, "s2": 1}),
        "reviewer_availability": _avail(
            {"r1": ["s1", "s2"], "r2": ["s1"]}
        ),
    }
    expected = [("m1", "s1"), ("m2", "s2")]
    for perm in (
        lambda b: b,
        lambda b: {**b, "assignments": list(reversed(b["assignments"]))},
        lambda b: {**b, "slots": list(reversed(b["slots"]))},
        lambda b: {
            **b,
            "reviewer_availability": list(reversed(b["reviewer_availability"])),
        },
    ):
        assert _schedule(perm(base)) == expected


def test_duplicate_reviewer_ids_within_assignment_dedup():
    body = {
        "assignments": [
            {"manuscript_id": "m1", "reviewer_ids": ["r1", "r1"]},
        ],
        "slots": _slots({"s1": 1}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    assert _schedule(body) == [("m1", "s1")]


def test_multiple_availability_declarations_take_union():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r1"]}),
        "slots": _slots({"s1": 1, "s2": 1}),
        "reviewer_availability": [
            {"reviewer_id": "r1", "slot_ids": ["s1"]},
            {"reviewer_id": "r1", "slot_ids": ["s2"]},
        ],
    }
    assert _schedule(body) == [("m1", "s1"), ("m2", "s2")]


# ---------- 不可行：不输出部分排期 ----------


def test_infeasible_returns_no_partial_schedule():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r1"]}),
        "slots": _slots({"s1": 1, "s2": 1}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    status, data = _post(body)
    assert status == 200
    assert data["status"] == "infeasible"
    assert data["schedule"] is None


def test_empty_availability_is_infeasible_not_an_error():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": _slots({"s1": 1}),
        "reviewer_availability": _avail({"r1": []}),
    }
    status, data = _post(body)
    assert status == 200
    assert data == {"status": "infeasible", "schedule": None}


# ---------- 非法输入：422 ----------


def _errors(body: dict) -> list[dict]:
    status, data = _post(body)
    assert status == 422, data
    assert "schedule" not in data
    return data["errors"]


def test_too_many_manuscripts():
    body = {
        "assignments": _assign(
            {f"m{i}": ["r1"] for i in range(9)}
        ),
        "slots": _slots({"s1": 9}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    errors = _errors(body)
    assert [e["type"] for e in errors] == ["too_many_manuscripts"]
    assert errors[0]["detail"]


def test_too_many_slots():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": _slots({f"s{i}": 1 for i in range(9)}),
        "reviewer_availability": _avail(
            {"r1": [f"s{i}" for i in range(9)]}
        ),
    }
    errors = _errors(body)
    assert [e["type"] for e in errors] == ["too_many_slots"]


def test_duplicate_manuscript_id():
    body = {
        "assignments": [
            {"manuscript_id": "m1", "reviewer_ids": ["r1"]},
            {"manuscript_id": "m1", "reviewer_ids": ["r2"]},
        ],
        "slots": _slots({"s1": 2}),
        "reviewer_availability": _avail({"r1": ["s1"], "r2": ["s1"]}),
    }
    errors = _errors(body)
    assert errors == [
        {
            "type": "duplicate_manuscript",
            "manuscript_id": "m1",
            "slot_id": None,
            "reviewer_id": None,
            "detail": errors[0]["detail"],
        }
    ]


def test_duplicate_slot_id():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": [
            {"id": "s1", "capacity": 1},
            {"id": "s1", "capacity": 1},
        ],
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    errors = _errors(body)
    assert [e["type"] for e in errors] == ["duplicate_slot"]
    assert errors[0]["slot_id"] == "s1"


def test_reviewer_without_availability_declaration():
    body = {
        "assignments": _assign({"m1": ["r1", "r2"]}),
        "slots": _slots({"s1": 1}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    errors = _errors(body)
    assert errors == [
        {
            "type": "reviewer_availability_missing",
            "manuscript_id": None,
            "slot_id": None,
            "reviewer_id": "r2",
            "detail": errors[0]["detail"],
        }
    ]


def test_unknown_slot_in_availability():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": _slots({"s1": 1}),
        "reviewer_availability": _avail({"r1": ["s1", "s9"]}),
    }
    errors = _errors(body)
    assert [e["type"] for e in errors] == ["unknown_slot"]
    assert errors[0]["slot_id"] == "s9"
    assert errors[0]["reviewer_id"] == "r1"


def test_availability_for_unassigned_reviewer_is_not_required():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": _slots({"s1": 1}),
        # 只声明 r1；r9 不在任何稿件名单中，无需出现（出现也不报错）
        "reviewer_availability": [
            {"reviewer_id": "r1", "slot_ids": ["s1"]},
            {"reviewer_id": "r9", "slot_ids": ["s1"]},
        ],
    }
    assert _schedule(body) == [("m1", "s1")]


def test_errors_collected_and_sorted_deterministically():
    # 同时存在多种非法输入：一次性返回全部问题并按 (type, 稿件, 时段, 评审人) 排序
    body = {
        "assignments": [
            {"manuscript_id": "m1", "reviewer_ids": ["r1"]},
            {"manuscript_id": "m2", "reviewer_ids": ["r2"]},
            {"manuscript_id": "m2", "reviewer_ids": ["r1"]},
        ],
        "slots": [
            {"id": "s1", "capacity": 1},
            {"id": "s1", "capacity": 1},
        ],
        "reviewer_availability": [
            {"reviewer_id": "r1", "slot_ids": ["s1", "sX"]},
            # r2 完全未声明
        ],
    }
    errors = _errors(body)
    types = [
        (e["type"], e.get("manuscript_id"), e.get("slot_id"), e.get("reviewer_id"))
        for e in errors
    ]
    assert types == [
        ("duplicate_manuscript", "m2", None, None),
        ("duplicate_slot", None, "s1", None),
        ("reviewer_availability_missing", None, None, "r2"),
        ("unknown_slot", None, "sX", "r1"),
    ]


def test_extra_fields_rejected():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": _slots({"s1": 1}),
        "reviewer_availability": _avail({"r1": ["s1"]}),
        "unexpected": 1,
    }
    status, _ = _post(body)
    assert status == 422


def test_negative_capacity_rejected():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "slots": [{"id": "s1", "capacity": -1}],
        "reviewer_availability": _avail({"r1": ["s1"]}),
    }
    status, data = _post(body)
    assert status == 422
