"""评审会讨论时段排期（POST /meeting-schedule）。

接收 `/assign` 可行结果中至多 8 篇已完成分配的稿件（稿件 -> 评审人名单）、
至多 8 个候选讨论时段（各带非负整数稿件容量）以及每位评审人可出席的时段，
在以下全局约束下求**完整排期**：

- 每篇稿件恰好排入一个时段；
- 该稿的**全部**评审人在该时段都可出席；
- 同一评审人在同一时段至多讨论一篇稿件（同一时段内评审人不得跨稿冲突）；
- 每个时段排入的稿件数不超过该时段容量。

多个可行方案按**稿件 ID 升序依次比较其对应时段 ID**，返回该向量字典序最小的
方案；实现为「稿件按 ID 排序、每稿按时段 ID 升序尝试」的带回溯深度优先搜索
（首个完整解即字典序最小解），并以前向检查（每篇未排稿件至少还剩一个当前可
放入的时段、剩余容量总和足够）剪枝——剪枝条件只可能因资源被占用而进一步恶化，
故不会漏掉任何可行解。相同业务输入（甚至不同的列表顺序）必然产生完全相同的
输出。

无完整方案时返回不可行且**不输出任何部分排期**；输入非法（数量超限、稿件 /
时段 ID 重复、分配名单中的评审人未声明可用时段、可用时段引用未知时段）时
收集全部确定性排序的问题清单，调用方整体拒绝（422）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: 单次请求允许的稿件 / 候选时段数量上限
MAX_MANUSCRIPTS = 8
MAX_SLOTS = 8


@dataclass(frozen=True)
class MeetingAssignment:
    """一篇已完成分配的稿件及其评审人名单（评审人重复由调用方去重）。"""

    manuscript_id: str
    reviewer_ids: tuple[str, ...]


@dataclass(frozen=True)
class MeetingSlot:
    """一个候选讨论时段：ID 与非负整数稿件容量。"""

    id: str
    capacity: int


@dataclass(frozen=True)
class ReviewerAvailability:
    """一位评审人声明可出席的时段列表（同一评审人多条声明取并集）。"""

    reviewer_id: str
    slot_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleIssue:
    """一条非法输入原因（结构化，整体 422 时返回全部问题）。"""

    type: str  # too_many_manuscripts / too_many_slots / duplicate_manuscript /
    # duplicate_slot / reviewer_availability_missing / unknown_slot
    manuscript_id: str | None
    slot_id: str | None
    reviewer_id: str | None
    detail: str


@dataclass(frozen=True)
class ScheduledItem:
    """一篇稿件在完整排期中的归属时段。"""

    manuscript_id: str
    slot_id: str


@dataclass(frozen=True)
class ScheduleResult:
    #: 非法输入问题清单（确定性排序）；非空时 feasible 为 False、schedule 为空
    issues: tuple[ScheduleIssue, ...]
    #: 是否存在完整排期（有非法输入时亦为 False）
    feasible: bool
    #: 完整排期（按稿件 ID 排序）；不可行 / 非法输入时为空元组
    schedule: tuple[ScheduledItem, ...]


def _issue_sort_key(issue: ScheduleIssue) -> tuple[str, str, str, str]:
    return (
        issue.type,
        issue.manuscript_id or "",
        issue.slot_id or "",
        issue.reviewer_id or "",
    )


def schedule_meeting(
    assignments: list[MeetingAssignment],
    slots: list[MeetingSlot],
    availability: list[ReviewerAvailability],
) -> ScheduleResult:
    """校验输入并求字典序最小的完整排期。

    存在任何非法输入时返回非空 ``issues``（调用方应整体 422，不输出排期）；
    输入合法但约束下无完整排期时 ``issues`` 为空、``feasible`` 为 False，
    同样不输出部分排期。
    """
    issues: list[ScheduleIssue] = []

    # ---- 1. 数量上限 ----
    manuscript_ids = [a.manuscript_id for a in assignments]
    if len(manuscript_ids) > MAX_MANUSCRIPTS:
        issues.append(ScheduleIssue(
            type="too_many_manuscripts",
            manuscript_id=None,
            slot_id=None,
            reviewer_id=None,
            detail=(
                f"稿件数量为 {len(manuscript_ids)}，超出上限 "
                f"{MAX_MANUSCRIPTS} 篇"
            ),
        ))

    slot_ids = [s.id for s in slots]
    if len(slot_ids) > MAX_SLOTS:
        issues.append(ScheduleIssue(
            type="too_many_slots",
            manuscript_id=None,
            slot_id=None,
            reviewer_id=None,
            detail=(
                f"候选时段数量为 {len(slot_ids)}，超出上限 {MAX_SLOTS} 个"
            ),
        ))

    # ---- 2. 稿件 / 时段 ID 重复（各重复 ID 记一条，与输入顺序无关） ----
    for mid, count in sorted(Counter(manuscript_ids).items()):
        if count > 1:
            issues.append(ScheduleIssue(
                type="duplicate_manuscript",
                manuscript_id=mid,
                slot_id=None,
                reviewer_id=None,
                detail=(
                    f"稿件 {mid} 在 assignments 中出现 {count} 次，"
                    f"每篇稿件至多一条分配记录"
                ),
            ))
    for sid, count in sorted(Counter(slot_ids).items()):
        if count > 1:
            issues.append(ScheduleIssue(
                type="duplicate_slot",
                manuscript_id=None,
                slot_id=sid,
                reviewer_id=None,
                detail=f"时段 {sid} 在 slots 中出现 {count} 次，时段 ID 必须唯一",
            ))

    known_slots = set(slot_ids)

    # 同一评审人多条可用声明取并集（重复条目不报错，结果只取决于声明集合）
    available: dict[str, set[str]] = {}
    for decl in availability:
        available.setdefault(decl.reviewer_id, set()).update(decl.slot_ids)

    # ---- 3. 可用时段引用未知时段（按 (评审人, 时段) 去重后各记一条） ----
    for rid in sorted(available):
        for sid in sorted(available[rid] - known_slots):
            issues.append(ScheduleIssue(
                type="unknown_slot",
                manuscript_id=None,
                slot_id=sid,
                reviewer_id=rid,
                detail=(
                    f"评审人 {rid} 的可用时段引用了未在 slots 中声明的未知时段 {sid}"
                ),
            ))

    # ---- 4. 分配名单中的评审人必须声明可用时段（缺整条声明才算未声明；
    #         声明空列表等于明确表示全程不可出席，属合法输入，径由求解判不可行） ----
    declared_reviewers = set(available)
    assigned_reviewers = {
        rid for a in assignments for rid in a.reviewer_ids
    }
    for rid in sorted(assigned_reviewers - declared_reviewers):
        issues.append(ScheduleIssue(
            type="reviewer_availability_missing",
            manuscript_id=None,
            slot_id=None,
            reviewer_id=rid,
            detail=(
                f"分配名单中的评审人 {rid} 未在 reviewer_availability 中声明"
                f"可出席的时段"
            ),
        ))

    if issues:
        issues.sort(key=_issue_sort_key)
        return ScheduleResult(issues=tuple(issues), feasible=False, schedule=())

    return _solve(assignments, slots, available)


def _solve(
    assignments: list[MeetingAssignment],
    slots: list[MeetingSlot],
    available: dict[str, set[str]],
) -> ScheduleResult:
    """输入已合法：求字典序最小完整排期。"""
    # 全部按 ID 排序：稿件顺序即字典序比较顺序，时段顺序为尝试先后
    manuscripts = sorted({a.manuscript_id for a in assignments})
    slot_order = sorted({s.id for s in slots})
    capacities = {s.id: s.capacity for s in slots}

    # 稿件 -> 去重后的评审人集合（名单内重复 ID 不造成自我冲突）
    reviewers_of: dict[str, frozenset[str]] = {}
    for a in assignments:
        reviewers_of[a.manuscript_id] = frozenset(a.reviewer_ids)

    # 每稿的静态候选时段：其全部评审人在该时段都可出席（动态的容量 / 冲突
    # 可行性在搜索中判定）
    options: dict[str, list[str]] = {}
    for mid in manuscripts:
        revs = reviewers_of[mid]
        options[mid] = [
            sid
            for sid in slot_order
            if all(sid in available[rid] for rid in revs)
        ]

    used_capacity = {sid: 0 for sid in slot_order}
    #: 时段 -> 已在该时段讨论稿件的评审人集合
    busy: dict[str, set[str]] = {sid: set() for sid in slot_order}
    remaining = manuscripts  # 已按 ID 排序

    def _fits(mid: str, sid: str) -> bool:
        """稿件 mid 当前能否放入时段 sid：容量未满且无评审人同时段冲突。"""
        if used_capacity[sid] >= capacities[sid]:
            return False
        return busy[sid].isdisjoint(reviewers_of[mid])

    def dfs(index: int) -> list[str] | None:
        if index == len(manuscripts):
            return []

        later = remaining[index + 1:]

        # 总量剪枝：剩余容量之和不足以容纳剩余稿件则必无解
        spare_total = sum(
            capacities[sid] - used_capacity[sid] for sid in slot_order
        )
        if spare_total < len(manuscripts) - index:
            return None

        mid = remaining[index]
        for sid in options[mid]:
            if not _fits(mid, sid):
                continue
            used_capacity[sid] += 1
            busy[sid].update(reviewers_of[mid])

            # 前向检查：每篇后续稿件至少还剩一个当前可放入的时段。资源只可能
            # 被进一步占用，现在无处可放则以后也无处可放，剪枝不排除任何可行解。
            completable = all(
                any(_fits(other, cand) for cand in options[other])
                for other in later
            )
            tail: list[str] | None = None
            if completable:
                tail = dfs(index + 1)
            if tail is not None:
                return [sid, *tail]

            busy[sid].difference_update(reviewers_of[mid])
            used_capacity[sid] -= 1

        return None

    chosen = dfs(0)
    if chosen is None:
        return ScheduleResult(issues=(), feasible=False, schedule=())

    schedule = tuple(
        ScheduledItem(manuscript_id=mid, slot_id=sid)
        for mid, sid in zip(manuscripts, chosen)
    )
    return ScheduleResult(issues=(), feasible=True, schedule=schedule)
