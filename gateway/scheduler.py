"""Priority scheduling of provider calls for the shared AI account (petrosa-llm-gateway#2).

One provider rate is shared by every caller. Requests wait in one queue ordered by an
effective priority (class rank, aged by waiting time); a request starts only when the
sliding rate window has room for its class and it is the best-placed eligible ticket.
``trading`` keeps a reserved share of the rate: other classes may use at most the rest.

The decision is a pure function of the queue, the window and the time, so the in-memory
store (tests, one replica) and the Redis store (two replicas acting as one) share it.
Time is injected (``clock`` and ``sleep``): no wall-clock waits in tests.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


@dataclass(frozen=True)
class ClassPolicy:
    """One priority class: lower rank is served first; bounded queue and age."""

    name: str
    rank: int
    max_queue: int
    max_age_s: float


@dataclass(frozen=True)
class SchedulingConfig:
    """Scheduler settings; ``rate_per_window`` is replaced by the learned rate (gw#3)."""

    classes: Mapping[str, ClassPolicy]
    rate_per_window: float
    window_s: float
    reserved_class: str
    reserved_share: float
    rank_gap_s: float
    poll_s: float

    def policy(self, name: str) -> ClassPolicy:
        try:
            return self.classes[name]
        except KeyError:
            raise SchedulingRefused("unknown_class", f"no class {name!r}") from None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SchedulingConfig:
        classes = {
            name: ClassPolicy(
                name=name,
                rank=int(spec["rank"]),
                max_queue=int(spec["max_queue"]),
                max_age_s=float(spec["max_age_s"]),
            )
            for name, spec in raw["classes"].items()
        }
        config = cls(
            classes=classes,
            rate_per_window=float(raw["rate_per_window"]),
            window_s=float(raw["window_s"]),
            reserved_class=str(raw["reserved_class"]),
            reserved_share=float(raw["reserved_share"]),
            rank_gap_s=float(raw["rank_gap_s"]),
            poll_s=float(raw["poll_s"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.reserved_class not in self.classes:
            raise ValueError(f"reserved_class {self.reserved_class!r} is not a class")
        if not 0.0 <= self.reserved_share < 1.0:
            raise ValueError("reserved_share must be in [0, 1)")
        for name, value in (
            ("rate_per_window", self.rate_per_window),
            ("window_s", self.window_s),
            ("poll_s", self.poll_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rank_gap_s < 0:
            raise ValueError("rank_gap_s must not be negative")


class SchedulingRefused(Exception):
    """A request the gateway will not run; ``code`` is the stable error code for callers."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Ticket:
    """A queued request. ``key`` orders the queue: rank, then arrival (aging)."""

    ticket_id: str
    cls: str
    enqueued_at: float
    deadline: float | None
    key: float


@dataclass(frozen=True)
class Started:
    """A start in the rate window."""

    ticket_id: str
    cls: str
    at: float


def queue_key(config: SchedulingConfig, cls: str, enqueued_at: float) -> float:
    """Effective-priority key, smaller first; constant over time, so it can be stored.

    A ticket of rank r enqueued at t sorts as ``r * rank_gap_s + t``: one rank step
    is worth ``rank_gap_s`` seconds of waiting, so a lower-priority ticket overtakes
    later higher-priority arrivals once it has waited that long (aging).
    """
    return config.policy(cls).rank * config.rank_gap_s + enqueued_at


def capacity(
    config: SchedulingConfig, window: list[Started], now: float
) -> dict[str, bool]:
    """Which classes may start now, from the starts inside the window."""
    recent = [s for s in window if s.at > now - config.window_s]
    total = len(recent)
    others = sum(1 for s in recent if s.cls != config.reserved_class)
    room = total < config.rate_per_window
    others_cap = config.rate_per_window * (1.0 - config.reserved_share)
    return {
        name: room and (name == config.reserved_class or others < others_cap)
        for name in config.classes
    }


def may_start(
    config: SchedulingConfig,
    ticket: Ticket,
    queue: list[Ticket],
    window: list[Started],
    now: float,
) -> bool:
    """True when ``ticket`` is the best-placed queued ticket among those allowed to start now."""
    allowed = capacity(config, window, now)
    if not allowed.get(ticket.cls, False):
        return False
    eligible = [t for t in queue if allowed.get(t.cls, False)]
    best = min(eligible, key=lambda t: (t.key, t.ticket_id), default=None)
    return best is not None and best.ticket_id == ticket.ticket_id


def estimated_start(
    config: SchedulingConfig,
    ticket: Ticket,
    queue: list[Ticket],
    window: list[Started],
    now: float,
) -> float:
    """A start-time estimate for deadline admission: the tickets ahead at the average spacing."""
    ahead = sum(
        1 for t in queue if (t.key, t.ticket_id) < (ticket.key, ticket.ticket_id)
    )
    if ahead == 0 and capacity(config, window, now).get(ticket.cls, False):
        return now
    spacing = config.window_s / config.rate_per_window
    return now + (ahead + 1) * spacing


class Store(Protocol):
    """Shared queue and rate-window state."""

    async def enqueue(
        self, config: SchedulingConfig, ticket: Ticket, now: float
    ) -> None: ...

    async def snapshot(self, now: float) -> tuple[list[Ticket], list[Started]]: ...

    async def try_start(
        self, config: SchedulingConfig, ticket: Ticket, now: float
    ) -> bool: ...

    async def remove(self, ticket: Ticket) -> None: ...


@dataclass
class InMemoryStore:
    """Single-process store; the Redis store gives the same semantics across replicas."""

    queue: dict[str, Ticket] = field(default_factory=dict)
    window: list[Started] = field(default_factory=list)

    async def enqueue(
        self, config: SchedulingConfig, ticket: Ticket, now: float
    ) -> None:
        waiting = sum(1 for t in self.queue.values() if t.cls == ticket.cls)
        if waiting >= config.policy(ticket.cls).max_queue:
            raise SchedulingRefused("queue_full", f"class {ticket.cls}")
        self.queue[ticket.ticket_id] = ticket

    async def snapshot(self, now: float) -> tuple[list[Ticket], list[Started]]:
        return list(self.queue.values()), list(self.window)

    async def try_start(
        self, config: SchedulingConfig, ticket: Ticket, now: float
    ) -> bool:
        self.window = [s for s in self.window if s.at > now - config.window_s]
        if not may_start(config, ticket, list(self.queue.values()), self.window, now):
            return False
        del self.queue[ticket.ticket_id]
        self.window.append(Started(ticket.ticket_id, ticket.cls, now))
        return True

    async def remove(self, ticket: Ticket) -> None:
        self.queue.pop(ticket.ticket_id, None)


_SEQUENCE = itertools.count()


#: The learned provider rate in requests per minute (``gateway.adaptive_rate``), read on
#: every decision so a 429 on one replica slows every replica at once.
RateSource = Callable[[], Awaitable[float]]


class Scheduler:
    """Admits requests in priority order within the shared rate."""

    def __init__(
        self,
        config: SchedulingConfig,
        store: Store,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
        rate: RateSource | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.store = store
        self.clock = clock
        self.sleep = sleep
        self.rate = rate

    async def effective_config(self) -> SchedulingConfig:
        """The config with the learned rate in place of ``rate_per_window`` (gw#3).

        Without a rate source the configured rate applies. The learned rate is in requests
        per minute and is scaled to the window.
        """
        if self.rate is None:
            return self.config
        rpm = await self.rate()
        return replace(self.config, rate_per_window=rpm * self.config.window_s / 60.0)

    async def admit(self, cls: str, budget_s: float | None = None) -> Ticket:
        """Wait until this request may start; raise ``SchedulingRefused`` when it never can.

        ``budget_s`` is the caller's remaining time: a request that cannot start before it
        runs out is refused at once (``deadline_unmeetable``) instead of queueing.
        """
        policy = self.config.policy(cls)
        now = self.clock()
        deadline = now + budget_s if budget_s is not None else None
        ticket = Ticket(
            ticket_id=f"{now:.6f}-{next(_SEQUENCE)}-{uuid.uuid4().hex[:8]}",
            cls=cls,
            enqueued_at=now,
            deadline=deadline,
            key=queue_key(self.config, cls, now),
        )
        if deadline is not None:
            queue, window = await self.store.snapshot(now)
            config = await self.effective_config()
            if estimated_start(config, ticket, queue, window, now) > deadline:
                raise SchedulingRefused(
                    "deadline_unmeetable", f"class {cls} budget {budget_s:.3f}s"
                )
        await self.store.enqueue(self.config, ticket, now)
        try:
            while True:
                now = self.clock()
                config = await self.effective_config()
                if await self.store.try_start(config, ticket, now):
                    return ticket
                if deadline is not None and now >= deadline:
                    raise SchedulingRefused("queue_expired", "deadline passed in queue")
                if now - ticket.enqueued_at >= policy.max_age_s:
                    raise SchedulingRefused("queue_expired", f"class {cls} max age")
                await self.sleep(self.config.poll_s)
        except BaseException:
            await self.store.remove(ticket)
            raise
