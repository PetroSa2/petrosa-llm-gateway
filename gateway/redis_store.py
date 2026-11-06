"""Redis-backed scheduler state: two gateway replicas share one queue and one rate window.

Every change is one optimistic transaction (WATCH / MULTI): read the queue and the window,
decide with the same pure functions as the in-memory store, write, and retry when another
replica changed the keys in between. The rate therefore holds across replicas.

Every key expires: each write refreshes, in the same transaction, a TTL equal to the
longest time any entry can matter (``key_ttl_s``). If every replica stops, the keys go
away on their own; the shared Redis runs ``noeviction`` and holds other services' state.

A waiting ticket refreshes its ``seen`` time on each poll. Tickets not seen for
``stale_after_s`` (a replica that died while its caller waited) are dropped by the next
transaction, so they never block the queue. The clock must be comparable across replicas
(wall-clock time), unlike a single process's monotonic clock.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

from gateway.scheduler import (
    SchedulingConfig,
    SchedulingRefused,
    Started,
    Ticket,
    may_start,
)

MAX_TRANSACTION_RETRIES = 20


def key_ttl_s(config: SchedulingConfig, stale_after_s: float) -> int:
    """How long any scheduler entry can matter: the window or the longest queue age, plus
    the time a dead replica's ticket takes to go stale."""
    longest_age = max((p.max_age_s for p in config.classes.values()), default=0.0)
    return math.ceil(max(config.window_s, longest_age) + stale_after_s)


class RedisStore:
    """Queue (hash of tickets), liveness (hash of seen times) and window (sorted set)."""

    def __init__(self, redis: Redis, prefix: str, stale_after_s: float) -> None:
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self.redis = redis
        self.queue_key = f"{prefix}:queue"
        self.seen_key = f"{prefix}:seen"
        self.window_key = f"{prefix}:window"
        self.stale_after_s = stale_after_s

    @staticmethod
    def _decode(raw: Any) -> str:
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    async def _read(
        self, pipe: Any, now: float
    ) -> tuple[list[Ticket], list[str], list[Started]]:
        """Live tickets, the ids of stale ones, and the window, from inside a WATCH."""
        tickets = {
            self._decode(k): Ticket(**json.loads(self._decode(v)))
            for k, v in (await pipe.hgetall(self.queue_key)).items()
        }
        seen = {
            self._decode(k): float(self._decode(v))
            for k, v in (await pipe.hgetall(self.seen_key)).items()
        }
        live, stale = [], []
        for ticket_id, ticket in tickets.items():
            if seen.get(ticket_id, ticket.enqueued_at) < now - self.stale_after_s:
                stale.append(ticket_id)
            else:
                live.append(ticket)
        window = []
        for member, at in await pipe.zrange(self.window_key, 0, -1, withscores=True):
            cls, _, ticket_id = self._decode(member).partition("|")
            window.append(Started(ticket_id, cls, float(at)))
        return live, stale, window

    async def _transaction(self, step) -> Any:
        for _ in range(MAX_TRANSACTION_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self.queue_key, self.seen_key, self.window_key)
                    return await step(pipe)
                except WatchError:
                    continue
        raise SchedulingRefused(
            "store_contention", "redis transaction retries exhausted"
        )

    def _drop_stale(self, pipe: Any, stale: list[str]) -> None:
        if stale:
            pipe.hdel(self.queue_key, *stale)
            pipe.hdel(self.seen_key, *stale)

    def _expire(self, pipe: Any, config: SchedulingConfig) -> None:
        """Refresh every key's TTL inside the current transaction."""
        ttl = key_ttl_s(config, self.stale_after_s)
        for key in (self.queue_key, self.seen_key, self.window_key):
            pipe.expire(key, ttl)

    async def enqueue(
        self, config: SchedulingConfig, ticket: Ticket, now: float
    ) -> None:
        async def step(pipe: Any) -> None:
            live, stale, _ = await self._read(pipe, now)
            waiting = sum(1 for t in live if t.cls == ticket.cls)
            if waiting >= config.policy(ticket.cls).max_queue:
                raise SchedulingRefused("queue_full", f"class {ticket.cls}")
            pipe.multi()
            self._drop_stale(pipe, stale)
            pipe.hset(self.queue_key, ticket.ticket_id, json.dumps(asdict(ticket)))
            pipe.hset(self.seen_key, ticket.ticket_id, repr(now))
            self._expire(pipe, config)
            await pipe.execute()

        await self._transaction(step)

    async def snapshot(self, now: float) -> tuple[list[Ticket], list[Started]]:
        live, _, window = await self._read(self.redis, now)
        return live, window

    async def try_start(
        self, config: SchedulingConfig, ticket: Ticket, now: float
    ) -> bool:
        async def step(pipe: Any) -> bool:
            live, stale, window = await self._read(pipe, now)
            window = [s for s in window if s.at > now - config.window_s]
            if ticket.ticket_id not in {t.ticket_id for t in live}:
                raise SchedulingRefused("queue_expired", "ticket left the shared queue")
            start = may_start(config, ticket, live, window, now)
            pipe.multi()
            self._drop_stale(pipe, stale)
            pipe.zremrangebyscore(self.window_key, "-inf", now - config.window_s)
            if start:
                pipe.hdel(self.queue_key, ticket.ticket_id)
                pipe.hdel(self.seen_key, ticket.ticket_id)
                pipe.zadd(self.window_key, {f"{ticket.cls}|{ticket.ticket_id}": now})
            else:
                pipe.hset(self.seen_key, ticket.ticket_id, repr(now))
            self._expire(pipe, config)
            await pipe.execute()
            return start

        return await self._transaction(step)

    async def remove(self, ticket: Ticket) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hdel(self.queue_key, ticket.ticket_id)
            pipe.hdel(self.seen_key, ticket.ticket_id)
            await pipe.execute()
