"""Who is talking to the agent. Tools read this to enforce role-based access
(e.g. only attendants and doctors may modify medical records)."""
from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass

WRITE_ROLES = {"attendant", "doctor"}


@dataclass
class Actor:
    user_id: str
    name: str
    role: str = "patient"          # patient (incl. caregivers) / attendant / doctor

    @property
    def can_edit_records(self) -> bool:
        return self.role in WRITE_ROLES

    def to_dict(self) -> dict:
        return asdict(self)


current_actor: contextvars.ContextVar[Actor | None] = contextvars.ContextVar("current_actor", default=None)


def get_actor() -> Actor:
    return current_actor.get() or Actor(user_id="anonymous", name="Anonymous", role="patient")
