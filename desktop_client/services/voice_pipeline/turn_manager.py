from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class VoiceTurnState(str, Enum):
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    WAITING_LLM = "waiting_llm"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class VoiceTurn:
    session_id: str
    turn_id: str = field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:10]}")
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:10]}")
    state: VoiceTurnState = VoiceTurnState.IDLE
    reply_buffer: str = ""
    created_at: float = field(default_factory=time.time)


class VoiceTurnManager:
    def __init__(self):
        self._turns: Dict[str, VoiceTurn] = {}

    def get_or_create(self, session_id: str) -> VoiceTurn:
        if session_id not in self._turns:
            self._turns[session_id] = VoiceTurn(session_id=session_id)
        return self._turns[session_id]

    def new_turn(self, session_id: str) -> VoiceTurn:
        turn = VoiceTurn(session_id=session_id)
        self._turns[session_id] = turn
        return turn

    def set_state(self, session_id: str, state: VoiceTurnState) -> VoiceTurn:
        turn = self.get_or_create(session_id)
        turn.state = state
        return turn

    def append_reply(self, session_id: str, text: str) -> VoiceTurn:
        turn = self.get_or_create(session_id)
        turn.reply_buffer += text
        return turn

    def clear_reply(self, session_id: str) -> VoiceTurn:
        turn = self.get_or_create(session_id)
        turn.reply_buffer = ""
        return turn

    def interrupt(self, session_id: str) -> VoiceTurn:
        return self.set_state(session_id, VoiceTurnState.INTERRUPTED)

    def end_turn(self, session_id: str) -> VoiceTurn:
        turn = self.get_or_create(session_id)
        turn.state = VoiceTurnState.IDLE
        turn.reply_buffer = ""
        return turn

    def current(self, session_id: str) -> Optional[VoiceTurn]:
        return self._turns.get(session_id)
