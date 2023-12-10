import uuid
from enum import Enum

from pydantic import BaseModel


class Actor(str, Enum):
    USER = 'USER'
    ASSISTANT = 'ASSISTANT'


class Turn(BaseModel):
    actor: Actor
    utterance: str

    def __str__(self) -> str:
        return '{0}: {1}'.format(self.actor, self.utterance)


class Dialogue(BaseModel):
    dialogue_id: uuid.UUID
    turns: list[Turn]

    def get_query(self) -> str:
        if self.turns:
            return self.turns[-1].utterance
        return ''

    def format_dialogue(self) -> list[str]:
        return [str(turn) for turn in self.turns]
