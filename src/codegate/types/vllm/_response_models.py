from typing import (
    Any,
    Iterable,
)

import pydantic

class VllmMessageError(pydantic.BaseModel):
    object: str
    message: str
    code: int

    def get_content(self) -> Iterable[Any]:
        yield self

    def get_text(self) -> str | None:
        return self.message

    def set_text(self, text) -> None:
        self.message = text
