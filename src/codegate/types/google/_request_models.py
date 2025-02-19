from typing import (
    Iterable,
    List,
)

import pydantic


class GenerateContentRequest(pydantic.BaseModel):
  model: str | None = None
  contents: List[Content] | None = None
  config: Config | None = None

  def get_messages(self) -> Iterable[Content]:
      if self.content is not None:
          for content in self.content:
              yield content
