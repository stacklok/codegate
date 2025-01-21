import datetime
from typing import List, Optional, Union

from pydantic import BaseModel

from codegate.pipeline.base import CodeSnippet


class ChatMessage(BaseModel):
    """
    Represents a chat message.
    """

    message: str
    timestamp: datetime.datetime
    message_id: str


class QuestionAnswer(BaseModel):
    """
    Represents a question and answer pair.
    """

    question: ChatMessage
    answer: Optional[ChatMessage]


class PartialQuestions(BaseModel):
    """
    Represents all user messages obtained from a DB row.
    """

    messages: List[str]
    timestamp: datetime.datetime
    message_id: str
    provider: Optional[str]
    type: str


class PartialQuestionAnswer(BaseModel):
    """
    Represents a partial conversation.
    """

    partial_questions: PartialQuestions
    answer: Optional[ChatMessage]


class Conversation(BaseModel):
    """
    Represents a conversation.
    """

    question_answers: List[QuestionAnswer]
    provider: Optional[str]
    type: str
    chat_id: str
    conversation_timestamp: datetime.datetime


class AlertConversation(BaseModel):
    """
    Represents an alert with it's respective conversation.
    """

    conversation: Conversation
    alert_id: str
    code_snippet: Optional[CodeSnippet]
    trigger_string: Optional[Union[str, dict]]
    trigger_type: str
    trigger_category: Optional[str]
    timestamp: datetime.datetime
