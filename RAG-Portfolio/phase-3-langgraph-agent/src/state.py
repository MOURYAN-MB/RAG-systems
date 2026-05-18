from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class AgentState(TypedDict):
    # add_messages merges new messages into the list instead of replacing it.
    # This is how the agent accumulates its full conversation + tool results.
    messages: Annotated[list, add_messages]