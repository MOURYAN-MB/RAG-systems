from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class SupervisorState(TypedDict):
    task: str                              # original user request
    messages: Annotated[list, add_messages]  # full agent conversation log
    research_notes: str                    # output from Research Agent
    final_report: str                      # output from Writer Agent
    next: str                              # routing decision: "research" | "write" | "FINISH"