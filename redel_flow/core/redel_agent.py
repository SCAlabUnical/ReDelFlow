"""ReDel (Reason, Act, Delegate, Observe) agent graph for distributed multi-agent systems.

ReDel extends ReAct with explicit DELEGATE and OBSERVE phases:
  1. REASON  - LLM analyses the problem and selects the next action.
  2. ACT     - Local tools are executed and their results are observed.
  3. DELEGATE- Sub-tasks are sent to neighbouring agents when the local
               scope is insufficient.
  4. OBSERVE - Results from tools or delegated computations are incorporated
               into the agent's state before the next reasoning step.

The graph exposes a single factory: `create_redel_agent`.
"""

import logging
from typing import Annotated, List, Optional, Sequence, Union

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from redel_flow.config.configurations import google_api_key
from redel_flow.core.llm.llm_validation import (
    CallAgentsWithMessages,
    SendResultToCaller,
)

logger = logging.getLogger(__name__)

MAX_ERRORS_BEFORE_STOP = 5
_SPECIAL_TOOLS = {"CallAgentsWithMessages", "SendResultToCaller"}

# ── Correction messages ────────────────────────────────────────────────────────

_CORRECTION_NO_TOOL = (
    "CORRECTION: You must always call a tool.\n"
    "  • Use 'CallAgentsWithMessages' to delegate work to other agents.\n"
    "  • Use 'SendResultToCaller' to return the final answer to the caller.\n"
    "  • Or call one of your local tools to retrieve information.\n"
    "Generate a new response that includes a valid tool call."
)

_CORRECTION_MIXED_TOOLS = (
    "CORRECTION: You cannot combine a delegation/reply tool with local tools in the same step.\n"
    "Choose exactly ONE action per step:\n"
    "  • Call one or more local tools (to retrieve information), OR\n"
    "  • Call 'CallAgentsWithMessages' to delegate, OR\n"
    "  • Call 'SendResultToCaller' to return the final answer.\n"
    "Generate a new response with only one type of action."
)

_CORRECTION_MULTI_SPECIAL = (
    "CORRECTION: You called more than one special tool in the same step.\n"
    "Use exactly one — either 'CallAgentsWithMessages' or 'SendResultToCaller' — never both.\n"
    "Generate a new response that calls only one of them."
)

# Formatted at runtime with the unknown names and available tools list.
_CORRECTION_UNKNOWN_TOOL = (
    "CORRECTION: You called unknown tool(s): {unknown}.\n"
    "Available local tools: {available}.\n"
    "Use 'CallAgentsWithMessages' to delegate or 'SendResultToCaller' to reply."
)


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_call_agents_args(
    args: dict, allowed: list[str]
) -> tuple[bool, str]:
    """Validate the arguments of a CallAgentsWithMessages tool call.

    Returns:
        (ok, error_message) — ok is True when the args are valid.
    """
    try:
        parsed = CallAgentsWithMessages.model_validate(args)
    except Exception as e:
        logger.error("Malformed CallAgentsWithMessages args: %s", args)
        error_detail = str(e)
        is_empty = "EMPTY_MESSAGES" in error_detail

        correction = (
            "CORRECTION: 'messages' cannot be empty.\n"
            "You must specify at least one agent and a message.\n"
            f"Available agents: {allowed}.\n"
            "Example: {\"agentA\": \"Write a story about a dragon.\"}"
        ) if is_empty else (
            "CORRECTION: Malformed 'CallAgentsWithMessages' arguments.\n"
            "Required fields:\n"
            "  - messages: dict mapping each agent name to its message.\n"
            "    Example: {'Agent1': 'What is the weather in Rome?'}\n"
            "  - current_state: your reasoning for contacting these agents.\n"
            "Escape inside message strings: \\\" for quotes, \\\\ for backslashes."
        )
        return False, correction

    unknown = [name for name in parsed.messages if name not in allowed]
    if unknown:
        logger.error("Unknown agent names %s (allowed: %s)", unknown, allowed)
        return False, (
            f"CORRECTION: Unknown agent name(s): {unknown}.\n"
            f"Available agents are: {allowed}.\n"
            "Fix the agent names and retry."
        )

    return True, ""


def _validate_send_result_args(args: dict) -> tuple[bool, str]:
    """Validate the arguments of a SendResultToCaller tool call."""
    try:
        SendResultToCaller.model_validate(args)
    except Exception:
        logger.error("Malformed SendResultToCaller args: %s", args)
        return False, (
            "CORRECTION: Malformed 'SendResultToCaller' arguments.\n"
            "Check the required fields and retry."
        )
    return True, ""


# ── Agent state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Persistent state of the ReDel agent across reasoning steps.

    Attributes:
        messages:         Full conversation history (LLM outputs, tool results,
                          correction messages).
        number_of_steps:  How many LLM reasoning iterations have been executed.
        neighbors_names:  Names of agents this agent is allowed to delegate to.
        number_of_errors: Consecutive error count; the agent aborts when this
                          reaches MAX_ERRORS_BEFORE_STOP.
        structured:       The validated final action once the agent is ready to
                          either delegate or reply.
        ready:            True when `structured` has been validated and the
                          graph can proceed to `finalize`.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int
    neighbors_names: List[str]
    number_of_errors: int
    structured: Optional[Union[CallAgentsWithMessages, SendResultToCaller]]
    ready: Optional[bool]


# ── Graph factory ──────────────────────────────────────────────────────────────

def create_redel_agent(real_tools: List[StructuredTool]) -> Runnable:
    """Build and compile the ReDel LangGraph state machine.

    Args:
        real_tools: Domain-specific tools the agent may call locally.

    Returns:
        Compiled LangGraph state machine.
    """
    all_tools = [CallAgentsWithMessages, SendResultToCaller] + real_tools
    local_tools_by_name: dict[str, StructuredTool] = {t.name: t for t in real_tools}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.4,
        max_retries=2,
        google_api_key=google_api_key,
    )
    model_with_tools = llm.bind_tools(all_tools, tool_choice="any")

    # ── Nodes ──────────────────────────────────────────────────────────────────

    def reason(state: AgentState) -> dict:
        """REASON phase: invoke the LLM to decide the next action."""
        if state.get("number_of_errors", 0) >= MAX_ERRORS_BEFORE_STOP:
            raise RuntimeError(
                f"Agent aborted: reached {MAX_ERRORS_BEFORE_STOP} consecutive errors."
            )
        response = model_with_tools.invoke(state["messages"], {"recursion_limit": 5})
        return {
            "messages": [response],
            "number_of_steps": state["number_of_steps"] + 1,
        }

    def handle_structural_error(state: AgentState) -> dict:
        """Inject a correction for structurally invalid LLM output.

        Called when the LLM produced:
          - no tool calls at all,
          - a mix of special and local tools,
          - more than one special tool, or
          - a local tool name that does not exist.
        The correction is appended to the message history so the next
        `reason` step has clear context to self-correct.
        """
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", [])
        names   = [tc["name"] for tc in tool_calls]
        special = [n for n in names if n in _SPECIAL_TOOLS]
        local   = [n for n in names if n not in _SPECIAL_TOOLS]

        if not tool_calls:
            correction = _CORRECTION_NO_TOOL
        elif special and local:
            correction = _CORRECTION_MIXED_TOOLS
        elif len(special) > 1:
            correction = _CORRECTION_MULTI_SPECIAL
        else:
            # Only local tool calls, but one or more names are unknown.
            unknown = [n for n in local if n not in local_tools_by_name]
            correction = _CORRECTION_UNKNOWN_TOOL.format(
                unknown=unknown,
                available=list(local_tools_by_name),
            )

        logger.warning("Structural error — %s", correction.splitlines()[0])
        return {
            "messages": [HumanMessage(content=correction)],
            "number_of_errors": state.get("number_of_errors", 0) + 1,
        }

    def execute_local_tools(state: AgentState) -> dict:
        """ACT phase: run every local tool call and collect results for OBSERVE phase.

        All tool names are guaranteed to exist in `local_tools_by_name`
        because the router validated them before reaching this node.
        Only runtime errors (bad arguments, external failures) are handled here.
        Results are appended to the message history for the next REASON step.
        """
        outputs = []
        for tc in state["messages"][-1].tool_calls:
            tool = local_tools_by_name[tc["name"]]
            try:
                result = tool.invoke(tc["args"])
                outputs.append(ToolMessage(
                    content=result,
                    name=tc["name"],
                    tool_call_id=tc["id"],
                ))
            except Exception as exc:
                logger.error("Error running tool '%s': %s", tc["name"], exc)
                outputs.append(HumanMessage(
                    content=(
                        f"CORRECTION: Tool '{tc['name']}' raised an error.\n"
                        "Check the arguments and retry, or choose a different action."
                    )
                ))

        return {"messages": outputs}

    def validate_special_action(state: AgentState) -> dict:
        """DELEGATE / REPLY phase: validate the single special tool call.

        On success  → sets `structured` and `ready=True` so the graph
                      can proceed to `finalize`.
        On failure  → appends a correction message and se for OBSERVE phasets `ready=False`
                      so the graph loops back to `reason`.
        """
        tc = state["messages"][-1].tool_calls[0]
        name, args = tc["name"], tc["args"]

        if name == "CallAgentsWithMessages":
            ok, err = _validate_call_agents_args(args, state["neighbors_names"])
            if not ok:
                return {
                    "messages": [HumanMessage(content=err)],
                    "ready": False,
                    "number_of_errors": state.get("number_of_errors", 0) + 1,
                }
            return {
                "structured": CallAgentsWithMessages.model_validate(args),
                "ready": True,
            }

        if name == "SendResultToCaller":
            ok, err = _validate_send_result_args(args)
            if not ok:
                return {
                    "messages": [HumanMessage(content=err)],
                    "ready": False,
                    "number_of_errors": state.get("number_of_errors", 0) + 1,
                }
            return {
                "structured": SendResultToCaller.model_validate(args),
                "ready": True,
            }

        # Unreachable given the router, included for safety.
        return {
            "messages": [HumanMessage(
                content=f"CORRECTION: Unexpected special tool '{name}'. "
                        "Use 'CallAgentsWithMessages' or 'SendResultToCaller'."
            )],
            "ready": False,
            "number_of_errors": state.get("number_of_errors", 0) + 1,
        }

    def finalize(state: AgentState) -> dict:
        """Return the validated structured action as the graph output."""
        if state.get("structured") is None:
            raise ValueError(
                "`finalize` reached without a validated structured payload."
            )
        return {"structured": state["structured"]}

    # ── Routers ────────────────────────────────────────────────────────────────

    def route_after_reason(state: AgentState) -> str:
        """Classify the LLM output and return the next node name.

        Priority rules:
          1. No tool calls at all              → handle_structural_error
          2. Mix of special + local tools      → handle_structural_error
          3. More than one special tool        → handle_structural_error
          4. Local tool(s) with unknown name   → handle_structural_error
          5. Only known local tool calls       → execute_local_tools
          6. Exactly one special tool          → validate_special_action
        """
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", [])

        if not tool_calls:
            return "handle_structural_error"

        names   = [tc["name"] for tc in tool_calls]
        special = [n for n in names if n in _SPECIAL_TOOLS]
        local   = [n for n in names if n not in _SPECIAL_TOOLS]

        if special and local:
            return "handle_structural_error"

        if len(special) > 1:
            return "handle_structural_error"

        if local:
            unknown = [n for n in local if n not in local_tools_by_name]
            if unknown:
                return "handle_structural_error"
            return "execute_local_tools"

        return "validate_special_action"   # len(special) == 1

    def route_after_validation(state: AgentState) -> str:
        return "finalize" if state.get("ready") else "reason"

    # ── Build graph ────────────────────────────────────────────────────────────

    builder = StateGraph(AgentState)

    builder.add_node("reason",                  reason)
    builder.add_node("handle_structural_error", handle_structural_error)
    builder.add_node("execute_local_tools",     execute_local_tools)
    builder.add_node("validate_special_action", validate_special_action)
    builder.add_node("finalize",                finalize)

    builder.set_entry_point("reason")

    builder.add_conditional_edges("reason", route_after_reason, {
        "handle_structural_error": "handle_structural_error",
        "execute_local_tools":     "execute_local_tools",
        "validate_special_action": "validate_special_action",
    })

    builder.add_edge("handle_structural_error", "reason")
    builder.add_edge("execute_local_tools",     "reason")

    builder.add_conditional_edges("validate_special_action", route_after_validation, {
        "reason":   "reason",
        "finalize": "finalize",
    })

    builder.set_finish_point("finalize")

    return builder.compile()