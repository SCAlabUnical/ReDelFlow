"""Conversation state management for the ReDel agent.

Encapsulates all mutable state and lock operations for an agent across
concurrent conversations. Exposes atomic operations so Agent never
touches the raw dictionaries directly.

"""

import asyncio
from langchain_core.messages import AIMessage, ToolMessage
import logging

logger = logging.getLogger(__name__)

class ConversationState:
    """Manages all mutable state for one agent across concurrent conversations.

    All methods that write state are async and acquire the appropriate lock
    internally. Callers never need to manage locks directly.
    """

    def __init__(self):
        # Completed conversation history per conversation
        self.history: dict[str, list[str]] = {}

        # In-progress history per (conv_id, pred_req_id)
        self.partial_history: dict[str, dict[str, list[str]]] = {}

        # Pending delegations: conv_id → pred_req_id → [(req_id, agent_name)]
        self.pending_results: dict[str, dict[str, list]] = {}

        # Results received from delegated agents: conv_id → req_id → (agent, response)
        self.invocation_results: dict[str, dict[str, tuple]] = {}

        # Maps req_id → pred_req_id
        self.req_to_pred: dict[str, str] = {}

        # Maps pred_req_id → caller agent name
        self.pred_to_caller: dict[str, str] = {}

        # Maps pred_req_id → agents that must not be called
        self.pred_to_not_callable_agents: dict[str, list] = {}

        self.results_lock = asyncio.Lock()
        self.requests_lock = asyncio.Lock()

    # ── Request initialisation ────────────────────────────────────────────────

    async def init_request(
        self,
        conv_id: str,
        pred_req_id: str,
        sender: str,
        not_callable_agents: list,
    ):
        """Register a new incoming request and initialise its state."""
        async with self.requests_lock:
            self.pred_to_caller[pred_req_id] = sender
            self.pred_to_not_callable_agents[pred_req_id] = not_callable_agents

            if conv_id not in self.history:
                self.history[conv_id] = []
                self.partial_history[conv_id] = {}

            if pred_req_id not in self.partial_history[conv_id]:
                self.partial_history[conv_id][pred_req_id] = []

    # ── History ───────────────────────────────────────────────────────────────

    async def append_partial(self, conv_id: str, pred_req_id: str, message: str):
        """Append a message to the in-progress history of a request."""
        async with self.requests_lock:
            self.partial_history[conv_id][pred_req_id].append(message)

    async def save_llm_result(
        self, conv_id: str, pred_req_id: str, message_to_save: str, result: dict
    ):
        """Append the incoming message and the LLM cycle result to partial history."""
        async with self.requests_lock:
            self.partial_history[conv_id][pred_req_id].append(message_to_save)
            tool_logs = self._extract_tool_logs(result)
            self.partial_history[conv_id][pred_req_id].extend(tool_logs)

    # ── Delegation ────────────────────────────────────────────────────────────

    async def prepare_delegation(
        self, conv_id: str, pred_req_id: str, current_state: str
    ):
        """Initialise pending/invocation structures before sending delegations."""
        async with self.requests_lock:
            self.partial_history[conv_id][pred_req_id].append(
                f"You reasoned: {current_state}"
            )
            if conv_id not in self.pending_results:
                self.pending_results[conv_id] = {}
            self.pending_results[conv_id][pred_req_id] = []

            if conv_id not in self.invocation_results:
                self.invocation_results[conv_id] = {}

    async def add_pending(
        self,
        conv_id: str,
        pred_req_id: str,
        req_id: str,
        agent_name: str,
        message: str,
    ):
        """Register one pending delegation."""
        async with self.requests_lock:
            self.req_to_pred[req_id] = pred_req_id
            self.pending_results[conv_id][pred_req_id].append((req_id, agent_name))
            self.partial_history[conv_id][pred_req_id].append(
                f"You invoked {agent_name} saying: {message}"
            )

    # ── Result collection ─────────────────────────────────────────────────────

    async def register_result(
        self, conv_id: str, req_id: str, sender: str, response: str
    ) -> tuple[str | None, bool]:
        """Save a result from a delegated agent.

        Returns:
            (pred_req_id, all_received) — pred_req_id is None if req_id unknown.
        """
        pred_req_id = self.req_to_pred.get(req_id)
        if not pred_req_id:
            return None, False

        async with self.results_lock:
            if conv_id not in self.invocation_results:
                self.invocation_results[conv_id] = {}
            self.invocation_results[conv_id][req_id] = (sender, response)

            pending_list = self.pending_results[conv_id][pred_req_id]
            self.pending_results[conv_id][pred_req_id] = [
                pair for pair in pending_list if pair[0] != req_id
            ]
            all_received = not self.pending_results[conv_id][pred_req_id]

        return pred_req_id, all_received

    async def collect_responses(
        self, conv_id: str, pred_req_id: str
    ) -> tuple[str, str]:
        """Collect and remove all results for a pred_req_id from invocation_results.

        Returns:
            (responses_text, sender_label) ready to be passed to interact_with_agent.
        """
        all_responses = self.invocation_results.get(conv_id, {})

        responses = []
        agents = []
        to_delete = []

        for req_id, (agent, response) in all_responses.items():
            if self.req_to_pred.get(req_id) == pred_req_id:
                to_delete.append(req_id)
                responses.append(f"You received the response from {agent}: {response}")
                agents.append(agent)

        async with self.results_lock:
            for req_id in to_delete:
                del self.invocation_results[conv_id][req_id]

        responses_text = "\n".join(responses)
        sender = (
            agents[0] if len(agents) == 1
            else f"{', '.join(agents[:-1])} and {agents[-1]}"
        )
        return responses_text, sender

    async def log_connection_error(self, conv_id: str, message: str):
        """Append a connection error to the main history."""
        async with self.requests_lock:
            if conv_id in self.history:
                self.history[conv_id].append(message)

    # ── Teardown ──────────────────────────────────────────────────────────────

    async def teardown(
        self,
        conv_id: str,
        pred_req_id: str,
        final_messages: list[str] | None = None,
    ) -> str:
        """Clean up all state for a completed or failed request.

        Appends any final_messages to partial history, merges it into
        the main history, and removes all mappings for pred_req_id.

        Returns:
            The caller's name (needed by Agent to route the response).
        """
        async with self.results_lock:
            if final_messages:
                for msg in final_messages:
                    self.partial_history[conv_id][pred_req_id].append(msg)

            # Remove req_id mappings belonging to this pred_req_id
            req_ids_to_delete = [
                req_id for req_id, pred in self.req_to_pred.items()
                if pred == pred_req_id
            ]
            for req_id in req_ids_to_delete:
                del self.req_to_pred[req_id]
                if req_id in self.invocation_results.get(conv_id, {}):
                    del self.invocation_results[conv_id][req_id]

            # Clean up empty dicts
            if conv_id in self.invocation_results and not self.invocation_results[conv_id]:
                del self.invocation_results[conv_id]

            if conv_id in self.pending_results and pred_req_id in self.pending_results[conv_id]:
                del self.pending_results[conv_id][pred_req_id]
                if not self.pending_results[conv_id]:
                    del self.pending_results[conv_id]

            caller = self.pred_to_caller.get(pred_req_id)

            if pred_req_id in self.pred_to_caller:
                del self.pred_to_caller[pred_req_id]
            if pred_req_id in self.pred_to_not_callable_agents:
                del self.pred_to_not_callable_agents[pred_req_id]

            # Merge partial history into main history
            merged = (
                self.history[conv_id]
                + self.partial_history[conv_id][pred_req_id]
            )
            self.history[conv_id] = merged
            del self.partial_history[conv_id][pred_req_id]

        return caller

    # ── Read accessors ────────────────────────────────────────────────────────
    # Called from Agent outside lock contexts — safe in asyncio (cooperative).

    def get_history(self, conv_id: str) -> list[str]:
        return self.history.get(conv_id, [])

    def get_partial_history(self, conv_id: str, pred_req_id: str) -> list[str]:
        return self.partial_history.get(conv_id, {}).get(pred_req_id, [])
    
    def flatten_history(self, conv_id: str, pred_req_id: str) -> str:
        return "\n".join(self.get_partial_history(conv_id, pred_req_id))

    def get_caller(self, pred_req_id: str) -> str:
        return self.pred_to_caller.get(pred_req_id, "")

    def get_not_callable(self, pred_req_id: str) -> list:
        return self.pred_to_not_callable_agents.get(pred_req_id, [])

    def get_pred_req_id(self, req_id: str) -> str | None:
        return self.req_to_pred.get(req_id)
    
    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_tool_logs(self, result: dict) -> list[str]:
        """Parse LLM cycle messages and extract tool usage logs for history."""
        try:
            messages = result.get("messages", [])
            tool_logs = []

            for msg in messages:
                if isinstance(msg, AIMessage):
                    if isinstance(msg.content, list):
                        for chunk in msg.content:
                            if chunk.get("type") == "text":
                                text = chunk.get("text", "").strip()
                                if text:
                                    tool_logs.append(f"You reasoned: {text}")

                    for call in msg.tool_calls:
                        if call["name"] in {"CallAgentsWithMessages", "SendResultToCaller"}:
                            continue
                        args_str = ", ".join(f"{k}={v}" for k, v in call["args"].items())
                        tool_logs.append(
                            f"You called the tool: {call['name']}({args_str}) → ..."
                        )

                elif isinstance(msg, ToolMessage):
                    if tool_logs:
                        last_log = tool_logs.pop()
                        tool_logs.append(
                            last_log.replace("→ ...", f"→ {msg.content.strip()}")
                        )

            return tool_logs

        except Exception as e:
            logger.error("Error extracting tool logs: %s", e)
            return []