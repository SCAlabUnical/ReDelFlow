"""Distributed agent implementation using gRPC and LangGraph for multi-agent communication.

This module defines the Agent class, which implements the ReDel (Reason, Act, Delegate, Observe)
paradigm - an extension of ReAct for distributed multi-agent systems where agents can:
- REASON: Analyze the current context and evaluate progress toward the goal
- ACT: Invoke local tools or external services to gather information
- DELEGATE: Transfer sub-tasks to neighboring agents when local scope is insufficient
- OBSERVE: Incorporate outcomes of tools or delegated computations into internal state

State management is delegated to ConversationState, keeping this class
focused on the ReDel logic and gRPC communication.
"""

import asyncio
import logging
import uuid

from grpc import aio

from redel_flow.config.configurations import (
    MAXIMUM_RETRY,
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TIMEOUT,
    HEARTBEAT_FAILURE_THRESHOLD,
)
from redel_flow.core.conversation_state import ConversationState
from redel_flow.core.redel_agent import create_redel_agent
from redel_flow.core.llm.llm_validation import CallAgentsWithMessages, SendResultToCaller
from redel_flow.core.llm.prompts import CONTEXT_PROMPT, INTERACTION_PROMPT
from redel_flow.core.proto import agent_pb2, agent_pb2_grpc
from redel_flow.core.proto import client_pb2, client_pb2_grpc
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class Agent(agent_pb2_grpc.AgentServicer):
    """gRPC service implementation for a distributed agent node using the ReDel paradigm."""

    def __init__(
        self,
        system_prompt,
        name,
        role,
        description,
        host,
        port,
        neighbors,
        tool_tips_section,
        tools,
        network_info,
    ):
        self.name = name
        self.role = role
        self.description = description
        self.host = host
        self.port = port
        self.neighbors = neighbors
        self.tools = tools
        self.tool_tips_section = tool_tips_section
        if self.tool_tips_section:
            tool_tips = f"""## Tool Usage Tips

            {self.tool_tips_section}
            """
        else:
            tool_tips = ""
        self.system_prompt = system_prompt.format(
            name=self.name,
            role=self.role,
            description=self.description,
            tool_tips=tool_tips,
        )
        self.redel_graph = create_redel_agent(real_tools=self.tools)
        self.network_info = network_info
        self.state = ConversationState()

        # Supervision tasks for outgoing delegations: req_id → task
        self._monitors: dict[str, asyncio.Task] = {}

    # ── gRPC service methods ──────────────────────────────────────────────────

    async def Invoke(self, request, context):
        """Handle incoming invocation requests from clients or other agents."""

        await self.state.init_request(
        request.conversation_id,
        request.request_id,
        request.sender,
        list(request.not_callable_agents),
        )
        
        task = asyncio.create_task(self.handle_request(
            request.conversation_id,
            request.request_id,
            request.sender,
            request.request
            ))
        task.add_done_callback(self.handle_call_back)
        return agent_pb2.InvokeResponse(conversation_id=request.conversation_id, ack=True)

    async def Heartbeat(self, request, context):
        """Report whether this node is still handling a delegated request."""
        return agent_pb2.HeartbeatResponse(
        conversation_id=request.conversation_id,
        request_id=request.request_id,
        working=self.state.is_handling(request.request_id),
        agent_name=self.name,
    )
    
    async def ReturnResult(self, request, context):
        """Handle result responses from previously invoked agents."""
        pred_req_id, all_received = await self.state.register_result(
            request.conversation_id,
            request.request_id,
            request.sender,
            request.response,
        )

        if pred_req_id is None:
            if self.state.was_abandoned(request.request_id):
                logger.info(
                    f"[{self.name}] Late result from {request.sender} for "
                    f"{request.request_id}: delegation already declared failed, discarding."
                )
            else:
                logger.error(f"[{self.name}] Unknown req_id {request.request_id}.")
            return agent_pb2.ReturnResultResponse(
                conversation_id=request.conversation_id, ack=False
            )

        self._stop_monitor(request.request_id)

        if all_received:
            task = asyncio.create_task(
                self.handle_response(request.conversation_id, pred_req_id)
            )
            task.add_done_callback(self.handle_call_back)

        return agent_pb2.ReturnResultResponse(
            conversation_id=request.conversation_id, ack=True
        )

    # ── Request / response handling ───────────────────────────────────────────

    async def handle_request(
        self, conversation_id, request_id, sender, request_text
    ):
        """Process an incoming request."""
        pred_req_id = request_id
        message_to_save = (
            f"You have been invoked by {sender} with the request: '{request_text}'"
        )
        return await self.interact_with_agent(
            conversation_id, pred_req_id,
            "Request", request_text, sender, message_to_save,
        )

    async def handle_response(self, conversation_id: str, pred_req_id: str):
        """Process collected responses from invoked agents and continue the workflow."""
        responses_text, sender = await self.state.collect_responses(
            conversation_id, pred_req_id
        )
        message_to_save = f"Results from invoked agents:\n{responses_text}"
        return await self.interact_with_agent(
            conversation_id, pred_req_id,
            "Results from invoked agents", responses_text, sender, message_to_save,
        )

    # ── ReDel cycle ───────────────────────────────────────────────────────────

    async def interact_with_agent(
        self, conversation_id, pred_req_id, message_type, message, sender, message_to_save
    ):
        """Execute one complete ReDel cycle for a single request."""
        try:
            callable_neighbors = {
                k: v for k, v in self.neighbors.items()
                if k not in self.state.get_not_callable(pred_req_id)
            }
            neighbor_names = (
                ', '.join(callable_neighbors.keys()) if callable_neighbors else 'no one'
            )
            neighbor_roles = (
                ', '.join([
                    f'{name} ({role}) with description: {description}'
                    for name, (_, role, description) in callable_neighbors.items()
                ]) if callable_neighbors else 'no one'
            )

            formatted_prompt_system = CONTEXT_PROMPT.format(
                history="\n".join(self.state.get_history(conversation_id)),
                history_current_request=self.state.flatten_history(conversation_id, pred_req_id),
                neighbor_names=neighbor_names,
                neighbor_roles=neighbor_roles,
            )
            formatted_prompt_interaction = INTERACTION_PROMPT.format(
                message_type=message_type,
                request=message,
                sender=sender,
            )

            input_data = {
                "messages": [
                    SystemMessage(self.system_prompt),
                    SystemMessage(content=formatted_prompt_system),
                    HumanMessage(content=formatted_prompt_interaction),
                ],
                "number_of_steps": 0,
                "neighbors_names": list(callable_neighbors.keys()),
                "number_of_errors": 0,
            }

            result = await self.redel_graph.ainvoke(
                input_data, {"recursion_limit": 1000}
            )
            structured = result["structured"]

            await self.state.save_llm_result(
                conversation_id, pred_req_id, message_to_save, result
            )
            logger.info(f"[{self.name}] LLM Response:\n{structured}")

        except Exception as e:
            logger.error(f"[{self.name}] Interaction Error with LLM: {e}")
            await self.state.append_partial(
                conversation_id, pred_req_id, message_to_save
            )
            return ("AGENT_INTERACTION_ERROR", [conversation_id, pred_req_id, sender])

        # ── DELEGATE ──────────────────────────────────────────────────────────
        if isinstance(structured, CallAgentsWithMessages):
            await self.state.prepare_delegation(
                conversation_id, pred_req_id, structured.current_state
            )
            tasks = []
            for agent_name, msg in structured.messages.items():
                req_id = str(uuid.uuid4())
                await self.state.add_pending(
                    conversation_id, pred_req_id, req_id, agent_name, msg
                )
                tasks.append(self.send_message_to_agent(
                    conversation_id, req_id, agent_name, msg,
                    self.state.get_not_callable(pred_req_id),
                ))
            results = await asyncio.gather(*tasks)
            return ("AGENT_CALLING_RESULTS", results)

        # ── RESPOND ───────────────────────────────────────────────────────────
        elif isinstance(structured, SendResultToCaller):
            caller = await self.state.teardown(
                conversation_id, pred_req_id,
                final_messages=[
                    f"You reasoned: {structured.current_state}",
                    f"Your final response was: {structured.result}",
                ],
            )
            address = self.network_info[caller]
            if caller != "Client":
                return await self.send_message_to_caller(
                    conversation_id, pred_req_id, address, structured.result,
                    MAXIMUM_RETRY, caller,
                )
            else:
                return await self.send_message_to_client(
                    conversation_id, address, structured.result, MAXIMUM_RETRY
                )

        else:
            logger.error(
                "[%s] Unexpected structured type: %s", self.name, type(structured)
            )
            return ("AGENT_INTERACTION_ERROR", [conversation_id, pred_req_id, sender])

    # ── Outbound communication ────────────────────────────────────────────────

    async def send_message_to_agent(
        self, conversation_id, req_id, agent_name, message, not_callable_agents
    ):
        """Send a delegation message to another agent via gRPC (DELEGATE phase)."""
        address = self.neighbors[agent_name][0]
        try:
            async with aio.insecure_channel(address) as channel:
                stub = agent_pb2_grpc.AgentStub(channel)
                result = await stub.Invoke(agent_pb2.InvokeRequest(
                    conversation_id=conversation_id,
                    request_id=req_id,
                    sender=self.name,
                    request=message,
                    not_callable_agents=not_callable_agents + [self.name],
                ))
                self._start_monitor(
                    conversation_id, req_id, agent_name, address
                )
                return ("OK", result)
        except Exception as e:
            logger.error(f"[{self.name}] Error calling agent {agent_name}: {e}")
            return ("AGENT_CALLING_ERROR", [req_id, conversation_id, agent_name])

    async def send_message_to_caller(
        self, conversation_id, pred_req_id, address, final_response, retry_count, caller
    ):
        """Send the final response back to the calling agent."""
        try:
            async with aio.insecure_channel(address) as channel:
                stub = agent_pb2_grpc.AgentStub(channel)
                result = await stub.ReturnResult(agent_pb2.ReturnResultRequest(
                    conversation_id=conversation_id,
                    sender=self.name,
                    request_id=pred_req_id,
                    response=final_response,
                ))
                if not result.ack:
                    return ("RESULT_REJECTED", [conversation_id, caller])
                return ("CALLING_CALLER_DONE", result)
        except Exception as e:
            logger.error(f"[{self.name}] Error contacting caller: {e}")
            if retry_count > 0:
                return await self.send_message_to_caller(
                    conversation_id, pred_req_id, address, final_response,
                    retry_count - 1, caller,
                )
            return ("CALLER_CALLING_ERROR", [conversation_id, caller])

    async def send_message_to_client(
        self, conversation_id, address, final_response, retry_count
    ):
        """Send the final response back to the client."""
        try:
            async with aio.insecure_channel(address) as channel:
                stub = client_pb2_grpc.ClientReceiverStub(channel)
                result = await stub.ReturnFinalResult(client_pb2.ReturnFinalResultRequest(
                    conversation_id=conversation_id,
                    final_response=f"[{self.name}] Response: {final_response}",
                ))
                return ("CALLING_CLIENT_DONE", result)
        except Exception as e:
            logger.error(f"[{self.name}] Error contacting client: {e}")
            if retry_count > 0:
                return await self.send_message_to_client(
                    conversation_id, address, final_response, retry_count - 1
                )
            return ("CLIENT_CALLING_ERROR", conversation_id)

    # ── Delegation supervision ────────────────────────────────────────────────

    def _start_monitor(self, conversation_id, req_id, agent_name, address):
        """Start supervising an accepted delegation."""
        self._monitors[req_id] = asyncio.create_task(self._monitor_delegation(
            conversation_id, req_id, agent_name, address
        ))

    def _stop_monitor(self, req_id):
        """Cancel the monitor of a delegation resolved by its delegate."""
        task = self._monitors.pop(req_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _probe_delegation(self, address, conversation_id, req_id, agent_name):
        """Probe a delegate. Returns WORKING, LOST or UNREACHABLE."""
        try:
            async with aio.insecure_channel(address) as channel:
                stub = agent_pb2_grpc.AgentStub(channel)
                response = await stub.Heartbeat(
                    agent_pb2.HeartbeatRequest(
                        conversation_id=conversation_id, request_id=req_id
                    ),
                    timeout=HEARTBEAT_TIMEOUT,
                )
        except Exception as e:
            logger.debug(f"[{self.name}] Heartbeat to {agent_name} failed: {e}")
            return "UNREACHABLE"

        if response.agent_name != agent_name:
            logger.warning(
                f"[{self.name}] Address {address} is now served by "
                f"{response.agent_name}, expected {agent_name}."
            )
            return "LOST"
        return "WORKING" if response.working else "LOST"

    async def _monitor_delegation(self, conversation_id, req_id, agent_name, address):
        """Supervise one delegation until it is resolved or declared failed.

        Cancelled by _stop_monitor when the result arrives. No deadline is
        imposed on the delegation itself: failure is declared only on positive
        evidence of unreachability or of lost state on the delegate side.
        """
        consecutive_failures = 0

        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
            outcome = await self._probe_delegation(
                address, conversation_id, req_id, agent_name
            )

            if outcome == "WORKING":
                consecutive_failures = 0
                continue

            if outcome == "LOST":
                reason = (
                    f"DELEGATION FAILURE: {agent_name} is reachable but is no longer "
                    f"handling your request; it lost the state of the delegation."
                )
                break

            consecutive_failures += 1
            if consecutive_failures >= HEARTBEAT_FAILURE_THRESHOLD:
                reason = (
                    f"DELEGATION FAILURE: {agent_name} accepted your request but "
                    f"became unreachable before returning a result."
                )
                break

        await self._declare_delegation_failed(
            conversation_id, req_id, agent_name, reason
        )

    async def _declare_delegation_failed(
        self, conversation_id, req_id, agent_name, reason
    ):
        """Turn a detected failure into an observation, as any other result."""
        logger.warning(f"[{self.name}] {reason}")

        resolved_pred, all_received = await self.state.register_result(
            conversation_id, req_id, agent_name, reason
        )

        self._monitors.pop(req_id, None)

        if resolved_pred is None:
            # The real response won the race; nothing to do.
            return

        self.state.mark_abandoned(req_id)

        if all_received:
            task = asyncio.create_task(
                self.handle_response(conversation_id, resolved_pred)
            )
            task.add_done_callback(self.handle_call_back)



    # ── Async callback handling ───────────────────────────────────────────────

    def handle_call_back(self, task):
        """Handle completion of asynchronous tasks."""
        try:
            result = task.result()
            asyncio.create_task(self._process_callback_result(result))
        except Exception as e:
            logger.error(f"[{self.name}] Error in async task: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _process_callback_result(self, result):
        """Process the result of completed asynchronous tasks."""
        message_type = result[0]

        if message_type == "CALLER_CALLING_ERROR":
            conversation_id, caller = result[1]
            await self.state.append_history_note(
                conversation_id,
                f"Your response to {caller} could not be delivered: {caller} was "
                f"unreachable and never received it.",
            )

        elif message_type == "CLIENT_CALLING_ERROR":
            await self.state.append_history_note(
                result[1],
                "Your response could not be delivered to the client, which was "
                "unreachable and never received it.",
            )

        elif message_type == "RESULT_REJECTED":
            conversation_id, caller = result[1]
            await self.state.append_history_note(
                conversation_id,
                f"Your response to {caller} was delivered but not accepted: {caller} "
                f"had already stopped waiting for it and continued without your "
                f"contribution.",
            )

        elif message_type == "AGENT_INTERACTION_ERROR":
            conversation_id = result[1][0]
            pred_req_id = result[1][1]

            caller = await self.state.teardown(
                conversation_id, pred_req_id,
                final_messages=[
                    "An error occurred and you couldn't continue processing the request."
                ],
            )
            address = self.network_info[caller]
            error_message = "AGENT ERROR OCCURRED, THE REQUEST COULDN'T BE PROCESSED"

            if caller != "Client":
                task = asyncio.create_task(self.send_message_to_caller(
                    conversation_id, pred_req_id, address, error_message, MAXIMUM_RETRY, caller
                ))
            else:
                task = asyncio.create_task(self.send_message_to_client(
                    conversation_id, address, error_message, MAXIMUM_RETRY
                ))
            task.add_done_callback(self.handle_call_back)

        elif message_type == "AGENT_CALLING_RESULTS":
            for item in result[1]:
                if item[0] == "AGENT_CALLING_ERROR":
                    req_id, conv_id, agent_name = item[1]
                    self.state.mark_abandoned(req_id)
                    pred_req_id, all_received = await self.state.register_result(
                        conv_id, req_id, agent_name,
                        f"DELEGATION FAILURE: {agent_name} could not be reached and "
                        f"never accepted your request.",
                    )
                    if all_received and pred_req_id:
                        task = asyncio.create_task(
                            self.handle_response(conv_id, pred_req_id)
                        )
                        task.add_done_callback(self.handle_call_back)

    # ── gRPC server ───────────────────────────────────────────────────────────

    async def serve(self):
        """Start the gRPC server and begin listening for requests."""
        server = aio.server()
        agent_pb2_grpc.add_AgentServicer_to_server(self, server)
        bind_address = f"{self.host}:{self.port}"
        server.add_insecure_port(f"[::]:{self.port}")
        await server.start()
        logger.info(f"[{self.name}] Listening on {bind_address}")
        try:
            await server.wait_for_termination()
        except (asyncio.CancelledError, KeyboardInterrupt):
            await server.stop(grace=0)
        except Exception as e:
            logger.error(f"[{self.name}] Server terminated due to error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await server.stop(grace=0)