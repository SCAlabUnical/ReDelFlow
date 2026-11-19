"""Simple gRPC client script for interacting with the distributed agent orchestrator."""

import argparse
import asyncio
import json
import logging
import uuid

from grpc import aio

from redel_flow.core.proto import agent_pb2, agent_pb2_grpc
from redel_flow.core.proto import client_pb2, client_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientReceiver(client_pb2_grpc.ClientReceiverServicer):
    """gRPC server implementation to receive final results from agents."""

    def __init__(self):
        self.pending_results = {}  # conversation_id → asyncio.Future

    async def ReturnFinalResult(self, request, context):
        conv_id = request.conversation_id
        logger.info(f"[CLIENT] ✅ ReturnFinalResult received for conv_id={conv_id}")
        if conv_id in self.pending_results:
            self.pending_results[conv_id].set_result(request.final_response)
        else:
            logger.error(f"[CLIENT] ⚠️ Unexpected conv_id {conv_id}")
        return client_pb2.ReturnFinalResultResponse(conversation_id=conv_id, ack=True)

    async def serve(self, port: int):
        server = aio.server()
        client_pb2_grpc.add_ClientReceiverServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{port}")
        await server.start()
        logger.info(f"[CLIENT] Listening on port {port} for ReturnFinalResult")
        return server


async def main(orchestrator_address: str, client_port: int):
    receiver = ClientReceiver()
    server = await receiver.serve(client_port)
    conv_id = str(uuid.uuid4())

    async with aio.insecure_channel(orchestrator_address) as channel:
        stub = agent_pb2_grpc.AgentStub(channel)

        while True:
            user_input = input("Enter message to send (or 'exit'): ")

            if user_input.lower() == "exit":
                break

            future = asyncio.get_running_loop().create_future()
            receiver.pending_results[conv_id] = future
            request_id = str(uuid.uuid4())

            logger.info(f"[CLIENT] Sending request with conv_id={conv_id}")
            try:
                await stub.Invoke(agent_pb2.InvokeRequest(
                    conversation_id=conv_id,
                    request_id=request_id,
                    sender="Client",
                    request=user_input
                ))
            except Exception as e:
                logger.error(f"[CLIENT] ❌ Error during submit: {e}")
                receiver.pending_results[conv_id].set_result("Orchestrator couldn't be reached")
                del receiver.pending_results[conv_id]
                continue

            result = await future

            print(f"\nFinal result: {result}\n")
            logger.info(f"[CLIENT] Received result for conv_id={conv_id}")

    await server.stop(grace=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReDel client.")
    parser.add_argument("--topology", required=True, help="Path to topology JSON file")
    args = parser.parse_args()

    with open(args.topology) as f:
        raw = json.load(f)

    orchestrator_cfg = raw.get("orchestrator", {})
    client_cfg = raw.get("client", {})

    orchestrator_address = f"{orchestrator_cfg['host']}:{orchestrator_cfg['port']}"
    client_port = int(client_cfg["port"])

    asyncio.run(main(orchestrator_address, client_port))