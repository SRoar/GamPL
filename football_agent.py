import os
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from uagents import Agent, Context, Model, Protocol
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage

# Import chat protocol components
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    EndSessionContent,
    StartSessionContent,
)

from football import get_team_info, FootballTeamRequest, FootballTeamResponse

agent = Agent()

# AI Agent for structured output (choose one)
AI_AGENT_ADDRESS = 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y'  # OpenAI Agent
# AI_AGENT_ADDRESS = 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx'  # Claude Agent

if not AI_AGENT_ADDRESS:
    raise ValueError("AI_AGENT_ADDRESS not set")

# Create the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Create structured output protocol
struct_output_client_proto = Protocol(
    name="StructuredOutputClientProtocol", version="0.1.0"
)

# Structured output models
class StructuredOutputPrompt(Model):
    prompt: str
    output_schema: dict[str, Any]

class StructuredOutputResponse(Model):
    output: dict[str, Any]

# Optional: Rate limiting protocol for direct requests
proto = QuotaProtocol(
    storage_reference=agent.storage,
    name="Football-Team-Protocol",
    version="0.1.0",
    default_rate_limit=RateLimit(window_size_minutes=60, max_requests=30),
)

# Chat protocol message handler
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Got a message from {sender}: {msg.content}")
    ctx.storage.set(str(ctx.session), sender)
    
    # Send acknowledgement
    await ctx.send(
        sender,
        ChatAcknowledgement(
            acknowledged_msg_id=msg.msg_id, 
            timestamp=datetime.now(timezone.utc)
        ),
    )

    # Process message content
    for content in msg.content:
        if isinstance(content, StartSessionContent):
            ctx.logger.info(f"Got a start session message from {sender}")
            continue
        elif isinstance(content, TextContent):
            ctx.logger.info(f"Got a message from {sender}: {content.text}")
            ctx.storage.set(str(ctx.session), sender)
            
            # Send to AI agent for structured output extraction
            await ctx.send(
                AI_AGENT_ADDRESS,
                StructuredOutputPrompt(
                    prompt=content.text, 
                    output_schema=FootballTeamRequest.schema()
                ),
            )
        else:
            ctx.logger.info(f"Got unexpected content from {sender}")

# Handle structured output response from AI agent
@struct_output_client_proto.on_message(StructuredOutputResponse)
async def handle_structured_output_response(
    ctx: Context, sender: str, msg: StructuredOutputResponse
):
    session_sender = ctx.storage.get(str(ctx.session))
    if session_sender is None:
        ctx.logger.error(
            "Discarding message because no session sender found in storage"
        )
        return

    if "<UNKNOWN>" in str(msg.output):
        error_response = ChatMessage(
            content=[TextContent(text="Sorry, I couldn't process your request. Please try again later.")],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc)
        )
        await ctx.send(session_sender, error_response)
        return

    try:
        # Parse the structured output
        prompt = FootballTeamRequest.parse_obj(msg.output)
        
        # Get team information
        team_info = await get_team_info(prompt.team_name)
        
        # Create response message
        response = ChatMessage(
            content=[TextContent(text=team_info)],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc)
        )
        
        await ctx.send(session_sender, response)
        
    except Exception as err:
        ctx.logger.error(f"Error processing structured output: {err}")
        error_response = ChatMessage(
            content=[TextContent(text="Sorry, I couldn't process your request. Please try again later.")],
            msg_id=uuid4(),
            timestamp=datetime.now(timezone.utc)
        )
        await ctx.send(session_sender, error_response)

# Chat protocol acknowledgement handler
@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}"
    )

# Optional: Direct request handler for structured requests
@proto.on_message(
    FootballTeamRequest, replies={FootballTeamResponse, ErrorMessage}
)
async def handle_request(ctx: Context, sender: str, msg: FootballTeamRequest):
    ctx.logger.info("Received team info request")
    try:
        results = await get_team_info(msg.team_name)
        ctx.logger.info("Successfully fetched team information")
        await ctx.send(sender, FootballTeamResponse(results=results))
    except Exception as err:
        ctx.logger.error(err)
        await ctx.send(sender, ErrorMessage(error=str(err)))

# Register protocols
agent.include(chat_proto, publish_manifest=True)
agent.include(struct_output_client_proto, publish_manifest=True)
agent.include(proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()