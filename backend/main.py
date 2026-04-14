import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles
from agent import agent

from stt.stt import STT
from events import *


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _stt_stream(audio_stream: AsyncIterator[bytes],) -> AsyncIterator[VoiceAgentEvent]:
    """Transforms a stream of audio bytes into a stream of VoiceAgentEvents using STT module for transcription."""
    
    stt = STT(sample_rate=16000)
    print(f"STT Initialized..")

    try:
        async for audio_chunk in audio_stream:
            stt_event = await stt.transcribe(audio_chunk)
            yield stt_event
    except Exception as e: 
        print(f"Error in STT stream: {e}")
    finally:
        print("STT stream stopped.")



async def _agent_stream(event_stream:AsyncIterator[VoiceAgentEvent]) -> AsyncIterator[VoiceAgentEvent]:
    """Processes a stream of VoiceAgentEvents through anagent and yields resulting events."""
    
    thread_id = str(uuid4())

    buffer: list[str] = []

    async for event in event_stream:
        yield event

        if event.type == "stt_output":

            print(f"Human: , {event.transcript}\n")

            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id }},
                stream_mode="messages",
            )

            async for message, metadata in stream:
                if isinstance(message, AIMessage):
                    yield AgentChunkEvent.create(text=message.text)
                    buffer.append(message.text)

            print(f"Agent response : {''.join(buffer)}\n\n{'-'*50}\n")
            buffer.clear()
            yield AgentEndEvent.create()





pipeline = (RunnableGenerator(_stt_stream) | RunnableGenerator(_agent_stream))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        try:
            while True:
                data = await websocket.receive_bytes()
                yield data
        except Exception as e:
            print(f"WebSocket audio stream ended: {e}")

    output_stream = pipeline.atransform(websocket_audio_stream())
    async for event in output_stream:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)