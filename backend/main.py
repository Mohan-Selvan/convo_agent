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
from tts.tts import TTS
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

    async def send_audio():
        async for audio_chunk in audio_stream:
            await stt.transcribe(audio_chunk)

    send_task = asyncio.create_task(send_audio())
    await asyncio.sleep(0.2)

    try:
        async for event in stt.receive_events():
            if event:
                print(event)
            yield event
    except Exception as e: 
        print(f"Error in STT stream: {e}")
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
            await stt.close()
            
        print("STT stream stopped.")



def should_barge_in(event: VoiceAgentEvent) -> bool:
    return event.type == "voice_start"


async def _agent_stream(event_stream: AsyncIterator[VoiceAgentEvent]) -> AsyncIterator[VoiceAgentEvent]:
    """Run the agent on each STT output. Cancel the in-flight turn when should_barge_in fires."""

    thread_id = str(uuid4())
    out: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    agent_task: asyncio.Task | None = None

    async def run_agent(transcript: str):
        buffer: list[str] = []
        try:
            print(f"Human: {transcript}")
            stream = agent.astream(
                {"messages": [HumanMessage(content=transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )
            async for message, _ in stream:
                if isinstance(message, AIMessage):
                    buffer.append(message.content)
                    await out.put(AgentChunkEvent.create(text=message.text))
            response = "".join(buffer)
            print(f"Agent Response: {response}")
            await out.put(AgentEndEvent.create(text=response))
        except asyncio.CancelledError:
            print(f"[barge-in] agent turn cancelled (partial={''.join(buffer)!r})")
            raise

    async def cancel_agent():
        nonlocal agent_task
        if agent_task and not agent_task.done():
            agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await agent_task

    async def pump():
        nonlocal agent_task
        try:
            async for event in event_stream:
                await out.put(event)

                if should_barge_in(event):
                    await cancel_agent()

                if event.type == "stt_output":
                    await cancel_agent()
                    agent_task = asyncio.create_task(run_agent(event.transcript))
        finally:
            await out.put(sentinel)

    pump_task = asyncio.create_task(pump())

    try:
        while True:
            item = await out.get()
            if item is sentinel:
                break
            yield item
    finally:
        pump_task.cancel()
        if agent_task:
            agent_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
        if agent_task:
            await asyncio.gather(agent_task, return_exceptions=True)

async def _tts_stream(event_stream: AsyncIterator[VoiceAgentEvent]) -> AsyncIterator[VoiceAgentEvent]:
    """Transforms AgentChunkEvents into TTSChunkEvents by synthesizing audio from text."""

    tts = TTS()

    async for event in event_stream:
        yield event

        if event and event.type == "agent_end":
            print(event.text)
            await tts.synthesize(event.text)


async def _stt_debug_stream(event_stream: AsyncIterator[VoiceAgentEvent]) -> AsyncIterator[VoiceAgentEvent]:
    """Debug stream that logs incoming events."""

    async for event in event_stream:

        if not event:
            continue

        if event.type == "voice_start":
            print("voice_start: user started speaking")

        if event.type == "voice_stop":
            print("voice_stop: user finished, transcribing...")

        if event.type == "stt_chunk" and len(event.transcript) > 0:
            print(f"chunk: {event.transcript} ")

        if event.type == "stt_output" and len(event.transcript) > 0:
            print(f"output: {event.transcript}")
            print("---")

        yield event


pipeline = (
    RunnableGenerator(_stt_stream)
    | RunnableGenerator(_stt_debug_stream)
    | RunnableGenerator(_agent_stream)
)
# TTS stage stays out until we wire streaming audio back to the client.

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