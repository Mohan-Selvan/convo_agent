# Voice Consultant (convo_agent)

A streaming voice agent. Audio in, audio out, with a real LLM in the middle and proper barge-in / interruption support. The entire pipeline can run locally with a low-tier to mid-tier hardware.

## TL;DR

`convo_agent` is a fully streaming voice pipeline: microphone audio comes in over a WebSocket, real-time STT produces partial and final transcripts, a LangChain/LangGraph agent (running locally on Ollama) generates a streamed reply, and a TTS engine speaks it back. If the user starts talking again while the agent is mid-response, the in-flight LLM call and the TTS playback are both cancelled. The whole thing is wired together as a `RunnableGenerator` pipeline so each stage is independently testable.

The default sample persona is a customer-service agent for a fictional property-investment firm, used purely to demonstrate the pipeline. Replace the persona prompt to point the agent at a different domain.

## Architecture

```
microphone audio (WebSocket bytes)
  -> STT stream (RealtimeSTT + faster-whisper, Silero VAD)
       emits voice_start / voice_stop / stt_chunk / stt_output events
  -> Agent stream (LangChain create_agent over Ollama, with InMemorySaver checkpointer)
       streams AIMessage chunks; barge-in cancels the in-flight task
  -> TTS stream (RealtimeTTS / Coqui)
       speaks each agent chunk; barge-in stops playback
```

Key behaviours:

- **Streaming end-to-end.** STT, LLM, and TTS all run as async generators chained through `RunnableGenerator`. The user starts hearing a reply before the LLM has finished generating.
- **Barge-in.** A `voice_start` event from the user cancels both the in-flight agent task and the current TTS playback. The system always favours the user.
- **TTS-aware prompting.** The system prompt for the LLM includes detailed voice-output guidelines (no markdown, no emoji, spelled-out numbers, short sentences, terminal punctuation for streaming chunk boundaries) so the model writes for the ear, not the screen.
- **Local-first.** STT runs on faster-whisper (`base.en` + `tiny.en` realtime), the LLM is an Ollama model running locally, and TTS uses Coqui. No cloud calls in the default path.

## Repo tour

```
.
├── backend/
│   ├── main.py          # FastAPI WebSocket entrypoint, RunnableGenerator pipeline
│   ├── agent.py         # LangChain agent + persona/TTS system prompt
│   ├── events.py        # event types for the streaming pipeline
│   ├── stt/stt.py       # RealtimeSTT-based STT class (and a Vosk variant)
│   ├── tts/tts.py       # TTS wrapper
│   ├── tts_test.py      # standalone TTS smoke test
│   ├── scripts/         # mic streamer, file streamer, format converter
│   └── requirements.txt
└── tests/
    └── vad_test.py      # voice-activity detection smoke test
```

## Setup

```bash
git clone https://github.com/Mohan-Selvan/convo_agent
cd convo_agent

python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```

You also need:

- **Ollama** running locally with a chat model pulled. The default in `agent.py` expects an Ollama server at `http://localhost:11434`. Pull a model:
  ```bash
  ollama pull gemma3
  ```
  Update the `model=` argument in `backend/agent.py` to match what you pulled.
- **Vosk model** (optional, only if you switch from RealtimeSTT to the Vosk variant). Download from https://alphacephei.com/vosk/models and set `VOSK_MODEL_PATH` in a `.env` file.

## Run it

Start the WebSocket server:

```bash
cd backend
python main.py
```

This serves a WebSocket at `ws://localhost:8000/ws`. Push raw 16 kHz PCM audio bytes into it and listen on the TTS device for the response.

For local mic-based testing, use the streamer in `backend/scripts/mic_streamer.py` to pipe microphone audio into the WebSocket.

## Note on the persona

`backend/agent.py` ships with a sample persona for a property-investment firm. It is illustrative scaffolding for the pipeline, not a real product or affiliation. Replace the persona prompt to point the agent at your own domain.

## Contact

mohanselvan.r.5814@gmail.com
