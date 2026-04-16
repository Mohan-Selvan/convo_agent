from typing import AsyncIterator

from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import os
from events import *
import asyncio
import json

load_dotenv()

class STT_Vosk:

    def __init__(self, sample_rate=16000):
        
        model_path = os.getenv("VOSK_MODEL_PATH", "")
        if not model_path:
            raise ValueError("VOSK_MODEL_PATH environment variable is not set.")
        
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(Model(model_path), sample_rate)

    async def transcribe(self, audio_chunk: bytes) -> STTEvent:
        
        loop = asyncio.get_running_loop()

        is_final: bool = await loop.run_in_executor(None, self.recognizer.AcceptWaveform, audio_chunk)

        if is_final:
            result = await loop.run_in_executor(None, self.recognizer.Result)
            result_dict = json.loads(result)
            return STTOutputEvent.create(transcript=result_dict.get("text", ""))
        else:
            partial_result = await loop.run_in_executor(None, self.recognizer.PartialResult)
            partial_result_dict = json.loads(partial_result)
            return STTChunkEvent.create(transcript=partial_result_dict.get("partial", ""))


from RealtimeSTT import AudioToTextRecorder
import torch
import asyncio

class STT:

    def __init__(self, sample_rate=16000):

        self.queue = asyncio.Queue()
        self.sample_rate = sample_rate
        self.is_closed = False
        self.recorder = AudioToTextRecorder(
            spinner=False,
            compute_type="auto",
            model="large-v2",
            realtime_model_type="tiny.en",
            language="en",
            sample_rate=16000,
            device="cpu", #"cuda" if torch.cuda.is_available() else "cpu",
            use_microphone=False,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self._on_transcription_update,
            on_realtime_transcription_stabilized=self._on_transcription_stabilized,
            silero_use_onnx=True,
            initial_prompt_realtime="""
End incomplete sentences with ellipses.
Examples:
Complete: "The sky is blue."
Incomplete: "When the sky..."
Complete: "She walked home."
Incomplete: "Because he..."
"""
        )

        self.recorder.start()

    def _on_transcription_update(self, transcript: str):
        self.queue.put_nowait(STTChunkEvent.create(transcript=transcript))

    def _on_transcription_stabilized(self, transcript: str):
        self.queue.put_nowait(STTOutputEvent.create(transcript=transcript))

    async def transcribe(self, audio_chunk: bytes) -> None:
        self.recorder.feed_audio(audio_chunk, original_sample_rate=self.sample_rate)
        await asyncio.sleep(0) 

    async def receive_events(self) -> AsyncIterator[VoiceAgentEvent]:
        while not self.is_closed:
            event = await self.queue.get()
            yield event

    def close(self):
        self.is_closed = True
        self.recorder.stop()

