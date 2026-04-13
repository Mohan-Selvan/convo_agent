from typing import AsyncIterator

from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import os
from events import *
import asyncio
import json

load_dotenv()

class STT:

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