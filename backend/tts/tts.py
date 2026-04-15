import asyncio
from typing import AsyncIterator
import sounddevice as sd
import numpy as np

from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines.system_engine import SystemEngine


class TTS:

    def _on_audio_chunk(self, chunk: bytes):
        sd.play(np.frombuffer(chunk, dtype=np.int16), samplerate=16000, blocking=False)


    def __init__(self):
        engine = SystemEngine(print_installed_voices=True)
        self.stream = TextToAudioStream(engine)
        self.stream.play_async(debug=False)


    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.stream.feed(text)

        print(f"Feeding text : {text}")

        if(not self.stream.is_playing()):
            self.stream.play_async(debug=False, muted=False)