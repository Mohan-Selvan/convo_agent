import asyncio
import sounddevice as sd
import websockets
import numpy as np

WS_URL = "ws://localhost:8000/ws"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


async def stream_mic():
    async with websockets.connect(WS_URL) as ws:

        def callback(indata, frames, time, status):
            if status:
                print(status)

            # convert float32 → int16 PCM
            audio = (indata * 32767).astype(np.int16)

            # send raw bytes
            asyncio.run_coroutine_threadsafe(
                ws.send(audio.tobytes()),
                loop
            )

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=callback
        ):
            print("🎤 Streaming mic... Press Ctrl+C to stop")
            await asyncio.Future()  # run forever


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(stream_mic())