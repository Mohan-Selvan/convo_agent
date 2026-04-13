import asyncio
import wave
import websockets
from tqdm import tqdm

async def steam_audio(ws_url:str, file_path:str, loop: bool = True):

    async with websockets.connect(ws_url) as websocket:
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            chunk_size = 1024  # Number of frames per chunk
            total_frames = wf.getnframes()
            total_chunks = total_frames // chunk_size

            print(f"Streaming audio: {file_path} (Sample Rate: {sample_rate}, Channels: {num_channels})")

            with tqdm(total=total_chunks, desc="Streaming Audio") as pbar:
                while True:
                    data = wf.readframes(chunk_size)
                    if not data:
                        if loop:
                            wf.rewind()  # Restart the audio file
                            pbar.reset()
                            continue
                        else:
                            break
                    await websocket.send(data)
                    await asyncio.sleep(chunk_size / sample_rate)  # Sleep to simulate real-time streaming
                    pbar.update(1)

if __name__ == "__main__":
    ws_url = "ws://localhost:8000/ws"
    file_path = "./data/OSR_us_000_0061_16k.wav" 
    asyncio.run(steam_audio(ws_url, file_path, loop=True))