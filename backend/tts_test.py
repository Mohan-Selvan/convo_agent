import numpy as np
import sounddevice as sd

from RealtimeTTS import TextToAudioStream, SystemEngine


def play_chunk(chunk: bytes):
    """Callback: play audio chunk"""
    audio = np.frombuffer(chunk, dtype=np.int16)

    # reshape for mono channel
    audio = audio.reshape(-1, 1)

    sd.play(audio, samplerate=16000)
    sd.wait()


def main():
    text = "Hello! This is a real time text to speech test."

    engine = SystemEngine()
    stream = TextToAudioStream(engine)

    stream.feed(text)

    # play with streaming callback
    stream.play(
        on_audio_chunk=lambda chunk: None,
        muted=False  # prevent double playback
    )


if __name__ == "__main__":
    main()