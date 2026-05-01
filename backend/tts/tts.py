from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines.system_engine import SystemEngine


class TTS:

    def __init__(self):
        self.engine = SystemEngine()
        self.stream = TextToAudioStream(self.engine)

    def feed(self, text: str) -> None:
        if not text:
            return
        self.stream.feed(text)
        if not self.stream.is_playing():
            self.stream.play_async(muted=False)

    def stop(self) -> None:
        if self.stream.is_playing():
            self.stream.stop()

    def is_playing(self) -> bool:
        return self.stream.is_playing()
