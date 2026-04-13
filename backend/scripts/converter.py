import soundfile as sf
import librosa

sample_rate = 16000
audio, sr = librosa.load("./data/OSR_us_000_0061_8k.wav", sr=sample_rate, mono=True)
sf.write("./data/OSR_us_000_0061_16k.wav", audio, sample_rate)
print("File converted to 16kHz mono and saved as audio_sample_16k.wav")