import librosa
import soundfile as sf

y, sr = librosa.load("audio.wav", sr=22050)  # Load audio to pitch shift.
# y is a numpy array of the wav file, sr = sample rate
y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=2)  # Shifted by 2 half steps (change as required).
sf.write("audio_pitch_shifted.wav", y_shifted, 22050)  # Name of pitch shifted audio.

# Once the audios are pitch shifted, they can be used in the training audio and the process remains the same.
