import os
import librosa
import soundfile as sf

input_folder = r"audio_side/audios/water_audios"
output_folder = r"audio_side/audios/processed_water_audios"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):

    if not file.endswith(".wav"):
        continue

    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)

    print("Processing:", file)

    audio, sr = librosa.load(
        input_path,
        sr=48000,
        mono=True
    )

    sf.write(
        output_path,
        audio,
        sr
    )

    print("Saved:", output_path)

print("Done.")