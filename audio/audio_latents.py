import torch
import torchaudio
import os

from kvae_audio.model.kvae_audio import KVAEAudio

from scipy.io import wavfile


device = "cuda" if torch.cuda.is_available() else "cpu"

model = KVAEAudio(use_attn = True)

checkpoint = torch.load( # Weights
    "kvae-audio/kvae-audio.pt",
    map_location=device
)

model.load_state_dict(checkpoint["state_dict"])

model = model.to(device)
model.eval()

audio_folder = "audio_side/audios/processed_water_audios"

latent_folder = "combined_latents/water_latents_audio"

os.makedirs(latent_folder, exist_ok=True)

for audio_file in os.listdir(audio_folder):

    if not audio_file.endswith(".wav"):
        continue

    audio_path = os.path.join(audio_folder, audio_file)

    print("Processing:", audio_file)

    try:

        sr, audio = wavfile.read(audio_path)

        audio = torch.tensor(audio, dtype=torch.float32)

        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T

        audio = audio.unsqueeze(0)

        audio = audio.to(device)

        with torch.no_grad():

            output = model(
                audio,
                sample_rate=sr,
                sample=False
            )

        z = output["z"] # latent

        z = z.squeeze(0)

        z = z.mean(dim=1)

        audio_id = os.path.splitext(audio_file)[0]

        torch.save(
            z.cpu(),
            os.path.join(
                latent_folder,
                f"{audio_id}.pt"
            )
        )

        print(
            "Saved:",
            audio_id,
            z.shape
        )

    except Exception as e:

        print("FAILED:", audio_file)
        print(e)
        raise

print(len(os.listdir(latent_folder)))