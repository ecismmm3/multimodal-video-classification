import os
import torch

video_folder = "combined_latents/water_latents_vid"
audio_folder = "combined_latents/water_latents_audio"

combined_path = "combined_latents/water_complete"

num = 1

for file in os.listdir(video_folder):

    if not file.endswith(".pt"):
        continue

    video_path = os.path.join(video_folder, file)
    audio_path = os.path.join(audio_folder, file)

    video_data = torch.load(video_path)
    audio_data = torch.load(audio_path)

    # add audio entry
    video_data["audio"] = audio_data

    # overwrite the video file (or save somewhere else)
    torch.save(video_data, os.path.join(combined_path, file))

    print(f'Saved {file} ({num})')

    num += 1

print("Done.")