# LATENTS: WATER = 0, WIND = 1

import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

import numpy as np
from decord import VideoReader, cpu

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = VideoMAEImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base"
)

model = VideoMAEModel.from_pretrained(
    "MCG-NJU/videomae-base"
)

model = model.to(device)

model.eval()

print("Model loaded")

path = "video_image/clips/wind_vids"

num = 1

for video in os.listdir(path):

    print(f"Starting {video}")

    vr = VideoReader(os.path.join(path, video), ctx = cpu())

    num_frames = len(vr)

    #print("Frames:", num_frames)

    indices = np.linspace(0, num_frames - 1, 16).astype(int) # 16 frames

    #print("Frames sampled", indices) # frames sampled

    frames = vr.get_batch(indices).asnumpy()
    #print(frames.shape) # (# frames, height, width, RGB channels)

    # Process frames

    inputs = processor(list(frames), return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract latent

    with torch.no_grad():
        outputs = model(**inputs)

    video_latent = outputs.last_hidden_state
    #print(video_latent.shape) # 1, 1568, 768 (batch size, # tokens/patches, # features per token)

    video_latent = video_latent.mean(dim=1)
    #print(video_latent.shape) # 1, 768 (batch size, average across all tokens)

    video_latent = video_latent.squeeze(0) # (1, 768) --> (768)

    video_id = os.path.splitext(video)[0]

    torch.save({"video": video_latent.cpu(),
                "label": 1}, 
                f'combined_latents/wind_latents/{video_id}.pt')

    print(f"Latent {num} saved.")

    num += 1