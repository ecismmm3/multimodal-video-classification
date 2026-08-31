# multimodal-video-classification

- research experiment classifying different categories of videos
- audio latents extracted from KVAE-Audio
- video latents extracted from VideoMAE
- custom classification model trained on combined latents

# Dataset
- AudioSet (https://research.google.com/audioset/download.html)
- Download unbalanced_trained_segments.csv for the largest dataset

# Tools
- Audio Class Hiearchy Generator <insert name>
- Choose two classes you would like to classify between
- Save their codes somewhere that you can look back on

## Third-Party Models

This project uses the following pretrained models:

- VideoMAE Base — MCG-NJU/VideoMAE
- KVAE-Audio — Kandinsky Lab

The pretrained models are used for feature/latent extraction. No pretrained model weights are included in this repository.
