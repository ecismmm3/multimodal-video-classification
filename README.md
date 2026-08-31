# multimodal-video-classification

# Downloads

* AudioSet (`https://research.google.com/audioset/download.html`).
* Download `unbalanced_trained_segments.csv` for the largest dataset.
* Download `ontology.json`.

# Choosing Classes

* Class Hierarchy Visualizer (`ontology.html`).
* Upload `ontology.json`.
* Choose two classes you would like to classify between.
* Save their codes somewhere that you can refer back to.

# Usage Steps (for each class)

* `sort_csv.py` - Extracts all rows with corresponding label from dataset into a separate CSV.
* `download_audio.py` - For each video in your new CSV, downloads audio only into a folder.
* `download_video.py` - For each video in your new CSV, downloads video only into a folder.
* `delete_no_pair.py` - Sometimes there may be an issue downloading both videos/audios from YouTube, so the script will skip over them. Since audio and video files are named and paired by YTID, running this script will delete the video and audio files that do not have a pair.

## Audio (`audio/`)

* `formatting.py` - Reformats each audio file for processing and saves it into a new folder.
* `audio_latents.py` - Utilizes `kvae_audio` model weights to extract latent representation (`.pt` file) into a new folder.

## Video (`video/`)

* `video_latents.py` - Utilizes `VideoMAE` to extract latent representation + class label (0/1), also as a `.pt` file, into a new folder.

## Dataset (`data/`)

* `combine.py` - Combines audio latent, video latent, and label into one `.pt` file.
* `split.py` - Splits `.pt` files from both classes randomly at a (80/10/10 train/test/val) split into respective folders.

## Training/Predicting (`scripts/`)

* `train.py` - Trains model with your data, saves weights to `classifier.pt`.

* `train_audio.py` - Classifier trained only on audio latents. Saves weights to `audio_only.pt`  

* `train_video.py` - Classifier trained only on video latents. Saves weights to `video_only.pt`

^ To test which modality had a higher impact on the prediction.

* `predict.py` - Loads `classifier.pt`, predicts on a test `.pt` file.

Usage: `python predict.py path/to/sample.pt`

# Experimental Data

## Overview
- Goal: Classify between wind and water
- Trained on clips of windy environments and "watery" conditions

## Training Stats (after 50 epochs)

| Model      | Modalities Used | Test Accuracy |
| ---------- | --------------- | ------------: |
| Multimodal | Audio + Video   |     **84.8%** |
| Video      | Video only      |     **81.8%** |
| Audio      | Audio only      |     **48.5%** |

**Conclusion:** This specific classification is highly dependent on video, much more than audio. The video latent captures both spatial and temporal information, which is useful for classification, while the audio latent preserves important acoustic information for reconstruction, but is not necessarily good for classification.

**Reasoning:** This could be because you can "see" water in a video, while wind is not directly visible. This makes the visual features of water more distinguishable, and the sounds of wind and water are too similar for the latent to provide consistent information.

# Third-Party Models

This project uses the following pretrained models:

* `VideoMAE Base` — `MCG-NJU/VideoMAE`.
* `KVAE-Audio` — `Kandinsky Lab`.

The pretrained models are used for feature/latent extraction. No pretrained model weights are included in this repository.
