# multimodal-video-classification

* research experiment classifying different categories of videos
* audio latents extracted from `KVAE-Audio`
* video latents extracted from `VideoMAE`
* custom classification model trained on combined latents

# Downloads

* AudioSet (`https://research.google.com/audioset/download.html`)
* Download `unbalanced_trained_segments.csv` for the largest dataset
* Download `ontology.json`

# Finding Classes

* Audio Class Hiearchy Generator (`ontology.html`)
* Upload `ontology.json`
* Choose two classes you would like to classify between
* Save their codes somewhere that you can refer back to

# Usage Steps (for each class)

* `sort_csv_1.py` - extracts all rows with corresponding label from dataset into a separate csv
* `download_audio.py` - for each video in your new csv, downloads audio only into a folder
* `download_video.py` - for each video in your new csv, downloads video only into a folder
* `delete_no_pair.py` - sometimes there may be an issue downloading both videos/audios from Youtube, so the script will skip over them. Since audio and video files are named and paired by YTID, running this script will delete the video and audio files that do not have a pair

## Audio

* `formatting.py` - reformats each audio file for processing and saves it into a new folder
* `audio_latents.py` - utilizes `kvae_audio` model weights to extract latent representation (`.pt` file) into a new folder

## Video

* `video_latents.py` - utilizes `VideoMAE` to extract latent representation + class label (0/1), also as a `.pt` file, into a new folder

## Dataset

* `combine.py` - combines audio latent, video latent, and label into one `.pt` file
* `split.py` - splits `.pt` files from both classes randomly at a (80/10/10 train/test/val) split into respective folders

## Training/Predicting

* `train.py` - trains model with your data, saves weights to `classifier.pt`

* `predict.py` - loads `classifier.pt`, predicts on a test `.pt` file

* `train_audio.py` - classifier trained only on audio latents

* `train_video.py` - classifier trained only on video latents

^ To test which modality had a higher impact on the prediction

Usage: `python predict.py path/to/sample.pt`

# Third-Party Models

This project uses the following pretrained models:

* `VideoMAE Base` — `MCG-NJU/VideoMAE`
* `KVAE-Audio` — `Kandinsky Lab`

The pretrained models are used for feature/latent extraction. No pretrained model weights are included in this repository.
