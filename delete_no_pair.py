import os

video_folder = "video_image/clips/wind_vids"
audio_folder = "audio_side/audios/wind_audios"

missing_vids = 0
missing_audios = 0

# Get IDs without extensions
video_ids = {
    os.path.splitext(f)[0]
    for f in os.listdir(video_folder)
    if f.endswith(".mp4")
}

audio_ids = {
    os.path.splitext(f)[0]
    for f in os.listdir(audio_folder)
    if f.endswith(".wav")
}

# Find missing pairs
missing_video = video_ids - audio_ids
missing_audio = audio_ids - video_ids

# Delete videos without audio
for vid in missing_video:
    path = os.path.join(video_folder, vid + ".mp4")
    os.remove(path)
    print("Delete: ", path)
    missing_vids += 1

# Delete audios without video
for aud in missing_audio:
    path = os.path.join(audio_folder, aud + ".wav")
    os.remove(path)
    print("Delete: ", path)
    missing_audios += 1

print("Done")
print(missing_vids)
print(missing_audios)

print(len(video_ids & audio_ids))