import csv
from yt_dlp import YoutubeDL
import subprocess
import os

num = 0  # successful clip counter

with open('water.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')

    for row in csv_reader:

        ytid = row[0]
        start = row[1]
        end = row[2]

        url = f"https://www.youtube.com/watch?v={ytid}"

        temp_audio = f"temp_audio_{ytid}.wav"
        output_audio = f"audio_side/files/water_audios/{ytid}.wav"

        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": f"temp_audio_{ytid}.%(ext)s",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }
                ],
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            subprocess.run([
                "ffmpeg",
                "-ss", str(start),
                "-to", str(end),
                "-i", temp_audio,
                "-ac", "1",
                "-ar", "16000",
                output_audio
            ], check=True)

            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            print(f"Saved {output_audio}")

            num += 1

        except Exception as e:
            print(f"Skipping {ytid}: {e}")

            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            continue


with open('audio.txt', 'a') as f:
    f.write(str(num) + "\n")