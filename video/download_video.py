import csv
from yt_dlp import YoutubeDL
import subprocess
import os

num = 0

with open('wind.csv') as f:
    csv_reader = csv.reader(f, delimiter = ',')
    for row in csv_reader:

        ytid = row[0] # Auto
        start = row[1]
        end = row[2]

        #ytid = "ClokfM86wRM" # Manual

        url = f"https://www.youtube.com/watch?v={ytid}"

        try: 
            ydl_opts = {
                "format": "mp4",
                "outtmpl": "video.%(ext)s",
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Auto:

            subprocess.run([
                "ffmpeg",
                "-ss", str(start),
                "-to", str(end),
                "-i", "video.mp4",
                "-c", "copy",
                f'video_image/clips/wind_vids/{ytid}.mp4'
            ])

            # Manual:

            # subprocess.run([
            #     "ffmpeg",
            #     "-ss", "7.000",
            #     "-to", "17.000",
            #     "-i", "video.mp4",
            #     "-c:v", "libx264",
            #     "-c:a", "aac",
            #     f"video_image/clips/clip_{num}.mp4"
            # ])

            os.remove('video.mp4')

            num += 1

        except Exception as e:
            print(f"Skipping {ytid}: {e}")
            if os.path.exists("video.mp4"):
                os.remove("video.mp4")
            continue

# with open('visual.txt', 'a') as f:
#     f.write("\n" + str(num))

