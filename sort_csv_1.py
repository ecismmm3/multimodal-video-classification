# Add files from eval_segments to "water" based on if they have the label

import csv

with open("eval_segments.csv", "r", newline="", encoding="utf-8") as infile, \
     open("water.csv", "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if any("/m/0838f" in field for field in row):
            writer.writerow([row[0],row[1], row[2],",".join(row[3:])])