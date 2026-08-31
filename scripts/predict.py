import torch
import torch.nn as nn
import sys

# class WindWaterClassifier(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.net = nn.Sequential(

#             nn.Linear(832, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(128, 2)
#         )


#     def forward(self, x):
#         return self.net(x)

class WindWaterClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU()
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )


    def forward(self, video, audio):

        video_features = self.video_proj(video)
        audio_features = self.audio_proj(audio)

        fused = torch.cat(
            [video_features, audio_features],
            dim=1
        )

        return self.classifier(fused)


def predict(file):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = WindWaterClassifier()

    checkpoint = torch.load(
        "wind_water_classifier.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)

    model.eval()

    data = torch.load(
        file,
        map_location="cpu"
    )


    video = data["video"].float()
    audio = data["audio"].float()

    video = (video - video.mean()) / (video.std() + 1e-8)
    audio = (audio - audio.mean()) / (audio.std() + 1e-8)

    video = video.to(device)
    audio = audio.to(device)

    with torch.no_grad():

        video = video.unsqueeze(0)
        audio = audio.unsqueeze(0)

        logits = model(video, audio)

        probs = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probs,
            dim=1
        ).item()

    labels = {
        0: "water",
        1: "wind"
    }


    print(
        "Prediction:",
        labels[prediction]
    )

    print(
        "Confidence:",
        probs[0][prediction].item()
    )

    print(
        "Actual label:",
        data["label"]
    )



if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python predict.py path/to/sample.pt"
        )

        exit()

    predict(sys.argv[1])