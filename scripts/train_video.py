import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class WindWaterDataset(Dataset):

    def __init__(self, folder):

        self.files = []

        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".pt"):
                    self.files.append(os.path.join(root, f))

        print(folder, "found", len(self.files), "files")


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        data = torch.load(
            self.files[idx],
            map_location="cpu"
        )

        video = data["video"].float()

        video = (video - video.mean()) / (video.std() + 1e-8)

        label = torch.tensor(
            data["label"]
        ).long()

        return video, label



class WindWaterClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Linear(
            768,
            2
        )


    def forward(self, x):
        return self.net(x)



def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            preds = torch.argmax(
                model(x),
                dim=1
            )

            correct += (
                preds == y
            ).sum().item()

            total += y.size(0)

    return correct / total



def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using:", device)


    train_dataset = WindWaterDataset(
        "ptmodel/dataset/train"
    )

    val_dataset = WindWaterDataset(
        "ptmodel/dataset/val"
    )

    test_dataset = WindWaterDataset(
        "ptmodel/dataset/test"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )


    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))


    model = WindWaterClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    best_val_acc = 0


    for epoch in range(50):

        model.train()

        total_loss = 0


        for video, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        ):

            video = video.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            loss = criterion(
                model(video),
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        val_acc = evaluate(
            model,
            val_loader,
            device
        )


        print(
            f"""
Epoch {epoch+1}
Loss: {total_loss / len(train_loader):.4f}
Val Accuracy: {val_acc:.4f}
"""
        )


        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "model_state": model.state_dict()
                },
                "video_only.pt"
            )

            print("Saved best model")



    checkpoint = torch.load(
        "video_only.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )


    print(
        "Final Test Accuracy:",
        evaluate(
            model,
            test_loader,
            device
        )
    )


if __name__ == "__main__":
    main()