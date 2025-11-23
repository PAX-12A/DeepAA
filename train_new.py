import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os, math, time
from sklearn.model_selection import train_test_split


# =========================
# 1️⃣ 数据加载部分
# =========================
class CharDataset(Dataset):
    def __init__(self, df, img_dir, input_size, tail='line_resize', slide=1):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.input_size = input_size
        self.tail = tail
        self.slide = slide

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y = row['x'] + 30, row['y'] + 30
        img_path = os.path.join(self.img_dir, row['file_name'] + self.tail + '.png')

        img = np.array(Image.open(img_path).convert('L'))
        h, w = img.shape
        mergin = (self.input_size - 18) // 2 + 30

        # add margin
        img_new = np.ones((h + 2*mergin, w + 2*mergin), dtype=np.uint8) * 255
        img_new[mergin:-mergin, mergin:-mergin] = img

        # random slide
        x += np.random.randint(-self.slide, self.slide + 1)
        y += np.random.randint(-self.slide, self.slide + 1)

        patch = img_new[y:y+self.input_size, x:x+self.input_size]
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(row['label'], dtype=torch.long)
        return patch, label


# =========================
# 2️⃣ 轻量CNN模型
# =========================
class LightCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 3️⃣ 训练流程
# =========================
def train():
    df_path = "data/data_500.csv"
    char_list_path = "data/char_list_500.csv"
    img_dir = "data/image_500/"
    input_size = 32
    batch_size = 128
    num_epoch = 3
    lr = 0.001

    df = pd.read_csv(df_path, encoding='cp932')
    char_list = pd.read_csv(char_list_path, encoding='cp932')
    num_label = char_list[char_list['frequency'] >= 10].shape[0]
    df = df[df['label'] < num_label]

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    train_ds = CharDataset(df_train, img_dir, input_size)
    val_ds = CharDataset(df_val, img_dir, input_size, slide=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightCNN(num_classes=num_label).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epoch):
        model.train()
        total_loss, total_acc = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).float().sum().item()

        print(f"[Epoch {epoch+1}/{num_epoch}] Train Loss: {total_loss/len(train_loader):.4f}  Acc: {total_acc/len(train_ds):.4f}")

        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_acc += (outputs.argmax(1) == labels).float().sum().item()
        print(f"           Val Loss: {val_loss/len(val_loader):.4f}  Acc: {val_acc/len(val_ds):.4f}")

    torch.save(model.state_dict(), "lightcnn.pth")
    print("✅ Training finished and model saved as lightcnn.pth")


if __name__ == "__main__":
    train()
