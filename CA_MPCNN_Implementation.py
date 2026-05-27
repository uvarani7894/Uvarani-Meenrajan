import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)

        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = torch.cat([x, out1], dim=1)
        out3 = F.relu(self.bn2(self.conv2(out2)))
        return torch.cat([x, out1, out3], dim=1)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        self.bn = nn.BatchNorm2d(out_channels * 4)

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b3 = F.relu(self.branch3(x))
        b5 = F.relu(self.branch5(x))
        bp = F.relu(self.pool(x))

        out = torch.cat([b1, b3, b5, bp], dim=1)
        return F.relu(self.bn(out))


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()

        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        value = self.value_conv(x).view(batch, -1, height * width)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)

        return self.gamma * out + x


class CAMPCNN(nn.Module):
    def __init__(self, num_classes):
        super(CAMPCNN, self).__init__()

        self.pfes_1 = nn.Conv2d(3, 32, kernel_size=1)
        self.pfes_3 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pfes_5 = nn.Conv2d(3, 32, kernel_size=5, padding=2)

        self.bn_pfes = nn.BatchNorm2d(96)

        self.dense1 = DenseBlock(96, growth_rate=32)
        self.inception1 = InceptionBlock(160, 32)

        self.attention = CrossAttention(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_reduce = nn.Conv2d(128, 64, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(64)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = F.relu(self.pfes_1(x))
        x3 = F.relu(self.pfes_3(x))
        x5 = F.relu(self.pfes_5(x))

        x = torch.cat([x1, x3, x5], dim=1)
        x = F.relu(self.bn_pfes(x))

        x = self.dense1(x)
        x = self.inception1(x)

        x = self.attention(x)
        x = self.pool(x)

        x = F.relu(self.bn_reduce(self.conv_reduce(x)))
        x = self.global_pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


def train_model(model, train_loader, epochs=200, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    return model


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset_path = "path_to_sugarcane_dataset"

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)

model = CAMPCNN(num_classes=num_classes)

model = train_model(model, train_loader, epochs=200, lr=0.0001)

evaluate_model(model, test_loader)

torch.save(model.state_dict(), "CA_MPCNN_sugarcane_model.pth")
