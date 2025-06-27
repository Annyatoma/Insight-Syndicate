import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        base_model = models.resnet50(pretrained=True)
        for name, param in base_model.named_parameters():
            if 'layer2' in name or 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"📊 Evaluation Metrics on Test Set:")
    print(f"  - Accuracy : {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall   : {rec:.4f}")
    print(f"  - F1-Score : {f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--weights', type=str, default='Gender_Classification_model.pt', help='Path to saved model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = CustomResNet50().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
