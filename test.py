import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')




# Force consistent class order to match training
class_order = ['female', 'male']  

class CustomImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = class_order
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx



# 🔹 Gender Classification Model (Task A)
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        base_model = models.resnet50(pretrained=False)
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


# 🔹 Face Embedding Model (Task B)
class FaceEmbeddingExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)


# 🔍 Determine Task Type
def detect_task(test_dir):
    subfolders = sorted([name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))])
    if 'male' in subfolders and 'female' in subfolders:
        return "A"
    else:
        return "B"


# 🧪 Evaluation for Task A
def evaluate_gender_classifier(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print_metrics(all_labels, all_preds, tag="Task A: Gender Classification")


# 🧪 Evaluation for Task B
def evaluate_face_recognition(model, test_dir, transform, device):
    refs = {}
    queries = []
    true_labels = []

    for person in os.listdir(test_dir):
        person_path = os.path.join(test_dir, person)
        if not os.path.isdir(person_path):
            continue

        # Clean images
        clean_embs = []
        for img_file in os.listdir(person_path):
            if img_file.lower() == 'distortion':
                continue
            path = os.path.join(person_path, img_file)
            if os.path.isfile(path):
                img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img)
                clean_embs.append(emb.cpu())
        if clean_embs:
            refs[person] = torch.cat(clean_embs, dim=0)

        # Distorted images
        dist_path = os.path.join(person_path, 'distortion')
        if os.path.isdir(dist_path):
            for f in os.listdir(dist_path):
                img_path = os.path.join(dist_path, f)
                if os.path.isfile(img_path):
                    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model(img).cpu()
                    queries.append(emb)
                    true_labels.append(person)

    # Cosine similarity evaluation
    all_preds = []
    for emb in queries:
        max_sim, pred_label = -1, None
        for ref_label, ref_embs in refs.items():
            if ref_embs.shape[0] == 0:
                continue
            sim = F.cosine_similarity(emb, ref_embs)
            if sim.max().item() > max_sim:
                max_sim = sim.max().item()
                pred_label = ref_label
        all_preds.append(pred_label)

    print_metrics(true_labels, all_preds, tag="Task B: Face Recognition")


# 📊 Print Metrics
def print_metrics(y_true, y_pred, tag="Evaluation"):
    acc = accuracy_score(y_true, y_pred)

    # Auto-detect task type
    unique_labels = sorted(set(y_true + y_pred))
    if tag.startswith("Task A") and len(unique_labels) == 2:
        avg_type = 'binary'  # For Gender Classification
    else:
        avg_type = 'macro'   # For Face Recognition 

    prec = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)

    print(f"\n📊 {tag} Metrics:")
    print(f"  - Accuracy : {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall   : {rec:.4f}")
    print(f"  - F1-Score : {f1:.4f}")



# 🚀 Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    args = parser.parse_args()

    test_dir = args.test_dir
    task_type = detect_task(test_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🧠 Task-Specific Transforms
    if task_type == "A":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    if task_type == "A":
        print("🔍 Detected Task A: Gender Classification")
        model = GenderClassifier().to(device)
        model.load_state_dict(torch.load("Gender_Classification_model.pt", map_location=device))
        test_dataset = CustomImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        evaluate_gender_classifier(model, test_loader, device)
        print("Class to Index Mapping:", test_dataset.class_to_idx) 
    else:
        print("🔍 Detected Task B: Face Recognition")
        base_model = models.resnet50(pretrained=False)
        base_model.fc = nn.Linear(base_model.fc.in_features, 877)
        base_model.load_state_dict(torch.load("face_recognition_model.pt", map_location=device), strict=False)
        embedding_model = FaceEmbeddingExtractor(base_model).to(device)
        embedding_model.eval()
        evaluate_face_recognition(embedding_model, test_dir, transform, device)


if __name__ == '__main__':
    main()
