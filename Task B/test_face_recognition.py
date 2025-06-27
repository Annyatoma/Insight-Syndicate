
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------- Model Definition ---------
class FaceEmbeddingExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)

# --------- Load Clean Reference Embeddings ---------
def extract_clean_references(root_dir, transform, model, device):
    refs = {}
    ref_names = []
    for person in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue
        clean_embeddings = []
        for file in os.listdir(person_path):
            if file.lower() == 'distortion':
                continue
            file_path = os.path.join(person_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(file_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img)
                clean_embeddings.append(emb.cpu())
        if clean_embeddings:
            refs[person] = torch.cat(clean_embeddings, dim=0)
            ref_names.append(person)
        else:
            refs[person] = torch.empty(0)
    return refs, ref_names

# --------- Load Distorted Queries ---------
def collect_distorted_queries(root_dir):
    query_paths = []
    true_labels = []
    for person in sorted(os.listdir(root_dir)):
        distortion_path = os.path.join(root_dir, person, "distortion")
        if not os.path.isdir(distortion_path):
            continue
        for img_file in os.listdir(distortion_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                query_paths.append(os.path.join(distortion_path, img_file))
                true_labels.append(person)
    return query_paths, true_labels

# --------- Evaluate Function ---------
def evaluate_embeddings(refs, query_paths, true_labels, model, transform, device):
    all_preds = []
    all_trues = []

    for path, true_label in zip(query_paths, true_labels):
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img).cpu()

        max_sim = -1
        pred_label = None
        for label, emb_list in refs.items():
            if emb_list.shape[0] == 0:
                continue
            sims = F.cosine_similarity(emb, emb_list)
            max_s = sims.max().item()
            if max_s > max_sim:
                max_sim = max_s
                pred_label = label

        all_preds.append(pred_label)
        all_trues.append(true_label)

    acc = accuracy_score(all_trues, all_preds)
    prec = precision_score(all_trues, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_trues, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)

    print("\n📊 Test Set Evaluation:")
    print(f"  - Accuracy : {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall   : {rec:.4f}")
    print(f"  - F1-Score : {f1:.4f}")

# --------- Main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help="Path to test dataset (person folders with distortion)")
    parser.add_argument('--weights', type=str, default="face_recognition_model.pt", help="Path to trained model weights")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and prepare model
    base_model = models.resnet50(pretrained=False)
    base_model.fc = nn.Linear(base_model.fc.in_features, 877)  
    base_model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    embedding_model = FaceEmbeddingExtractor(base_model).to(device)
    embedding_model.eval()

    # Extract reference embeddings (clean)
    refs, _ = extract_clean_references(args.test_dir, transform, embedding_model, device)

    # Extract query images (distorted)
    query_paths, true_labels = collect_distorted_queries(args.test_dir)

    # Evaluate
    evaluate_embeddings(refs, query_paths, true_labels, embedding_model, transform, device)

if __name__ == '__main__':
    main()
