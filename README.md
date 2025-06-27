# 👤 Gender Classification Under Challenging Conditions – COMSYS-2025 Hackathon (Task A)

This project addresses **Task A** of the COMSYS-2025 Hackathon: binary gender classification of face images captured under visually degraded real-world conditions using a ResNet50-based architecture in PyTorch.

---

## 📂 Dataset Structure

Task_A/
├── train/
│ ├── male/
│ └── female/
├── val/
│ ├── male/
│ └── female/


Images are annotated into two classes:
- `male`
- `female`

---

## 🚀 Model Architecture

We use **ResNet-50** pretrained on ImageNet as the feature extractor, followed by a custom neural network classifier.

Key highlights:
- Layers `layer2`, `layer3`, `layer4` are **fine-tuned**.
- Other layers are **frozen** to retain pre-learned visual features.
- Final classifier layers are:
  - Linear(2048 → 1024) → ReLU → Dropout
  - Linear(1024 → 512) → ReLU → Dropout
  - Linear(512 → 2) for binary classification

---

## 🛠 Dependencies

Install all dependencies via:

```bash
pip install -r requirements.txt

    requirements.txt
        torch>=2.0.1
        torchvision>=0.15.2
        torchaudio>=2.0.2
        numpy>=1.24.4
        scikit-learn>=1.3.0
        Pillow>=9.5.0
        argparse>=1.4.0


🏋️‍♂️ Training Instructions
Run the training process inside the Gender_Classification.ipynb notebook

Training saves:

Model with best validation accuracy
All epoch-wise metrics
You can modify paths as needed in the notebook.

📊 Evaluation Metrics

After training, the best model is evaluated using:
Accuracy
Precision
Recall
F1-score

On both train and validation sets.

Evaluation is done using: from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Our Evaluation result

📊 Evaluation on Training Set:
  - Accuracy : 0.9346
  - Precision: 0.9426
  - Recall   : 0.9821
  - F1-Score : 0.9620

📊 Evaluation on Validation Set:
  - Accuracy : 0.9455
  - Precision: 0.9651
  - Recall   : 0.9679
  - F1-Score : 0.9665

🧪 Test Script
           
Run the following script to evaluate the model on the test set

python test_Gender_Classification_model.py --data_dir /path/to/test_data --weights Gender_Classification_model.pt

Replace /path/to/test_data with the actual test folder path having the same structure (male/, female/).

This script:

Loads the best model
Computes all 4 evaluation metrics on the test dataset
Runs on CPU or GPU automatically

🤝 Acknowledgments
           
This project is part of the COMSYS-2025 Hackathon organized by COMSYS Educational Trust, Kolkata.
We aim to contribute toward building robust facial analysis systems under real-world challenging visual conditions.

🧠 Authors & Team Roles
Team Member         	Role
SOUMABRATA BHOWMIK	    ML & Deep Learning (Model Design, Training, Optimization)
ANNYATOMA DAS	        Dataset Setup, Evaluation, Metric Analysis & Submission
SAHELI MONDAL	        Testing, Packaging, Analysis, Documentation
