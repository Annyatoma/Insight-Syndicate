
# 🚀 Task B: Face Recognition - COMSYS 2025 FACECOM Challenge

This project implements a robust **multi-class face recognition system** for distorted and clean facial images using deep learning. It uses a **ResNet-50**-based embedding extractor and cosine similarity for evaluation.

---

## 📁 Dataset Structure

The dataset is structured with folders per individual, each containing:

- Clean images
- A subfolder `distortion/` containing distorted versions

Task_B/
├── train/
│ ├── person_1/
│ │ ├── image_1.jpg
│ │ └── distortion/
│ │ ├── distorted_image_1.jpg
│ └── ...
└── val/
├── person_x/
│ ├── image_2.jpg
│ └── distortion/


---

Here is a diagram representing the model Face Recognition System Task B:
![Model Architecture](Diagram_Face_Recognition_Model_Task_B.png)


## 📌 Objective

- Train a **ResNet-50** classifier to extract embeddings for each identity.
- Validate performance using **cosine similarity** between distorted images and clean reference images.
- Evaluate with **Accuracy, Precision, Recall, F1-Score** for both train and validation datasets.

---

## 🛠️ Setup Instructions

### ✅ Install Dependencies

Run this to install required packages:

```bash
pip install -r requirements.txt

    requirements.txt
        torch>=2.0.0
        torchvision>=0.15.0
        numpy>=1.23.0
        Pillow>=9.4.0
        scikit-learn>=1.1.0
        matplotlib>=3.5.0

📓 How the Code Works

1. Data Preparation

Uses a custom FaceRecognitionDataset to load clean and distorted images.
Labels are auto-assigned per person folder.
Data augmentations applied to training set only.

2. Model Architecture

Based on ResNet-50 pretrained model.
Final layer replaced with a classification head (output = num classes).
All layers frozen except layer4 and fc.

3. Training

Model is trained on the clean + distorted training set.
Uses CrossEntropyLoss with label smoothing.
Optimized using AdamW and a step-wise LR scheduler.

4. Saving Model

The model is saved using:
    torch.save(model.state_dict(), "face_recognition_model.pt")

5. Evaluation

After training:

The model is evaluated using a second class (FaceEmbeddingExtractor) to extract embeddings.
Cosine similarity is computed between distorted images and clean reference embeddings.
Evaluation metrics:
Accuracy
Precision
Recall
F1 Score

    Our Evaluation result

📊 Evaluation Metrics on Training Set:
  - Accuracy : 1.0000
  - Precision: 1.0000
  - Recall   : 1.0000
  - F1-Score : 1.0000

📊 Evaluation Metrics on Validation Set:
  - Accuracy : 0.9990
  - Precision: 0.9989
  - Recall   : 0.9991
  - F1-Score : 0.9989


##🧪 Testing the Model

##📝 Test Script: test_face_recognition.py
You can test the saved model on a new test dataset using:

python test_face_recognition.py --test_dir /path/to/test_data --weights face_recognition_model.pt
Replace /path/to/test_data with the actual test folder path having the same structure
Make sure test data has the same structure as the validation dataset.

📈 Performance

The model achieved very high validation accuracy (close to 0.999).
Evaluated on both clean and distorted faces.
Consistent metrics across train and val sets ensure no overfitting or data leakage.

🤝 Acknowledgments

This project is part of the COMSYS-2025 Hackathon organized by COMSYS Educational Trust, Kolkata.
We aim to contribute toward building robust facial analysis systems under real-world challenging visual conditions.

🧠 Authors & Team Roles
Team Member         	    Role
SOUMABRATA BHOWMIK	        ML & Deep Learning (Model Design, Training, Optimization)
ANNYATOMA DAS	            Dataset Setup, Evaluation, Metric Analysis & Submission
SAHELI MONDAL	            Testing, Packaging, Analysis, Documentation
