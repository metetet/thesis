# https://huggingface.co/prithivMLmods/Deepfake-Real-Class-Siglip2
# The code below evaluates the model on the faces and art datasets

import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os

# CUDA check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
model_name = "prithivMLmods/Deepfake-Real-Class-Siglip2"
model = SiglipForImageClassification.from_pretrained(model_name).to(device)
processor = AutoImageProcessor.from_pretrained(model_name)

# Load data
art_dataset_path = 'archive/datasets/art_512x512'
faces_dataset_path = 'archive/datasets/faces_512x512'

def get_image_paths(dataset_path):
    image_paths = []
    labels = []
    for label in ['0', '1']:
        folder_path = os.path.join(dataset_path, 'test', label)
        for filename in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(int(label))  # 0 for real, 1 for AI
    return image_paths, labels

# Evaluate the model on the dataset
def evaluate_model(image_paths, true_labels, batch_size=32):
    preds = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_images = [Image.open(path).convert("RGB") for path in image_paths[i:i + batch_size]]
        inputs = processor(batch_images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
        
        preds.extend(pred_classes)
    
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    auc = roc_auc_score(true_labels, preds)
    
    return accuracy, precision, recall, f1, auc

faces_paths, faces_labels = get_image_paths(faces_dataset_path)
art_paths, art_labels = get_image_paths(art_dataset_path)

faces_metrics = evaluate_model(faces_paths, faces_labels)
print(f"Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): {faces_metrics}")

art_metrics = evaluate_model(art_paths, art_labels)
print(f"Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): {art_metrics}")

# Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.47625, 0.46184738955823296, 0.2875, 0.35439137134052384, 0.47625000000000006)
# Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.52625, 0.5155555555555555, 0.87, 0.6474418604651163, 0.52625)
