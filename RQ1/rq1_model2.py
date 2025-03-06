# https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector
# This model outputs 3 classes: real, sd, dalle. Since my goal is to only see real and AI, dalle and sd are combined into one class 
# The code below evaluates the model on the faces and art datasets

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import pipeline
import os
import torch

# CUDA check 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
classifier = pipeline("image-classification", model="NYUAD-ComNets/NYUAD_AI-generated_images_detector", device=0 if torch.cuda.is_available() else -1)

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
        batch_images = [Image.open(path) for path in image_paths[i:i + batch_size]]
        predictions = classifier(batch_images)
        
        # Combine sd and dalle predictions into one class
        for pred in predictions:
            pred_class = 1 if pred[0]['label'] in ['sd', 'dalle'] else 0
            preds.append(pred_class)
    
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

# Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.596875, 0.6124818577648766, 0.5275, 0.5668233713901947, 0.5968749999999999)
# Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.64125, 0.6782334384858044, 0.5375, 0.5997210599721061, 0.6412500000000001)