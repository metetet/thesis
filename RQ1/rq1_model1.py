# https://huggingface.co/Organika/sdxl-detector#validation-metrics
# The code below evaluates the model on the faces and art datasets

from transformers import pipeline
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os
import torch

# CUDA check 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
classifier = pipeline("image-classification", model="Organika/sdxl-detector", device=0 if torch.cuda.is_available() else -1)

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
    
        for pred in predictions:
            #print(pred)
            pred_class = 1 if pred[0]['label'] == 'artificial' else 0 # labels are 'human' and 'artificial'
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

# Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.565625, 0.5404157043879908, 0.8775, 0.6688899475940924, 0.565625)
# Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.811875, 0.7988023952095809, 0.83375, 0.8159021406727828, 0.811875)