# The code below evaluates the fine-tuned model on the faces and art datasets

from transformers import pipeline
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os
import torch

# CUDA check 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
classifier = pipeline("image-classification", model="./sdxl-fine-tune-mixed", device=0 if torch.cuda.is_available() else -1)

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
            pred_class = 1 if pred[0]['label'] == '1' else 0 # labels are 0 and 1 for real and AI respectively (load_dataset() uses folder names)
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

# Fine-Tuned on Human Dataset (sdxl-fine-tune)
# Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.99125, 0.9937185929648241, 0.98875, 0.9912280701754386, 0.99125)
# Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.1875, 0.043795620437956206, 0.03, 0.03560830860534125, 0.1875)

# Fine-Tuned on Art Dataset (sdxl-fine-tune-art)
# Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.421875, 0.36323851203501095, 0.2075, 0.26412092283214, 0.42187499999999994)
# Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.9575, 0.9815789473684211, 0.9325, 0.9564102564102565, 0.9575)

# Fine-Tuned on Mixed Dataset (sdxl-fine-tune-mixed)
# Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.943125, 0.9265944645006017, 0.9625, 0.944206008583691, np.float64(0.943125))
# Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.959375, 0.9476248477466505, 0.9725, 0.9599012954966071, np.float64(0.9593750000000001))