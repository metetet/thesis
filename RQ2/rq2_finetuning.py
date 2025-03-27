# https://huggingface.co/Organika/sdxl-detector#validation-metrics
# The code below fine-tunes the model on the faces and art datasets (separately) and evaluates the model on their validation sets

from transformers import pipeline
from transformers import AutoModelForImageClassification
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os
import torch

# CUDA check 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
classifier = pipeline("image-classification", model="Organika/sdxl-detector", device=0 if torch.cuda.is_available() else -1)
# print(classifier.model.eval())

# Load data
art_dataset_path = 'archive/datasets/art_512x512'
faces_dataset_path = 'archive/datasets/faces_512x512'

def get_image_paths(dataset_path, split):
    image_paths = []
    labels = []
    for label in ['0', '1']:
        folder_path = os.path.join(dataset_path, split, label)
        for filename in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(int(label))  # 0 for real, 1 for AI
    return image_paths, labels

faces_paths_train, faces_labels_train = get_image_paths(faces_dataset_path, 'train')
faces_paths_val, faces_labels_val = get_image_paths(faces_dataset_path, 'validate')