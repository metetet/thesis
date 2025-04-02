# The code below evaluates the fine-tuned model on the faces and art datasets

from transformers import pipeline
from PIL import Image, ImageOps
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os
import torch
import random

# CUDA check 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
classifier = pipeline("image-classification", model="./sdxl-fine-tune-art", device=0 if torch.cuda.is_available() else -1)

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

# Function to randomly augment a single image
def augment_image(image):
    transform_list = []

    if random.random() < 0.3: # 30% chance to resize and crop
        transform_list.append(transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)))
    if random.random() < 0.15: # 15% chance to invert colors
        transform_list.append("invert")
    if random.random() < 0.2: # 20% chance to shift the image
        transform_list.append("shift")
    if random.random() < 0.3: # 30% chance to rotate the image 15 degrees
        transform_list.append(transforms.RandomRotation(degrees=15))
    if random.random() < 0.4: # 40% chance to flip the image horizontally
        transform_list.append(transforms.RandomHorizontalFlip())
    if random.random() < 0.2: # 20% chance to flip the image vertically
        transform_list.append(transforms.RandomVerticalFlip())

    # Apply a maximum of 3 transformations so that the images are not too distorted
    if len(transform_list) > 3:
        transform_list = random.sample(transform_list, 3) 

    final_transforms = []
    for transform in transform_list:
        if transform == "invert":
            image = ImageOps.invert(image)
        elif transform == "shift":
            pixels_to_move = 50
            move_in_x = random.randint(-pixels_to_move, pixels_to_move)
            move_in_y = random.randint(-pixels_to_move, pixels_to_move)
            image = image.transform(image.size, Image.AFFINE, (1, 0, move_in_x, 0, 1, move_in_y))
        else:
            final_transforms.append(transform)

    if final_transforms:
        image = transforms.Compose(final_transforms)(image)

    return image

# Evaluate the model on the dataset
def evaluate_model(image_paths, true_labels, batch_size=32):
    preds = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_images = [augment_image(Image.open(path)) for path in image_paths[i:i + batch_size]]
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
print(f"Augmented Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): {faces_metrics}")

art_metrics = evaluate_model(art_paths, art_labels)
print(f"Augmented Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): {art_metrics}")

# Fine-Tuned on Human Dataset (sdxl-fine-tune) on augmented datasets
# Augmented Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.83625, 0.9803571428571428, 0.68625, 0.8073529411764706, np.float64(0.8362499999999999))
# Augmented Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.335625, 0.041811846689895474, 0.015, 0.02207911683532659, np.float64(0.33562499999999995))

# Fine-Tuned on Art Dataset (sdxl-fine-tune-art) on augmented datasets
# Augmented Faces Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.403125, 0.41223103057757643, 0.455, 0.4325609031491384, np.float64(0.40312499999999996))
# Augmented Art Dataset Metrics (Accuracy, Precision, Recall, F1, AUC): (0.95125, 0.9287410926365796, 0.9775, 0.9524969549330086, np.float64(0.9512500000000002))