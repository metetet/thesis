import os
import torch
from transformers import AutoImageProcessor, SwinForImageClassification, TrainingArguments, Trainer
import evaluate
from datasets import load_dataset, DatasetDict
import numpy as np
import random
from torch.utils.data import Dataset
from torchinfo import summary

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
model_path = './sdxl-fine-tune'
dataset_path = 'archive/datasets/art_512x512'

# Load processor and dataset
processor = AutoImageProcessor.from_pretrained(model_path)
ds = load_dataset("imagefolder", data_dir=dataset_path)
print("Original dataset:", ds)

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, hf_dataset, processor, device):
        self.dataset = hf_dataset
        self.processor = processor
        self.device = device
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = self.processor(images=item['image'], return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),  # Remove batch dimension
            'labels': torch.tensor(item['label'])
        }

# Create few-shot dataset
def create_few_shot_set(dataset, set_size, seed):
    random.seed(seed)
    
    label0_indices = [i for i, label in enumerate(dataset['train']['label']) if label == 0]
    label1_indices = [i for i, label in enumerate(dataset['train']['label']) if label == 1]
    
    samples_per_class = set_size // 2
    selected_indices = (random.sample(label0_indices, samples_per_class) + 
                       random.sample(label1_indices, samples_per_class))
    random.shuffle(selected_indices)
    
    return DatasetDict({
        'train': dataset['train'].select(selected_indices),
        'validation': dataset['validation']
    })

# Create few-shot datasets
few_shot_ds = create_few_shot_set(ds, set_size=6400, seed=42)
print("Few-shot dataset:", few_shot_ds)

# Create custom datasets
train_dataset = CustomDataset(few_shot_ds['train'], processor, device)
val_dataset = CustomDataset(few_shot_ds['validation'], processor, device)

# Verify dataset samples
sample = train_dataset[0]
print("\nSample verification:")
print("Keys:", sample.keys())
print("Pixel values shape:", sample['pixel_values'].shape)
print("Label:", sample['labels'])

# Collate function - modified to handle device placement correctly
def collate_fn(batch):
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    labels = torch.tensor([x['labels'] for x in batch])
    return {
        'pixel_values': pixel_values.to(device),
        'labels': labels.to(device)
    }

# Metrics
acc_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')

def compute_metrics(p):
    acc = acc_metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    f1 = f1_metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    return {"Accuracy": acc["accuracy"], "F1": f1["f1"]}

# Load model
labels = ds['train'].features['label'].names
print("\nLabels:", labels)

model = SwinForImageClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
).to(device)

# Freeze model
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Training arguments - disable pin_memory since we're handling device placement manually
training_args = TrainingArguments(
    output_dir=model_path + "_few_shot",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=1,
    load_best_model_at_end=False,
    report_to="none",
    dataloader_pin_memory=False  # Disable pin memory as we handle device placement
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# Train
print("\nStarting training...")
train_results = trainer.train()

# Save results
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Evaluate
metrics = trainer.evaluate(val_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

print("\nTraining complete!")
print("Training metrics:", train_results.metrics)
print("Validation metrics:", metrics)