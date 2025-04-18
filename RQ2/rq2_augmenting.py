# This is a helper method for RQ2 used to augment a percentage of images in the faces dataset.

from PIL import Image, ImageOps
from torchvision import transforms
import random
import os
from shutil import copy2
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Load data
faces_dataset_path = 'archive/datasets/faces_512x512'

# Function to randomly augment a single image
def augment_image(image):
    transform_list = []

    if random.random() < 0.95:
        transform_list.append(transforms.RandomResizedCrop(size=(512, 512), scale=(0.4, 1.0)))
    if random.random() < 0.30: # keep this lower than the others to avoid too many inverted samples
        transform_list.append("invert")
    if random.random() < 0.95:
        transform_list.append("shift")
    if random.random() < 0.95:
        transform_list.append(transforms.RandomRotation(degrees=25))
    if random.random() < 0.95:
        transform_list.append(transforms.RandomHorizontalFlip())
    if random.random() < 0.95:
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

# Function to apply the augmentation to a percentage of images
def augment_images(dataset_path, percentage, output_path, seed):
    set_seed(seed)
    for split in ["train", "test", "val"]:
        split_path = os.path.join(dataset_path, split)
        output_split_path = os.path.join(output_path, split)

        for label in ["0", "1"]:
            image_path = os.path.join(split_path, label)
            output_image_path = os.path.join(output_split_path, label)
            os.makedirs(output_image_path, exist_ok=True)

            set_of_images = os.listdir(image_path)
            num_images_to_augment = int(len(set_of_images) * percentage) # must be a whole number
            images_to_augment = set(random.sample(set_of_images, num_images_to_augment))

            for image in set_of_images:
                src = os.path.join(image_path, image)
                dst = os.path.join(output_image_path, image)

                if image in images_to_augment:
                    img = Image.open(src).convert("RGB")
                    img = augment_image(img)
                    img.save(dst)
                else:
                    copy2(src, dst)

        print(f"Augmented {percentage*100}% of images in {split} split")

augment_images(faces_dataset_path, 0.05, 'archive/datasets/faces_aug5', seed=42)