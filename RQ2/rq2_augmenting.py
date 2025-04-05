# This is a helper method for RQ2 used to augment a percentage of images in the faces dataset.

from PIL import Image, ImageOps
from torchvision import transforms
import random

# Load data
faces_dataset_path = 'archive/datasets/faces_512x512'

# Function to randomly augment a single image
def augment_image(image):
    transform_list = []

    if random.random() < 0.95:
        transform_list.append(transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)))
    if random.random() < 0.95:
        transform_list.append("invert")
    if random.random() < 0.95:
        transform_list.append("shift")
    if random.random() < 0.95:
        transform_list.append(transforms.RandomRotation(degrees=15))
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
def augment_images(dataset_path, percentage, output_path):
    