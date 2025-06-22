import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.data import Dataset

IMAGE_SIZE = 224
BACKGROUND_LABEL = 0
BORDER_LABEL = 1
CONTENT_LABEL = 2


def display(display_list):
    clear_output(wait=True)
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def parse_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image, dtype=np.uint8)
    mask_path = img_path.replace('raw', 'segmentation_mask').replace('jpg', 'png')
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))
    mask = np.array(mask, dtype=np.uint8)
    # Transform mask colors into labels
    mask = np.where(mask == 255, BACKGROUND_LABEL, mask)
    mask = np.where(mask == 29, BACKGROUND_LABEL, mask)
    mask = np.where((mask == 76) | (mask == 134), BORDER_LABEL, mask)
    mask = np.where(mask == 149, CONTENT_LABEL, mask)
    # 强制所有非0/1/2的像素归为背景
    mask = np.where((mask != BACKGROUND_LABEL) & (mask != BORDER_LABEL) & (mask != CONTENT_LABEL), BACKGROUND_LABEL, mask)
    print('mask unique values:', np.unique(mask))  # 调试用
    return {'image': image, 'segmentation_mask': mask}


class ComicDataset(Dataset):
    def __init__(self, folder, shuffle=True):
        self.image_paths = [os.path.join(folder, 'raw', f) for f in sorted(os.listdir(os.path.join(folder, 'raw'))) if f.endswith('.jpg')]
        if shuffle:
            np.random.shuffle(self.image_paths)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        sample = parse_image(self.image_paths[idx])
        image = sample['image'].astype(np.float32) / 255.0
        mask = sample['segmentation_mask'].astype(np.int64)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).long()  # 确保为long类型
        return image, mask


def load_data_set():
    return {
        'test': ComicDataset('./dataset/test/', shuffle=False),
        'train': ComicDataset('./dataset/training/')
    }


def predicted_pixel_to_class(x):
    return np.argmax(x)


def map_prediction_to_mask(predicted_image):
    return np.argmax(predicted_image, axis=-1)


def files_in_folder(folder):
    return sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

def count_files_in_folder(folder):
    return len(files_in_folder(folder))

def compare_accuracy_per_label(true_mask, predicted_mask):
    total_background_labels = np.sum(true_mask == BACKGROUND_LABEL)
    total_border_labels = np.sum(true_mask == BORDER_LABEL)
    total_content_labels = np.sum(true_mask == CONTENT_LABEL)
    properly_predicted_background_pixels = np.sum((true_mask == BACKGROUND_LABEL) & (predicted_mask == BACKGROUND_LABEL))
    properly_predicted_border_pixels = np.sum((true_mask == BORDER_LABEL) & (predicted_mask == BORDER_LABEL))
    properly_predicted_content_pixels = np.sum((true_mask == CONTENT_LABEL) & (predicted_mask == CONTENT_LABEL))
    background_accuracy = properly_predicted_background_pixels / (total_background_labels + 1e-8)
    border_accuracy = properly_predicted_border_pixels / (total_border_labels + 1e-8)
    content_accuracy = properly_predicted_content_pixels / (total_content_labels + 1e-8)
    return background_accuracy, border_accuracy, content_accuracy

def compare_accuracy(true_masks, predictions):
    background_acc = 0.0
    border_acc = 0.0
    content_acc = 0.0
    for index in range(len(predictions)):
        partial_back_acc, partial_border_acc, partial_content_acc = compare_accuracy_per_label(true_masks[index], predictions[index])
        background_acc += partial_back_acc
        border_acc += partial_border_acc
        content_acc += partial_content_acc
    pred_num = len(predictions)
    return background_acc / pred_num, border_acc / pred_num, content_acc / pred_num

def label_to_rgb(labeled_pixel):
    if labeled_pixel == BACKGROUND_LABEL:
        return 0
    if labeled_pixel == CONTENT_LABEL:
        return 127
    if labeled_pixel == BORDER_LABEL:
        return 255

def labeled_prediction_to_image(predicted_result):
    color_matrix = np.vectorize(label_to_rgb)(predicted_result)
    return Image.fromarray(np.uint8(color_matrix))
