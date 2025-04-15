"""
hugging_face.py

This script demonstrates how to load a trained ConditionalPixelCNN model,
compute class predictions for each test image, and then write the predictions 
to a CSV file named 'hugging_face.csv'.

Usage (example):
  python hugging_face.py \
    --data_dir data \
    --batch_size 32 \
    --mode test

It expects a CSV file containing test image paths at './data/test.csv'.
"""

from torchvision import datasets, transforms
import numpy as np
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
import torch
import os

NUM_CLASSES = len(my_bidict)


def get_label(model, model_input, device):
    """
    Compute predictions for each sample in model_input by evaluating
    negative log-likelihood for each possible class in [0..NUM_CLASSES-1].

    Returns:
      predicted_labels: LongTensor of shape (B,) with predicted class IDs
      all_ll: Tensor of shape (B, NUM_CLASSES) holding per-class log-likelihood
    """
    model.eval()
    B = model_input.size(0)
    # We'll store the (negative) log-likelihood for each sample & each class.
    all_ll = torch.zeros(B, NUM_CLASSES, device=device)

    with torch.no_grad():
        for c in range(NUM_CLASSES):
            # Create label vector of size B, all set to candidate class c.
            label_vec = torch.full((B,), c, dtype=torch.long, device=device)
            # Use the proper argument name "label" that matches your model's forward.
            out = model(model_input, label=label_vec, sample=False)

            # For each sample in the batch, compute the per-sample NLL.
            for i in range(B):
                single_in  = model_input[i:i+1]
                single_out = out[i:i+1]
                loss_val   = discretized_mix_logistic_loss(single_in, single_out)
                all_ll[i, c] = -loss_val  # negative, so higher => more likely

    predicted_labels = torch.argmax(all_ll, dim=1)
    return predicted_labels, all_ll


def classify(model, data_loader, device, csv_test_file, csv_output_file_name):
    """
    1. Reads image filenames from 'csv_test_file' (e.g., ./data/test.csv).
    2. For each batch from 'data_loader', uses get_label(...) to obtain predictions.
    3. Writes each filename + predicted label to 'csv_output_file_name' (e.g. hugging_face.csv).
    """

    # 1) Read all image names from csv_test_file (like './data/test.csv').
    img_names = []
    with open(csv_test_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assume each row looks like "test/xxxxx.png, -1" or similar.
            line = row[0].split(',')[0]
            # Remove the "test/" prefix if desired.
            img_name = line.replace('test/', '')
            img_names.append(img_name)

    # 2) Run inference on each batch in data_loader, storing predictions
    model.eval()
    img_idx = 0
    all_logits = []

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item  # 'categories' may be dummy for test set.
        model_input = model_input.to(device)
        answer, logit = get_label(model, model_input, device)
        all_logits.append(logit.detach().cpu().numpy())

        # 3) Write out predictions (one line per image) matching the order in img_names.
        for pred_label in answer.cpu().numpy():
            filename_in_csv = f"test/{img_names[img_idx]}"
            img_names[img_idx] = [filename_in_csv, str(pred_label)]
            img_idx += 1

    # 4) Write the predictions to CSV.
    with open(csv_output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in img_names:
            writer.writerow(row)
    print(f"Done! Wrote predictions to {csv_output_file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': False}

    # Use the same transforms as in training (e.g., resizing and rescaling).
    ds_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        rescaling
    ])

    # Construct the dataset/dataloader in test mode.
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(
            root_dir=args.data_dir,
            mode=args.mode,
            transform=ds_transforms
        ),
        batch_size=args.batch_size,
        shuffle=False,  # Order must be preserved.
        **kwargs
    )

    # Instantiate your model EXACTLY as in training.
    # Here we use PixelCNN with the hyperparameters used during training.
    model = PixelCNN(
        nr_resnet=2,        # Update to match your training (was 2 during training)
        nr_filters=80,      # Update to match your training (80 filters)
        nr_logistic_mix=5,  # Update to match your training
        input_channels=3,
        nr_classes=4,       # For CPEN455 dataset, assuming 4 classes.
        emb_dim=80          # Typically set equal to nr_filters.
    ).to(device)

    # Update the checkpoint path to your trained model file.
    ckpt_path = os.path.join('models', 'pcnn_cpen455_load_model_107.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model parameters loaded from {ckpt_path}")

    # Classify and write predictions to "hugging_face.csv"
    classify(
        model=model,
        data_loader=dataloader,
        device=device,
        csv_test_file=os.path.join(args.data_dir, 'test.csv'),
        csv_output_file_name='hugging_face.csv'
    )
