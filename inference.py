import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from load_data import load_mura_data
from util import plot_training_history, choose_model
from model import Vit_b_16, SimpleCNN, ResNet34, Swin_V2_T, RegNet_Y_16GF, efficientnet_v2_m
import gc
import shutil
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from torch.amp import GradScaler,autocast
from PIL import Image
import torchvision.transforms as transforms
import numpy as np



MODEL_MAP = {"Vit_b_16": Vit_b_16, "SimpleCNN": SimpleCNN, "ResNet34": ResNet34 , "Swin_V2_T": Swin_V2_T, "RegNet_Y_16GF": RegNet_Y_16GF, "efficientnet_v2_m": efficientnet_v2_m}  # Add other models as needed
def test(args):
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = 4 
    # # load data    # valid代替test
    test_loader, _ = load_mura_data(args.data_path, mode='test', batch_size = 1, num_workers=num_workers)

    # Select the model dynamically
    model_class = MODEL_MAP.get(args.model)
    if model_class is None:
        raise ValueError(f"Model {args.model} not found in available models.")
    
    
    criterion = nn.CrossEntropyLoss()
    print(f"Training model: {args.model}")
    model = model_class(num_classes=2)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    study_folders = []
    predicted_labels = []
    predicted_probs = []
    ground_truth_labels = []
    imagenames = []
    correct, total = 0, 0
    with torch.no_grad():
        for image, imagename in tqdm(test_loader):
            print(f"Processing image: {imagename}")
            image.to(device)
            
            outputs = model(image)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            # correct += (preds == labels).sum().item()
            # total += labels.size(0)

            predicted_labels.extend(preds.cpu().numpy())
            predicted_probs.extend(probs.cpu().numpy())
            imagenames.append(imagename)
            #ground_truth_labels.extend(labels.cpu().numpy())

    # val_acc = correct / total
    # f1 = f1_score(ground_truth_labels, predicted_labels)
    # try:
    #     auc = roc_auc_score(ground_truth_labels, predicted_probs)
    # except:
    #     auc = float('nan')        
    
    # print(f"[Valid/ Test] Acc: {val_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    # Save the results to a CSV file
    #results = pd.DataFrame({'study_folder': study_folders, 'predicted_prob': predicted_probs, 'predicted_label': predicted_labels, 'ground_truth_label': ground_truth_labels}, columns=['study_folder', 'predicted_prob', 'predicted_label', 'ground_truth_label'])
    results = pd.DataFrame({'imagename': imagenames, 'predicted_label': predicted_labels, 'predicted_prob': predicted_probs}, columns=['imagename', 'predicted_label', 'predicted_prob'])
    csv_path = os.path.join(args.save_path, f"md{args.model}_testset_result.csv")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results.to_csv(csv_path, index=False)
    print(f"Classification results have been saved to: {csv_path}")



def get_args(project_root, selected_model):
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--data_path', type=str, default=os.path.join(project_root), help='path of the input data')
    parser.add_argument('--save_path', type=str, default=os.path.join(project_root, "test_result", f"{selected_model}"), help='Path to save model checkpoints')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default=selected_model, help='Model to run (Unet or ResNet34UNet or ResNet34UNet_modified)')
    return parser.parse_args()

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    selected_model = choose_model()
    args = get_args(project_root, selected_model)
    test(args)


# resnet34
# python inference.py --model_path "C:\Users\Vivo\Downloads\model_epoch_15.pth"

# effitient net
# python inference.py --model_path "C:\Users\Vivo\Downloads\best_model.pth"