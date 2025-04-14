import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import multiprocessing
import argparse
from load_data import load_mura_data, mura_collate_fn
from util import plot_training_history, choose_model
from model import SimpleCNN
import gc
import shutil
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

def get_optimizer(model, learning_rate, optimizer_type, weight_decay,momentum, **kwargs):
    """get the optimizer for training"""
    # get the optimizer type
    momentum = kwargs.get('momentum', 0.9)  # default momentum value

    if optimizer_type == 'adam':
        print(f"Using Adam - learning rate: {learning_rate}, weight decay: {weight_decay}")
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    elif optimizer_type == 'sgd':
        print(f"Using SGD - learning rate: {learning_rate}, momentum: {momentum}, weight decay: {weight_decay}")
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_type == 'adamw':
        print(f"Using Adamw - learning rate: {learning_rate}, weight decay: {weight_decay}")
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

MODEL_MAP = {"SimpleCNN": SimpleCNN,}

def train(args):
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_workers = 0
    # load data
    train_loader = load_mura_data(args.data_path, mode='train', batch_size = args.batch_size, num_workers=num_workers)
    valid_loader = load_mura_data(args.data_path, mode='valid', batch_size = args.batch_size, num_workers=num_workers)
    

    # Select the model dynamically
    model_class = MODEL_MAP.get(args.model)
    if model_class is None:
        raise ValueError(f"Model {args.model} not found in available models.")
    
    print(f"Training model: {args.model}")
    model = model_class(num_classes=2)
    model = model.to(device)

    # set loss function and optimizer
    # Lossfunction : BCE 
    criterion = nn.CrossEntropyLoss()
    # create the specific optimizer
    optimizer = get_optimizer(model, args.learning_rate, args.optimizer, args.weight_decay, args.momentum)

    # track the best validation metric
    train_acc = 0.0
    best_epoch = -1
    best_loss = float('inf')
    best_model_path = ""
    history = []
    
    # early stopping parameters
    early_stop_patience = args.early_stop_patience
    no_improve_epochs = 0
    
    # Create run ID for this training session
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"bs{args.batch_size}_lr{args.learning_rate:.1e}_opt{args.optimizer}_wd{args.weight_decay:.1e}"
    
    # Create save directory
    save_path = os.path.join(args.save_path, run_id)
    os.makedirs(save_path, exist_ok=True)
    
    # Define memory cleanup frequency (every N epochs)
    memory_cleanup_freq = 2  # Adjust as needed
    
    # training process
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        best_val_loss = float('inf')
        correct, total = 0, 0

        for images_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            images_batch = [imgs.mean(dim=0) for imgs in images_batch]
            inputs = torch.stack(images_batch).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Clear batch variables to free up memory
            del images, masks, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}],[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")


        # validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for images_batch, labels, image_paths in tqdm(valid_loader, desc="Validating"):
                images_batch = [imgs.mean(dim=0) for imgs in images_batch]
                inputs = torch.stack(images_batch).to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss_avg = val_loss / len(valid_loader)
        val_acc = correct / total
        f1 = f1_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = float('nan')

        print(f"[Valid] Loss: {val_loss_avg:.4f} | Acc: {val_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        history.append((epoch + 1, train_loss, train_acc, val_loss, val_acc, f1, auc))
        
        
        # Save the current model (every epoch)
        epoch_model_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved: {epoch_model_path}")
        
        # Track the best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            best_model_path = epoch_model_path
            best_val_acc = val_acc
            best_val_f1 = f1
            best_val_auc = auc
            
            # Create a symbolic link or copy to mark as best model
            best_marker_path = os.path.join(save_path, f"best_model.pth")
            if os.path.exists(best_marker_path):
                os.remove(best_marker_path)
            # Either create a symlink (Unix) or copy the file (Windows)
            try:
                os.symlink(epoch_model_path, best_marker_path)
            except (OSError, AttributeError):
                shutil.copy2(epoch_model_path, best_marker_path)
                
            print(f"Best model updated: Epoch {best_epoch}, Best Loss {best_val_loss:.4f}")
            # reset the early stopping counter
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs. Best loss: {best_val_loss:.4f} at epoch {best_epoch}")
            
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping triggered. No improvement for {early_stop_patience} epochs.")
                break
        
        # Perform memory cleanup periodically
        if (epoch + 1) % memory_cleanup_freq == 0:
            print("Performing memory cleanup...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Plot training history
    plot_training_history(history, save_path)
    
    # Create a summary of training history
    df = pd.DataFrame(history, columns=["Epoch", "Train Loss", "Train accuracy", "Val Loss", "Val accuracy", "F1 Score", "AUC"])
    print("\nTraining Summary:")
    print(df.to_string(index=False))
    
    # Save training history to CSV
    history_path = os.path.join(save_path, f"training_history_{current_time}.csv")
    df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    return best_model_path, best_val_loss, best_epoch, history, best_val_acc, best_val_f1, best_val_auc

def get_args(project_root,selected_model):
    parser = argparse.ArgumentParser(description='Train the UNet/resnetunet on images and target masks')
    parser.add_argument('--data_path', type=str, default=r"D:\PUPU\2025_medicalimage_and_AI", help='path of the input data')
    parser.add_argument('--save_path', type=str, default=os.path.join(project_root, "saved_models_and_others"), help='Path to save model checkpoints')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='optimizer to use')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='patience for early stopping')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer (L2 regularization)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--model', type=str, default=selected_model, help='Model to run')
    return parser.parse_args()


if __name__ == "__main__":
    project_root = r"D:\PUPU\2025_medicalimage_and_AI"  
    selected_model = choose_model()
    args = get_args(project_root, selected_model)
    best_model_path, best_val_loss, best_epoch, history,  best_val_acc, best_val_f1, best_val_auc = train(args)

    print(f"訓練完成，最佳模型: {best_model_path}")
    print(f"最佳acc分數: {best_val_acc:.4f}")
    print(f"最佳f1分數: {best_val_f1:.4f}")
    print(f"最佳auc分數: {best_val_auc:.4f}")
    print(f"最佳損失值: {best_val_loss:.4f}")
    print(f"最佳輪次: {best_epoch}")