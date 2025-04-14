import torch
from torch import Tensor
import datetime
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import ttk


# history; epoch + 1, train_loss, train_acc, val_loss, val_acc, f1, auc
def plot_training_history(history, save_path):
    """create training history plot and visualize the training process""" 
    epochs = [item[0] for item in history]
    train_loss = [item[1] for item in history]
    val_loss = [item[3] for item in history]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    
    loss_plot_path = os.path.join(save_path, f'loss_plot_{current_time}.png')
    plt.savefig(loss_plot_path)
    print(f"loss graph save to : {loss_plot_path}")
    
    # create dice coefficient plot and visualize
    plt.figure(figsize=(10, 5))
    train_acc = [item[2] for item in history]
    val_acc = [item[4] for item in history]
    val_f1 = [item[5] for item in history]
    val_auc = [item[6] for item in history]
    
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.plot(epochs, val_f1, 'g-', label='Validation F1 Score')
    plt.plot(epochs, val_auc, 'y-', label='Validation AUC')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.grid(True)
    
    dice_plot_path = os.path.join(save_path, f'dice_plot{current_time}.png')
    plt.savefig(dice_plot_path)
    print(f"Dice graph save to : {dice_plot_path}")

def choose_model():
    def on_submit():
        selected_model.set(combo.get())  #set the selected model
        root.destroy() 

    root = tk.Tk()
    root.title("Choose Model")
    root.geometry("300x150")

    selected_model = tk.StringVar(value="SimpleCNN")  # default model
    label = tk.Label(root, text="Select the model to run:")
    label.pack(pady=10)

    # create a combobox
    combo = ttk.Combobox(root, textvariable=selected_model)
    combo['values'] = ("SimpleCNN")
    combo.current(0) 
    combo.pack(pady=10)

    # create a submit button
    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(pady=10)
    root.mainloop()
    return selected_model.get()