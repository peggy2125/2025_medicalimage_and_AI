# We will evaluate the model on the validation set using the dice score and the loss function
from tqdm import tqdm 
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
    
def validate_evaluation(model, valid_loader, criterion, device):
    ''' 
    evaluate the model on the validation set 
        args: model(torch.nn.Module): the model to be evaluated
             valid_loader(torch.utils.data.DataLoader): the validation set
             criterion(torch.nn.Module): the loss function
             device(torch.device): the device to run the model on
        return: val_loss(float): the average loss on the validation set
               val_dice(float): the average dice score on the validation set
    ''' 
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images_batch, labels in tqdm(valid_loader, desc="Validating"):
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