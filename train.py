import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets.indian_pines import IndianPinesDataset
from models.hyperspectral_net import HyperspectralNet
import numpy as np
import csv

NUM_BANDS = 200
NUM_CLASSES = 17
BASE_FILTERS = 32

EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_RATIO = 0.2
VAL_RATIO = 0.1

PATCH_SIZE = 16
STRIDE = 8               
BATCH_SIZE = 8          
SAMPLES_PER_EPOCH = 512  

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

train_ds = IndianPinesDataset('data/indian_pines/Indian_pines_corrected.mat',
                               'data/indian_pines/Indian_pines_gt.mat', 
                               split='train', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                               patch_size=PATCH_SIZE, stride=STRIDE, use_patches=True)
                               
val_ds   = IndianPinesDataset('data/indian_pines/Indian_pines_corrected.mat',
                               'data/indian_pines/Indian_pines_gt.mat', 
                               split='val', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                               use_patches=False)
                               
test_ds  = IndianPinesDataset('data/indian_pines/Indian_pines_corrected.mat',
                               'data/indian_pines/Indian_pines_gt.mat', 
                               split='test', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                               use_patches=False)

sampler = WeightedRandomSampler(
    weights=train_ds.patch_weights, 
    num_samples=SAMPLES_PER_EPOCH, 
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

model = HyperspectralNet(
    num_bands=NUM_BANDS,
    num_classes=NUM_CLASSES,
    base_filters=BASE_FILTERS
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

def compute_metrics(pred, target, num_classes):
    pred = pred.argmax(dim=1)
    mask = target != 0
    pred_m = pred[mask]
    target_m = target[mask]

    if target_m.numel() == 0:
        return 0, 0, 0, 0, 0

    oa = (pred_m == target_m).sum().item() / target_m.numel()

    ious, dices, precisions, recalls = [], [], [], []
    for c in range(1, num_classes):
        pred_c = pred_m == c
        target_c = target_m == c
        
        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()
        union = tp + fp + fn

        if union > 0:
            ious.append(tp / union)
            dices.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
            precisions.append(tp / (tp + fp + 1e-8))
            recalls.append(tp / (tp + fn + 1e-8))

    miou      = sum(ious)       / len(ious)       if ious       else 0
    dice      = sum(dices)      / len(dices)      if dices      else 0
    precision = sum(precisions) / len(precisions) if precisions else 0
    recall    = sum(recalls)    / len(recalls)    if recalls    else 0

    return oa, miou, dice, precision, recall


print(f"\nTraining on {DEVICE}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Steps per epoch: {SAMPLES_PER_EPOCH // BATCH_SIZE}")

csv_path = 'training_log.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_oa', 'val_miou', 'val_dice', 'val_precision', 'val_recall'])

best_val_miou = 0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    
    for data, labels in train_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        # 1. Standard 32-bit forward pass (NO AUTOCAST)
        output = model(data)
        loss = criterion(output, labels)
            
        # 2. Standard backward pass (NO SCALER)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_train_loss)

    # Validating EVERY epoch now
    model.eval()
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            val_data, val_labels = val_data.to(DEVICE), val_labels.to(DEVICE)
            val_output = model(val_data)
            val_loss = criterion(val_output, val_labels)
            val_oa, val_miou, val_dice, val_precision, val_recall = compute_metrics(val_output, val_labels, NUM_CLASSES)

    print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val OA: {val_oa:.4f} | Val mIoU: {val_miou:.4f}")

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{val_loss:.4f}", f"{val_oa:.4f}", 
                       f"{val_miou:.4f}", f"{val_dice:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}"])

    if val_miou > best_val_miou:
        best_val_miou = val_miou
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

# Final test evaluation block exactly as requested
print(f"\nBest model at epoch {best_epoch} with Val mIoU: {best_val_miou:.4f}")
print("Running final test evaluation...")

model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        output = model(data)
        test_oa, test_miou, test_dice, test_precision, test_recall = compute_metrics(output, labels, NUM_CLASSES)

print(f"Test OA:        {test_oa:.4f}")
print(f"Test mIoU:      {test_miou:.4f}")
print(f"Test Dice:      {test_dice:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")

with open('test_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['best_epoch', 'test_oa', 'test_miou', 'test_dice', 'test_precision', 'test_recall'])
    writer.writerow([best_epoch, f"{test_oa:.4f}", f"{test_miou:.4f}", f"{test_dice:.4f}", 
                    f"{test_precision:.4f}", f"{test_recall:.4f}"])

print(f"\nLogs saved to {csv_path}")
print(f"Test results saved to test_results.csv")