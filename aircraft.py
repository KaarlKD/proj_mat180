!pip install timm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import timm
import pickle
from google.colab import drive

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


from torchvision.datasets import FGVCAircraft

DATASET_NAME = 'FGVC-Aircraft'
NUM_CLASSES = 100


stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

print(f"downloading {DATASET_NAME} ...")
train_dataset_raw = FGVCAircraft(root='./data', split='trainval', annotation_level='variant', download=True, transform=train_transform)
test_dataset = FGVCAircraft(root='./data', split='test', annotation_level='variant', download=True, transform=test_transform)


print("extract LT distribution.")

try:
    raw_targets = np.array(train_dataset_raw._labels)
except:

    raw_targets = np.array([train_dataset_raw.targets[i] if hasattr(train_dataset_raw, 'targets') else train_dataset_raw[i][1] for i in range(len(train_dataset_raw))])


img_max = int(np.max(np.bincount(raw_targets)))    


IMBALANCE_RATIO = 0.1
img_num_per_cls = [int(img_max * (IMBALANCE_RATIO**(cls_idx / (NUM_CLASSES - 1.0)))) for cls_idx in range(NUM_CLASSES)]

def gen_imbalanced_aircraft(targets, img_num_per_cls):
    classes = np.unique(targets)
    idx_to_keep = []
    for t in classes:
        idx = np.where(targets == t)[0]
        np.random.shuffle(idx)
        num_to_keep = min(len(idx), img_num_per_cls[t])
        idx_to_keep.extend(idx[:num_to_keep])
    return idx_to_keep

indices = gen_imbalanced_aircraft(raw_targets, img_num_per_cls)
train_dataset = Subset(train_dataset_raw, indices)


BATCH_SIZE = 32 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ size: {len(train_dataset)} | catego num: {NUM_CLASSES}")

plt.figure(figsize=(12, 4))
plt.bar(range(NUM_CLASSES), img_num_per_cls, color='tab:blue')
plt.title(f'FGVC-Aircraft Long-Tailed Distribution (Max={img_max}, Min={img_num_per_cls[-1]})')
plt.xlabel('Aircraft Class ID')
plt.ylabel('Number of Samples')
plt.show()






def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def evaluate_per_class(model, loader, num_classes):
    model.eval()
    class_correct = [0. for _ in range(num_classes)]
    class_total = [0. for _ in range(num_classes)]
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return [100 * class_correct[i] / max(1, class_total[i]) for i in range(num_classes)]

def run_experiment(model_key, epochs=30):
    model_names = {
        'ResNet18': 'resnet18',
        'EfficientNet-B0': 'efficientnet_b0',
        'ConvNeXt-Tiny': 'convnext_tiny',
        'TinyViT-5M': 'tiny_vit_5m_224',
        'EfficientViT-M0': 'efficientvit_m0'
    }

    print(f"\n start : {model_key}")

    try:
        model = timm.create_model(model_names[model_key], pretrained=True, num_classes=NUM_CLASSES).to(device)
    except Exception as e:
        print(f"build {model_key} fail: {e}")
        return None, None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'test_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)
        scheduler.step()
        history['test_acc'].append(test_acc)

        print(f"[{model_key}] Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    per_class_acc = evaluate_per_class(model, test_loader, NUM_CLASSES)
    return history, per_class_acc

# ==================================
EPOCHS = 50
target_models = ['ResNet18', 'EfficientNet-B0', 'TinyViT-5M', 'EfficientViT-M0']
all_results = {}

for m in target_models:
    hist, per_cls = run_experiment(m, epochs=EPOCHS)
    if hist:
        all_results[m] = {'hist': hist, 'per_cls': per_cls}

    torch.cuda.empty_cache()

#draw
plt.figure(figsize=(20, 7))

### left
plt.subplot(1, 2, 1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, m in enumerate(all_results.keys()):
    plt.plot(all_results[m]['hist']['test_acc'], label=f"{m}", color=colors[i], linewidth=2)
plt.title('FGVC-Aircraft Test Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

###right
plt.subplot(1, 2, 2)
x = np.arange(NUM_CLASSES)
for i, m in enumerate(all_results.keys()):
    import pandas as pd
    smooth_data = pd.Series(all_results[m]['per_cls']).rolling(window=5, min_periods=1).mean()
    plt.plot(x, smooth_data, label=m, color=colors[i], linewidth=2, alpha=0.8)

plt.title('Smoothed Per-class Accuracy (Head to Tail)', fontsize=14)
plt.xlabel('Class Index (0=Head, 99=Tail)', fontsize=12)
plt.ylabel('Smoothed Accuracy (%)', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\n Finish")
