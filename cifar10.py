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

#set seed
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


##second part
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np


def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 5000
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls

def gen_imbalanced_data(dataset, img_num_per_cls):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    idx_to_keep = []
    for t in classes:
        idx = np.where(targets == t)[0]
        np.random.shuffle(idx)
        idx = idx[:img_num_per_cls[t]]
        idx_to_keep.extend(idx)
    return idx_to_keep

##resize
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std), 
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

# download
print("loading cifar10")
train_dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# longtail distribution
IMBALANCE_RATIO = 0.01
num_per_cls = get_img_num_per_cls(10, 'exp', IMBALANCE_RATIO)
indices = gen_imbalanced_data(train_dataset_raw, num_per_cls)
train_dataset = Subset(train_dataset_raw, indices)

print(f"size: {len(train_dataset_raw)}")
print(f"size after: {len(train_dataset)}")

plt.figure(figsize=(8, 5))
plt.bar(range(10), num_per_cls, color='tab:blue', alpha=0.8)
plt.title(f'Long-Tailed Distribution (Ratio={int(1/IMBALANCE_RATIO)})', fontsize=12)
plt.xlabel('Class ID', fontsize=10)
plt.ylabel('Number of Samples', fontsize=10)
plt.xticks(range(10))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

##third part, key training code
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

def evaluate_per_class(model, loader):
    model.eval()
    class_correct = [0. for i in range(10)]
    class_total = [0. for i in range(10)]
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
    return [100 * class_correct[i] / max(1, class_total[i]) for i in range(10)]

def run_experiment(model_key, epochs=30):
    model_names = {
        'ResNet18': 'resnet18',
        'EfficientNet-B0': 'efficientnet_b0',
        # 'ConvNeXt-Tiny': 'convnext_tiny',
        'TinyViT-5M': 'tiny_vit_5m_224',
        'EfficientViT-M0': 'efficientvit_m0',
        # 'Swin-Tiny': 'swin_tiny_patch4_window7_224'
    }

    print(f"\nstart: {model_key}")

    try:
        model = timm.create_model(model_names[model_key], pretrained=False, num_classes=10).to(device)
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

    per_class_acc = evaluate_per_class(model, test_loader)
    return history, per_class_acc


##run model
EPOCHS = 50  # set epoches
target_models = [
    'ResNet18', 'EfficientNet-B0',
    'TinyViT-5M', 'EfficientViT-M0'
]
all_results = {}

for m in target_models:
    hist, per_cls = run_experiment(m, epochs=EPOCHS)
    if hist:
        all_results[m] = {'hist': hist, 'per_cls': per_cls}

    # empty the cache
    torch.cuda.empty_cache()

#plot
plt.figure(figsize=(18, 7))

# left graph
plt.subplot(1, 2, 1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, m in enumerate(all_results.keys()):
    plt.plot(all_results[m]['hist']['test_acc'], label=m, color=colors[i], linewidth=2)
plt.title('Overall Test Accuracy Comparison', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# right graph
plt.subplot(1, 2, 2)
x = np.arange(10)
width = 0.14
for i, m in enumerate(all_results.keys()):
    plt.bar(x + (i-2.5)*width, all_results[m]['per_cls'], width, label=m, color=colors[i])

plt.title('Per-class Accuracy (Head to Tail)', fontsize=14)
plt.xlabel('Class Index (0=Many samples, 9=Few samples)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(x)
plt.legend(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\nfinish")
