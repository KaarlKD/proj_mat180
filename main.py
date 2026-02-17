import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import timm
import random
import os 

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


print("prepare dataset")

#resize
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

# download
train_dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# imbalance
IMBALANCE_RATIO = 0.01  
num_per_cls = get_img_num_per_cls(10, 'exp', IMBALANCE_RATIO)
print(f"每个类别的样本数量分布: {num_per_cls}")

indices = gen_imbalanced_data(train_dataset_raw, num_per_cls)
train_dataset = torch.utils.data.Subset(train_dataset_raw, indices)

# dataloader
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



#def train
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

#cal accu
def evaluate_per_class(model, loader):
    model.eval()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    return [100 * class_correct[i] / class_total[i] for i in range(10)]



#main part
def run_experiment(model_name, epochs=10):
    print(f"\nStarted Training: {model_name}")
  
    if model_name == 'ResNet18':
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    elif model_name == 'ViT-Tiny':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    
    model = model.to(device)


  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)
        scheduler.step()
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        

    per_class_acc = evaluate_per_class(model, test_loader)
    return history, per_class_acc



#compare, change epochs
EPOCHS = 50

print("==============================")
print("Running ResNet-18 Experiment")
print("==============================")
resnet_hist, resnet_per_class = run_experiment('ResNet18', epochs=EPOCHS)

print("\n==============================")
print("Running ViT-Tiny Experiment")
print("==============================")
vit_hist, vit_per_class = run_experiment('ViT-Tiny', epochs=EPOCHS)



#plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(resnet_hist['test_acc'], label='ResNet18', marker='o')
plt.plot(vit_hist['test_acc'], label='ViT-Tiny', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Overall Test Accuracy vs Epochs')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
classes = np.arange(10)
width = 0.35
plt.bar(classes - width/2, resnet_per_class, width, label='ResNet18')
plt.bar(classes + width/2, vit_per_class, width, label='ViT-Tiny')
plt.xlabel('Class Index (0=Many, 9=Few)')
plt.ylabel('Accuracy per Class (%)')
plt.title('Performance on Imbalanced Classes')
plt.xticks(classes)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("\nfinish")
