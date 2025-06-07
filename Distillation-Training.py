# Advanced Knowledge Distillation with Teacher Boosting and Optimization

# Knowledge Distillation with EfficientNet-B0 + Early Stopping + Gradual Alpha

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy
import matplotlib.pyplot as plt

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform_train = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(96, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# Load STL10 dataset
train_set = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
test_set = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Define teacher model
teacher_model = models.mobilenet_v2(pretrained=True)
teacher_model.classifier[1] = nn.Linear(1280, 10)
teacher_model = teacher_model.to(device)

# Define student model using EfficientNet-B0
student_model = models.efficientnet_b0(pretrained=False)
student_model.classifier[1] = nn.Linear(student_model.classifier[1].in_features, 10)
student_model = student_model.to(device)

# Loss and optimizer
ce_loss = nn.CrossEntropyLoss()
kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.0005)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

# Train teacher model
def train_teacher(model, loader, optimizer, epochs=25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Teacher] Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# Distillation loss with gradual alpha
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.9):
    soft_targets = nn.functional.log_softmax(student_logits / T, dim=1)
    soft_teacher = nn.functional.softmax(teacher_logits / T, dim=1)
    kd_loss = kd_loss_fn(soft_targets, soft_teacher) * (T * T)
    ce = ce_loss(student_logits, labels)
    return alpha * kd_loss + (1. - alpha) * ce

# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

# Train student model with distillation and early stopping
def train_kd(student, teacher, loader, optimizer, test_loader, epochs=30):
    teacher.eval()
    student.train()
    start = time.time()
    best_acc = 0.0
    best_model = copy.deepcopy(student.state_dict())
    patience = 5
    trigger = 0
    val_acc_history = []

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher(images)
            student_outputs = student(images)

            if epoch < 5:
                loss = ce_loss(student_outputs, labels)
            else:
                alpha = min(0.1 + 0.05 * (epoch - 5), 0.9)
                loss = distillation_loss(student_outputs, teacher_outputs, labels, alpha=alpha)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(student, test_loader)
        val_acc_history.append(val_acc)
        print(f"[Student] Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(student.state_dict())
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered")
                break

    student.load_state_dict(best_model)
    print(f"Training completed in {time.time() - start:.2f} seconds. Best Accuracy: {best_acc:.2f}%")

    # Plot accuracy trend
    plt.plot(val_acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Student Accuracy During Training')
    plt.grid(True)
    plt.savefig("student_accuracy_plot.png")
    plt.show()

# ---- EXECUTION ----
print("Training teacher model...")
train_teacher(teacher_model, train_loader, optimizer_teacher)
teacher_acc = evaluate(teacher_model, test_loader)
print(f"Teacher Test Accuracy: {teacher_acc:.2f}%")

print("\nTraining student model (EfficientNet-B0) with knowledge distillation and early stopping...")
train_kd(student_model, teacher_model, train_loader, optimizer_student, test_loader)
final_acc = evaluate(student_model, test_loader)
print(f"Final Student Accuracy: {final_acc:.2f}%")
