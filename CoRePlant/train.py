import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def worker_init_fn(worker_id, seed=1):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        logits = torch.div(
            torch.matmul(feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)),
            self.temperature,
        )
        return F.cross_entropy(logits, labels)


def update_teacher_model(student_model_classifier, teacher_model_classifier, global_step, alpha=0.99):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_params, student_params in zip(teacher_model_classifier.parameters(), student_model_classifier.parameters()):
        teacher_params.data.mul_(alpha).add_(student_params.data, alpha=1 - alpha)


def update_consistency_loss(outputs, ema_output, original_loss, alpha=0.5):
    # Compute softmax and log_softmax
    if False:
        sm = nn.Softmax(dim=1)
        log_sm = nn.LogSoftmax(dim=1)
        kl_distance = nn.KLDivLoss(reduction="none")

        # Calculate KL divergence without reduction
        loss_kl = torch.sum(kl_distance(log_sm(outputs), sm(ema_output)), dim=1)

        # Apply exponential weighting to the original loss
        exp_loss_kl = torch.exp(-loss_kl)

        # Compute the weighted original loss
        # Here we are not using original_loss directly because we are checking whether student is consistent with teacher
        # If it is consistent then we are taking max proportion of supervised_loss or else less proportion
        weighted_original_loss = torch.mean(original_loss * exp_loss_kl)

        # Combine weighted original loss with mean KL divergence
        final_loss = weighted_original_loss + torch.mean(loss_kl)

        return final_loss

    log_probs_student = F.log_softmax(outputs, dim=1)
    probs_teacher = F.softmax(ema_output, dim=1)

    # KLDivLoss with reduction='batchmean' is efficient and numerically stable
    consistency_loss = F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")

    # Combine with supervised loss (weighted sum)
    total_loss = alpha * original_loss + (1 - alpha) * consistency_loss
    return total_loss


def mean_teacher_train(
    student,
    teacher,
    train_loader,
    test_loader,
    unlabeled_loader,
    unlabeled_student_loader,
    num_classes=5,
    lr=1e-4,
    num_epochs=35,
):
    best_model_wts = deepcopy(student.state_dict())
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    classifier_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)  # Only train classifier
    torch.cuda.empty_cache()
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    global_step = 0

    for epoch in range(num_epochs):
        student.train()
        running_loss = 0.0
        train_accuracy.reset()

        # Iterate through labeled and unlabeled data together
        labeled_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader)
        unlabeled_iter_student_input = iter(unlabeled_student_loader)
        num_batches = max(len(train_loader), len(unlabeled_loader))

        for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            # Get next batch from labeled and unlabeled data
            try:
                labeled_inputs, labels = next(labeled_iter)
            except StopIteration:
                # Recycle the labeled data if it's exhausted
                labeled_iter = iter(train_loader)
                labeled_inputs, labels = next(labeled_iter)

            try:
                unlabeled_inputs, lbp_histograms = next(unlabeled_iter)
                unlabeled_inputs_student_input, lbp_histograms_student_input = next(unlabeled_iter_student_input)
            except StopIteration:
                # Recycle the unlabeled data if it's exhausted
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_inputs, lbp_histograms = next(unlabeled_iter)
                unlabeled_iter_student_input = iter(unlabeled_student_loader)
                unlabeled_inputs_student_input, lbp_histograms_student_input = next(unlabeled_iter_student_input)

            # Transfer data to device
            labeled_inputs, labels = labeled_inputs.to(device), labels.to(device)
            unlabeled_inputs, lbp_histograms = unlabeled_inputs.to(device), lbp_histograms.to(device)
            unlabeled_inputs_student_input, lbp_histograms_student_input = unlabeled_inputs_student_input.to(device), lbp_histograms_student_input.to(device)

            # Forward pass through the student model for labeled data
            optimizer.zero_grad()
            outputs_labeled = student(labeled_inputs, None)
            supervised_loss = classifier_loss_fn(outputs_labeled, labels)

            # Forward pass through the student model for unlabeled data
            outputs_unlabeled = student(unlabeled_inputs_student_input, lbp_histograms_student_input)

            # Process the teacher model outputs
            with torch.no_grad():
                ema_output = teacher(unlabeled_inputs, lbp_histograms)

            total_loss = update_consistency_loss(
                outputs_unlabeled,  # Unlabeled student predictions
                ema_output,  # Teacher model predictions
                supervised_loss,  # Supervised contrastive loss
            )

            # Combine losses and update model
            #             total_loss = supervised_loss + consistency_loss
            total_loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            running_loss += total_loss.item() * (labeled_inputs.size(0) + unlabeled_inputs.size(0))
            preds = torch.argmax(outputs_labeled, dim=1)
            train_accuracy.update(preds, labels)

        epoch_loss = running_loss / (len(train_loader.dataset) + len(unlabeled_loader.dataset))
        epoch_acc = train_accuracy.compute().item()
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch + 1} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Evaluate student model on validation data
        student.eval()
        running_loss = 0.0
        test_accuracy.reset()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                loss = classifier_loss_fn(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                test_accuracy.update(preds, labels)

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = test_accuracy.compute().item()
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)
        print(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save best student model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = deepcopy(student.state_dict())

        # Update teacher model using EMA
        update_teacher_model(student, teacher, global_step, alpha=0.99)
        global_step += 1
        scheduler.step()

    print(f"Best Val Acc: {best_acc:.4f}")
    student.load_state_dict(best_model_wts)
    results = (train_losses, test_losses, train_accuracies, test_accuracies)
    return student, results


def plot(results):
    train_losses, test_losses, train_accuracies, test_accuracies = results

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def validate(model, validation_loader, num_classes=5):
    model.eval()

    val_accuracy = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass").to(device)
    val_precision = torchmetrics.Precision(num_classes=num_classes, average="macro", task="multiclass").to(device)
    val_recall = torchmetrics.Recall(num_classes=num_classes, average="macro", task="multiclass").to(device)
    val_f1 = torchmetrics.F1Score(num_classes=num_classes, average="macro", task="multiclass").to(device)

    all_labels, all_preds, all_probs = [], [], []
    images = []

    with torch.no_grad():
        for inputs, labels in tqdm(validation_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            val_accuracy.update(preds, labels)
            val_precision.update(preds, labels)
            val_recall.update(preds, labels)
            val_f1.update(preds, labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            images.extend(inputs.cpu())

    val_acc = val_accuracy.compute().item()
    val_prec = val_precision.compute().item()
    val_rec = val_recall.compute().item()
    val_f1_score = val_f1.compute().item()

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    print(f"Validation F1 Score: {val_f1_score:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.show()


# torch.cuda.empty_cache()
# student_model, results = mean_teacher_train(student_model_efficient, student_model_classifier, teacher_model_classifier, train_loader, unlabeled_loader, test_loader, num_epochs=35)
# evaluate_and_display(teacher_model_classifier, validation_loader)
