import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.metrics import accuracy_score
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model,
    train_dataloader_labeled,
    train_dataloader_unlabeled,
    lr=1e-4,
    num_epochs=35,
):
    model.train()
    reconstruction_criterion = nn.MSELoss()  # Unsupervised Loss
    classification_criterion = nn.CrossEntropyLoss()  # Supervised Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        total_recon_loss = 0.0
        total_class_loss = 0.0
        total_loss = 0.0
        y_true, y_pred = [], []

        # Create iterators for both labeled and unlabeled dataloaders
        labeled_iter = iter(train_dataloader_labeled)
        unlabeled_iter = iter(train_dataloader_unlabeled)

        # Number of batches to iterate
        num_batches = max(
            len(train_dataloader_labeled), len(train_dataloader_unlabeled)
        )

        # Progress bar
        for i in tqdm(
            range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ):
            # Get the next batch from labeled and unlabeled data
            try:
                labeled_inputs, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_dataloader_labeled)
                labeled_inputs, labels = next(labeled_iter)

            try:
                unlabeled_inputs = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(train_dataloader_unlabeled)
                unlabeled_inputs = next(unlabeled_iter)

            # Transfer data to device
            labeled_inputs, labels = labeled_inputs.to(DEVICE), labels.to(DEVICE)
            unlabeled_inputs = unlabeled_inputs.to(DEVICE)

            # Inject Gaussian noise
            labeled_inputs += torch.clamp(
                torch.randn_like(labeled_inputs) * 0.1, -0.2, 0.2
            )
            unlabeled_inputs += torch.clamp(
                torch.randn_like(unlabeled_inputs) * 0.1, -0.2, 0.2
            )

            # Forward pass for both labeled and unlabeled data
            reconstructed, classification = model.train_forward(
                labeled_inputs
            )  # Labeled data forward pass
            reconstructed_unlabeled, _ = model.train_forward(
                unlabeled_inputs
            )  # Unlabeled data forward pass

            # print(f"Reconstructed shape: {reconstructed.shape}, Classification shape: {classification.shape}")
            # print(f"Reconstructed unlabeled shape: {reconstructed_unlabeled.shape}")
            # print(f"Labeled inputs shape: {labeled_inputs.shape}, unlabled inputs shape: {unlabeled_inputs.shape}")
            # Compute losses
            recon_loss_labeled = reconstruction_criterion(
                reconstructed, labeled_inputs
            )  # Unsupervised loss (labeled)
            recon_loss_unlabeled = reconstruction_criterion(
                reconstructed_unlabeled, unlabeled_inputs
            )  # Unsupervised loss (unlabeled)
            class_loss_labeled = classification_criterion(
                classification, labels
            )  # Supervised loss (labeled)

            # Total loss
            total_recon_loss += recon_loss_labeled.item() + recon_loss_unlabeled.item()
            total_class_loss += class_loss_labeled.item()

            # Combine the loss: unsupervised loss for both, supervised loss for labeled data
            loss = recon_loss_labeled + recon_loss_unlabeled + class_loss_labeled
            total_loss += loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy for labeled data
            predicted = torch.argmax(classification, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        # After each epoch, calculate accuracy and other metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss:.4f}, "
            f"Reconstruction Loss: {total_recon_loss:.4f}, "
            f"Classification Loss: {total_class_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
    return model


def validate_model(
    model, validation_loader, num_classes, epoch, csv_path="validation_metrics.csv"
):
    model.eval()

    # Initialize metrics for multi-class evaluation
    accuracy = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass").to(
        DEVICE
    )
    precision = torchmetrics.Precision(
        num_classes=num_classes, task="multiclass", average="macro"
    ).to(DEVICE)
    recall = torchmetrics.Recall(
        num_classes=num_classes, task="multiclass", average="macro"
    ).to(DEVICE)
    f1 = torchmetrics.F1Score(
        num_classes=num_classes, task="multiclass", average="macro"
    ).to(DEVICE)
    confusion = torchmetrics.ConfusionMatrix(
        num_classes=num_classes, task="multiclass"
    ).to(DEVICE)

    all_labels, all_preds = [], []
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(validation_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Update metrics
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)
            confusion.update(preds, labels)

            # Store results for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            num_samples += len(labels)

    # Compute final scores
    val_acc = accuracy.compute().item()
    val_prec = precision.compute().item()
    val_rec = recall.compute().item()
    val_f1 = f1.compute().item()
    cm = (
        confusion.compute().cpu().numpy()
    )  # Confusion matrix (num_classes x num_classes)

    # For multi-class: compute TP, FP, FN, TN
    TP = cm.diagonal()
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Aggregate totals (across all classes)
    total_TP = TP.sum()
    total_FP = FP.sum()
    total_FN = FN.sum()
    total_TN = TN.sum()

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Number of samples validated on: {num_samples}")
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}, TN: {total_TN}")

    # Save results to CSV
    results = {
        "epoch": epoch,
        "accuracy": val_acc,
        "precision": val_prec,
        "recall": val_rec,
        "f1": val_f1,
        "tp": int(total_TP),
        "tn": int(total_TN),
        "fp": int(total_FP),
        "fn": int(total_FN),
        "num_samples": num_samples,
    }

    df = pd.DataFrame([results])

    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)  # Create new file
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)  # Append

    return results
