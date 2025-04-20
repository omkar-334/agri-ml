import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set models to evaluation mode
student_model_classifier.eval()
teacher_model_classifier.eval()

# For 3 classes
class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]


# Function to extract embeddings (optionally with LBP)
def extract_embeddings(model_classifier, dataloader, use_lbp=False):
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if use_lbp:
                images, lbp_features, labels = batch
                images, lbp_features = images.to(device), lbp_features.to(device)
            else:
                images, labels = batch
                images = images.to(device)

            # Ensure 4D input
            if images.ndim == 3:
                images = images.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

            # Convert grayscale to RGB if needed
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            # Extract embeddings
            embeddings = model_classifier.encoder(images)
            if use_lbp:
                embeddings = torch.cat((embeddings, lbp_features), dim=1)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_embeddings, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


# Plot t-SNE results with custom colors
def plot_tsne(embeddings, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, perplexity=3, init="random", learning_rate="auto", random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    # Custom colors for each class
    # custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    custom_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",
    ]  # Purple

    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], color=custom_colors[i], label=class_name, alpha=0.7, edgecolor="k", linewidth=0.5)

    plt.legend(title="Classes")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- t-SNE for Student Model ---
student_embeddings, student_labels = extract_embeddings(student_model_classifier, train_loader, use_lbp=False)
plot_tsne(student_embeddings, student_labels, title="t-SNE: Student Model (5 Classes)")

# --- t-SNE for Teacher Model ---
teacher_embeddings, teacher_labels = extract_embeddings(teacher_model_classifier, train_loader, use_lbp=False)
plot_tsne(teacher_embeddings, teacher_labels, title="t-SNE: Teacher Model (5 Classes)")
print(f"Embeddings shape: {teacher_embeddings.shape}")
print(f"Labels shape: {teacher_labels.shape}")
