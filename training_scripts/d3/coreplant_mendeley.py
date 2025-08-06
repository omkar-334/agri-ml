import os
import sys

import numpy as np
import torch

from CoRePlant.coreplant import Classifier
from CoRePlant.dataset import get_mean_teacher_dataloaders
from CoRePlant.train import mean_teacher_train, validate

torch.manual_seed(42)
np.random.seed(42)

# Create logs directory
os.makedirs("logs", exist_ok=True)

NUM_CLASSES = 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for labeled_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    log_filename = f"logs/coreplant_mendeley_{int(labeled_ratio * 100)}pct.log"

    # Redirect stdout and stderr to log file
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    print(
        f"\n=== Training with {int(labeled_ratio * 100)}% of labeled training data ===\n"
    )
    train_loader, test_loader, unlabeled_loader, unlabeled_student_loader = (
        get_mean_teacher_dataloaders("data3/mendeley", labeled_ratio, 16)
    )

    student = Classifier(512, 256, NUM_CLASSES).to(device)
    teacher = Classifier(512, 256, NUM_CLASSES).to(device)

    # # Synchronize initial weights
    teacher.encoder.load_state_dict(student.encoder.state_dict())
    teacher.load_state_dict(student.state_dict())

    student_model, results = mean_teacher_train(
        student,
        teacher,
        train_loader,
        test_loader,
        unlabeled_loader,
        unlabeled_student_loader,
        NUM_CLASSES,
    )

    print("Student Model:")
    validate(student_model, test_loader, NUM_CLASSES)
    print("Teacher Model:")
    validate(teacher, test_loader, NUM_CLASSES)
