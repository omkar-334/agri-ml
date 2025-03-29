import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=0.0001, num_epochs=80, batch_size=32, scheduler=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        # add L2 regularization to the optimizer
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr, lr_decay=1e-6, weight_decay=1e-5, initial_accumulator_value=0, eps=1e-10)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.96) if scheduler else None
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            del outputs, loss

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del outputs, loss

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self.calculate_metrics(all_preds, all_labels)

        return avg_loss, metrics

    @staticmethod
    def calculate_metrics(predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def print_metrics(self, metrics, phase, epoch=None, train_loss=None, test_loss=None, val_loss=None, filename="metrics_log.txt"):
        log_entry = [f"\n{phase} Metrics:", "-" * 50]
        if epoch == 1:
            log_entry.append(f"Running experiment with batch_size={self.batch_size}, lr={self.lr}")
        if epoch is not None:
            log_entry.append(f"Epoch: {epoch}")
        if train_loss is not None:
            log_entry.append(f"Train Loss: {train_loss:.4f}")
        if test_loss is not None:
            log_entry.append(f"Test Loss: {test_loss:.4f}")
        if val_loss is not None:
            log_entry.append(f"Validation Loss: {val_loss:.4f}")
        for metric, value in metrics.items():
            log_entry.append(f"{metric.capitalize()}: {value:.4f}")
        log_entry.append("-" * 50)

        log_text = "\n".join(log_entry)
        print(log_text)

        with open(filename, "a") as f:
            f.write(log_text + "\n")

    def train(self):
        try:
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss, val_metrics = self.validate()

                self.print_metrics(val_metrics, "Train", epoch, train_loss, val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            if self.best_model_state is not None:
                torch.save(self.best_model_state, "interrupted_model.pt")

    def test(self):
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        test_loss, test_metrics = self.validate()
        self.print_metrics(test_metrics, "Test", test_loss=test_loss)
