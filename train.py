import torch
import torch.amp as amp 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=0.2, num_epochs=80):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.lr = lr

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.96)
        self.criterion = nn.CrossEntropyLoss().to(self.device)  
        self.scaler = amp.GradScaler("cuda")

        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for images, labels in pbar:
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True) 
            
            with amp.autocast("cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            del outputs, loss

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

            with amp.autocast("cuda"):
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

    @staticmethod
    def print_metrics(metrics, phase):
        print(f"\n{phase} Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("-" * 50)

    def train(self):
        try:
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss, val_metrics = self.validate()

                print(f"\nEpoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
                self.print_metrics(val_metrics, "Validation")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                # if torch.cuda.is_available():
                #     print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            if self.best_model_state is not None:
                torch.save(self.best_model_state, "interrupted_model.pt")

    def test(self):
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        test_loss, test_metrics = self.validate()
        print("\nBest Model Performance on Test Set:")
        self.print_metrics(test_metrics, "Test")
