import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetModel(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        # self.model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        # self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=True)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, embedding_size)

        # Ensure the parameters are set to require gradients if necessary
        for param in self.model.parameters():
            param.requires_grad = True  # or False, depending on whether you want to train them

    def forward(self, x):
        embeddings = self.model(x)
        return embeddings


class Classifier(nn.Module):
    def __init__(self, input_shape=512, hidden_units=256, num_classes=5, lbp_feature_size=26, dropout_rate=0.5):
        super().__init__()
        self.encoder = EfficientNetModel(embedding_size=input_shape)
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Ensure input_shape and lbp_feature_size are integers
        input_shape = int(input_shape)
        lbp_feature_size = int(lbp_feature_size)

        # Define layers for input with LBP features
        self.fc1_with_lbp = nn.Linear(input_shape + lbp_feature_size, hidden_units)
        self.bn1_with_lbp = nn.BatchNorm1d(hidden_units)

        # Define layers for input without LBP features
        self.fc1_without_lbp = nn.Linear(input_shape, hidden_units)
        self.bn1_without_lbp = nn.BatchNorm1d(hidden_units)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_classes)

    def forward(self, x, lbp_features=None):
        x = self.encoder(x)
        if lbp_features is not None and not torch.any(lbp_features):  # no LBP features
            lbp_features = None

        if lbp_features is not None:
            # Check dimensions of LBP features
            #             print(f"LBP features shape: {lbp_features.shape}")
            x = torch.cat((x, lbp_features), dim=1)
            #             print(f"Concatenated input shape: {x.shape}")
            x = F.relu(self.bn1_with_lbp(self.fc1_with_lbp(self.dropout1(x))))
        else:
            x = F.relu(self.bn1_without_lbp(self.fc1_without_lbp(self.dropout1(x))))

        x = F.relu(self.bn2(self.fc2(self.dropout2(x))))
        x = self.fc3(x)
        return x
