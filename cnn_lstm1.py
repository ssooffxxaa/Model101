import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_utils import prepare_train_test_data

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self, input_channels=768, dropout_rate=0.3):
        super().__init__()
        self.feature_reduction = nn.Sequential(
            # First CNN block
            nn.Conv1d(input_channels, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Second CNN block
            nn.Conv1d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Third CNN block
            nn.Conv1d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.output_channels = 96

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.feature_reduction(x)


class CNN_LSTM(nn.Module):
    def __init__(self, sequence_length=20, input_channels=768, hidden_size=256,
                 num_classes=4, num_lstm_layers=2, dropout_rate=0.3):
        super().__init__()

        self.cnn = CNN(input_channels, dropout_rate)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.cnn.output_channels,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)

        # Classification using the last output
        x = self.classifier(lstm_out[:, -1, :])
        return x


def train_and_evaluate(X_train, y_train, X_test, y_test, class_weights=None):
    # Normalize the data
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    X_test_normalized = scaler.transform(X_test_reshaped)

    X_train_normalized = X_train_normalized.reshape(X_train.shape)
    X_test_normalized = X_test_normalized.reshape(X_test.shape)

    # Convert to tensor
    X_train = torch.FloatTensor(X_train_normalized)
    X_test = torch.FloatTensor(X_test_normalized)

    # Model parameters with modified learning rate and increased epochs
    config = {
        'hidden_size': 256,
        'batch_size': 32,
        'learning_rate': 0.0005,  # Reduced from 0.001 to 0.0005
        'num_epochs': 200,        # Increased from 100 to 200
        'dropout_rate': 0.3
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN_LSTM(
        sequence_length=20,
        hidden_size=config['hidden_size'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # For tracking best model
    best_f1 = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

            all_preds = []
            all_labels = []

            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()

            print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], '
                  f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    print("\nFinal Model Evaluation (Best Model):")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    return model


if __name__ == "__main__":
    # Get data from preprocessing
    X_train, X_test, y_train, y_test, class_weights = prepare_train_test_data()

    # Train and evaluate model
    model = train_and_evaluate(X_train, y_train, X_test, y_test, class_weights)