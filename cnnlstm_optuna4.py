import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_utils import prepare_train_test_data
import optuna

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, input_channels=768, dropout_rate=0.3, activation='ReLU'):
        super().__init__()
        self.activation = getattr(nn, activation)()

        self.feature_reduction = nn.Sequential(
            # First CNN block
            nn.Conv1d(input_channels, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            self.activation,
            nn.Dropout(dropout_rate),

            # Second CNN block
            nn.Conv1d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            self.activation,
            nn.Dropout(dropout_rate),

            # Third CNN block
            nn.Conv1d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            self.activation,
            nn.Dropout(dropout_rate)
        )

        self.output_channels = 96

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.feature_reduction(x)


class CNN_LSTM(nn.Module):
    def __init__(self, sequence_length=20, input_channels=768, hidden_size=256,
                 num_classes=4, num_lstm_layers=2, dropout_rate=0.3, activation='ReLU'):
        super().__init__()

        self.cnn = CNN(input_channels, dropout_rate, activation)
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
            getattr(nn, activation)(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        x = self.classifier(lstm_out[:, -1, :])
        return x


def compute_accuracy(model, data_loader, device):
    """Calculate accuracy for the given model and data loader"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (correct / total) * 100
    return accuracy, all_preds, all_labels


def train_model(trial, X_train, y_train, X_test, y_test, device, class_weights=None):
    """Train model with hyperparameters suggested by Optuna"""

    # Hyperparameters from Optuna
    hidden_size = trial.suggest_int("lstm_hidden_units", 50, 150)
    num_lstm_layers = trial.suggest_int("lstm_hidden_layers", 1, 5)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ['ReLU', 'Sigmoid', 'Tanh'])
    num_epochs = trial.suggest_int("epochs", 50, 200)


    # Print trial parameters
    print("\nTrial Parameters:")
    print(f"Batch Size: {batch_size}")
    print(f"Activation Function: {activation}")
    print(f"Hidden Units: {hidden_size}")
    print(f"Hidden Layers: {num_lstm_layers}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}")

    # Create model
    model = CNN_LSTM(
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training tracking
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
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

            # Calculate accuracies
            train_acc, _, _ = compute_accuracy(model, train_loader, device)
            val_acc, _, _ = compute_accuracy(model, test_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_trial_{trial.number}.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            print("-" * 30)

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted!")

    # Final trial summary
    print(f"\nTrial #{trial.number} Summary:")
    print("=" * 30)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Parameters:")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print("=" * 50)

    return best_val_acc


def objective(trial):
    """Optuna objective function"""
    X_train, X_test, y_train, y_test, class_weights = prepare_train_test_data()

    # Normalize the data
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    X_test_normalized = scaler.transform(X_test_reshaped)

    X_train_normalized = torch.FloatTensor(X_train_normalized.reshape(X_train.shape))
    X_test_normalized = torch.FloatTensor(X_test_normalized.reshape(X_test.shape))

    return train_model(trial, X_train_normalized, y_train, X_test_normalized, y_test, device, class_weights)


def main():
    print("\nStarting Optuna optimization...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
