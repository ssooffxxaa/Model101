import torch
import torch.nn as nn
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from data_utils import prepare_train_test_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, activation):
        super(CNN, self).__init__()
        self.activation_name = activation
        self.activation = getattr(nn, activation)()

        self.feature_reduction = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=2),
            self.activation,
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),

            nn.Conv1d(256, 128, kernel_size=3, padding=2),
            nn.BatchNorm1d(128),
            self.activation,
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.feature_reduction(x)


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size, num_lstm_layers, activation, sequence_length=20, num_classes=4):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(activation)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.cnn_output_size = 128

        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            getattr(nn, activation)(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        x = self.classifier(lstm_out[:, -1, :])
        return x


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100


def train_model(trial, X_train, y_train, X_test, y_test, device, class_weights=None):
    # Hyperparameters from Optuna
    hidden_size = trial.suggest_int("lstm_hidden_units", 50, 150)
    num_lstm_layers = trial.suggest_int("lstm_hidden_layers", 1, 5)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
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
        activation=activation
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_test, y_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_acc = 0

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
            train_acc = compute_accuracy(model, train_loader, device)
            val_acc = compute_accuracy(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_trial_{trial.number}.pth')

            # Print progress
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            print("-" * 30)

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
    X_train, X_test, y_train, y_test, class_weights = prepare_train_test_data()
    return train_model(trial, X_train, y_train, X_test, y_test, device, class_weights)
//



def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
