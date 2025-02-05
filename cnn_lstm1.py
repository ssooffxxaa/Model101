import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_data_from_files, prepare_data, balance_classes, prepare_cross_validation_data

# Device configuration
# device = torch.device("cuda:0" )
# print(device)

import os


def plot_learning_curves(train_losses, val_losses, val_accuracies, val_f1s, fold):
    """
    Plot learning curves for a single fold with improved colors and save to file
    """
    # Remove seaborn style
    plt.style.use('default')  # Use default style instead

    fig = plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, color='#2ecc71', label='Training Loss', linewidth=2)
    plt.plot(val_losses, color='#e74c3c', label='Validation Loss', linewidth=2)
    plt.title(f'Learning Curves - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, color='#3498db', label='Validation Accuracy', linewidth=2)
    plt.plot(val_f1s, color='#9b59b6', label='Validation F1 Score', linewidth=2)
    plt.title(f'Validation Metrics - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Define save path
    save_path = r"D:\Model101\.venv\Scripts"
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    # Save the plot
    plt.savefig(os.path.join(save_path, f'fold_{fold + 1}_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_average_curves(all_folds_train_losses, all_folds_val_losses,
                        all_folds_val_accuracies, all_folds_val_f1s, config):
    """
    Plot average learning curves across all folds with improved colors and save to file
    """
    # Remove seaborn style
    plt.style.use('default')  # Use default style instead

    fig = plt.figure(figsize=(15, 5))

    # Plot average losses
    plt.subplot(1, 2, 1)
    mean_train_losses = np.mean(all_folds_train_losses, axis=0)
    std_train_losses = np.std(all_folds_train_losses, axis=0)
    mean_val_losses = np.mean(all_folds_val_losses, axis=0)
    std_val_losses = np.std(all_folds_val_losses, axis=0)

    epochs = range(1, config['num_epochs'] + 1)

    # Training loss with confidence interval
    plt.plot(epochs, mean_train_losses, color='#2ecc71', label='Avg Training Loss', linewidth=2)
    plt.fill_between(epochs, mean_train_losses - std_train_losses,
                     mean_train_losses + std_train_losses, color='#2ecc71', alpha=0.2)

    # Validation loss with confidence interval
    plt.plot(epochs, mean_val_losses, color='#e74c3c', label='Avg Validation Loss', linewidth=2)
    plt.fill_between(epochs, mean_val_losses - std_val_losses,
                     mean_val_losses + std_val_losses, color='#e74c3c', alpha=0.2)

    plt.title('Average Learning Curves Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot average metrics
    plt.subplot(1, 2, 2)
    mean_val_accuracies = np.mean(all_folds_val_accuracies, axis=0)
    std_val_accuracies = np.std(all_folds_val_accuracies, axis=0)
    mean_val_f1s = np.mean(all_folds_val_f1s, axis=0)
    std_val_f1s = np.std(all_folds_val_f1s, axis=0)

    # Accuracy with confidence interval
    plt.plot(epochs, mean_val_accuracies, color='#3498db',
             label='Avg Validation Accuracy', linewidth=2)
    plt.fill_between(epochs, mean_val_accuracies - std_val_accuracies,
                     mean_val_accuracies + std_val_accuracies, color='#3498db', alpha=0.2)

    # F1 score with confidence interval
    plt.plot(epochs, mean_val_f1s, color='#9b59b6',
             label='Avg Validation F1 Score', linewidth=2)
    plt.fill_between(epochs, mean_val_f1s - std_val_f1s,
                     mean_val_f1s + std_val_f1s, color='#9b59b6', alpha=0.2)

    plt.title('Average Validation Metrics Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Define save path
    save_path = r"D:\Model101\.venv\Scripts"
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    # Save the plot
    plt.savefig(os.path.join(save_path, 'average_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

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


def train_and_evaluate_cv(X, y, n_splits=5):
    """
    Train and evaluate model using cross validation
    """
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Model configuration
    config = {
        'hidden_size': 256,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'num_epochs': 1000,
        'dropout_rate': 0.3
    }

    # Lists to store metrics for each fold
    fold_metrics = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Lists to store all folds data for average plots
    all_folds_train_losses = []
    all_folds_val_losses = []
    all_folds_val_accuracies = []
    all_folds_val_f1s = []

    # Perform cross validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print("-" * 50)

        # Lists to store metrics for current fold
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1s = []

        # [Original data preparation code remains unchanged...]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

        X_train_normalized = scaler.fit_transform(X_train_reshaped)
        X_val_normalized = scaler.transform(X_val_reshaped)

        X_train_normalized = X_train_normalized.reshape(X_train.shape)
        X_val_normalized = X_val_normalized.reshape(X_val.shape)

        X_train_tensor = torch.FloatTensor(X_train_normalized).to(device)
        y_train_tensor = torch.LongTensor(y_train.numpy()).to(device)
        X_val_tensor = torch.FloatTensor(X_val_normalized).to(device)
        y_val_tensor = torch.LongTensor(y_val.numpy()).to(device)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train.numpy()),
            y=y_train.numpy()
        )

        model = CNN_LSTM(
            sequence_length=20,
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate']
        ).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        best_val_f1 = 0
        best_model_state = None

        for epoch in range(config['num_epochs']):
            # Training phase
            model.train()
            total_train_loss = 0
            total_val_loss = 0

            for sequences, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for sequences, labels in val_loader:
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            val_accuracy = accuracy_score(val_labels, val_preds)
            val_accuracies.append(val_accuracy)

            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            val_f1s.append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()

            print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], '
                  f'Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        # Plot learning curves for this fold
        plot_learning_curves(train_losses, val_losses, val_accuracies, val_f1s, fold)

        # Store metrics for average plots
        all_folds_train_losses.append(train_losses)
        all_folds_val_losses.append(val_losses)
        all_folds_val_accuracies.append(val_accuracies)
        all_folds_val_f1s.append(val_f1s)

        # [Original final evaluation code remains unchanged...]
        model.load_state_dict(best_model_state)
        model.eval()
        final_preds = []
        final_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

        fold_metrics.append({
            'accuracy': accuracy_score(final_labels, final_preds),
            'f1': f1_score(final_labels, final_preds, average='weighted'),
            'classification_report': classification_report(final_labels, final_preds, digits=4)
        })

        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {fold_metrics[-1]['accuracy']:.4f}")
        print(f"F1 Score: {fold_metrics[-1]['f1']:.4f}")
        print("\nClassification Report:")
        print(fold_metrics[-1]['classification_report'])

    # Plot average curves across all folds
    plot_average_curves(all_folds_train_losses, all_folds_val_losses,
                        all_folds_val_accuracies, all_folds_val_f1s, config)

    print("\nAverage Metrics Across All Folds:")
    print(
        f"Average Accuracy: {np.mean([m['accuracy'] for m in fold_metrics]):.4f} ± {np.std([m['accuracy'] for m in fold_metrics]):.4f}")
    print(
        f"Average F1 Score: {np.mean([m['f1'] for m in fold_metrics]):.4f} ± {np.std([m['f1'] for m in fold_metrics]):.4f}")

    return fold_metrics

if __name__ == "__main__":
    # Prepare data
    X, y = prepare_cross_validation_data()

    # Perform cross validation
    fold_metrics = train_and_evaluate_cv(X, y, n_splits=5)
