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
from data_utils import prepare_train_test_data

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
    save_path = r"C:\CNNLSTM\scripts"
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
    save_path = r"C:\CNNLSTM\scripts"
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


def train_and_evaluate_cv(X_train, y_train, X_test, y_test, n_splits=5):
    """
    Train and evaluate model using cross validation on training data only,
    then evaluate on test set.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    config = {
        'hidden_size': 256,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'num_epochs': 100,
        'dropout_rate': 0.3
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold_metrics = []
    all_folds_train_losses, all_folds_val_losses = [], []
    all_folds_val_accuracies, all_folds_val_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print("-" * 50)

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        X_tr, y_tr = torch.FloatTensor(X_tr).to(device), torch.LongTensor(y_tr.numpy()).to(device)
        X_val, y_val = torch.FloatTensor(X_val).to(device), torch.LongTensor(y_val.numpy()).to(device)

        class_weights = compute_class_weight(
            class_weight='balanced', classes=np.unique(y_tr.cpu().numpy()), y=y_tr.cpu().numpy()
        )

        model = CNN_LSTM(sequence_length=20, hidden_size=config['hidden_size'], dropout_rate=config['dropout_rate']).to(
            device)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config['batch_size'])

        best_val_f1, best_model_state = 0, None
        train_losses, val_losses, val_accuracies, val_f1s = [], [], [], []

        for epoch in range(config['num_epochs']):
            model.train()
            total_train_loss = sum(
                criterion(model(seq), labels).backward() or optimizer.step() or criterion(model(seq), labels).item() for
                seq, labels in train_loader) / len(train_loader)
            train_losses.append(total_train_loss)

            model.eval()
            val_preds, val_labels, total_val_loss = [], [], 0
            with torch.no_grad():
                for seq, labels in val_loader:
                    outputs = model(seq)
                    total_val_loss += criterion(outputs, labels).item()
                    val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            val_accuracies.append(val_accuracy)
            val_f1s.append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()

            print(
                f'Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {total_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        plot_learning_curves(train_losses, val_losses, val_accuracies, val_f1s, fold)
        all_folds_train_losses.append(train_losses)
        all_folds_val_losses.append(val_losses)
        all_folds_val_accuracies.append(val_accuracies)
        all_folds_val_f1s.append(val_f1s)

        model.load_state_dict(best_model_state)
        model.eval()
        final_preds, final_labels = [], []

        with torch.no_grad():
            for seq, labels in val_loader:
                outputs = model(seq)
                final_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

        fold_metrics.append({
            'accuracy': accuracy_score(final_labels, final_preds),
            'f1': f1_score(final_labels, final_preds, average='weighted'),
            'classification_report': classification_report(final_labels, final_preds, digits=4)
        })

    plot_average_curves(all_folds_train_losses, all_folds_val_losses, all_folds_val_accuracies, all_folds_val_f1s,
                        config)

    print("\nEvaluating on test set:")
    model.load_state_dict(best_model_state)
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config['batch_size'])

    test_preds, test_labels = [], []
    with torch.no_grad():
        for seq, labels in test_loader:
            outputs = model(seq)
            test_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print("\nTest Set Results:")
    print(f"Test Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Test F1 Score: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4))

    return fold_metrics


if __name__ == "__main__":
    # เรียกใช้ฟังก์ชัน prepare_train_test_data แทน prepare_cross_validation_data
    X_train, X_test, y_train, y_test = prepare_train_test_data()

    print("Data shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ทำการ train และ evaluate โมเดล
    fold_metrics = train_and_evaluate_cv(X_train, y_train, X_test, y_test, n_splits=5)
