import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_utils import prepare_train_test_data
import optuna
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import shutil
from typing import List, Dict, Tuple
import time

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TrainingVisualizer:
    def __init__(self):
        self.save_dir = r"D:\Model1101\.venv\Scripts\Result"

        # สร้างโฟลเดอร์ถ้ายังไม่มี
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # สร้างโฟลเดอร์ย่อยสำหรับแต่ละ trial
        self.trial_dir = os.path.join(self.save_dir, 'trial_plots')
        if not os.path.exists(self.trial_dir):
            os.makedirs(self.trial_dir)

        print(f"Results will be saved in: {self.save_dir}")
        self.metrics = {
            'train_acc': [],
            'val_acc': [],
            'epochs': [],
            'train_losses': [],
            'val_losses': []
        }
        self.all_trial_results = []

    def update_metrics(self, epoch: int, train_acc: float, val_acc: float, train_loss: float, val_loss: float):
        """Update metrics after each epoch"""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)

    def plot_training_progress(self, trial_num: int, save_to_trial_folder: bool = True):
        """Plot training and validation metrics over epochs"""
        # สร้าง figure ขนาด 15x15 นิ้ว และแบ่งเป็น 2 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

        # Plot 1: Accuracy
        ax1.plot(self.metrics['epochs'], self.metrics['train_acc'],
                 label='Training Accuracy', marker='o', markersize=4, linewidth=2)
        ax1.plot(self.metrics['epochs'], self.metrics['val_acc'],
                 label='Validation Accuracy', marker='s', markersize=4, linewidth=2)
        ax1.set_title(f'Training and Validation Accuracy - Trial {trial_num}', fontsize=14, pad=20)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Plot 2: Loss
        ax2.plot(self.metrics['epochs'], self.metrics['train_losses'],
                 label='Training Loss', color='red', marker='o', markersize=4, linewidth=2)
        ax2.plot(self.metrics['epochs'], self.metrics['val_losses'],
                 label='Validation Loss', color='blue', marker='s', markersize=4, linewidth=2)
        ax2.set_title(f'Training and Validation Loss - Trial {trial_num}', fontsize=14, pad=20)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', which='major', labelsize=10)

        # ปรับระยะห่างระหว่าง subplot
        plt.tight_layout(pad=4.0)

        # บันทึกกราฟ
        if save_to_trial_folder:
            save_path = os.path.join(self.trial_dir, f'training_progress_trial_{trial_num}.png')
        else:
            save_path = os.path.join(self.save_dir, f'training_progress_trial_{trial_num}.png')

        # บันทึกด้วย DPI ที่สูงขึ้นเพื่อความคมชัด
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, trial_num: int,
                              save_to_trial_folder: bool = True):
        """Plot confusion matrix"""
        class_names = ['Non-request', 'Both hands', 'Left hand', 'Right hand']
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Confusion Matrix - Trial {trial_num}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        if save_to_trial_folder:
            save_path = os.path.join(self.trial_dir, f'confusion_matrix_trial_{trial_num}.png')
        else:
            save_path = os.path.join(self.save_dir, f'confusion_matrix_trial_{trial_num}.png')

        plt.savefig(save_path)
        plt.close()

    def save_trial_result(self, trial_result: Dict):
        """Save trial result for final comparison"""
        self.all_trial_results.append(trial_result)

    def reset_metrics(self):
        """Reset metrics for new trial"""
        self.metrics = {
            'train_acc': [],
            'val_acc': [],
            'epochs': [],
            'train_losses': [],
            'val_losses': []
        }

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


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def compute_accuracy(model, data_loader, device, criterion):
    """Calculate accuracy and loss for the given model and data loader"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            clear_gpu_cache()

    accuracy = (correct / total) * 100
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss, all_preds, all_labels


def train_model(trial, X_train, y_train, X_test, y_test, device, class_weights=None):
    """Train model with hyperparameters suggested by Optuna"""

    # Hyperparameters
    hidden_size = trial.suggest_int("lstm_hidden_units", 50, 150)
    num_lstm_layers = trial.suggest_int("lstm_hidden_layers", 1, 5)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ["ReLU", "Sigmoid", "Tanh"])
    num_epochs = trial.suggest_int("epochs", 50, 200)

    # Print trial parameters
    print(f"\nTrial #{trial.number} Parameters:")
    print("=" * 30)
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    print("=" * 30)

    # Initialize model and training components
    model = CNN_LSTM(
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize visualizer and tracking variables
    visualizer = TrainingVisualizer()
    best_val_acc = 0
    patience_counter = 0
    best_predictions = None
    best_true_labels = None

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0

            # Training loop
            for sequences, labels in train_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            # Calculate average loss for the epoch
            avg_loss = total_loss / batch_count

            # Compute accuracies
            train_acc, train_loss, train_preds, train_labels = compute_accuracy(model, train_loader, device, criterion)
            val_acc, val_loss, val_preds, val_labels = compute_accuracy(model, test_loader, device, criterion)

            # Calculate F1 scores
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            # Clear GPU cache after epoch completion
            clear_gpu_cache()

            # Update visualizer
            visualizer.update_metrics(epoch + 1, train_acc, val_acc, train_loss, val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_predictions = val_preds
                best_true_labels = val_labels
                torch.save(model.state_dict(),
                           os.path.join(visualizer.save_dir, f'best_model_trial_{trial.number}.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Accuracy: {val_acc:.2f}%")
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            print("-" * 30)

        # Generate final plots
        visualizer.plot_training_progress(trial.number)
        if best_predictions is not None and best_true_labels is not None:
            visualizer.plot_confusion_matrix(best_true_labels, best_predictions, trial.number)

        # Final trial summary
        print(f"\nTrial #{trial.number} Summary:")
        print("=" * 50)
        print("Accuracy Metrics:")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print("\nModel Parameters:")
        print(f"Batch Size: {batch_size}")
        print(f"Number of Epochs: {num_epochs}")
        print(f"Learning Rate: {lr:.6f}")
        print(f"Weight Decay: {weight_decay:.6f}")
        print(f"Hidden Size: {hidden_size}")
        print(f"LSTM Layers: {num_lstm_layers}")
        print(f"Dropout Rate: {dropout_rate}")
        print(f"Activation Function: {activation}")
        print("=" * 50)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(val_labels, val_preds,
                                    target_names=['Non-request', 'Both hands', 'Left hand', 'Right hand']))

        return best_val_acc, {
            'trial_num': trial.number,
            'best_val_acc': best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1
        }
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None, {
            'trial_num': trial.number,
            'error': str(e)
        }
    finally:
        print(f"Trial #{trial.number} completed.")

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

    val_acc, trial_result = train_model(
        trial, X_train_normalized, y_train,
        X_test_normalized, y_test, device, class_weights
    )
    return val_acc


def main():
    print("\nStarting Optuna optimization...")

    trial_results = []
    study = optuna.create_study(direction="maximize")
    start_time = time.time()

    try:
        for _ in range(100):  # 10 trials
            trial = study.ask()
            best_val_acc = objective(trial)
            study.tell(trial, best_val_acc)

            if isinstance(best_val_acc, tuple):
                best_val_acc, trial_result = best_val_acc
                if trial_result:
                    trial_results.append(trial_result)

        # Plot trial comparison
        visualizer = TrainingVisualizer()
        visualizer.plot_trial_comparison(trial_results)

        # Print best trial results
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value:.2f}%")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted!")

    finally:
        # Calculate study statistics
        total_time = time.time() - start_time
        accuracies = [trial.value for trial in study.trials if trial.value is not None]
        trials_data = [(trial.number, trial.value) for trial in study.trials if trial.value is not None]

        best_trial_idx = np.argmax(accuracies)
        worst_trial_idx = np.argmin(accuracies)

        # Print comprehensive summary report
        print("\n" + "=" * 50)
        print("🎯 STUDY SUMMARY REPORT")
        print("=" * 50)

        print("\n📊 Overall Statistics:")
        print(f"Total number of trials completed: {len(study.trials)}")
        print(f"Total time taken: {total_time / 3600:.2f} hours")
        print(f"Average time per trial: {total_time / len(study.trials) / 60:.2f} minutes")

        print("\n📈 Accuracy Statistics:")
        print(f"Best accuracy: {max(accuracies):.2f}% (Trial #{trials_data[best_trial_idx][0]})")
        print(f"Worst accuracy: {min(accuracies):.2f}% (Trial #{trials_data[worst_trial_idx][0]})")
        print(f"Mean accuracy across all trials: {np.mean(accuracies):.2f}%")
        print(f"Median accuracy: {np.median(accuracies):.2f}%")
        print(f"Standard deviation: {np.std(accuracies):.2f}%")

        print("\n🏆 Best Trial Parameters (Trial #{study.best_trial.number}):")
        for param_name, param_value in study.best_trial.params.items():
            if isinstance(param_value, float):
                print(f" {param_name}: {param_value:.6f}")
            else:
                print(f" {param_name}: {param_value}")

        print("\n📉 Trial Distribution:")
        ranges = [
            (0, 60),
            (60, 70),
            (70, 80),
            (80, 90),
            (90, 95),
            (95, 100)
        ]
        print("Accuracy distribution:")
        accuracies_array = np.array(accuracies)
        for start, end in ranges:
            mask = (accuracies_array >= start) & (accuracies_array < end)
            count = np.sum(mask)
            if count > 0:
                trials_in_range = [trials_data[i][0] for i, m in enumerate(mask) if m]
                percentage = (count / len(accuracies_array)) * 100
                print(f" {start}-{end}%: {count} trials ({percentage:.1f}%)")
                print(f" Trials: {', '.join(map(str, trials_in_range))}")

        print("\n💾 Results Location:")
        print(f"Detailed results and plots saved in: {visualizer.save_dir}")
        print(f"Individual trial plots saved in: {visualizer.trial_dir}")

        print("\n" + "=" * 50)
        print("Study completed successfully!")
        print("=" * 50)

        # Save study results
        study_results = {
            'best_value': study.best_value,
            'best_params': study.best_trial.params,
            'n_trials': len(study.trials)
        }


if __name__ == "__main__":
    main()
