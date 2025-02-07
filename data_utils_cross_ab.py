import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def get_class_name(feature_value: int) -> str:
    """แปลงค่า feature เป็นชื่อคลาส"""
    class_names = {
        0: "Non-request",
        1: "Both hands",
        2: "Left hand",
        3: "Right hand"
    }
    return class_names.get(feature_value, "Unknown")


def load_and_separate_data():
    """โหลดและแยกข้อมูลปกติและไม่ปกติ"""
    BASE_DATA_PATH = r"C:\CNNLSTM\scripts\separated_data"

    normal_data = []
    normal_labels = []
    abnormal_data = []
    abnormal_labels = []

    print("Loading and separating normal and abnormal data...")

    # Map filenames to class indices
    filename_to_class = {
        'Non_request': 0,
        'Both_hand': 1,
        'Left_hand': 2,
        'Right_hand': 3
    }

    for filename in os.listdir(BASE_DATA_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(BASE_DATA_PATH, filename)

            # Determine if file is normal or abnormal and get class
            is_abnormal = filename.startswith('abnormal_')
            for class_name, class_idx in filename_to_class.items():
                if class_name in filename:
                    # Load data
                    data = pd.read_csv(file_path)
                    features = data.iloc[:, 1:-2].values
                    labels = np.full(len(features), class_idx)

                    if is_abnormal:
                        abnormal_data.append(features)
                        abnormal_labels.append(labels)
                    else:
                        normal_data.append(features)
                        normal_labels.append(labels)
                    break

    # Combine all data
    normal_data = np.vstack(normal_data)
    normal_labels = np.concatenate(normal_labels)
    abnormal_data = np.vstack(abnormal_data)
    abnormal_labels = np.concatenate(abnormal_labels)

    # Print initial distribution
    print("\nInitial data distribution:")
    print("\nNormal data:")
    unique_labels, counts = np.unique(normal_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples")
    print(f"Total normal samples: {len(normal_labels)}")

    print("\nAbnormal data:")
    unique_labels, counts = np.unique(abnormal_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples")
    print(f"Total abnormal samples: {len(abnormal_labels)}")

    return normal_data, normal_labels, abnormal_data, abnormal_labels


def balance_data(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Under-sampling đểทำให้แต่ละคลาสมีจำนวนเท่ากัน"""
    unique_labels = np.unique(labels)
    min_size = min(np.sum(labels == label) for label in unique_labels)

    balanced_data = []
    balanced_labels = []

    for label in unique_labels:
        idx = labels == label
        label_data = data[idx]
        if len(label_data) > min_size:
            indices = np.random.choice(len(label_data), size=min_size, replace=False)
            balanced_data.append(label_data[indices])
            balanced_labels.extend([label] * min_size)
        else:
            balanced_data.append(label_data)
            balanced_labels.extend([label] * len(label_data))

    return np.vstack(balanced_data), np.array(balanced_labels)


def prepare_windows(data: np.ndarray, labels: np.ndarray, window_size: int = 20, stride: int = 1) -> Tuple[
    np.ndarray, np.ndarray]:
    """สร้าง sliding windows จากข้อมูล"""
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        label = labels[i + window_size - 1]
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)


def prepare_train_test_data():
    """เตรียมข้อมูลสำหรับ train และ test"""
    # 1. โหลดและแยกข้อมูล
    normal_data, normal_labels, abnormal_data, abnormal_labels = load_and_separate_data()

    # 2. Balance ข้อมูลแต่ละประเภท
    print("\nBalancing data...")
    normal_data_balanced, normal_labels_balanced = balance_data(normal_data, normal_labels)
    abnormal_data_balanced, abnormal_labels_balanced = balance_data(abnormal_data, abnormal_labels)

    # แสดงผลหลัง balance
    print("\nBalanced data distribution:")
    print("\nNormal data (balanced):")
    unique_labels, counts = np.unique(normal_labels_balanced, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples")
    print(f"Total normal samples after balancing: {len(normal_labels_balanced)}")

    print("\nAbnormal data (balanced):")
    unique_labels, counts = np.unique(abnormal_labels_balanced, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples")
    print(f"Total abnormal samples after balancing: {len(abnormal_labels_balanced)}")

    # 3. สร้าง windows
    X_normal, y_normal = prepare_windows(normal_data_balanced, normal_labels_balanced)
    X_abnormal, y_abnormal = prepare_windows(abnormal_data_balanced, abnormal_labels_balanced)

    # 4. แบ่ง train/test โดยรักษาสัดส่วน 90:10 (normal:abnormal)
    def create_stratified_split(X_normal, y_normal, X_abnormal, y_abnormal, test_size=0.2):
        # คำนวณจำนวนตัวอย่างที่ต้องการสำหรับแต่ละส่วน
        n_normal = int(len(X_normal) * 0.9)  # 90% ของข้อมูลปกติ
        n_abnormal = int(n_normal * (1 / 9))  # 10% ของข้อมูลทั้งหมด

        # สุ่มเลือกข้อมูล
        normal_indices = np.random.choice(len(X_normal), size=n_normal, replace=False)
        abnormal_indices = np.random.choice(len(X_abnormal), size=n_abnormal, replace=False)

        # รวมข้อมูล
        X = np.concatenate([X_normal[normal_indices], X_abnormal[abnormal_indices]])
        y = np.concatenate([y_normal[normal_indices], y_abnormal[abnormal_indices]])

        # แบ่ง train/test
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = create_stratified_split(X_normal, y_normal, X_abnormal, y_abnormal)

    # แปลงเป็น tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # แสดงการกระจายข้อมูลสุดท้าย
    print("\nFinal data distribution:")
    print("\nTraining set (80%):")
    unique_labels, counts = np.unique(y_train.numpy(), return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_train) * 100:.2f}%)")

    print("\nTest set (20%):")
    unique_labels, counts = np.unique(y_test.numpy(), return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_test) * 100:.2f}%)")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_train_test_data()
    print("\nPreprocessing completed successfully!")
    print("\n--------------------------------------------------------")
