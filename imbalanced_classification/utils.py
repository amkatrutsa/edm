import os
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def get_loss(model, criterion, dataloader, device):
    total_loss = 0.
    with torch.no_grad():
        for data in dataloader:
            # get the inputs
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            current_loss = criterion(outputs, labels.to(device))

            # print statistics
            total_loss += current_loss.item()
        total_loss = total_loss / len(dataloader)
    return total_loss


def get_loss_per_class(model, criterion, dataloader, device, num_classes):
    total_loss_per_class = torch.zeros(num_classes)
    total_samples_per_class = torch.zeros(num_classes)
    with torch.no_grad():
        for data in dataloader:
            # get the inputs
            inputs, labels = data

            unique_labels = torch.unique(labels)
            for l in unique_labels:
                outputs = model(inputs[labels == l, :].to(device))
                # print(outputs, l)
                current_loss = criterion(outputs, labels[labels == l].to(device))
                total_loss_per_class[l] += current_loss
                total_samples_per_class[l] += labels[labels == l].shape[0]
    return total_loss_per_class / total_samples_per_class


def get_accuracy_per_class(model, dataloader, device, num_classes):
    correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    with torch.no_grad():
        for data in dataloader:
            # get the inputs
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            _, predicted_labels = torch.max(outputs, 1)
            c = (predicted_labels == labels).squeeze()
            for i in range(labels.shape[-1]):
                label = labels[i]
                correct[label] += c[i].item()
                class_total[label] += 1

            # print statistics

    return correct / class_total

def process_data():
    if os.path.exists("./data.csv"):
        raw_df = pd.read_csv("./data.csv")
    else:
        raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
        raw_df.to_csv("./data.csv")

    neg, pos = np.bincount(raw_df['Class'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    cleaned_df = raw_df.copy()

    # You don't want the `Time` column.
    cleaned_df.pop('Time')

    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount') + eps)

    train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42)
    # train_df, val_df = train_test_split(train_df, test_size=0.2)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Class'))
    bool_train_labels = train_labels != 0
    # val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))

    train_features = np.array(train_df)
    # val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    # val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    # val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    num_ones_train = np.sum(train_labels == 1)
    num_zero_train = np.sum(train_labels == 0)
    num_ones_test = np.sum(test_labels == 1)
    num_zero_test = np.sum(test_labels == 0)

    # np.savez("./data", train_features=train_features, train_labels=train_labels,
    #          test_features=test_features, test_labels=test_labels)

    print('Training labels shape:', train_labels.shape)
    # print('Validation labels shape:', val_labels.shape)
    print('Test labels shape:', test_labels.shape)

    print("Positive class in train = {}".format(num_ones_train / train_labels.shape[-1]))
    print("Positive class in test = {}".format(num_ones_test / test_labels.shape[-1]))

    print('Training features shape:', train_features.shape)
    # print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)

    return train_features, train_labels, test_features, test_labels

def exact_linesearch4quadratic(alpha, t, M):
    cross_inner_prod = M[t, :] @ alpha
    if cross_inner_prod >= M[t, t]:
        gamma = 1
    elif cross_inner_prod >= alpha @ M @ alpha:
        gamma = 0
    else:
        diff = alpha.clone()
        diff[t] -= 1
        gamma = (diff @ M @ alpha) / (diff @ M @ diff)
    return gamma


def frank_wolfe4quadratic(gradients, max_iter, tol):
    num_classes = gradients.shape[1]
    alpha = torch.ones(num_classes).to(gradients) / num_classes
    M = gradients.t() @ gradients
    for i in range(max_iter):
        alpha_M = torch.sum(M * alpha, dim=1)
        t = torch.argmin(alpha_M)
        gamma = exact_linesearch4quadratic(alpha, t, M)
        alpha = (1 - gamma) * alpha
        alpha[t] += gamma
        if gamma < tol:
            break
    return alpha
