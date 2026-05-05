import os
import torch
import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter, defaultdict
from pathlib import Path


def preprocess_data(file_path, device='cpu', augmented=False, noise=0.1):
    """
    Preprocess the data loaded using the provided file path.

    Args:
        - file_path (str): The file path to the folder where the data is stored.

    Returns:
        - time (PyTorch tensor): A list of the time sequences of the data.
        - train_list (PyTorch tensor): A list of the sequences of the data.
        - state_list (PyTorch tensor): A list of the state sequences of the data.
        - train_names (list): A list of the names of the data variables (except time and states).
        - state_names (list): A list of the names of the states.
    """
    # Load data from file
    raw_data = load_data(file_path)

    print(file_path)

    if 'MHEALTH' in file_path:
        # Load the data into a pandas DataFrame
        time_list = []
        data_list = []
        state_list = []
        data_names_list = []
        state_names_list = []


        for dataframe in raw_data:
            dataframe  = dataframe.iloc[::5, :]
            # Create the time as a sequence of integers (e.g., 0, 1, 2, ..., n)
            time = torch.tensor(dataframe.index, dtype=torch.float, device=device)

            # Separate the data and the state (last column is the state)
            data_values = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float, device=device)
            state_values = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float, device=device)

            # Filter out rows where state is 0
            valid_indices = state_values != 0  # Create a mask for rows where state is not 0

            # Generate default names for the data and state if names are empty
            data_names_list.append([])
            state_names_list.append([])

            # Apply the mask to filter the data
            time = time[valid_indices]
            data_values = data_values[valid_indices]
            state_values = state_values[valid_indices]

            # Normalize the data (optional, can skip if not needed)
            mean = torch.mean(data_values, dim=0)
            std = torch.std(data_values, dim=0)
            std[std == 0] = 1e-8  # Avoid division by zero
            data_values = (data_values - mean) / std

            time_list.append(time)
            data_list.append(data_values)
            state_list.append(state_values)

        print('Data preprocessing done!')

        return time_list, data_list, state_list, data_names_list, state_names_list

    elif 'HAR' in file_path:
        # Load the data into a pandas DataFrame
        time_list = []
        data_list = []
        state_list = []
        data_names_list = []
        state_names_list = []

        for dataframe in raw_data:
            # Create the time as a sequence of integers (e.g., 0, 1, 2, ..., n)
            time = torch.tensor(dataframe.index, dtype=torch.float, device=device)

            # Separate the data and the state (last column is the state)
            data_values = torch.tensor(dataframe.iloc[:, :-2].values, dtype=torch.float, device=device)
            state_values = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float, device=device)

            # Filter out rows where state is 0
            valid_indices = state_values != 0  # Create a mask for rows where state is not 0

            # Generate default names for the data and state if names are empty
            data_names_list.append([])
            state_names_list.append([])

            # Apply the mask to filter the data
            time = time[valid_indices]
            data_values = data_values[valid_indices]
            state_values = state_values[valid_indices]

            # Normalize the data (optional, can skip if not needed)
            mean = torch.mean(data_values, dim=0)
            std = torch.std(data_values, dim=0)
            std[std == 0] = 1e-8  # Avoid division by zero
            data_values = (data_values - mean) / std

            time_list.append(time)
            data_list.append(data_values)
            state_list.append(state_values)

        print('Data preprocessing done!')
        return time_list, data_list, state_list, None, None

    elif 'PAMAP2' in file_path:
        # Load the data into a pandas DataFrame
        time_list = []
        data_list = []
        state_list = []
        data_names_list = []
        state_names_list = []

        for dataframe in raw_data:
            # Create the time as a sequence of integers (e.g., 0, 1, 2, ..., n)
            time = torch.tensor(dataframe.index, dtype=torch.float, device=device)

            # Separate the data and the state (last column is the state)
            data_values = torch.tensor(dataframe.iloc[:, 2:].values, dtype=torch.float, device=device)
            state_values = torch.tensor(dataframe.iloc[:, 1].values, dtype=torch.float, device=device)

            # Filter out rows where state is 0
            valid_indices = state_values != 0  # Create a mask for rows where state is not 0

            # Generate default names for the data and state if names are empty
            data_names_list.append([])
            state_names_list.append([])

            # Apply the mask to filter the data
            time = time[valid_indices]
            data_values = data_values[valid_indices]
            state_values = state_values[valid_indices]

            # Normalize the data (optional, can skip if not needed)
            mean = torch.mean(data_values, dim=0)
            std = torch.std(data_values, dim=0)
            std[std == 0] = 1e-8  # Avoid division by zero
            data_values = (data_values - mean) / std

            time_list.append(time)
            data_list.append(data_values)
            state_list.append(state_values)

        print('Data preprocessing done!')
        return time_list, data_list, state_list, None, None

    else:
        raise Exception('Missing data!')


def load_data(keyword):
    """
    Load data from CSV files in a specified directory based on the provided keyword.

    Args:
        - keyword (str): A keyword to identify the directory containing CSV files.

    Returns:
        - dataframes (list): A list of Pandas DataFrames loaded from CSV files.
    """
    # Determine the path based on the keyword
    home_path = Path.home()
    cwd_relative_path = Path.cwd().relative_to(home_path)
    data_dir = f'{cwd_relative_path}/Data/{keyword}'

    if 'Siemens' in keyword:
        path = os.path.join(home_path, data_dir, 'data.csv')
        dataframes = pd.read_csv(path).reset_index(drop=True)
    elif 'simu_tank' in keyword:
        path = os.path.join(home_path, data_dir, 'data.csv')
        dataframes = pd.read_csv(path).iloc[1000:].reset_index(drop=True)
    elif 'MHEALTH' in keyword:
        paths = Path(os.path.join(home_path, data_dir)).glob('data*.csv')
        dataframes = [pd.read_csv(dataset) for dataset in paths]
    elif 'HAR' in keyword:
        paths = Path(os.path.join(home_path, data_dir)).glob('data*.csv')
        dataframes = [pd.read_csv(dataset) for dataset in paths]
    elif 'PAMAP2' in keyword:
        paths = Path(os.path.join(home_path, data_dir)).glob('data*.csv')
        dataframes = [pd.read_csv(dataset) for dataset in paths]
    else:
        path = os.path.join(home_path, data_dir, 'data*.csv')
        dataframes = pd.read_csv(path)

    return dataframes


def augment_data(data, noise, device):
    noise = torch.randn(data.size(), device=device) * noise
    return data + noise


def compute_purity(cluster_assignments_list, class_assignments_list, device="cpu"):
    """Computes the purity between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Args:
        cluster_assignments (ndarray): List of cluster assignments for every point.
        class_assignments (ndarray): List of class assignments for every point.

    Returns:
        float: The purity value.
    """

    purity_list = []

    for cluster_assignments, class_assignments in zip(cluster_assignments_list, class_assignments_list):
        # Ensure tensors are on the same device
        #cluster_assignments = cluster_assignments.to(device)
        #class_assignments = class_assignments.to(device)


        # Get the unique clusters and classes
        unique_clusters = torch.unique(cluster_assignments)
        unique_classes = torch.unique(class_assignments)

        # Create a count matrix on the GPU
        count_matrix = torch.zeros(len(unique_clusters), len(unique_classes), device=device)

        # Populate the count matrix
        for cluster, cls in zip(cluster_assignments, class_assignments):
            cluster_idx = (unique_clusters == cluster).nonzero(as_tuple=True)[0].item()
            class_idx = (unique_classes == cls).nonzero(as_tuple=True)[0].item()
            count_matrix[cluster_idx, class_idx] += 1

        # Calculate the purity
        max_intersections = torch.max(count_matrix, dim=1).values
        purity = max_intersections.sum() / len(cluster_assignments)
        purity_list.append(purity.item())

    return purity_list



def split_data(time, data, states, ratio=0.8):

    split_index = int(len(time) * ratio)

    train_time = time[:split_index]
    valid_time = time[split_index:]

    train_data = data[:split_index]
    valid_data = data[split_index:]

    train_states = states[:split_index]
    valid_states = states[split_index:]

    return train_time, valid_time, train_data, valid_data, train_states, valid_states


def summarize_seed_results(seed_results: list[dict]) -> dict:
    """
    Compute mean and standard deviation across seed-level results.
    """
    metric_names = [
        "loss",
        "purity",
        "nmi",
        "avg_label_entropy",
        "mutual_information",
        "entropy_labels",
        "entropy_states",
        "classification_accuracy",
        "classification_macro_f1",
    ]

    summary = {}

    for metric_name in metric_names:
        values = [result[metric_name] for result in seed_results]

        summary[f"mean_{metric_name}"] = float(np.mean(values))
        summary[f"std_{metric_name}"] = float(np.std(values))

    summary["per_seed_results"] = seed_results

    return summary

def compute_state_label_metrics(latent_data_list, valid_states):
    """
    Compute metrics based on the joint distribution of discrete states S and labels Y.

    Args:
        latent_data_list: list of 1D tensors of state IDs (e.g., SOM node indices)
        valid_states:     list of 1D tensors of ground-truth labels (same shapes)

    Returns:
        metrics: dict with keys:
            - "avg_label_entropy": average H(Y | S)  (lower = more interpretable)
            - "mutual_information": I(Y; S)
            - "entropy_labels": H(Y)
            - "entropy_states": H(S)
        per_state_entropy: dict[state_id -> H(Y | S = state_id)]
    """
    # counts for S, Y, and (S,Y)
    state_counts = Counter()
    label_counts = Counter()
    joint_counts = defaultdict(Counter)  # joint_counts[s][y]

    for latent, labels in zip(latent_data_list, valid_states):
        latent_np = latent.detach().cpu().numpy().ravel()
        labels_np = labels.detach().cpu().numpy().ravel()
        for s, y in zip(latent_np, labels_np):
            s = int(s)
            y = int(y)
            state_counts[s] += 1
            label_counts[y] += 1
            joint_counts[s][y] += 1

    total = sum(state_counts.values())
    if total == 0:
        # no data; avoid division by zero
        metrics = {
            "avg_label_entropy": np.nan,
            "mutual_information": np.nan,
            "entropy_labels": np.nan,
            "entropy_states": np.nan,
        }
        return metrics, {}

    # Entropy of labels H(Y)
    H_Y = 0.0
    for y, c_y in label_counts.items():
        p_y = c_y / total
        H_Y -= p_y * np.log(p_y + 1e-12)

    # Entropy of states H(S)
    H_S = 0.0
    for s, c_s in state_counts.items():
        p_s = c_s / total
        H_S -= p_s * np.log(p_s + 1e-12)

    # Mutual information I(Y; S)
    I = 0.0
    for s, c_s in state_counts.items():
        p_s = c_s / total
        for y, c_sy in joint_counts[s].items():
            p_sy = c_sy / total
            p_y = label_counts[y] / total
            I += p_sy * np.log((p_sy + 1e-12) / ((p_s + 1e-12) * (p_y + 1e-12)))

    # Average conditional entropy H(Y | S)
    avg_H_Y_given_S = 0.0
    per_state_entropy = {}
    for s, c_s in state_counts.items():
        H = 0.0
        for y, c_sy in joint_counts[s].items():
            p_y_given_s = c_sy / c_s
            H -= p_y_given_s * np.log(p_y_given_s + 1e-12)
        per_state_entropy[s] = H

        p_s = c_s / total
        avg_H_Y_given_S += p_s * H

    metrics = {
        "avg_label_entropy": float(avg_H_Y_given_S),
        "mutual_information": float(I),
        "entropy_labels": float(H_Y),
        "entropy_states": float(H_S),
    }
    return metrics, per_state_entropy


def window_sequences(latent_data_list, label_seq_list, window_size, step_size):
    """
    Turn (possibly long) state/label sequences into many fixed-size windows.

    Args:
        latent_data_list: list of 1D tensors of state IDs (can be length 1)
        label_seq_list:   list of 1D tensors of labels (same shapes)
        window_size:      int, number of time steps per window
        step_size:        int, stride between window starts

    Returns:
        window_latents: list of 1D tensors, each length = window_size (state IDs)
        window_labels:  list of 1D tensors, each length = window_size (labels)
    """
    window_latents = []
    window_labels = []

    assert len(latent_data_list) == len(label_seq_list)

    for latent, labels in zip(latent_data_list, label_seq_list):
        assert latent.shape[0] == labels.shape[0]
        L = latent.shape[0]

        # Slide windows: [start, start+window_size)
        for start in range(0, L - window_size + 1, step_size):
            end = start + window_size
            window_latents.append(latent[start:end])
            window_labels.append(labels[start:end])

    return window_latents, window_labels


def build_bag_of_states_features(latent_data_list, label_seq_list, num_states=None, label_strategy="majority"):
    """
    Build bag-of-states features (histograms) and one label per window.

    Args:
        latent_data_list: list of 1D tensors of state IDs for each window
        label_seq_list:   list of 1D tensors of labels for each window
        num_states:       int or None. If None, inferred as max state id + 1.
        label_strategy:   "majority" (default) or "first" for window label

    Returns:
        X: np.ndarray of shape (num_windows, num_states)
        y: np.ndarray of shape (num_windows,)
    """
    assert len(latent_data_list) == len(label_seq_list)

    # Infer number of states if not provided
    if num_states is None:
        max_state = 0
        for latent in latent_data_list:
            latent_np = latent.detach().cpu().numpy().ravel()
            if latent_np.size > 0:
                max_state = max(max_state, int(latent_np.max()))
        num_states = max_state + 1

    features = []
    seq_labels = []

    for latent, labels in zip(latent_data_list, label_seq_list):
        latent_np = latent.detach().cpu().numpy().ravel()
        labels_np = labels.detach().cpu().numpy().ravel()

        # Bag-of-states: histogram over states in the window
        hist = np.bincount(latent_np, minlength=num_states).astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()  # fraction of time spent in each state
        features.append(hist)

        # One label per window
        if label_strategy == "majority":
            counts = Counter(labels_np.tolist())
            seq_label = counts.most_common(1)[0][0]
        elif label_strategy == "first":
            seq_label = int(labels_np[0])
        else:
            raise ValueError(f"Unknown label_strategy: {label_strategy}")
        seq_labels.append(seq_label)

    X = np.stack(features, axis=0)
    y = np.array(seq_labels, dtype=np.int64)
    return X, y

def evaluate_state_based_classifier_with_windows(
    train_latent_list,
    train_label_seq_list,
    test_latent_list,
    test_label_seq_list,
    window_size,
    step_size,
    num_states=None,
    label_strategy="majority",
):
    """
    Full pipeline:
      long sequences -> windows -> bag-of-states -> classifier.

    Args:
        train_latent_list, train_label_seq_list: lists of 1D tensors (train split)
        test_latent_list,  test_label_seq_list:  lists of 1D tensors (test split)
        window_size:  window length in time steps
        step_size:    stride between windows
        num_states:   number of states; if None, inferred from training data
        label_strategy: "majority" or "first" window label

    Returns:
        acc:       test accuracy
        macro_f1:  test macro-averaged F1
        clf:       trained sklearn classifier
    """
    # 1) Windowing
    train_win_latents, train_win_labels = window_sequences(
        train_latent_list, train_label_seq_list,
        window_size=window_size,
        step_size=step_size,
    )
    test_win_latents, test_win_labels = window_sequences(
        test_latent_list, test_label_seq_list,
        window_size=window_size,
        step_size=step_size,
    )

    # 2) Bag-of-states features
    X_train, y_train = build_bag_of_states_features(
        train_win_latents, train_win_labels,
        num_states=num_states,
        label_strategy=label_strategy,
    )
    X_test, y_test = build_bag_of_states_features(
        test_win_latents, test_win_labels,
        num_states=X_train.shape[1] if num_states is None else num_states,
        label_strategy=label_strategy,
    )

    # 3) Classifier
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return acc, macro_f1, clf
