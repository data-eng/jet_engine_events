import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


def load_dataloader(csv_path, batch_size, input_cols=slice(2, 10), label_col=10):
    """
    Loads gzipped time series CSV, groups by TSid, and returns a DataLoader.

    Parameters:
        csv_path (str): Path to the .csv.gz file.
        batch_size (int): Batch size for the DataLoader.
        input_cols (slice): Slice for the input columns.
        label_col (int): Index of the label column (0-based).

    Returns:
        tuple: (DataLoader, sequences_tensor, labels_tensor)
    """
    df = pd.read_csv(csv_path, compression="gzip")
    df = df.sort_values(by=["TSid", "Timestamp"]).reset_index(drop=True)

    sequences, labels = [], []
    for tsid in df["TSid"].unique():
        ts_data = df[df["TSid"] == tsid]
        input_seq = ts_data.iloc[:, input_cols].values.astype(np.float32)
        label_seq = (ts_data.iloc[:, label_col] - 1).values.astype(np.int64)
        sequences.append(torch.tensor(input_seq))
        labels.append(torch.tensor(label_seq))

    sequences_tensor = torch.stack(sequences)
    labels_tensor = torch.stack(labels)

    dataset = TensorDataset(sequences_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, sequences_tensor, labels_tensor


def save_all(dataloader, sequences, labels, out_dir):
    """
    Saves the TensorDataset and raw tensors to disk.

    Parameters:
        dataloader (DataLoader): The DataLoader whose dataset will be saved.
        sequences (Tensor): Raw input sequences tensor.
        labels (Tensor): Raw label sequences tensor.
        out_dir (str): Directory to save the files.
    """
    os.makedirs(out_dir, exist_ok=True)

    torch.save(
        dataloader.dataset, os.path.join(out_dir, "jet_engines_events_dataset.pt")
    )
    torch.save(sequences, os.path.join(out_dir, "jet_engines_events_seqs.pt"))
    torch.save(labels, os.path.join(out_dir, "jet_engines_events_labels.pt"))

    print(
        f"Saved TensorDataset to: {os.path.join(out_dir, 'jet_engines_events_dataset.pt')}"
    )
    print(
        f"Saved raw sequences to: {os.path.join(out_dir, 'jet_engines_events_seqs.pt')}"
    )
    print(
        f"Saved raw labels to: {os.path.join(out_dir, 'jet_engines_events_labels.pt')}"
    )


# =============== OPTIONAL STANDALONE EXECUTION ===============
if __name__ == "__main__":
    # Adjusted for your folder structure
    CSV_PATH = os.path.join("..", "data", "jet_engines_events_dataset.csv.gz")
    OUTPUT_DIR = os.path.join("..", "data", "processed")
    BATCH_SIZE = 32

    print(f"Creating DataLoader from {CSV_PATH}...")
    dataloader, seqs, labels = load_dataloader(CSV_PATH, batch_size=BATCH_SIZE)
    print(f"DataLoader created with {len(dataloader)} batches of size {BATCH_SIZE}")

    save_all(dataloader, seqs, labels, OUTPUT_DIR)
