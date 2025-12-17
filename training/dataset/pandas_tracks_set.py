import json
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class TrackTableDataset(Dataset):
    def __init__(
        self,
        csv_dir,
        mapping_json_path='track_generator/data/category_mappings.json',
        x_cols=("x", "y"),
        y_col="tag"
    ):
        self.csv_files = sorted(Path(csv_dir).glob("*.csv"))

        with open(mapping_json_path, "r") as f:
            self.mappings = json.load(f)

        self.categorical_columns = ['tag', 'aug_tag', 'is_valid']
        self.x_cols = list(x_cols)
        self.y_col = y_col

    def _apply_mappings(self, df):
        for col in self.categorical_columns:
            df[col] = (
                df[col]
                .astype(str)
                .map(self.mappings[col])
            )
        return df

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.csv_files[idx])
        df = self._apply_mappings(df)

        # Inputs: (T, 2)
        X = torch.tensor(
            df[self.x_cols].values,
            dtype=torch.float32
        )       

        # Targets: (T,)
        Y = torch.tensor(
            df[self.y_col].values,
            dtype=torch.long
        )

        return X, Y

def collate_tracks(batch):
    """
    batch: list of (X_i, Y_i)
    X_i: (T_i, 2)
    Y_i: (T_i,)
    """
    Xs, Ys = zip(*batch)

    lengths = torch.tensor([x.size(0) for x in Xs], dtype=torch.long)

    X_padded = pad_sequence(
        Xs,
        batch_first=True,
        padding_value=0.0
    )  # (B, N_max, 2)

    Y_padded = pad_sequence(
        Ys,
        batch_first=True,
        padding_value=-100
    )  # (B, N_max)

    # mask: True for valid points
    B, N_max, _ = X_padded.shape
    mask = torch.arange(N_max)[None, :] < lengths[:, None]
    mask = mask.bool()  # (B, N_max)

    return X_padded, Y_padded, mask, lengths

def get_dataloader():
    loader = DataLoader(
        TrackTableDataset(
            csv_dir="../track_generator/data/augmented_tracks",
            mapping_json_path="../track_generator/data/category_mappings.json",
            x_cols=("x", "y"),
            y_col="tag"
        ),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_tracks
    )
    return loader