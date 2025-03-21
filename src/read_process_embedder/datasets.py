import lightning as pl
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# Not currently used, but could be useful if model is extended to pointer output
class UnorderedNumbersDataset(Dataset):
    def __init__(self, seq_length=5, num_samples=1_000):
        self.X: Float[Tensor, "batch {seq_length}"] = torch.randint(
            0, 1, (num_samples, seq_length)
        )
        self.y: Float[Tensor, "batch {seq_length}"] = self.X.sort(dim=-1).values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MomentsDataset(Dataset):
    """
    Features consist of n numbers, distributed according to a normal distribution with some mean and std
    Output is the mean and std of the numbers
    """

    def __init__(
        self,
        means: list[float],
        stds: list[float],
        seq_length=5,
        num_samples=1_000,
    ):
        self.X: Float[Tensor, "{num_samples} {seq_length} 1"] = torch.zeros(
            num_samples, seq_length, 1
        )
        self.y: Float[Tensor, "{num_samples} {seq_length} 1"] = torch.zeros(
            num_samples, 2
        )
        # For each sample, pick a mean and std uniformly at random
        for i in range(num_samples):
            mean = means[torch.randint(0, len(means), (1,))]
            variance = stds[torch.randint(0, len(stds), (1,))]
            self.X[i] = torch.normal(mean, variance, (seq_length, 1))
            self.y[i] = torch.tensor([mean, variance])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataModuleFromDataset(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, train_ratio=0.8):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_size = int(self.train_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
