import lightning as pl
import torch
from jaxtyping import Float
from torch import Tensor, nn, optim
from torchrl.modules import MLP


class ReadProcessEmbedder(pl.LightningModule):
    """
    A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391).
    """

    def __init__(
        self,
        feature_size: int,  # size of each element in input sequence. Usually 1 (scalar sequence)
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: list[int] = [32, 32],
        writing_block_cells: list[int] = [32, 32],
        memory_size: int = 16,  # each element in the sequence gets converted into a memory with this size
        processing_steps: int = 5,  # RNN processing steps. Paper shows that 5-10 is good.
        criterion: nn.Module = nn.MSELoss(),  # feel free to change this to any other loss function
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_size = feature_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps
        self.lr = lr

        self.reading_block = MLP(
            in_features=feature_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
        )
        # An RNN without input! The input will always be zero and the sequence length will be 1
        self.rnn = nn.GRU(input_size=1, hidden_size=2 * memory_size, batch_first=True)
        # The RNN output has to be projected to the memory size
        self.proj = nn.Linear(2 * memory_size, memory_size)

        # After the processing steps, the final memory is passed through a MLP to produce the output
        # The paper uses a pointer network but we only want a single feature vector
        self.write_block = MLP(
            in_features=2 * memory_size,
            out_features=output_size,
            num_cells=writing_block_cells,
        )

        self.criterion = criterion

    def forward(
        self, input_set: Float[Tensor, "N L {self.feature_size}"]
    ) -> Float[Tensor, "N {self.output_size}"]:
        batch_size = input_set.shape[0]
        seq_length = input_set.shape[1]  # noqa: F841, variable is used in array type hints
        memories: Float[Tensor, "{batch_size} {seq_length} {self.memory_size}"] = (
            self.reading_block(input_set)
        )
        h = torch.zeros(
            1, batch_size, 2 * self.memory_size, device=self.device
        )  # initial hidden state
        for _ in range(self.processing_steps):
            q: Float[Tensor, "{batch_size} 1 {2*self.memory_size}"] = self.rnn(
                torch.zeros(batch_size, 1, 1, device=self.device), h
            )[0]
            q: Float[Tensor, "{batch_size} {self.memory_size}"] = self.proj(
                q.squeeze(1)
            )
            # Take the dotproduct of the query with each memory
            e: Float[Tensor, "{batch_size} {seq_length}"] = torch.bmm(
                memories, q.unsqueeze(-1)
            ).squeeze(-1)
            # Softmax over sequence dimension
            a: Float[Tensor, "{batch_size} {seq_length}"] = torch.softmax(e, dim=-1)
            # Reshape a to (batch_size, 1, seq_length) for block matrix multiplication
            a: Float[Tensor, "{batch_size} 1 {seq_length}"] = a.unsqueeze(1)
            # Linear combination of memories. (1,seq_length) x (seq_length,memory_size) -> (1,memory_size)
            r: Float[Tensor, "{batch_size} 1 {self.memory_size}"] = torch.bmm(
                a, memories
            )
            r: Float[Tensor, "{batch_size} {self.memory_size}"] = r.squeeze(1)
            # Concatenate r with q to produce next long-term memory
            h: Float[Tensor, "{batch_size} {2*self.memory_size}"] = torch.cat(
                [q, r], dim=-1
            )
            # Conform to torch standard for RNNs
            h: Float[Tensor, "1 {batch_size} {2*self.memory_size}"] = h.unsqueeze(0)
        # Final h is the output of the process block
        h: Float[Tensor, "{batch_size} {2*self.memory_size}"] = h.squeeze(0)
        output: Float[Tensor, "{batch_size} {self.output_size}"] = self.write_block(h)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
