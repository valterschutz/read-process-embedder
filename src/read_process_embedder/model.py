from jaxtyping import Shaped
from torch import Tensor, nn
from torchrl.modules import MLP


class ReadProcessEmbedder(nn.Module):
    def __init__(
        self,
        input_size: int,
        memory_size: int = 16,  # shim
        reading_block_cells: list[int] = [32, 32],  # shim
        lstm_size: int = 16,  # shim
        processing_steps: int = 5,  # vinyals
    ):
        self.reading_block = MLP(
            in_features=input_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
        )
        self.process_block = nn.LSTM(
            input_size=0,
            hidden_size=lstm_size,
        )

    def forward(self, input_set: Shaped[Tensor, "B N"]) -> Shaped[Tensor, "B N"]:
        memories: Shaped[Tensor, "B N {self.memory_size}"] = self.reading_block(
            input_set.unsqueeze(-1)
        )
        input_set = self.process_block(input_set)
        return input_set
