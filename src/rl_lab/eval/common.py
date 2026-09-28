# rl_lab/eval/common.py

import torch


def find_common_length(
    sequences: list[list[int]],
    direction: str = "left",
) -> int:
    min_len = min(map(len, sequences))

    indices = (
        range(min_len)
        if direction == "left"
        else range(-1, -min_len - 1, -1)
    )

    for i, idx in enumerate(indices):
        token = sequences[0][idx]

        if any(sequence[idx] != token for sequence in sequences[1:]):
            return i

    return min_len


def pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
) -> torch.Tensor:
    max_len = max(map(len, sequences))

    output = torch.full(
        (len(sequences), max_len),
        pad_token_id,
        dtype=torch.long,
    )

    for i, sequence in enumerate(sequences):
        output[i, :len(sequence)] = torch.tensor(sequence)

    return output