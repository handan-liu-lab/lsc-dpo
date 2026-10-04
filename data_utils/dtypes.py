from __future__ import annotations

from typing import Any

import torch


def normalize_torch_dtype(torch_dtype: Any) -> Any:
    if torch_dtype in ("auto", None):
        return torch_dtype
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        try:
            return getattr(torch, torch_dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch_dtype value: {torch_dtype}") from exc
    raise TypeError(
        "torch_dtype must be None, 'auto', a torch dtype name string, or an instance of torch.dtype. "
        f"Received {type(torch_dtype).__name__}."
    )
