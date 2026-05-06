"""Load prompts from text file."""

import os
from typing import List

from .schemas import Prompt


def load_prompts(path: str, num: int = None) -> List[Prompt]:
    """Load prompts from file. Blank lines and #-prefixed lines skipped.
    
    Returns list[Prompt] with 1-based index.
    If num is set, returns only first num prompts.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    prompts = []
    idx = 1
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prompts.append(Prompt(index=idx, text=stripped))
        idx += 1
    
    if num is not None:
        prompts = prompts[:num]
    
    return prompts