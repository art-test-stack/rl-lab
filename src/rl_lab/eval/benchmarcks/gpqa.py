"""
GPQA evaluation.
https://huggingface.co/datasets/Idavidrein/gpqa
"""

from datasets import load_dataset
from .common import Task

class GPQA(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)

        self.ds = load_dataset("Idavidrein/gpqa", subset, split=split).shuffle(seed=42)

        @property
        def eval_type(self):
            return 'generative'