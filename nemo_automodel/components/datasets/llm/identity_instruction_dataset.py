# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from datasets import load_dataset

from nemo_automodel.components.datasets.utils import SFTSingleTurnPreprocessor


class IdentityInstructionDataset:
    """A dataset wrapper for identity instruction data, tailored for single-turn supervised fine-tuning (SFT).

    This class loads and preprocesses instruction datasets using a tokenizer and a custom preprocessing
    pipeline for language model fine-tuning. The dataset consists of question-answer pairs.

    Attributes:
        dataset (Dataset): The processed dataset ready for model training or evaluation.
    """

    def __init__(
        self,
        path_or_dataset,
        tokenizer,
        split="train",
        num_samples_limit=None,
        trust_remote_code=True,
        pad_to_max_length=True,
    ):
        """Initialize the IdentityInstructionDataset wrapper.

        Args:
            path_or_dataset (str or Dataset): Path to the dataset (JSONL file) or a HuggingFace Dataset object.
            tokenizer (PreTrainedTokenizer): The tokenizer used to process text.
            split (str, optional): Dataset split to use (e.g., 'train', 'validation'). Defaults to 'train'.
                For local JSONL files, this is typically ignored.
            num_samples_limit (int, optional): Maximum number of samples to load. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
            pad_to_max_length (bool, optional): Whether to pad sequences to max length in the dataset.
                If False, sequences will have variable lengths and padding will be handled by the collate function.
                Defaults to True.

        Notes:
            If num_samples_limit is an integer, it limits the dataset size using slicing.
        """
        # Load dataset from JSONL file or HF dataset
        # Check if it's a local file path
        is_local_file = (
            isinstance(path_or_dataset, str)
            and not path_or_dataset.startswith(("http://", "https://"))
            and Path(path_or_dataset).exists()
        )
        
        if is_local_file:
            # Local JSONL file
            raw_datasets = load_dataset("json", data_files=path_or_dataset, split="train")
            # Apply sample limit for JSONL files
            if isinstance(num_samples_limit, int):
                raw_datasets = raw_datasets.select(range(min(num_samples_limit, len(raw_datasets))))
        else:
            # HF dataset or URL
            if isinstance(num_samples_limit, int):
                split = f"{split}[:{num_samples_limit}]"
            raw_datasets = load_dataset(path_or_dataset, split=split, trust_remote_code=trust_remote_code)
        
        processor = SFTSingleTurnPreprocessor(tokenizer)
        processor.pad_to_max_length = pad_to_max_length
        self.dataset = processor.process(raw_datasets, self)

    def get_context(self, examples):
        """Extracts the context part of each example (question).

        Args:
            examples (dict): A dictionary containing example data with a "question" key.

        Returns:
            list[str]: List of question strings (used as context).
        """
        # Return question as context, or empty string if not present
        return examples.get("question", [""] * len(examples.get("answer", [])))

    def get_target(self, examples):
        """Extracts the target part of each example (answer).

        Args:
            examples (dict): A dictionary with "answer" key.

        Returns:
            list[str]: The answer strings.
        """
        return examples["answer"]

    def __getitem__(self, index):
        """Get a processed example by index.

        Args:
            index (int): Index of the example.

        Returns:
            dict: A tokenized and preprocessed example.
        """
        ans = self.dataset[index]
        return ans

    def __len__(self):
        """Get the number of examples in the dataset.

        Returns:
            int: Length of the processed dataset.
        """
        return len(self.dataset)
