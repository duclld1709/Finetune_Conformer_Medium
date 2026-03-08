import os
os.environ["HF_HOME"] = r"D:\hf_datasets"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_datasets\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_datasets\models"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_datasets\hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torchaudio
import soundfile as sf
import io
import os
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from typing import Tuple, Union, Optional
import glob
import json
from pathlib import Path
from tqdm import tqdm

from utils import normalize_transcript


class ViMD(Dataset):
    """
    ViMD Dataset class thiết kế để thay thế LIBRISPEECH.
    Trả về tuple: (waveform, sample_rate, transcript)
    """

    def __init__(
        self,
        dataset_name: str = "ViMD",  
        split: str = "train",       
        target_sample_rate: Optional[int] = 16000,
        min_duration_sec: float = 0.3,
        max_silence_ratio: float = 0.95,
        filter_bad_samples: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.target_sr = target_sample_rate
        self.min_duration_sec = min_duration_sec
        self.max_silence_ratio = max_silence_ratio
        self.filter_bad_samples = filter_bad_samples

        # Load dataset từ Hugging Face (giữ nguyên cấu trúc decode=False để xử lý bytes)
        print(f"Loading ViMD dataset split: {split}...")
#-------------------------------------------------------------KHÚC NÀY ĐÃ SỬA---------------------------------------------------        
        # self._dataset = load_dataset(dataset_name, split=split)
        if split == 'train':
            data_dir = r"D:\hf_datasets\hub\datasets--nguyendv02--ViMD_Dataset\snapshots\3a5b30157034e7eadd5c75fae1a820c6f9383398\data"
            train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))

            self._dataset = load_dataset(
                "parquet",
                data_files=train_files,
                split="train"
            )
        if split == 'test':
            data_dir = r"D:\hf_datasets\hub\datasets--nguyendv02--ViMD_Dataset\snapshots\3a5b30157034e7eadd5c75fae1a820c6f9383398\data"
            train_files = sorted(glob.glob(os.path.join(data_dir, "test-*.parquet")))

            self._dataset = load_dataset(
                "parquet",
                data_files=train_files,
                split="train"
            )
        if split == 'valid':
            data_dir = r"D:\hf_datasets\hub\datasets--nguyendv02--ViMD_Dataset\snapshots\3a5b30157034e7eadd5c75fae1a820c6f9383398\data"
            train_files = sorted(glob.glob(os.path.join(data_dir, "valid-*.parquet")))

            self._dataset = load_dataset(
                "parquet",
                data_files=train_files,
                split="train"
            )
#-------------------------------------------------------------KHÚC NÀY ĐÃ SỬA--------------------------------------------------
        # Lọc các cột cần thiết để tối ưu bộ nhớ nếu cần
        self._dataset = self._dataset.select_columns(["audio", "text"])
        self._dataset = self._dataset.cast_column("audio", Audio(decode=False))

        self._valid_indices = list(range(len(self._dataset)))

        if self.filter_bad_samples:
            self._valid_indices = self._build_valid_indices()

    def _build_valid_indices(self):

        valid = []

        for idx in tqdm(range(len(self._dataset)), desc=f"Filtering {self.split}"):

            try:
                waveform, sr, transcript = self._load_raw(idx)

                if transcript is None:
                    continue

                if len(transcript.strip()) == 0:
                    continue

                duration = waveform.shape[-1] / sr

                if duration < self.min_duration_sec:
                    continue

                silence_ratio = (waveform.abs() < 1e-4).float().mean().item()

                if silence_ratio >= self.max_silence_ratio:
                    continue

                valid.append(idx)

            except Exception:
                continue

        print(f"Kept {len(valid)}/{len(self._dataset)} samples")

        return valid

    def _load_raw(self, idx: int) -> Tuple[torch.Tensor, int, str]:

        item = self._dataset[idx]

        transcript = item.get("text", "")

        audio_bytes = item["audio"]["bytes"]

        with io.BytesIO(audio_bytes) as f:
            array, sr = sf.read(f, dtype="float32")

        waveform = torch.from_numpy(array)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if self.target_sr and sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                self.target_sr
            )
            sr = self.target_sr

        return waveform, sr, transcript

    def __len__(self):

        return len(self._valid_indices)

    def __getitem__(self, n: int):

        idx = self._valid_indices[n]

        waveform, sr, transcript = self._load_raw(idx)

        transcript = normalize_transcript(transcript)

        return waveform, sr, transcript

# Ví dụ sử dụng:
# dataset = ViMD(dataset_name="path/to/vimd", split="train")
# waveform, sr, text = dataset[0]


class CachedFeatureDataset(Dataset):
    """
    Dataset wrapper that loads pre-extracted spectrogram features stored on disk.
    Expect each sample to be saved as a torch file that includes
    spectrogram, label, input_length, label_length, and text.
    """

    _DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "float64": torch.float64,
    }

    def __init__(self, cache_dir: Union[str, Path], split: Optional[str] = "train", dtype: str = "float32"):
        self.cache_root = Path(cache_dir)
        self.split = split
        self.split_dir = self.cache_root / split if split else self.cache_root

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Cached feature directory not found: {self.split_dir}")

        metadata_path = self.split_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.samples = metadata.get("samples", metadata)
        else:
            self.samples = [{"file": p.name} for p in sorted(self.split_dir.glob("*.pt"))]

        if not self.samples:
            raise RuntimeError(f"No cached feature files found in {self.split_dir}")

        # 🔹 FIX: cache full file paths
        self.files = [self.split_dir / sample["file"] for sample in self.samples]

        dtype = dtype.lower()
        if dtype not in self._DTYPE_MAP:
            raise ValueError(f"Unsupported dtype {dtype}, choose from {list(self._DTYPE_MAP.keys())}")

        self.torch_dtype = self._DTYPE_MAP[dtype]

        # Optional metadata for smart batching
        self.frame_lengths = [
            sample.get("frames") or sample.get("input_length")
            for sample in self.samples
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):

        data = torch.load(self.files[index], map_location="cpu")

        spectrogram = data["spectrogram"].to(self.torch_dtype)
        label = data["label"]
        input_length = data["input_length"]
        label_length = data["label_length"]
        text = data["text"]

        return spectrogram, label, input_length, label_length, text
