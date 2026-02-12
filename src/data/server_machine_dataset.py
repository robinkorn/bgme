from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_FILENAME_RE = re.compile(r"^machine-(?P<machine>\d+)-(?P<part>\d+)\.txt$")


@dataclass(frozen=True, slots=True)
class ServerMachineDatasetConfig:
    """Config for `data/ServerMachineDataset`.

    The dataset files are plain text CSV (no header). Each row is one timestamp.
    """

    root: Path = Path("data/ServerMachineDataset")
    delimiter: str = ","
    dtype: type = np.float32


@dataclass(frozen=True, slots=True)
class InterpretationEvent:
    """Human interpretation metadata (range + anomaly type ids)."""

    start: int
    end: int
    anomaly_types: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TestStreamItem:
    """One streamed test sample with optional label."""

    x: np.ndarray  # shape (n_features,)
    y: int | None
    machine: int
    part: int
    t: int


class ServerMachineDataset:
    """Loader for `data/ServerMachineDataset`.

    Provides:
    - Train loading (optionally concatenated across files)
    - Test loading + test labels
    - Test streaming iterator (sample-wise or mini-batch)
    """

    def __init__(self, config: ServerMachineDatasetConfig | None = None):
        self.config = config or ServerMachineDatasetConfig()
        self.root = Path(self.config.root)
        self._train_dir = self.root / "train"
        self._test_dir = self.root / "test"
        self._test_label_dir = self.root / "test_label"
        self._interpretation_dir = self.root / "interpretation_label"

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root.resolve()} (expected data/ServerMachineDataset)"
            )

    # discovery
    def list_files(self, split: str) -> list[Path]:
        """List dataset files for a split: 'train' | 'test' | 'test_label' | 'interpretation_label'."""
        split_dir = self._split_dir(split)
        files = [p for p in split_dir.iterdir() if p.is_file() and _FILENAME_RE.match(p.name)]
        files.sort(key=lambda p: self._parse_filename(p.name))
        return files

    def list_ids(self, split: str) -> list[tuple[int, int]]:
        """Return sorted list of (machine, part) pairs present in a split."""
        return [self._parse_filename(p.name) for p in self.list_files(split)]

    # loading
    def load_train(
        self,
        machines=None,
        parts=None,
        concat: bool = True,
    ):
        """Load training data.

        If `concat=True`, returns one array (N, D).
        If `concat=False`, returns dict[(machine, part)] -> array.
        """

        return self._load_split_matrix("train", machines=machines, parts=parts, concat=concat)

    def load_test(self, machine: int, part: int) -> np.ndarray:
        """Load one test file as (T, D)."""
        path = self._file_path(self._test_dir, machine, part)
        return self._load_matrix(path)

    def load_test_labels(self, machine: int, part: int) -> np.ndarray:
        """Load one test label file as (T,) int array (0/1)."""
        path = self._file_path(self._test_label_dir, machine, part)
        y = np.loadtxt(path, dtype=np.int8)
        if y.ndim != 1:
            y = np.asarray(y).reshape(-1)
        return y

    def load_interpretation_events(self, machine: int, part: int) -> list[InterpretationEvent]:
        """Parse interpretation_label file into a list of events."""
        path = self._file_path(self._interpretation_dir, machine, part)
        events: list[InterpretationEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # Example: 15849-16368:1,9,10,12,13,14,15
            try:
                range_part, types_part = line.split(":", 1)
                start_s, end_s = range_part.split("-", 1)
                types = tuple(int(x) for x in types_part.split(",") if x.strip())
                events.append(InterpretationEvent(int(start_s), int(end_s), types))
            except Exception as e:
                raise ValueError(f"Failed to parse interpretation label line: {line!r} in {path}") from e
        return events

    # streaming
    def iter_test_stream(
        self,
        machine: int,
        part: int,
        batch_size: int = 1,
        with_labels: bool = True,
    ):
        """Stream test data in order.

        Yields (X, y, meta):
        - X: (B, D)
        - y: (B,) or None
        - meta: {machine, part, t_start, t_end}
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        X = self.load_test(machine, part)
        y = self.load_test_labels(machine, part) if with_labels else None

        if y is not None and len(y) != len(X):
            raise ValueError(
                f"Length mismatch for machine-{machine}-{part}: X has {len(X)}, y has {len(y)}"
            )

        T = len(X)
        for t0 in range(0, T, batch_size):
            t1 = min(t0 + batch_size, T)
            xb = X[t0:t1]
            yb = y[t0:t1] if y is not None else None
            meta = {"machine": machine, "part": part, "t_start": t0, "t_end": t1}
            yield xb, yb, meta

    def iter_test_stream_all(
        self,
        machines=None,
        parts=None,
        batch_size: int = 1,
        with_labels: bool = True,
    ):
        """Stream multiple test files sequentially in filename order."""

        for machine, part in self._filtered_ids("test", machines=machines, parts=parts):
            yield from self.iter_test_stream(machine, part, batch_size=batch_size, with_labels=with_labels)

    def iter_test_samples(
        self,
        machine: int,
        part: int,
        with_labels: bool = True,
    ):
        """Yield one test sample at a time."""

        X = self.load_test(machine, part)
        y = self.load_test_labels(machine, part) if with_labels else None
        if y is not None and len(y) != len(X):
            raise ValueError(
                f"Length mismatch for machine-{machine}-{part}: X has {len(X)}, y has {len(y)}"
            )
        for t, x in enumerate(X):
            yield TestStreamItem(
                x=np.asarray(x, dtype=self.config.dtype),
                y=int(y[t]) if y is not None else None,
                machine=int(machine),
                part=int(part),
                t=int(t),
            )

    # internals
    def _split_dir(self, split: str) -> Path:
        split = split.strip().lower()
        if split == "train":
            return self._train_dir
        if split == "test":
            return self._test_dir
        if split == "test_label":
            return self._test_label_dir
        if split == "interpretation_label":
            return self._interpretation_dir
        raise ValueError("split must be one of: train, test, test_label, interpretation_label")

    def _parse_filename(self, name: str) -> tuple[int, int]:
        m = _FILENAME_RE.match(name)
        if not m:
            raise ValueError(f"Unexpected filename: {name}")
        return int(m.group("machine")), int(m.group("part"))

    def _file_path(self, split_dir: Path, machine: int, part: int) -> Path:
        path = split_dir / f"machine-{int(machine)}-{int(part)}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.resolve()}")
        return path

    def _filtered_ids(
        self,
        split: str,
        machines,
        parts,
    ) -> list[tuple[int, int]]:
        ids = self.list_ids(split)
        if machines is not None:
            mset = set(int(m) for m in machines)
            ids = [mp for mp in ids if mp[0] in mset]
        if parts is not None:
            pset = set(int(p) for p in parts)
            ids = [mp for mp in ids if mp[1] in pset]
        return ids

    def _load_split_matrix(
        self,
        split: str,
        machines,
        parts,
        concat: bool,
    ):
        out: dict[tuple[int, int], np.ndarray] = {}
        for machine, part in self._filtered_ids(split, machines=machines, parts=parts):
            split_dir = self._split_dir(split)
            path = self._file_path(split_dir, machine, part)
            out[(machine, part)] = self._load_matrix(path)

        if not out:
            raise ValueError(f"No files matched for split={split!r} machines={machines} parts={parts}")

        if not concat:
            return out

        mats = [out[k] for k in sorted(out.keys())]
        return np.vstack(mats)

    def _load_matrix(self, path: Path) -> np.ndarray:
        X = np.loadtxt(path, delimiter=self.config.delimiter, dtype=self.config.dtype)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X
