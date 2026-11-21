"""Paired, reproducible experiments on the official CIFAR-10 Python release."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from urllib.request import urlopen

import numpy as np
import psutil

SIZES = (1000, 2500, 5000, 10000)
SEEDS = (17, 29, 43, 61, 79)
MODEL_NAMES = ("MLP", "Small CNN", "Medium CNN")
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
ARCHIVE_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
EPOCHS = 3
BATCH_SIZE = 128
LR = 0.002
RSS_INTERVAL_S = 0.005
MIB = 1024 ** 2
METRICS = (
    "fit_time_ms", "predict_time_ms",
    "fit_peak_rss_mib", "predict_peak_rss_mib",
    "fit_rss_baseline_mib", "predict_rss_baseline_mib",
    "fit_cpu_time_s", "predict_cpu_time_s",
    "fit_energy_j", "predict_energy_j",
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
)
KEYS = ("model", "n_train", "n_test", "n_params")


def prepare_data(archive_path, output):
    if not archive_path.is_file():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive_path.with_suffix(".download")
        with urlopen(ARCHIVE_URL, timeout=60) as response, temporary.open("wb") as target:
            for block in iter(lambda: response.read(2 ** 20), b""):
                target.write(block)
        with temporary.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != ARCHIVE_SHA256:
            raise ValueError("Downloaded CIFAR-10 archive failed the expected SHA-256 check.")
        temporary.replace(archive_path)

    with archive_path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
        stream.seek(0)
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != ARCHIVE_MD5 or sha256 != ARCHIVE_SHA256:
        raise ValueError("Supply the official CIFAR-10 Python archive with the documented checksums.")

    # The verified release uses pickled NumPy arrays. Read only the six known
    # members directly, without extracting the archive into the filesystem.
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        train_x = np.lib.format.open_memmap(output / "train_x.npy", mode="w+",
                                           dtype=np.uint8, shape=(50_000, 3, 32, 32))
        train_y = np.empty(50_000, dtype=np.int64)
        for batch in range(1, 6):
            stream = archive.extractfile(f"cifar-10-batches-py/data_batch_{batch}")
            with stream:
                entry = pickle.load(stream, encoding="bytes")
            start = (batch - 1) * 10_000
            train_x[start:start + 10_000] = entry[b"data"].reshape(-1, 3, 32, 32)
            train_y[start:start + 10_000] = entry[b"labels"]
        train_x.flush()

        with archive.extractfile("cifar-10-batches-py/test_batch") as stream:
            test = pickle.load(stream, encoding="bytes")
        np.save(output / "train_y.npy", train_y, allow_pickle=False)
        np.save(output / "test_x.npy", test[b"data"].reshape(-1, 3, 32, 32), allow_pickle=False)
        np.save(output / "test_y.npy", np.asarray(test[b"labels"], dtype=np.int64), allow_pickle=False)
    return sha256


def balanced_indices(labels, seed, size):
    generator = np.random.default_rng(seed)
    # One permutation per class gives nested subsets for each seed.
    per_class = [generator.permutation(np.flatnonzero(labels == label))
                 for label in range(len(CLASSES))]
    return np.concatenate([indices[:size // len(CLASSES)] for indices in per_class])


class ResidentSampler:
    """Sample process RSS outside model code at a nominal 5 ms interval."""

    def __init__(self):
        self.process = psutil.Process()
        self.done = threading.Event()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def capture(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def sample(self):
        while not self.done.wait(RSS_INTERVAL_S):
            self.capture()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.done.set()
        self.thread.join()
        self.capture()


def measure(operation, wattage_per_thread):
    with ResidentSampler() as rss:
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        result = operation()
        time_ms = (time.perf_counter() - start_wall) * 1000
        cpu_time = time.process_time() - start_cpu
    return result, {
        "time_ms": time_ms,
        "cpu_time_s": cpu_time,
        "energy_j": wattage_per_thread * cpu_time,
        "rss_baseline_mib": rss.baseline / MIB,
        "peak_rss_mib": rss.peak / MIB,
    }


def classification_metrics(expected, predicted):
    n_classes = len(CLASSES)
    cm = np.bincount(expected * n_classes + predicted, minlength=n_classes ** 2)
    cm = cm.reshape(n_classes, n_classes)
    correct = cm.diagonal()
    predicted_per_class = cm.sum(axis=0)
    actual_per_class = cm.sum(axis=1)

    def safe_divide(x, y):
        return np.divide(x, y, out=np.zeros_like(x, dtype=np.float64), where=y != 0)

    return {
        "accuracy": float(correct.sum() / cm.sum()),
        "precision_macro": float(safe_divide(correct, predicted_per_class).mean()),
        "recall_macro": float(safe_divide(correct, actual_per_class).mean()),
        "f1_macro": float(safe_divide(2 * correct, predicted_per_class + actual_per_class).mean()),
    }


def evaluate_trial(data, model_name, n_train, seed, threads, base_power_w):
    import torch
    from models import MODELS

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    train_source = np.load(data / "train_x.npy", mmap_mode="r", allow_pickle=False)
    train_labels = np.load(data / "train_y.npy", allow_pickle=False)
    index = balanced_indices(train_labels, seed, n_train)
    train_x = torch.from_numpy(train_source[index])
    train_y = torch.from_numpy(train_labels[index])
    test_x = torch.from_numpy(np.load(data / "test_x.npy", allow_pickle=False))
    test_y = np.load(data / "test_y.npy", allow_pickle=False)

    model = MODELS[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    order = torch.Generator().manual_seed(seed)
    average = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
    deviation = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
    wattage_per_thread = base_power_w / (os.cpu_count() or 1)

    def normalize(images):
        return images.float().div_(255).sub_(average).div_(deviation)

    # Warm forward/backward kernels without applying an optimizer update.
    model(normalize(train_x[:BATCH_SIZE])).sum().backward()
    model.zero_grad(set_to_none=True)

    def train():
        model.train()
        for _ in range(EPOCHS):
            indices = torch.randperm(n_train, generator=order)
            for batch in indices.split(BATCH_SIZE):
                optimizer.zero_grad(set_to_none=True)
                logits = model(normalize(train_x[batch]))
                criterion(logits, train_y[batch]).backward()
                optimizer.step()

    def infer():
        model.eval()
        with torch.inference_mode():
            batches = (model(normalize(batch)).argmax(1)
                       for batch in test_x.split(BATCH_SIZE))
            return torch.cat(list(batches)).numpy()

    _, fit = measure(train, wattage_per_thread)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, train_x, train_y, train_source
    predicted, inference = measure(infer, wattage_per_thread)
    return {
        "model": model_name, "n_train": n_train, "seed": seed,
        "n_test": len(test_y),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        **{f"fit_{key}": value for key, value in fit.items()},
        **{f"predict_{key}": value for key, value in inference.items()},
        **classification_metrics(test_y, predicted),
    }


def write_csv(path, rows, columns):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summaries(runs, sizes):
    rows = []
    for model in MODEL_NAMES:
        for size in sizes:
            group = [run for run in runs if run["model"] == model and run["n_train"] == size]
            if not group:
                continue
            row = {key: group[0][key] for key in KEYS}
            row["n_runs"] = len(group)
            for name in METRICS:
                values = [run[name] for run in group]
                row[f"{name}_mean"] = statistics.mean(values)
                row[f"{name}_sd"] = statistics.stdev(values) if len(values) > 1 else ""
            rows.append(row)
    return rows


def system_metadata(args, sha256):
    from importlib.metadata import version

    # Windows reports the exact product name in the processor registry entry.
    cpu_model = platform.processor()
    if sys.platform == "win32":
        import winreg
        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                cpu_model = winreg.QueryValueEx(handle, "ProcessorNameString")[0]
        except OSError:
            pass
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "dataset": "CIFAR-10 official Python release",
        "archive_url": ARCHIVE_URL, "archive_md5": ARCHIVE_MD5, "archive_sha256": sha256,
        "n_train_source": 50000, "n_test": 10000, "n_classes": 10,
        "classes": CLASSES,
        "n_train": args.sizes, "seeds": args.seeds, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "run_order_seed": 2026, "preprocessing": "fixed per-channel normalization; no augmentation",
        "device": "CPU only", "torch_intra_op_threads": args.threads, "torch_inter_op_threads": 1,
        "host_gpu_adapters": args.gpu_adapter, "gpu_used": False,
        "cpu_model": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": os.cpu_count(),
        "system_ram_mib": psutil.virtual_memory().total / MIB,
        "os": platform.platform(), "python": platform.python_version(),
        "torch": version("torch"), "numpy": np.__version__, "psutil": psutil.__version__,
        "rss_interval_ms": RSS_INTERVAL_S * 1000,
        "energy_method": "Estimated process CPU J = CPU base power W * process CPU s / logical CPU count",
        "energy_power_base_w": args.base_power_w,
        "rss_method": "Process RSS sampled during each phase, including measurement boundaries",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/cifar-10-python.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--base-power-w", type=float, required=True,
                        help="Published processor base power for the machine that executes the trials")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-adapter", action="append", default=[],
                        help="Host GPU name to record in metadata; this option may be repeated")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if not math.isfinite(args.base_power_w) or args.base_power_w <= 0 or args.threads < 1:
        parser.error("Base power and thread count must be positive")
    if any(n < 10 or n > 50000 or n % 10 for n in args.sizes):
        parser.error("Each training size must be a multiple of 10 between 10 and 50000")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Training sizes and seeds must be unique")

    args.output.mkdir(parents=True, exist_ok=True)
    old_summary = args.output / "summary.csv"
    old_summary.unlink(missing_ok=True)
    trials = [(model, size, seed) for model in MODEL_NAMES
              for size in args.sizes for seed in args.seeds]
    random.Random(2026).shuffle(trials)
    completed = []
    columns = (*KEYS, "seed", *METRICS)
    write_csv(args.output / "runs.csv", completed, columns)
    with tempfile.TemporaryDirectory(dir=args.output) as temp:
        data = Path(temp)
        sha256 = prepare_data(args.archive, data)
        (args.output / "environment.json").write_text(
            json.dumps(system_metadata(args, sha256), indent=2) + "\n", encoding="utf-8")
        for number, (model, size, seed) in enumerate(trials, start=1):
            command = [sys.executable, __file__, "--worker", str(data.resolve()), model,
                       str(size), str(seed), str(args.threads), str(args.base_power_w)]
            worker = subprocess.run(command, capture_output=True, text=True)
            if worker.returncode != 0:
                raise RuntimeError(f"{model}, n={size}, seed={seed}:\n{worker.stderr}")
            result = json.loads(worker.stdout)
            completed.append(result)
            write_csv(args.output / "runs.csv", completed, columns)
            print(f"{number:02d}/{len(trials)} {model:10s} n={size:5d} seed={seed:2d} "
                  f"train={result['fit_time_ms']/1000:.2f}s "
                  f"test={result['predict_time_ms']/1000:.2f}s "
                  f"acc={result['accuracy']:.4f}", flush=True)
    rows = summaries(completed, args.sizes)
    write_csv(old_summary, rows, list(rows[0]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, data, model, size, seed, threads, base_power = sys.argv
        print(json.dumps(evaluate_trial(Path(data), model, int(size), int(seed),
                                        int(threads), float(base_power))))
    else:
        main()
