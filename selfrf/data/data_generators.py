import os
from pathlib import Path
import platform
import random
from functools import partial
from typing import List, Optional, Callable, Tuple
import json

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from torchsig.utils.writer import DatasetWriter
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.types import ModulatedRFMetadata
from torchsig.datasets.signal_classes import torchsig_signals


class DatasetLoader:
    """Dataset Loader takes on the responsibility of defining how a SignalDataset
    is loaded into memory (usually in parallel)

    Args:
        dataset (SignalDataset): Dataset from which to pull data
        seed (int): seed for the underlying dataset
        num_workers (int, optional): Defaults to os.cpu_count().
        batch_size (int, optional): Defaults to os.cpu_count().
    """

    @staticmethod
    def worker_init_fn(worker_id: int, seed: int):
        seed = seed + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _determine_num_workers(num_workers: Optional[int] = None) -> int:
        if num_workers is None:
            num_workers = os.cpu_count() // 2

        if platform.system() == 'Darwin':  # macOS
            available_cpus = os.cpu_count()
            return min(num_workers, available_cpus) if num_workers else available_cpus // 2

        if hasattr(os, "sched_getaffinity"):
            if len(os.sched_getaffinity(0)) < num_workers:
                try:
                    os.sched_setaffinity(0, range(num_workers))
                    return min(num_workers, len(os.sched_getaffinity(0)))
                except OSError:
                    pass
        return num_workers

    def __init__(
        self,
        dataset: SignalDataset,
        seed: int,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: Optional[int] = 4,
    ) -> None:
        num_workers = self._determine_num_workers(num_workers)
        batch_size = batch_size if batch_size is not None else os.cpu_count() // 2
        prefetch_factor = None if num_workers <= 1 else prefetch_factor

        multiprocessing_context = None if num_workers <= 1 else torch.multiprocessing.get_context(
            "fork")

        self.loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=partial(DatasetLoader.worker_init_fn, seed=seed),
            multiprocessing_context=multiprocessing_context,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        self.length = int(len(dataset) / batch_size)

    def __len__(self):
        return self.length

    def __next__(self):
        data, label = next(iter(self.loader))
        return data, label

    def __iter__(self):
        return iter(self.loader)


class COCODatasetWriter(DatasetWriter):
    """
    Writer to store a COCO-style dataset for RF signals.

    IQ frames are stored as NumPy binary files(.npy) and annotations are collected
    into a single JSON file following the COCO format.

    Directory layout:
        dataset_name/
            data/
                iq_0.npy  # IQ samples in complex64 format
                iq_1.npy
                ...
            annotations/
                instances_dataset_name.json  # COCO-styled format annotations

    RF COCO format attributes:
        - data: list of IQ frame metadata including:
            - id: unique identifier
            - file_name: relative path to .npy file
            - sample_count: number of IQ samples
        - annotations: list of signal annotations including:
            - id: unique identifier
            - image_id: corresponding IQ frame id
            - class_name: signal type(e.g. "ook", "fsk")
            - class_index: numeric class identifier
            - snr: signal-to-noise ratio
            - bandwidth: normalized signal bandwidth
            - center_freq: normalized center frequency
            - start/stop: temporal bounds within frame
        - categories: (optional) signal class definitions

    Methods:
        write(batch): Process a batch of IQ frames and their annotations
        finalize(): Write accumulated annotations to JSON file
        exists(): Check if dataset already exists at target location
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.dataset_name = self.path.name
        self.data_dir = os.path.join(self.path, "data")
        self.annot_dir = os.path.join(self.path, "annotations")

        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.annot_dir, exist_ok=True)

        # Initialize COCO annotation structure.
        self.annotations_dict = {
            "iq_frames": [],
            "annotations": [],
            "categories": []
        }

        self.iq_frame_id = 0
        self.annotation_id = 0

    def exists(self) -> bool:
        """
        Check if the annotation file exists in the specified directory.

        Returns:
            bool: True if the annotation file exists, False otherwise.
        """
        return os.path.exists(os.path.join(self.annot_dir, f"instances_{self.dataset_name}.json"))

    def write(self, batch):
        """
        Writes IQ data and annotations to disk and updates the annotations dictionary.

        This method handles both narrowband and wideband dataset formats. For each IQ frame and its
        corresponding annotations, it:
        1. Saves the IQ data as a .npy file
        2. Creates metadata entries for the IQ frame
        3. Processes the annotations based on their type (tuple for narrowband, list for wideband)

        Args:
            batch (tuple): A tuple containing two elements:
                - datas: List/array of IQ data frames
                - annots: List of annotations for each IQ frame. Each annotation can be either:
                    - tuple: For narrowband dataset
                    - list: For wideband dataset

        Note:
            The method updates the internal annotations dictionary and increments the IQ frame counter.
            Assumes self.data_dir exists and is writable.
        """
        datas, annots = batch

        # loop through each IQ frame and its annotations (can be a tuple or list)
        for data, annotations in zip(datas, annots):
            # Save data as .npy file
            filename = f"iq_{self.iq_frame_id}.npy"
            data_path = os.path.join(self.data_dir, filename)
            np.save(data_path, data)

            # Create data metadata entry
            data_entry = {
                "id": self.iq_frame_id,
                "file_name": filename,
                "sample_count": data.shape[0],
            }
            self.annotations_dict["iq_frames"].append(data_entry)

            # if annotations are a tuple, e.g. narrowband dataset
            if isinstance(annotations, tuple):
                self._handle_narrowband_annotations(annotations, data.shape[0])

            # if frame contains lists of annotations, e.g. wideband dataset
            if isinstance(annotations, list):
                self._handle_wideband_annotations(annotations)

            self.iq_frame_id += 1

    def write_annotations(self):
        """
        Write COCO format annotations dictionary to a JSON file.

        The annotations are written to a file named 'instances_{dataset_name}.json' in the
        annotations directory specified during class initialization.

        The method uses the internal annotations dictionary (self.annotations_dict) which
        should be populated before calling this method.

        Returns:
            None. The method writes to a file and prints a confirmation message.
        """
        annotation_file = os.path.join(
            self.annot_dir, f"instances_{self.dataset_name}.json")
        with open(annotation_file, "w") as f:
            json.dump(self.annotations_dict, f, indent=2)
        print(f"Annotations written to {annotation_file}")

    def _handle_narrowband_annotations(self, annotation: Tuple, num_samples: int):
        modulation_class, snr = annotation
        annotation = ModulatedRFMetadata(
            sample_rate=0.0,
            num_samples=num_samples,
            complex=True,
            lower_freq=-0.25,
            upper_freq=0.25,
            center_freq=0.0,
            bandwidth=0.5,
            start=0.0,
            stop=1.0,
            duration=1.0,
            bits_per_symbol=0.0,
            samples_per_symbol=0.0,
            excess_bandwidth=0.0,
            class_name=torchsig_signals().class_list[modulation_class],
            class_index=modulation_class,
            snr=snr,
        )
        annotation["id"] = self.annotation_id
        annotation["iq_frame_id"] = self.iq_frame_id
        self.annotations_dict["annotations"].append(annotation)
        self.annotation_id += 1

    def _handle_wideband_annotations(self, annotations: List):
        for annotation in annotations:
            annotation["id"] = self.annotation_id
            annotation["iq_frame_id"] = self.iq_frame_id
            self.annotations_dict["annotations"].append(annotation)
            self.annotation_id += 1


class DatasetCreator:
    """Class is whose sole responsibility is to interface a dataset (a generator)
    with a DatasetLoader and a DatasetWriter to produce a static dataset with a
    parallelized generation scheme and some specified storage format.

    Args:
        dataset (SignalDataset): dataset class
        path (str): path to store the static dataset
        loader (DatasetLoader): DatasetLoader.
        writer (DatasetWriter, optional): DatasetWriter. Defaults to LMDBDatasetWriter.
    """

    def __init__(
        self,
        path: str,
        loader: DatasetLoader,
        writer: Optional[DatasetWriter] = None,
    ) -> None:
        self.loader = loader
        self.writer = writer or COCODatasetWriter(path)
        self.path = path

    def create(self):
        """Creates the dataset by iterating over the data loader and writing batches.

        If the dataset already exists at the specified path, the function returns without
        regenerating it. Otherwise, it processes each batch from the loader and writes it
        using the provided writer. If the writer has a 'write_annotations' method, it is
        called after all batches are processed.

        """
        if self.writer.exists():
            print("Dataset already exists in {}. Not regenerating".format(self.path))
            return

        for batch in tqdm.tqdm(self.loader, total=len(self.loader)):
            self.writer.write(batch)

        if hasattr(self.writer, 'write_annotations'):
            self.writer.write_annotations()
