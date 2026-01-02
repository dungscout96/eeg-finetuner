from eegdash import EEGDashDataset
import numpy as np
import torch
import mne
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
import yaml
from sklearn.model_selection import train_test_split as sk_train_test_split

DATASET_REGISTRY = { # example
    "motor": ["ds002718", "ds003646"],
    "sleep": ["ds003654", "ds003645"],
    "cognitive": ["ds003505", "ds003654"],
    "emotion": ["ds003681"],
    "bci": ["DS005863"]
}

class DatasetPipeline:
    def __init__(self, dataset):
        self.config = yaml.safe_load(open(f"eeg_finetuner/preprocessing_configs/{dataset}.yaml"))
        print("initializing", self.config)
        self.dataset = EEGDashDataset(cache_dir="/Users/dtyoung/Documents/Research/LEM-SCCN/standardized-finetuning/eegdash_cache", **self.config["data_query"])

    def process_dataset(self):
        '''
        Docstring for process_dataset
        
        :param self: Description
        '''
        windows_ds = self.preprocess_dataset()
        return self.data_split(windows_ds)

    def preprocess_dataset(self):
        """Apply preprocessing pipeline to an EEG dataset.

        Returns numpy arrays: (n_trials, n_channels, n_times)
        """

        # Define preprocessing steps
        preprocessors = [
            Preprocessor("set_eeg_reference", ref_channels="average", projection=True),
            Preprocessor("resample", sfreq=self.config["sampling_rate"]),
            Preprocessor("filter", l_freq=self.config["low_frequency"], h_freq=self.config["high_frequency"]),
            # Preprocessor(
            #     "pick_channels", ch_names=[ch.lower() for ch in self.config["channels"]], ordered=False
            # ),
        ]

        # Apply preprocessing
        preprocess(self.dataset, preprocessors)

        # Extract windowed trials around stimulus onset
        trial_start = int(self.config["trial_start_offset"] * self.config["sampling_rate"])
        trial_stop = int(self.config["trial_stop_offset"] * self.config["sampling_rate"])
        window_size_samples = int(self.config["trial_duration"] * self.config["sampling_rate"])
        
        # Ensure trial is at least as long as window_size_samples
        trial_stop_offset_samples = max(trial_stop, trial_start + window_size_samples)
        
        # Filter out events that are too close to the edges to avoid out-of-bounds errors
        for ds in self.dataset.datasets:
            raw = ds.raw
            sfreq = raw.info['sfreq']
            
            onsets = raw.annotations.onset
            durations = raw.annotations.duration
            descriptions = raw.annotations.description
            
            # Convert onsets to samples
            onset_samples = np.round(onsets * sfreq).astype(int)
            
            # Create mask for valid events
            mask = (
                (onset_samples + trial_start >= 0) & 
                (onset_samples + trial_stop_offset_samples <= raw.n_times)
            )
            
            if not np.all(mask):
                print(f"Filtering {np.sum(~mask)} out-of-bounds events from {raw.filenames[0] if raw.filenames else 'unknown'}")
                raw.set_annotations(mne.Annotations(
                    onset=onsets[mask],
                    duration=durations[mask],
                    description=descriptions[mask],
                    orig_time=raw.annotations.orig_time
                ))

        print("window_size_samples:", window_size_samples)
        windows_ds = create_windows_from_events(
            self.dataset,
            trial_start_offset_samples=trial_start,
            trial_stop_offset_samples=trial_stop_offset_samples,
            window_size_samples=window_size_samples, # TODO check
            window_stride_samples=2, # TODO check
            drop_last_window=True,
            preload=True,
            drop_bad_windows=True,
        )


        return windows_ds
    
    def data_split(self, windows_ds):
        """Split preprocessed data into training and validation sets.
        Args:
            windows_ds: Braindecode WindowsDataset object
        Returns:
            train_dataloader, val_dataloader, test_dataloader
        """
        train_ratio = self.config.get("train_ratio", 0.8)
        val_ratio = self.config.get("val_ratio", 0.1)
        test_ratio = self.config.get("test_ratio", 0.1)
        
        train_set, val_set, test_set = train_test_split(
            windows_ds, 
            subject_based=True, 
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio
        )
        
        batch_size = self.config.get("batch_size", 32)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set else None
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False) if test_set else None
        
        return train_loader, val_loader, test_loader

class WindowsDatasetWrapper:
    def __init__(self, windows_ds):
        self.windows_ds = windows_ds
        self.window_metadata = windows_ds.get_metadata()

    def __len__(self):
        return len(self.windows_ds)

    def __getitem__(self, idx):
        data, label, *_ = self.windows_ds[idx]
        metadata = self.window_metadata.iloc[idx]
        return data, label, metadata

def train_test_split(windows_ds, 
                    subject_based=True,
                    balanced_classes=True,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split a Braindecode windows dataset into training and testing sets.
        Default to subject-based split.
        If subject_based is False, do random trial-based split.
        If balanced_classes is True, ensure each split has balanced class distribution.

    Args:
        windows_ds: Braindecode WindowsDataset object
        train_ratio: float, ratio of data to use for training
        val_ratio: float, ratio of data to use for validation
        test_ratio: float, ratio of data to use for testing
    Returns:
        train_set, val_set, test_set: Braindecode WindowsDataset objects
    """
    # TODO ensure balanced classes in subject-based split
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    metadata = windows_ds.get_metadata()

    if subject_based:
        if 'subject' not in metadata.columns:
            print("Warning: 'subject' column not found in metadata. Falling back to trial-based split.")
            subject_based = False

    if subject_based:
        subjects = metadata['subject'].unique()
        if len(subjects) < 2:
            print("Warning: Only one subject found. Falling back to trial-based split.")
            subject_based = False
        else:
            train_subs, temp_subs = sk_train_test_split(
                subjects, 
                test_size=(val_ratio + test_ratio), 
                random_state=42
            )
            
            if (val_ratio + test_ratio) > 0 and len(temp_subs) > 0:
                val_size_adj = val_ratio / (val_ratio + test_ratio)
                if val_size_adj >= 1.0:
                    val_subs, test_subs = temp_subs, []
                elif val_size_adj <= 0.0:
                    val_subs, test_subs = [], temp_subs
                elif len(temp_subs) < 2:
                    val_subs, test_subs = temp_subs, []
                else:
                    val_subs, test_subs = sk_train_test_split(
                        temp_subs, 
                        train_size=val_size_adj, 
                        random_state=42
                    )
            else:
                val_subs, test_subs = [], []

            train_idx = np.where(metadata['subject'].isin(train_subs))[0]
            val_idx = np.where(metadata['subject'].isin(val_subs))[0]
            test_idx = np.where(metadata['subject'].isin(test_subs))[0]

    if not subject_based:
        indices = np.arange(len(windows_ds))
        stratify = metadata['target'] if (balanced_classes and 'target' in metadata.columns) else None
        
        train_idx, temp_idx = sk_train_test_split(
            indices, 
            test_size=(val_ratio + test_ratio), 
            stratify=stratify, 
            random_state=42
        )
        
        if (val_ratio + test_ratio) > 0:
            if stratify is not None:
                temp_stratify = stratify.iloc[temp_idx]
            else:
                temp_stratify = None
            
            val_size_adj = val_ratio / (val_ratio + test_ratio)
            if val_size_adj >= 1.0:
                val_idx, test_idx = temp_idx, []
            elif val_size_adj <= 0.0:
                val_idx, test_idx = [], temp_idx
            elif len(temp_idx) < 2:
                val_idx, test_idx = temp_idx, []
            else:
                val_idx, test_idx = sk_train_test_split(
                    temp_idx, 
                    train_size=val_size_adj, 
                    stratify=temp_stratify, 
                    random_state=42
                )
        else:
            val_idx, test_idx = [], []

    # Use braindecode's split if available
    # if hasattr(windows_ds, 'split'):
    #     print("Using braindecode's built-in split method.")
    #     print(train_idx, val_idx, test_idx)
    #     split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    #     # Filter out empty splits
    #     split_dict = {k: v for k, v in split_dict.items() if len(v) > 0}
    #     datasets = windows_ds.split(split_dict)
    #     return datasets.get('train'), datasets.get('valid'), datasets.get('test')
    # else:
    from torch.utils.data import Subset
    return (
        Subset(windows_ds, train_idx) if len(train_idx) > 0 else None,
        Subset(windows_ds, val_idx) if len(val_idx) > 0 else None,
        Subset(windows_ds, test_idx) if len(test_idx) > 0 else None
    )

    