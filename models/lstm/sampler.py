import random
from torch.utils.data import Sampler, BatchSampler


class FileBatchSampler(BatchSampler):
    """Yield batches grouped by file to maximize per-batch file locality.

    - Shuffles file order each epoch.
    - Optionally shuffles sample order within each file.
    - Forms batches of size `batch_size` from each file before moving on.
    """

    def __init__(self, dataset, batch_size, drop_last=False,
                 shuffle_files=True, shuffle_within=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within

        # Build mapping from file_idx -> list of dataset positions
        self.file_to_positions = {}
        for pos, triple in enumerate(self.dataset.index):
            fidx = triple[0]
            self.file_to_positions.setdefault(fidx, []).append(pos)

    def __iter__(self):
        file_indices = list(self.file_to_positions.keys())
        if self.shuffle_files:
            random.shuffle(file_indices)

        for fidx in file_indices:
            positions = list(self.file_to_positions[fidx])
            if self.shuffle_within:
                random.shuffle(positions)
            # chunk into batches
            for i in range(0, len(positions), self.batch_size):
                batch = positions[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        # Total number of batches across all files
        n = 0
        for fidx, positions in self.file_to_positions.items():
            full, rem = divmod(len(positions), self.batch_size)
            n += full
            if rem and not self.drop_last:
                n += 1
        return n

