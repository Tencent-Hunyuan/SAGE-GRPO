import warnings
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler
from typing import Iterator, Sized
from torch.utils.data import Sampler


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, start_index: int = 0) -> None:
        self.data_source = data_source
        self._start_index = start_index
        self.recompute_sizes()

    def recompute_sizes(self):
        self.num_samples = len(self.data_source) - self._start_index
        self.total_size = self.num_samples

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if self._start_index != value:
            self._start_index = value
            self.recompute_sizes()

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._start_index, len(self.data_source)))

    def __len__(self) -> int:
        return self.num_samples


class RepeatRandomDistributedSampler(TorchDistributedSampler):
    """
    A distributed sampler that repeats the indices of a dataset in a structured manner and 
    supports moving the start index of the dataset to a specific position.
    This feature is useful when we want to resume training from a specific position in the dataset.
    The (`num_replicas` * `batch_size`) must be exactly divisible by `mini_repeat_count`.

    Example:
    ```python
    >>> sampler = RepeatRandomDistributedSampler(
            ["a", "b", "c", "d", "e", "f", "g"], num_replicas=4, mini_repeat_count=2, batch_size=3, repeat_count=4,
        )
    >>> list(sampler)
    <-rank0-> | <-rank1-> | <-rank2-> | <-rank3->
    [4, 4, 3,   [3, 0, 0,   [1, 1, 2,   [2, 6, 6    --> total_batch_size := 6
     4, 4, 3,    3, 0, 0,    1, 1, 2,    2, 6, 6,
     4, 4, 3,    3, 0, 0,    1, 1, 2,    2, 6, 6,
     4, 4, 3]    3, 0, 0]    1, 1, 2]    2, 6, 6]
    ```
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=True,
                 start_index=0, raise_start_index_expired=False, batch_size=1, repeat_count=1, mini_repeat_count=1):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            raise RuntimeError('Using `dist.get_world_size()` is dangerous.')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            raise RuntimeError('Using `dist.get_rank()` is dangerous.')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.repeat_count = repeat_count
        self.mini_repeat_count = mini_repeat_count
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self._start_index = start_index
        self.shuffle = shuffle
        self.seed = seed

        # Used with MultiResolutionBucketIndexV2. When batch_size > 1, the interleaved indices will become
        # interleaved batches.
        self.batch_size = batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {self.batch_size}.")

        assert (
            (self.batch_size * self.num_replicas) % self.mini_repeat_count == 0
        ), f"""Invalid configuration: (num_replicas ({self.num_replicas}) x batch_size ({self.batch_size}))
        must be exactly divisible by mini_repeat_count ({self.mini_repeat_count})"""

        # Total batch_size for num_replicas processes.
        self.total_batch_size = (self.batch_size * self.num_replicas) // self.mini_repeat_count
        assert drop_last is True, "When use 'RepeatRandomDistributedSampler', drop_last should be True."

        # Define a flag to indicate whether the start_index is expired. The start_index is expired if and only
        # if the start_index is not 0 and the first epoch (i.e., the first time to create the iterator in the
        # current run) is finished. This flag will warn (or raise error) the user if the second epoch is started
        # but the start_index is not reset to 0.
        self.start_index_expired = False
        self.raise_start_index_expired = raise_start_index_expired

        self.recompute_sizes()

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if value % self.total_batch_size != 0:
            new_value = value // self.total_batch_size * self.total_batch_size
            message = f"start_index should be divisible by total_batch_size({self.total_batch_size}). Reset start_index from {value} to {new_value}."
            try:
                from loguru import logger
                logger.warning(message)
            except ImportError:
                warnings.warn(message)
            self._start_index = new_value
        else:
            self._start_index = value
        self.recompute_sizes()

    def recompute_sizes(self):
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # Otherwise, split to nearest available length that is evenly divisible.
        # This is to ensure each rank receives the same amount of data when
        # using this Sampler.
        self.total_size = (len(self.dataset) - self._start_index) // self.total_batch_size * self.total_batch_size
        self.num_samples = (len(self.dataset) - self._start_index) // self.total_batch_size * self.batch_size * self.repeat_count

    def blockwise_sample_iter(self, indices):
        # This function is used to create a blockwise iterator for the indices.
        # The indices are divided into blocks with size of self.batch_size.
        # The indices in the same block are from the same rank.

        total_len = len(indices)
        step = self.num_replicas * self.batch_size
        current_block_start = self.rank * self.batch_size
        while current_block_start < total_len:
            current_block_end = min(current_block_start + self.batch_size, total_len)

            # Get the indices in the current block
            for i in range(current_block_start, current_block_end):
                yield indices[i]

            # Move to the next block
            current_block_start += step
    
    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        if self.start_index_expired and self._start_index != 0:
            message = "The start_index is expired. Please reset the start_index to 0 before starting next epoch."
            if self.raise_start_index_expired:
                raise ValueError(message)
            else:
                try:
                    from loguru import logger
                    logger.warning(message)
                except ImportError:
                    warnings.warn(message)

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            indices = indices[self._start_index:]
        else:
            indices = list(range(self._start_index, len(self.dataset)))  # type: ignore[arg-type]

        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (total_batch_size = 3)
        indices = [indices[i : i + self.total_batch_size] for i in range(0, len(indices), self.total_batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indices = [chunk for chunk in indices if len(chunk) == self.total_batch_size]

        repeated_indices = []
        for chunk in indices:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        repeated_indices.append(index)
        indices = repeated_indices

        # subsample with start_index
        if self.batch_size == 1:
            indices = indices[self.rank : len(indices) : self.num_replicas]
            assert len(indices) == self.num_samples

            self.start_index_expired = True
            print(f"Iterator of DistributedSamplerWithStartIndex created.")
            return iter(indices)

        else:
            self.start_index_expired = True
            print(f"Iterator of DistributedSamplerWithStartIndex created.")
            return self.blockwise_sample_iter(indices)