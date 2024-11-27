import torch
import logging
from torch.utils.data import (
    WeightedRandomSampler,
    Sampler,
)
import numpy as np
from typing import List
from speechbrain.dataio.dataset import DynamicItemDataset
from collections import Counter
from scipy.stats import lognorm

logger = logging.getLogger(__name__)

class DynamicBalancingBatchSampler(Sampler):
    """A sampler that combines dynamic batching with balancing by a specified key.

    Arguments
    ---------
    dataset : torch.utils.data.Dataset
        Pytorch Dataset from which elements will be sampled.
    max_batch_length : int
        Upper limit for the sum of the length of examples in a batch.
    num_buckets : int
        Number of discrete buckets used to group examples together.
    length_func : callable
        Function used to get length of each example from the dataset.
    key : str
        The key from which samples will be balanced.
    num_samples : int
        Number of samples to draw. Defaults to the size of the dataset.
    shuffle : bool
        Whether or not shuffle examples between each epoch.
    batch_ordering : string
        If ``random``, batches are randomly permuted; otherwise ``ascending`` or ``descending`` sorted by length.
    max_batch_ex: int
        If set, it limits the maximum number of examples that can be in a batch superseeding max_batch_length
        in instances where the amount of examples will exceed the value specified here.
    bucket_boundaries : list
        Overrides bucket_length_multiplier and left_bucket_length by specifying manually
        the buckets' right boundaries.
    lengths_list: list
        Overrides length_func by passing a list containing the length of each example
        in the dataset. This argument must be set when the dataset is a plain
        Pytorch Dataset object and not a DynamicItemDataset object as length_func
        cannot be used on Pytorch Datasets.
    seed : int
        The base seed to use for the random number generator.
    epoch : int
        The epoch to start at.
    drop_last : bool
         If ``True``, the sampler will drop the last examples which
         have not been grouped.
    verbose: bool
        If ``True``, log also the stats for each batch at the first epoch.
    """

    def __init__(
        self,
        dataset,
        max_batch_length: int,
        num_buckets: int = None,
        length_func=lambda x: x["duration"],
        key: str = None,
        num_samples: int = None,
        shuffle: bool = True,
        batch_ordering: str = "random",
        max_batch_ex: int = None,
        bucket_boundaries: List[int] = [],
        lengths_list: List[int] = None,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ):
        self._dataset = dataset
        self._ex_lengths = {}
        self.verbose = verbose
        self.key = key
        self.num_samples = num_samples or len(dataset)

        # Balancing logic
        if key is not None:
            self.weights = self._compute_weights(dataset, key)
            self.replacement = True
        else:
            self.weights = torch.ones(len(dataset))
            self.replacement = False

        # logger.info("weights:")
        # logger.info(self.weights)

        # We do not put a default on num_buckets to encourage users to play with this parameter
        if num_buckets is None and len(bucket_boundaries) == 0:
            raise RuntimeError(
                "Please specify either num_buckets or bucket boundaries."
                "Check the docs, and/or the tutorial !"
            )

        if lengths_list is not None:
            # take length of examples from this argument and bypass length_key
            for indx in range(len(lengths_list)):
                self._ex_lengths[str(indx)] = lengths_list[indx]
        else:
            # use length func
            if not isinstance(dataset, DynamicItemDataset):
                raise NotImplementedError(
                    "Dataset should be a Speechbrain DynamicItemDataset when using length function"
                )
            for indx in range(len(self._dataset)):
                self._ex_lengths[str(indx)] = length_func(
                    self._dataset.data[self._dataset.data_ids[indx]]
                )

        if len(bucket_boundaries) > 0:
            if not all([x >= 0 for x in bucket_boundaries]):
                raise ValueError(
                    "All elements in bucket boundaries should be non-negative (>= 0)."
                )
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError(
                    "Bucket_boundaries should not contain duplicates."
                )
            np.testing.assert_array_equal(
                np.array(bucket_boundaries),
                np.array(sorted(bucket_boundaries)),
                err_msg="The arg bucket_boundaries should be an ascending sorted list of non negative values values!",
            )
            self._bucket_boundaries = np.array(sorted(bucket_boundaries))
        else:
            # use num_buckets
            self._bucket_boundaries = np.array(
                self._get_boundaries_through_warping(
                    max_batch_length=max_batch_length,
                    num_quantiles=num_buckets,
                )
            )

        self._max_batch_length = max_batch_length
        self._shuffle_ex = shuffle
        self._batch_ordering = batch_ordering
        self._seed = seed
        self._drop_last = drop_last
        if max_batch_ex is None:
            max_batch_ex = np.inf
        self._max_batch_ex = max_batch_ex
        # Calculate bucket lengths - how often does one bucket boundary fit into max_batch_length?
        self._bucket_lens = [
            max(1, int(max_batch_length / self._bucket_boundaries[i]))
            for i in range(len(self._bucket_boundaries))
        ] + [1]
        self._epoch = epoch
        self._generate_batches()

    def _compute_weights(self, dataset, key):
        with dataset.output_keys_as([key]):
            class_ids = [item[key] for item in dataset]
            class_counter = Counter(class_ids)
        weights = 1 / torch.tensor(
            [class_counter[class_id] for class_id in class_ids]
        )
        return weights

    def get_durations(self, batch):
        """Gets durations of the elements in the batch."""
        return [self._ex_lengths[str(idx)] for idx in batch]

    def _get_boundaries_through_warping(
        self, max_batch_length: int, num_quantiles: int,
    ) -> List[int]:

        # NOTE: the following lines do not cover that there is only one example in the dataset
        logger.info("Batch quantisation in latent space")
        # linspace set-up
        num_boundaries = num_quantiles + 1
        # create latent linearly equal spaced buckets
        latent_boundaries = np.linspace(
            1 / num_boundaries, num_quantiles / num_boundaries, num_quantiles,
        )
        # get quantiles using lognormal distribution
        quantiles = lognorm.ppf(latent_boundaries, 1)
        # scale up to to max_batch_length
        bucket_boundaries = quantiles * max_batch_length / quantiles[-1]
        # compute resulting bucket length multipliers
        length_multipliers = [
            bucket_boundaries[x + 1] / bucket_boundaries[x]
            for x in range(num_quantiles - 1)
        ]
        # logging
        logger.debug(
            "Latent bucket boundary - buckets: {} - length multipliers: {}".format(
                list(map("{:.2f}".format, bucket_boundaries)),
                list(map("{:.2f}".format, length_multipliers)),
            )
        )
        return list(sorted(bucket_boundaries))

    def _permute_batches(self):

        if self._batch_ordering == "random":
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(
                len(self._batches), generator=g
            ).tolist()  # type: ignore
            tmp = []
            for idx in sampler:
                tmp.append(self._batches[idx])
            self._batches = tmp

        elif self._batch_ordering == "ascending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
            )
        elif self._batch_ordering == "descending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                reverse=True,
            )
        else:
            raise NotImplementedError

    def _generate_batches(self):
        logger.info("DynamicBalancingBatchSampler: Generating dynamic batches")
        if self._shuffle_ex:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = WeightedRandomSampler(
                self.weights, len(self._dataset), replacement=self.replacement, generator=g
            )
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            # length of pre-sampled audio
            item_len = self._ex_lengths[str(idx)]
            # bucket to fill up most padding
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            stats_tracker[bucket_id]["min"] = min(
                stats_tracker[bucket_id]["min"], item_len
            )
            stats_tracker[bucket_id]["max"] = max(
                stats_tracker[bucket_id]["max"], item_len
            )
            stats_tracker[bucket_id]["tot"] += item_len
            stats_tracker[bucket_id]["n_ex"] += 1
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            if (
                len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                or len(bucket_batches[bucket_id]) >= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

        # Dump remaining batches
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

        self._permute_batches()  # possibly reorder batches

        if self._epoch == 0:  # only log at first epoch
            # frames per batch & their padding remaining
            boundaries = [0] + self._bucket_boundaries.tolist()

            for bucket_indx in range(len(self._bucket_boundaries)):
                try:
                    num_batches = stats_tracker[bucket_indx]["tot"] // (
                        self._max_batch_length
                    )
                    pad_factor = (
                        stats_tracker[bucket_indx]["max"]
                        - stats_tracker[bucket_indx]["min"]
                    ) / (
                        stats_tracker[bucket_indx]["tot"]
                        / stats_tracker[bucket_indx]["n_ex"]
                    )
                except ZeroDivisionError:
                    num_batches = 0
                    pad_factor = 0

                logger.debug(
                    (
                        "DynamicBalancingBatchSampler: Bucket {} with boundary {:.1f}-{:.1f} and "
                        + "batch_size {}: Num Examples {:.1f}, Num Full Batches {:.3f}, Pad Factor {:.3f}."
                    ).format(
                        bucket_indx,
                        boundaries[bucket_indx],
                        boundaries[bucket_indx + 1],
                        self._bucket_lens[bucket_indx],
                        stats_tracker[bucket_indx]["n_ex"],
                        num_batches,
                        pad_factor * 100,
                    )
                )

            if self.verbose:
                batch_stats = {
                    "tot_frames": [],
                    "tot_pad_frames": [],
                    "pad_%": [],
                }
                for batch in self._batches:
                    tot_frames = sum(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    batch_stats["tot_frames"].append(tot_frames)
                    max_frames = max(
                        [self._ex_lengths[str(idx)] for idx in batch]
                    )
                    tot_pad = sum(
                        [
                            max_frames - self._ex_lengths[str(idx)]
                            for idx in batch
                        ]
                    )
                    batch_stats["tot_pad_frames"].append(tot_pad)
                    batch_stats["pad_%"].append(tot_pad / tot_frames * 100)

                padding_details = "Batch {} with {:.1f} frames with {} files - {:.1f} padding, {:.2f} (%) of total."
                padding_details = "DynamicBalancingBatchSampler: " + padding_details
                for i in range(len(self._batches)):
                    logger.debug(
                        padding_details.format(
                            i,
                            batch_stats["tot_frames"][i],
                            len(self._batches[i]),
                            batch_stats["tot_pad_frames"][i],
                            batch_stats["pad_%"][i],
                        )
                    )

    def __iter__(self):
        for batch in self._batches:
            yield batch
        if self._shuffle_ex:  # re-generate examples if ex_ordering == "random"
            self._generate_batches()
        if self._batch_ordering == "random":
            # we randomly permute the batches only --> faster
            self._permute_batches()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        self._generate_batches()

    def __len__(self):
        return len(self._batches)
