"""Tiled ensemble - prediction merging job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.tiled_ensemble.components.ensemble_tiling import EnsembleTiler
from anomalib.pipelines.tiled_ensemble.components.helper_functions import get_ensemble_tiler
from anomalib.pipelines.tiled_ensemble.components.predictions import EnsemblePredictions, PredictionMergingMechanism
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

logger = logging.getLogger(__name__)


class MergeJob(Job):
    """Job for merging tile-level predictions into image-level predictions.

    Args:
        predictions (EnsemblePredictions): object containing ensemble predictions.
        tiler (EnsembleTiler): ensemble tiler used for untiling.
    """

    name = "pipeline"

    def __init__(self, predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        super().__init__()
        self.predictions = predictions
        self.tiler = tiler

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run merging job that merges all batches of tile-level predictions into image-level predictions.

        Args:
            task_id: not used in this case

        Returns:
            list[Any]: list of merged predictions.
        """
        del task_id  # not needed here

        merger = PredictionMergingMechanism(self.predictions, self.tiler)

        logger.info("Merging predictions.")

        # merge all batches
        merged_predictions = [
            merger.merge_tile_predictions(batch_idx)
            for batch_idx in tqdm(range(merger.num_batches), desc="Prediction merging")
        ]

        return merged_predictions  # noqa: RET504

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing to save in this job."""


class MergeJobGenerator(JobGenerator):
    """Generate MergeJob."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return MergeJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: EnsemblePredictions = None,
    ) -> Generator[Job, None, None]:
        """Return a generator producing a single merging job.

        Args:
            args: tiled ensemble pipeline args.
            prev_stage_result (EnsemblePredictions): ensemble predictions from predict step.

        Returns:
            Generator[Job, None, None]: MergeJob generator
        """
        tiler = get_ensemble_tiler(args)
        yield MergeJob(prev_stage_result, tiler)