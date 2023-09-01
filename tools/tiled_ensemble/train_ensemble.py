"""Anomalib Training Script for ensemble of models.

This script reads the name of the model or config file from command
line, trains and evaluates ensemble of anomaly models to get quantitative and qualitative
results.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

sys.path.append(".")

import logging
import time
import warnings
from argparse import ArgumentParser, Namespace
from itertools import product

from pytorch_lightning import Trainer, seed_everything
from tools.tiled_ensemble import (
    EnsembleTiler,
    get_ensemble_callbacks,
    get_ensemble_datamodule,
    get_prediction_storage,
    log_metrics,
    post_process,
    prepare_ensemble_configurable_parameters,
)

from anomalib.config import get_configurable_parameters
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.models import get_model
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--ensemble_config", type=str, required=True, help="Path to an ensemble configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def train(args: Namespace):
    """Train a tiled ensemble.

    Images are split to tiles and separate model is trained for each tile location.

    Args:
        args (Namespace): The arguments from the command line.
    """
    start_time = time.time()
    total_test_duration = 0

    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.model_config)
    # update and prepare config for ensemble
    config = prepare_ensemble_configurable_parameters(ens_config_path=args.ensemble_config, config=config)

    experiment_logger = get_experiment_logger(config)

    # instantiate tiler used for splitting images into tiles
    tiler = EnsembleTiler(
        tile_size=config.ensemble.tiling.tile_size,
        stride=config.ensemble.tiling.stride,
        image_size=config.dataset.image_size,
    )
    # prepare datamodule with tiling mechanism
    datamodule = get_ensemble_datamodule(config, tiler)
    # prepare storage objects that handle storage of tiled predictions
    ensemble_predictions, validation_predictions = get_prediction_storage(config)

    logger.info(
        "Tiled ensemble training started. Separate models will be trained for %d tile locations.",
        tiler.num_tiles,
    )
    # go over all tile positions and train
    for i, tile_index in enumerate(product(range(tiler.num_patches_h), range(tiler.num_patches_w))):
        logger.info(
            "Start of procedure for tile at position %s. Progress: %d/%d positions.", tile_index, i, tiler.num_tiles
        )

        if config.project.get("seed") is not None:
            seed_everything(config.project.seed)

        # configure callbacks for ensemble
        ensemble_callbacks = get_ensemble_callbacks(config, tile_index)
        # set tile position inside dataloader
        datamodule.collate_fn.tile_index = tile_index

        model = get_model(config)

        trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=ensemble_callbacks)
        logger.info("Training the model.")
        trainer.fit(model=model, datamodule=datamodule)

        if config.dataset.test_split_mode == TestSplitMode.NONE:
            logger.info("No test set provided. Skipping prediction on test data.")
        else:
            logger.info("Predicting on predict test data.")

            test_start_time = time.time()

            current_predictions = trainer.predict(
                model=model, dataloaders=datamodule.test_dataloader(), ckpt_path="best"
            )
            ensemble_predictions.add_tile_prediction(tile_index, current_predictions)

            test_predict_duration = time.time() - test_start_time
            # accumulate test time over all models
            total_test_duration += test_predict_duration

        # if val data == test data, or no validation data is provided - don't compute
        if config.dataset.val_split_mode not in [ValSplitMode.SAME_AS_TEST, ValSplitMode.NONE]:
            logger.info("Predicting on validation data.")
            current_val_predictions = trainer.predict(
                model=model, dataloaders=datamodule.val_dataloader(), ckpt_path="best"
            )
            validation_predictions.add_tile_prediction(tile_index, current_val_predictions)

    # postprocess, visualization and metric pipeline
    results, test_pipe_start = post_process(
        config=config,
        tiler=tiler,
        ensemble_predictions=ensemble_predictions,
        validation_predictions=validation_predictions,
    )
    log_metrics(results)

    end_time = time.time()

    total_duration = end_time - start_time
    # get duration from start time returned (this way we don't count validation pipeline into test)
    test_pipe_duration = end_time - test_pipe_start
    total_test_duration += test_pipe_duration

    results["total_time"] = total_duration
    results["test_time"] = total_test_duration
    return results


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
