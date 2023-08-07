"""Anomalib Training Script for ensemble of models.

This script reads the name of the model or config file from command
line, trains and evaluates ensemble of anomaly models to get quantitative and qualitative
results.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
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
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--ens_config", type=str, required=True, help="Path to an ensemble configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def train(args: Namespace):
    """Train an anomaly model.

    Args:
        args (Namespace): The arguments from the command line.
    """

    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    # update and prepare config for ensemble
    config = prepare_ensemble_configurable_parameters(ens_config_path=args.ens_config, config=config)

    experiment_logger = get_experiment_logger(config)

    tiler = EnsembleTiler(config)
    datamodule = get_ensemble_datamodule(config, tiler)
    ensemble_predictions, validation_predictions = get_prediction_storage(config)

    # go over all tile positions and train
    for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
        logger.info("Start of procedure for tile %s", tile_index)

        if config.project.get("seed") is not None:
            seed_everything(config.project.seed)

        # configure callbacks for ensemble
        ensemble_callbacks = get_ensemble_callbacks(config, tile_index)
        # set tile position inside dataloader
        datamodule.custom_collate_fn.tile_index = tile_index

        model = get_model(config)

        trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=ensemble_callbacks)
        logger.info("Training the model.")
        trainer.fit(model=model, datamodule=datamodule)

        logger.info("Predicting on predict (test) data.")
        current_predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path="best")
        ensemble_predictions.add_tile_prediction(tile_index, current_predictions)

        # if val data == test data, don't recompute
        if config.dataset.val_split_mode != "same_as_test":
            logger.info("Predicting on validation data.")
            current_val_predictions = trainer.predict(
                model=model, dataloaders=datamodule.val_dataloader(), ckpt_path="best"
            )
            validation_predictions.add_tile_prediction(tile_index, current_val_predictions)

    # postprocess, visualization and metric pipeline
    computed_metrics = post_process(
        config=config,
        tiler=tiler,
        ensemble_predictions=ensemble_predictions,
        validation_predictions=validation_predictions,
    )
    log_metrics(computed_metrics)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
